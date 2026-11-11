"""Optional local-only HF/PEFT trainer. See docs/MODELS.md before using it."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="existing local model directory")
    p.add_argument("--data", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--identity", required=True)
    p.add_argument("--output", required=True, help="new directory")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--rank-budget", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--task-weight", type=float, default=1.0)
    p.add_argument("--target-modules", default="q_proj,v_proj")
    p.add_argument("--inhibition", help="JSON object domain -> positive lambda")
    args = p.parse_args()
    if not Path(args.model).is_dir() or Path(args.output).exists():
        p.error("model must be a local directory and output must not exist")
    if args.max_steps < 1 or args.max_length < 8:
        p.error("positive max-steps and max-length >= 8 required")
    data, anchors, nodes = [read_jsonl(x) for x in (args.data, args.anchors, args.identity)]
    domains = sorted({x["domain"] for x in data})
    if len({a["id"] for a in anchors}) != len(anchors) or len({n["id"] for n in nodes}) != len(nodes):
        p.error("duplicate anchor/identity IDs")
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from calms_savant.identity import IdentityNode, IdentityGraph
    from calms_savant.lora import allocate_ranks
    from calms_savant.policy import RiskConfig
    from calms_savant.engine import atomic_json
    from calms_savant.integrations.torch_guard import TorchLoRAGuard
    torch.manual_seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            p.error("tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    inhibition = json.loads(Path(args.inhibition).read_text()) if args.inhibition else {d: 1.0 for d in domains}
    if set(inhibition) != set(domains):
        p.error("inhibition keys must equal training domains")
    ranks = allocate_ranks(inhibition, base_rank=4, budget=args.rank_budget, max_rank=8)
    def adapter_config(domain):
        return LoraConfig(task_type="CAUSAL_LM", r=ranks[domain], lora_alpha=ranks[domain],
                          lora_dropout=0.0, bias="none", target_modules=args.target_modules.split(","))
    model = get_peft_model(base, adapter_config(domains[0]), adapter_name=domains[0])
    for domain in domains[1:]:
        model.add_adapter(domain, adapter_config(domain))
    model.to(args.device)
    model.config.use_cache = False
    def tokens(texts):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                         max_length=args.max_length).to(args.device)
    def encode(text):
        training = model.training
        model.eval()
        try:
            with torch.no_grad():
                batch = tokens([text])
                hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                return pooled.detach().float().cpu().numpy()
        finally:
            model.train(training)
    def probe():
        active = model.active_adapter
        result = {}
        try:
            for domain in domains:
                model.set_adapter(domain)
                for anchor in anchors:
                    result[domain+"::"+anchor["id"]] = encode(anchor["text"])
            return result
        finally:
            model.set_adapter(active)
    model.set_adapter(domains[0])
    prior_nodes = []
    for node in nodes:
        domain = node.get("domain", "*")
        if domain != "*" and domain not in domains:
            p.error("identity domain is absent from training data")
        model.set_adapter(domains[0] if domain == "*" else domain)
        prior_nodes.append(IdentityNode(node["id"], encode(node["text"])[0], domain))
    graph = IdentityGraph(prior_nodes)
    guard = TorchLoRAGuard(model, graph, probe, learning_rate=args.learning_rate,
        config=replace(RiskConfig(checkpoint_capacity=2), task_weight=args.task_weight))
    logs = []
    for index in range(args.max_steps):
        row = data[index % len(data)]
        model.set_adapter(row["domain"])
        representation = encode(row["text"])
        batch = tokens([row["text"]])
        if int(batch["attention_mask"].sum()) < 2:
            p.error("training texts must contain at least two tokens")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        report = guard.step(lambda: model(**batch, labels=labels).loss,
                            representation, domain=row["domain"], inhibition=inhibition[row["domain"]])
        logs.append({"record_index": index % len(data), "domain": row["domain"], **report})
        if guard.frozen:
            break
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(out/"adapters", safe_serialization=True, selected_adapters=domains)
    tokenizer.save_pretrained(out/"tokenizer")
    atomic_json(out/"training_log.json", {"implementation": "optional_hf_peft_direct_sgd",
        "model_local_path": str(Path(args.model).resolve()), "ranks": ranks, "frozen": guard.frozen,
        "config": vars(args), "torch": torch.__version__, "records": logs,
        "notes": ["No PSR/FRS evaluation performed.", "One active semantic adapter at a time.",
                  "Only final/restored adapters are saved; no exact training-resume state."]})
    print(json.dumps({"output": str(out.resolve()), "frozen": guard.frozen, "steps": len(logs)}))


if __name__ == "__main__":
    main()
