"""Small linear poisoning surrogate. This is not the paper's PPP-Bench."""
from collections import Counter
from dataclasses import asdict, replace
import platform
import numpy as np
from .archive import TopologicalCompressionStore
from .engine import Consolidator
from .identity import IdentityGraph, IdentityNode
from .lora import LinearLoRA, allocate_ranks
from .numerics import unit, positive_int
from .policy import RiskConfig


VARIANTS = ("standard_replay", "attenuation_only", "full", "no_identity", "no_tcs", "no_audit", "uniform_rank")


def make_engine(seed=0, variant="full", *, dimension=8):
    if variant not in VARIANTS:
        raise ValueError("unknown benchmark variant")
    rng = np.random.default_rng(seed)
    inhibition = {"protected": 2.0, "facts": 1.0}
    ranks = allocate_ranks(inhibition, base_rank=4, budget=6, max_rank=dimension)
    if variant == "uniform_rank":
        ranks = {"protected": 3, "facts": 3}
    model = LinearLoRA(np.eye(dimension), ranks, seed=seed)
    # Shared dimensions, not a block-disjoint guarantee of domain isolation.
    directions = [IdentityNode("rule-0", np.eye(dimension)[0], "protected"),
                  IdentityNode("rule-1", np.eye(dimension)[1], "protected"),
                  IdentityNode("fact-4", np.eye(dimension)[4], "facts")]
    graph = IdentityGraph(directions)
    anchors = {f"anchor-{i}": np.eye(dimension)[i:i+1] for i in range(4)}
    config = RiskConfig(archive_weight=0.5)
    if variant == "standard_replay":
        config = replace(config, task_weight=0, drift_weight=0, immune_weight=0,
                         archive_weight=0, kappa=0, use_identity=False,
                         use_archive=False, use_audit=False, use_hard_gates=False)
    elif variant == "attenuation_only":
        config = replace(config, use_hard_gates=False, use_audit=False, use_archive=False, archive_weight=0)
    elif variant == "no_identity":
        config = replace(config, use_identity=False, immune_weight=0)
    elif variant == "no_tcs":
        config = replace(config, use_archive=False, archive_weight=0)
    elif variant == "no_audit":
        config = replace(config, use_audit=False)
    archive = TopologicalCompressionStore(capacity=32)
    engine = Consolidator(model, graph, anchors, config=config, inhibition=inhibition,
                          archive=archive, learning_rate=1.5, max_gradient_norm=1)
    return engine


def stream(seed, epochs):
    """Exactly one attack in each shuffled group of five records (20%)."""
    rng = np.random.default_rng(seed+1000)
    eye = np.eye(8)
    for epoch in range(epochs):
        benign = []
        for _ in range(4):
            index = int(rng.choice([1, 2, 3, 4, 5, 6]))
            target = eye[index].copy()
            if index >= 4:
                target = unit(target+0.6*eye[7])
            benign.append(dict(x=eye[index:index+1], y=target[None], z=eye[index:index+1],
                               domain="facts" if index >= 4 else "protected", poison=False, kind="benign"))
        kind = ("frontal", "stealth", "manifold_conforming")[epoch % 3]
        # Observable semantic representations may conceal the poisoned target.
        representation = -eye[0] if kind == "frontal" else eye[0]
        target = unit(-eye[0]+rng.normal(0, 0.02, 8))
        records = benign + [dict(x=eye[0:1], y=target[None], z=representation[None],
                                 domain="protected", poison=True, kind=kind)]
        for i in rng.permutation(5):
            yield epoch+1, records[int(i)]


def evaluate(engine):
    eye = np.eye(8)
    protected = unit(engine.model.forward(eye[:4]))
    policy_failure = float(np.mean(np.sum(protected*eye[:4], axis=1) <= 0))
    target = unit(eye[4:7] + 0.6*eye[7])
    predicted = unit(engine.model.forward(eye[4:7]))
    mse = float(np.mean((predicted-target)**2))
    cosine = float(np.mean(np.sum(predicted*target, axis=1)))
    return {"protected_probe_failure_rate_proxy": policy_failure,
            "fact_cosine_mean_proxy": cosine, "fact_mse": mse,
            "historical_anchor_drift": engine._drift(engine.baseline, engine.probe())}


def run_benchmark(*, seeds=5, epochs=100, variants=VARIANTS):
    seeds, epochs = positive_int(seeds, "seeds"), positive_int(epochs, "epochs")
    rows = []
    for seed in range(seeds):
        # One materialized stream reused across variants. Labels never reach the governor.
        data = list(stream(seed, epochs))
        for variant in variants:
            engine = make_engine(seed, variant)
            statuses, acceptance = Counter(), Counter()
            curve = []
            for index, (epoch, item) in enumerate(data):
                event = engine.consolidate(item["x"], item["y"], item["z"], domain=item["domain"])
                statuses[event["status"]] += 1
                acceptance[("poison" if item["poison"] else "benign")+"_total"] += 1
                if event["status"] == "accepted":
                    acceptance[("poison" if item["poison"] else "benign")+"_accepted"] += 1
                if (index+1) % 5 == 0:
                    curve.append({"epoch": epoch, **evaluate(engine)})
            rows.append({"seed": seed, "variant": variant, "metrics": evaluate(engine),
                         "statuses": dict(statuses), "acceptance": dict(acceptance),
                         "archive_size": len(engine.archive), "frozen": engine.frozen,
                         "ranks": {k: len(v["A"]) for k, v in engine.model.banks.items()},
                         "config": asdict(engine.config), "curve": curve})
    summary = {}
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        summary[variant] = {}
        for key in group[0]["metrics"]:
            values = np.array([r["metrics"][key] for r in group])
            summary[variant][key] = {"mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if seeds > 1 else None,
                "sem": float(values.std(ddof=1)/np.sqrt(seeds)) if seeds > 1 else None}
    return {"schema_version": 1, "benchmark": "linear_poisoning_surrogate_v1",
            "is_paper_reproduction": False, "epochs": epochs, "seeds": seeds,
            "attack_fraction": 0.2, "python": platform.python_version(), "numpy": np.__version__,
            "summary": summary, "runs": rows,
            "limits": ["No language model or external judges.", "Metrics are linear proxies, not PSR/FRS.",
                       "TCS recurrence penalty is an explicit implementation extension.",
                       "Frozen runs can preserve probes by refusing later benign learning."]}


def nullspace_demo(seed=0):
    """Actual parameter change with zero anchor drift, despite identity geometry."""
    eye = np.eye(4)
    model = LinearLoRA(eye, {"open": 2}, seed=seed)
    # Deliberately restrict A to unobserved input directions for an exact counterexample.
    model.banks["open"]["A"][:, :2] = 0
    graph = IdentityGraph([IdentityNode("protected-direction", eye[0])])
    engine = Consolidator(model, graph, {"seen-0": eye[:1], "seen-1": eye[1:2]},
        config=RiskConfig(task_weight=0, archive_weight=0), learning_rate=1.0)
    initial = engine.model.forward(eye[3:4]).copy()
    events = [engine.consolidate(eye[3:4], -eye[3:4], eye[:1], domain="open") for _ in range(100)]
    final = engine.model.forward(eye[3:4])
    return {"scenario": "unobserved_input_subspace", "steps": len(events),
            "max_proposed_anchor_drift": max(e.get("proposed_drift", 0) for e in events),
            "initial_unobserved_output": initial.tolist(), "final_unobserved_output": final.tolist(),
            "behavior_reversed": bool(final[0, 3] < 0), "frozen": engine.frozen,
            "interpretation": "A finite trusted anchor set does not cover every behavioral change."}
