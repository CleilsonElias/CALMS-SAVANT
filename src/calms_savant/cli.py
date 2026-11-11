"""Local CLI; neither network calls nor model downloads in the core."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .benchmark import make_engine, stream, evaluate, run_benchmark, nullspace_demo
from .engine import Consolidator, atomic_json
from .consensus import seed_consensus


def main(argv=None):
    parser = argparse.ArgumentParser(prog="calms-savant")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a local low-rank consolidation example")
    demo.add_argument("--directory", default="demo_run")
    demo.add_argument("--seed", type=int, default=0)
    bench = sub.add_parser("benchmark", help="run the linear surrogate, not paper PPP-Bench")
    bench.add_argument("--output", default="benchmark.json")
    bench.add_argument("--seeds", type=int, default=5)
    bench.add_argument("--epochs", type=int, default=100)
    check = sub.add_parser("inspect", help="validate and summarize saved model state")
    check.add_argument("state")
    resume = sub.add_parser("resume", help="unfreeze an inspected state with an explicit review note")
    resume.add_argument("state")
    resume.add_argument("--review-note", required=True)
    resume.add_argument("--output", required=True)
    null = sub.add_parser("nullspace", help="show the finite-anchor blind spot")
    null.add_argument("--output", default="nullspace.json")
    con = sub.add_parser("consensus", help="compare seed-consensus heuristics on JSON input")
    con.add_argument("input", help="JSON list of dictionaries: node ID -> vector")
    con.add_argument("--method", choices=["medoid_mad", "paper_mean_std"], default="medoid_mad")
    con.add_argument("--output", default="consensus.json")
    args = parser.parse_args(argv)
    if args.command == "demo":
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=False)
        engine = make_engine(args.seed)
        events = []
        for epoch, record in stream(args.seed, 12):
            event = engine.consolidate(record["x"], record["y"], record["z"], domain=record["domain"])
            events.append({**event, "evaluation_only_poison_label": record["poison"]})
        atomic_json(directory/"events.json", events)
        engine.save(directory/"state.json")
        summary = {"kind": "synthetic_linear_demo", "statuses": dict(Counter(e["status"] for e in events)),
                   "metrics": evaluate(engine), "archive_size": len(engine.archive), "frozen": engine.frozen}
        atomic_json(directory/"summary.json", summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "benchmark":
        result = run_benchmark(seeds=args.seeds, epochs=args.epochs)
        atomic_json(args.output, result)
        print(json.dumps({"output": str(Path(args.output).absolute()), "summary": result["summary"]}, indent=2))
    elif args.command == "nullspace":
        result = nullspace_demo()
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "consensus":
        candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
        r = seed_consensus(candidates, method=args.method)
        result = {"method": r.method, "accepted": r.accepted, "rejected": r.rejected,
                  "threshold": r.threshold, "scores": r.scores.tolist(),
                  "directions": {k: v.tolist() for k, v in r.directions.items()}, "certified_safe": False}
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        engine = Consolidator.load(args.state)
        if args.command == "resume":
            if Path(args.output).resolve() == Path(args.state).resolve():
                parser.error("resume output must be a separate path")
            engine.resume(review_note=args.review_note)
            engine.save(args.output)
        print(json.dumps({"step": engine.step_count, "frozen": engine.frozen,
                          "archive_size": len(engine.archive), "event_hash": engine.last_event_hash}, indent=2))


if __name__ == "__main__":
    main()
