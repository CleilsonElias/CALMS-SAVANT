"""Run and record actual checks; never turn skipped optional work into a pass."""
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"src"))
    os.chdir(root)
    from calms_savant.engine import atomic_json
    from calms_savant.consensus import seed_consensus
    from calms_savant.benchmark import nullspace_demo
    records = []
    def command(args, **kwargs):
        result = subprocess.run(args, capture_output=True, text=True, **kwargs)
        records.append({"command": " ".join(args), "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        return result.stdout
    with (root/"validation/tests.log").open("w", encoding="utf-8") as f:
        suite = unittest.defaultTestLoader.discover(str(root/"tests"))
        result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("test suite failed; see validation/tests.log")
    with tempfile.TemporaryDirectory(prefix="calms-release-") as tmp:
        demo = str(Path(tmp)/"demo")
        command([sys.executable, "run_calms_savant.py", "demo", "--directory", demo])
        command([sys.executable, "run_calms_savant.py", "inspect", str(Path(demo)/"state.json")])
        # Verify the installed console entry point when this distribution is installed.
        import shutil
        if shutil.which("calms-savant"):
            command(["calms-savant", "inspect", str(Path(demo)/"state.json")])
    command([sys.executable, "examples/defensive_replay.py"])
    command([sys.executable, "-m", "compileall", "-q", "src", "examples", "scripts"])
    atomic_json(root/"validation/nullspace.json", nullspace_demo())
    candidates = json.loads((root/"examples/seeds.json").read_text())
    consensus = {}
    for method in ("paper_mean_std", "medoid_mad"):
        r = seed_consensus(candidates, method=method)
        consensus[method] = {"accepted": r.accepted, "rejected": r.rejected,
                             "scores": r.scores.tolist(), "threshold": r.threshold,
                             "distances": r.distances.tolist(), "certified_safe": False}
    atomic_json(root/"validation/consensus_counterexample.json", consensus)
    versions = {}
    for package in ("numpy", "scipy", "torch", "transformers", "peft", "hnswlib", "matplotlib", "reportlab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    atomic_json(root/"validation/validation.json", {
        "version": "0.1.0", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "tests": {"discovered": result.testsRun, "passed": result.testsRun-len(result.skipped),
                  "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
                  "failures": len(result.failures), "errors": len(result.errors)},
        "commands": records,
        "benchmark": {"file": "benchmark_linear.json", "seeds": 5, "epochs": 100,
                       "variants": 7, "is_paper_reproduction": False},
        "not_executed": ["real_transformer_training", "original_PPP_Bench", "HNSW_backend",
                         "remote_GitHub_CI", "GPU_latency_or_memory_benchmark"],
        "optional_integration": {"torch_guard": "syntax_checked_not_executed",
                                 "hf_peft_trainer": "syntax_checked_not_executed"}})
    print(json.dumps({"tests": result.testsRun, "passed": result.testsRun-len(result.skipped),
                      "skipped": len(result.skipped), "commands": len(records)}, indent=2))


if __name__ == "__main__":
    main()
