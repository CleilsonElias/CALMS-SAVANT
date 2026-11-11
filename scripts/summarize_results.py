"""Export exact summary CSV and a scientific figure from raw benchmark output."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root/"validation/benchmark_linear.json").read_text(encoding="utf-8"))
    summary = raw["summary"]
    variants = ["standard_replay", "attenuation_only", "full", "no_identity", "no_tcs", "no_audit", "uniform_rank"]
    with (root/"validation/benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "metric", "mean", "sample_sd", "sem", "seeds", "epochs"])
        for variant in variants:
            for key, values in summary[variant].items():
                writer.writerow([variant, key, values["mean"], values["sample_sd"], values["sem"], raw["seeds"], raw["epochs"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["Replay", "Attenuation", "Full", "No identity", "No TCS", "No audit", "Uniform rank"]
    colors = ["#9c5551", "#c48b50", "#286b80", "#7d909b", "#7d909b", "#7d909b", "#7d909b"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    metrics = [("protected_probe_failure_rate_proxy", "Protected probe failure", (0, 0.3)),
               ("historical_anchor_drift", "Historical anchor drift", (0, 0.32)),
               ("fact_cosine_mean_proxy", "Factual cosine (linear proxy)", (0.99, 1.001))]
    for ax, (key, title, ylim) in zip(axes, metrics):
        values = [summary[v][key]["mean"] for v in variants]
        errors = [summary[v][key]["sem"] or 0 for v in variants]
        ax.bar(range(len(variants)), values, yerr=errors, capsize=3, color=colors, width=0.72)
        ax.set_xticks(range(len(variants)), labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CALMS-SAVANT | Synthetic low-rank poisoning surrogate", fontsize=15, x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.86, "Five seeds, 100 epochs, 20% attack records. Bars: mean +/- SEM. No LLM or paper PPP-Bench reproduction.", fontsize=9, color="#465564")
    fig.text(0.06, 0.025, "No-identity and no-audit ablations also preserve protected probes: component necessity is not established here.", fontsize=9, color="#465564")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.3, wspace=0.38)
    fig.savefig(root/"validation/benchmark_linear.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
