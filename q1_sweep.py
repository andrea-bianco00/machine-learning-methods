"""Generate Q1 Monte Carlo tables and plots without a notebook or web server."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from q1 import run_experiment


def generate(output: Path, repetitions: int = 400, seed: int = 42) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "sample_size": [(n, 8, 1.0) for n in (50, 100, 200, 400, 800)],
        "dimension": [(200, p, 1.0) for p in (1, 3, 8, 16, 32)],
        "noise_sd": [(200, 8, s) for s in (0.25, 0.5, 1.0, 1.5, 2.0)],
    }
    rows = []
    for group, configs in scenarios.items():
        results = [run_experiment(n, p, sigma, repetitions, seed + i)
                   for i, (n, p, sigma) in enumerate(configs)]
        for result in results:
            rows.append({"scenario": group, **result})
        xname = {"sample_size": "n", "dimension": "p", "noise_sd": "sigma"}[group]
        xs = [r[xname] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [r["theoretical_risk"] for r in results], "o-",
                label="Theory: σ²(p+1)/n")
        ax.errorbar(xs, [r["mean_in_design_excess_risk"] for r in results],
                    yerr=[1.96 * r["mc_standard_error"] for r in results],
                    fmt="s", capsize=4, label="Monte Carlo mean ± 1.96 SE")
        ax.set(
            xlabel={"n": "Training sample size n", "p": "Number of covariates p",
                    "sigma": "Noise standard deviation σ"}[xname],
            ylabel="In-design excess prediction risk",
            title={"sample_size": "Sample size and excess risk",
                   "dimension": "Model dimension and excess risk",
                   "noise_sd": "Noise and excess risk"}[group])
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"q1_{group}.png", dpi=170)
        plt.close(fig)
    with (output / "q1_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repetitions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.repetitions, args.seed)
