"""Reproducible Gaussian-design OLS experiment for ENSAI Q1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def simulate_once(n: int, p: int, sigma: float, rng: np.random.Generator) -> dict[str, float]:
    """Return in-design excess risk and squared parameter error for one sample."""
    if n <= p + 1 or p < 0 or sigma < 0:
        raise ValueError("Require n > p + 1, p >= 0 and sigma >= 0")
    x = rng.standard_normal((n, p))
    z = np.column_stack((np.ones(n), x))
    theta_star = np.linspace(-0.5, 0.5, p + 1)
    y = z @ theta_star + sigma * rng.standard_normal(n)
    theta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
    delta = theta_hat - theta_star
    return {"in_design_excess_risk": float(np.mean((z @ delta) ** 2)),
            "squared_parameter_error": float(delta @ delta)}


def run_experiment(n: int, p: int, sigma: float, repetitions: int, seed: int) -> dict:
    if repetitions < 2:
        raise ValueError("Require at least two repetitions for uncertainty estimation")
    rng = np.random.default_rng(seed)
    values = np.array([simulate_once(n, p, sigma, rng)["in_design_excess_risk"]
                       for _ in range(repetitions)])
    theoretical = sigma**2 * (p + 1) / n
    se = float(values.std(ddof=1) / np.sqrt(repetitions))
    return {"n": n, "p": p, "sigma": sigma, "repetitions": repetitions,
            "seed": seed, "theoretical_risk": theoretical,
            "mean_in_design_excess_risk": float(values.mean()),
            "mc_standard_error": se,
            "ci95_normal_approx": [float(values.mean() - 1.96 * se),
                                   float(values.mean() + 1.96 * se)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/q1.json"))
    args = parser.parse_args()
    result = run_experiment(args.n, args.p, args.sigma, args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
