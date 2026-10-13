"""ENSAI Q2: reproducible Gaussian design and exploratory plots."""
import argparse
import json
from pathlib import Path
import numpy as np


def generate_data(n=3000, p=8, noise_variance=0.5, seed=42):
    if n < 2 or p < 2 or not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("Require n>=2, p>=2, finite noise_variance>=0")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = x[:, 0] + np.maximum(x[:, 1], 0) + np.sqrt(noise_variance) * rng.standard_normal(n)
    return x, y


def theoretical_moments(noise_variance=0.5):
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("Noise variance must be finite and nonnegative")
    variance = 1.5 - 1 / (2 * np.pi) + noise_variance
    return {"mean_y": float(1 / np.sqrt(2 * np.pi)), "variance_y": float(variance),
            "correlation_x1_y": float(1 / np.sqrt(variance)),
            "correlation_x2_y": float(0.5 / np.sqrt(variance)),
            "correlation_other_features_y": 0.0}


def analyze(x, y):
    if x.ndim != 2 or len(y) != len(x) or x.shape[1] < 2:
        raise ValueError("Expected X (n,p), p>=2, matching y")
    r = y - x[:, 0]
    neg = x[:, 1] <= 0
    pos = ~neg
    return {"n": int(len(y)), "p": int(x.shape[1]), "mean_y": float(np.mean(y)),
            "variance_y": float(np.var(y)),
            "correlations": [float(np.corrcoef(x[:, j], y)[0, 1]) for j in range(x.shape[1])],
            "partial_residual_mean_x2_nonpositive": float(r[neg].mean()) if neg.any() else None,
            "partial_residual_mean_x2_positive": float(r[pos].mean()) if pos.any() else None}


def plot_data(x, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-3.7, 3.7, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(x[:, 0], y, s=5, alpha=0.16, rasterized=True)
    axes[0].plot(grid, grid + 1 / np.sqrt(2 * np.pi), color="black", label="E[Y | X1]")
    axes[0].set(xlabel="X1", ylabel="Y", title="Q2a · Linear signal")
    axes[0].legend()
    axes[1].scatter(x[:, 1], y - x[:, 0], s=5, alpha=0.16, rasterized=True)
    axes[1].plot(grid, np.maximum(grid, 0), color="black", label="E[Y − X1 | X2]")
    axes[1].set(xlabel="X2", ylabel="Y − X1", title="Q2b · ReLU signal")
    axes[1].legend()
    corrs = [np.corrcoef(x[:, j], y)[0, 1] for j in range(x.shape[1])]
    axes[2].bar(range(1, len(corrs) + 1), corrs)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set(xlabel="Feature index", ylabel="Sample correlation with Y",
                title="Q2c · Marginal associations")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--noise-variance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/q2.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/q2_signals.png"))
    args = parser.parse_args()
    x, y = generate_data(args.n, args.p, args.noise_variance, args.seed)
    summary = {"seed": args.seed, "noise_variance": args.noise_variance,
               "observed": analyze(x, y), "theoretical": theoretical_moments(args.noise_variance)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_data(x, y, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
