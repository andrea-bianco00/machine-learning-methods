"""ENSAI Q3: nested-sample OLS, empirical risk, exact population risk."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

SIZES = (100, 500, 1000, 1500, 2000, 2500, 3000)

def approximation_error():
    return 0.25 - 1 / (2 * np.pi)

def generate(n=3000, p=8, noise_sd=0.5, seed=42):
    if n < 2 or p < 2 or not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("Require n>=2, p>=2 and finite nonnegative noise_sd")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    return x, x[:, 0] + np.maximum(x[:, 1], 0) + noise_sd * rng.standard_normal(n)

def evaluate(n_values=SIZES, seed=42, noise_sd=0.5, p=8):
    n_values = tuple(n_values)
    if not n_values or min(n_values) <= p + 1:
        raise ValueError("Every sample size must exceed p+1")
    x, y = generate(max(n_values), p, noise_sd, seed)
    beta_star = np.zeros(p + 1)
    beta_star[:3] = [1 / np.sqrt(2 * np.pi), 1, 0.5]
    rows = []
    for n in n_values:
        model = LinearRegression().fit(x[:n], y[:n])
        beta_hat = np.r_[model.intercept_, model.coef_]
        estimation_error = float(np.sum((beta_hat - beta_star)**2))
        rows.append({"n": int(n),
                     "empirical_risk": float(np.mean((y[:n] - model.predict(x[:n]))**2)),
                     "population_risk": float(noise_sd**2 + approximation_error() + estimation_error),
                     "noise_variance": float(noise_sd**2),
                     "approximation_error": float(approximation_error()),
                     "estimation_error": estimation_error})
    return rows

def save(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "q3.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "q3.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [r["empirical_risk"] for r in rows], "o-", label="Training empirical risk")
    ax.plot(ns, [r["population_risk"] for r in rows], "s-", label="Exact population risk")
    ax.axhline(rows[0]["noise_variance"] + approximation_error(), color="black",
               linestyle="--", label="Best linear-model population risk")
    ax.set(xlabel="Training sample size", ylabel="Mean squared error",
           title="Q3: Empirical vs population risk")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "q3_risks.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [r["estimation_error"] for r in rows], "o-", label="Estimation error")
    ax.axhline(approximation_error(), color="black", linestyle="--",
               label="Linear approximation error")
    ax.set(xlabel="Training sample size", ylabel="Squared-error contribution",
           title="Q3: Estimation vs approximation")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "q3_components.png", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-sd", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = evaluate(seed=args.seed, noise_sd=args.noise_sd)
    save(rows, args.output)
    print(json.dumps(rows, indent=2))
