"""ENSAI Q4: GD vs SGD with equal component-gradient budgets and eta_k=1/k."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def make_data(n=3000, p=8, noise_sd=0.5, seed=42):
    if n < 2 or p < 2 or noise_sd < 0 or not np.isfinite(noise_sd):
        raise ValueError("Require n>=2, p>=2 and finite nonnegative noise_sd")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = x[:, 0] + np.maximum(x[:, 1], 0) + noise_sd * rng.standard_normal(n)
    return np.column_stack((np.ones(n), x)), y


def loss(z, y, theta):
    return float(np.mean((z @ theta - y) ** 2))


def component_gradient(z_i, y_i, theta):
    return 2 * z_i * (z_i @ theta - y_i)


def full_gradient(z, y, theta):
    return 2 * z.T @ (z @ theta - y) / len(y)


def optimize(z, y, budget, method, seed=0, checkpoint_every=None):
    n, d = z.shape
    if budget < 0 or method not in ("gd", "sgd"):
        raise ValueError("budget must be nonnegative; method gd or sgd")
    theta = np.zeros(d)
    rng = np.random.default_rng(seed)
    if checkpoint_every is None:
        checkpoint_every = n
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    history = [(0, loss(z, y, theta))]
    spent = 0
    steps = budget // n if method == "gd" else budget
    for k in range(1, steps + 1):
        if method == "gd":
            theta -= full_gradient(z, y, theta) / k
            spent += n
        else:
            i = rng.integers(n)
            theta -= component_gradient(z[i], y[i], theta) / k
            spent += 1
        if spent % checkpoint_every == 0 or k == steps:
            history.append((spent, loss(z, y, theta)))
    return theta, history


def experiment(n=3000, p=8, budget=30000, repeats=5, data_seed=42, sgd_seed=100):
    if repeats < 2 or budget < n or budget % n:
        raise ValueError("Require repeats>=2 and budget>=n divisible by n")
    z, y = make_data(n, p, seed=data_seed)
    theta_opt = np.linalg.lstsq(z, y, rcond=None)[0]
    f_opt = loss(z, y, theta_opt)
    gd_theta, gd_history = optimize(z, y, budget, "gd")
    runs = [optimize(z, y, budget, "sgd", seed=sgd_seed + j)[1] for j in range(repeats)]
    assert all([t for t, _ in run] == [t for t, _ in gd_history] for run in runs)
    rows = []
    for j, (spent, gd_value) in enumerate(gd_history):
        vals = np.array([run[j][1] for run in runs])
        rows.append({"component_gradients": spent, "gd_objective": gd_value,
                     "sgd_mean_objective": float(vals.mean()),
                     "sgd_sd_objective": float(vals.std(ddof=1)),
                     "sgd_min_objective": float(vals.min()),
                     "sgd_max_objective": float(vals.max()),
                     "ols_minimum": f_opt})
    return rows, {"n": n, "p": p, "budget": budget, "repeats": repeats,
                  "data_seed": data_seed, "sgd_seed_start": sgd_seed,
                  "gd_steps": budget // n, "sgd_steps_per_run": budget,
                  "ols_minimum": f_opt, "gd_final": loss(z, y, gd_theta),
                  "sgd_final_mean": rows[-1]["sgd_mean_objective"]}


def save(rows, summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "q4.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "q4.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    x = np.array([r["component_gradients"] for r in rows])
    gd = np.array([r["gd_objective"] for r in rows])
    sgd = np.array([r["sgd_mean_objective"] for r in rows])
    sd = np.array([r["sgd_sd_objective"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, gd, "o-", label="GD (deterministic on fixed data)")
    ax.plot(x, sgd, "s-", label="SGD (mean of independent index runs)")
    ax.fill_between(x, np.maximum(sgd - sd, 0), sgd + sd, alpha=0.2,
                    label="SGD ±1 sample SD (across runs)")
    ax.axhline(summary["ols_minimum"], color="black", ls="--", label="OLS minimum")
    ax.set(xlabel="Component-gradient evaluations", ylabel="Full training objective f(θ)",
           title="Q4: GD vs SGD under equal gradient-evaluation budget")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "q4_convergence.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sgd-seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows, summary = experiment(budget=args.budget, repeats=args.repeats,
                               data_seed=args.data_seed, sgd_seed=args.sgd_seed)
    save(rows, summary, args.output)
    print(json.dumps(summary, indent=2))
