"""ENSAI Q5: two-point zeroth-order least-squares optimization.

Costs are measured in *component oracle calls*: one gradient of f_i or
one value of f_i. They are different oracle types, not equal wall-time.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def data(n=3000, p=8, noise_sd=0.5, seed=42):
    if n < 2 or p < 2 or not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError('n>=2, p>=2, noise_sd finite and nonnegative required')
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = x[:, 0] + np.maximum(x[:, 1], 0) + noise_sd * rng.standard_normal(n)
    return np.column_stack([np.ones(n), x]), y


def objective(z, y, theta):
    return float(np.mean((z @ theta - y)**2))


def full_gradient(z, y, theta):
    return 2 * z.T @ (z @ theta - y) / len(y)


def component_gradient(zi, yi, theta):
    return 2 * zi * (zi @ theta - yi)


def direction(rng, d, distribution='gaussian'):
    if distribution == 'gaussian':
        return rng.standard_normal(d)
    if distribution == 'rademacher':
        return rng.choice(np.array([-1., 1.]), size=d)
    if distribution == 'sphere':
        v = rng.standard_normal(d)
        return np.sqrt(d) * v / np.linalg.norm(v)
    raise ValueError('distribution must be gaussian, rademacher or sphere')


def zero_gradient(z, y, theta, u, h=0.01, stochastic_index=None):
    """Two-point estimator. E[g_hat | theta] = gradient for quadratic loss.

    For full objective, each evaluation uses all n component values;
    for stochastic objective, each uses one component value.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError('h must be positive and finite')
    if stochastic_index is None:
        fp = objective(z, y, theta + h*u)
        fm = objective(z, y, theta - h*u)
    else:
        zi, yi = z[stochastic_index], y[stochastic_index]
        fp = (zi @ (theta+h*u) - yi)**2
        fm = (zi @ (theta-h*u) - yi)**2
    return ((fp - fm)/(2*h)) * u


def optimize(z, y, method, budget, seed=100, h=0.01,
             distribution='gaussian', checkpoints=10):
    if method not in ('gd', 'sgd', 'zgd', 'zsgd'):
        raise ValueError('invalid method')
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    if not isinstance(checkpoints, int) or checkpoints <= 0:
        raise ValueError('checkpoints must be a positive integer')
    if not np.isfinite(h) or h <= 0:
        raise ValueError('h must be positive and finite')
    n, d = z.shape
    cost = {'gd': n, 'sgd': 1, 'zgd': 2*n, 'zsgd': 2}[method]
    steps = budget // cost  # no budget overrun
    theta = np.zeros(d)
    rng = np.random.default_rng(seed)
    history = [{'calls': 0, 'objective': objective(z, y, theta)}]
    for k in range(1, steps+1):
        if method == 'gd':
            grad = full_gradient(z, y, theta)
        elif method == 'sgd':
            i = int(rng.integers(n))
            grad = component_gradient(z[i], y[i], theta)
        elif method == 'zgd':
            grad = zero_gradient(z, y, theta, direction(rng, d, distribution), h)
        else:
            i = int(rng.integers(n))
            grad = zero_gradient(z, y, theta, direction(rng, d, distribution), h, i)
        theta -= grad / k
        if k == steps or k % max(1, steps // checkpoints) == 0:
            history.append({'calls': k*cost, 'objective': objective(z, y, theta)})
    return theta, history, {'steps': steps, 'oracle_calls': steps*cost,
                            'unused_budget': budget - steps*cost,
                            'oracle': 'component_gradient' if method in ('gd','sgd') else 'component_value'}


def experiment(budget=30000, repeats=5, seed=42, h=0.01):
    if repeats < 2 or budget < 6000:
        raise ValueError('repeats>=2, budget>=6000 required for n=3000')
    z, y = data(seed=seed)
    ols = np.linalg.lstsq(z, y, rcond=None)[0]
    optimum = objective(z, y, ols)
    histories, metadata = {}, {}
    for method in ('gd', 'sgd', 'zgd', 'zsgd'):
        runs = []
        for j in range(1 if method == 'gd' else repeats):
            _, history, info = optimize(z, y, method, budget, seed=100+j, h=h)
            runs.append(history)
        histories[method] = runs
        metadata[method] = {**info, 'runs': len(runs),
                            'final_objectives': [r[-1]['objective'] for r in runs],
                            'final_mean': float(np.mean([r[-1]['objective'] for r in runs]))}
    return histories, {'budget_per_method': budget, 'n': len(y), 'd': z.shape[1],
                       'noise_sd': 0.5, 'data_seed': seed, 'h': h,
                       'direction': 'gaussian N(0,I)', 'ols_minimum': optimum,
                       'methods': metadata,
                       'budget_note': 'First-order: component-gradient calls; zero-order: component-function-value calls. Diagnostic full-loss evaluations excluded.'}


def save(histories, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'q5.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    with (output/'q5.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'run', 'calls', 'objective'])
        w.writeheader()
        for method, runs in histories.items():
            for j, run in enumerate(runs):
                for point in run:
                    w.writerow({'method':method, 'run':j, **point})
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    for ax, pair in zip(axes, [('gd','zgd'), ('sgd','zsgd')]):
        for method in pair:
            for j, run in enumerate(histories[method]):
                ax.plot([p['calls'] for p in run], [p['objective'] for p in run],
                        marker='o' if method in ('gd','zgd') else None,
                        alpha=0.85 if j==0 else 0.25,
                        label=method.upper() if j==0 else None)
        ax.axhline(summary['ols_minimum'], ls='--', color='black', label='OLS minimum')
        ax.set(xlabel='Component oracle calls (different oracle types)',
               ylabel='Full training objective', title=' vs '.join(m.upper() for m in pair))
        ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Q5: first-order vs two-point zero-order optimization')
    fig.tight_layout()
    fig.savefig(output/'q5_convergence.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=30000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h', type=float, default=0.01)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    histories, summary = experiment(args.budget, args.repeats, args.seed, args.h)
    save(histories, summary, args.output)
    print(json.dumps(summary, indent=2))
