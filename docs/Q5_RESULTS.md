# Q5 — Zeroth-order optimization (ENSAI assignment)

This section reconstructs the university exercise and **distinguishes the original notebook from corrected, reproducible experiments**. The main comparison is GD, SGD, two-point zeroth-order GD (ZGD) and stochastic zeroth-order SGD (ZSGD) for least-squares regression.

## Mathematical audit

For a quadratic objective and a direction U, the centered two-point estimator is

```text
g_hat(theta,U) = [f(theta+hU) - f(theta-hU)] U / (2h).
```

It equals `U Uᵀ ∇f(theta)` exactly in exact arithmetic for quadratic objectives. Therefore `E[g_hat | theta] = ∇f(theta)` when `E[UUᵀ]=I`. Gaussian N(0,I), independent Rademacher signs, and sphere directions scaled by sqrt(d) satisfy this covariance condition. An unscaled unit-sphere direction has covariance I/d and would require an additional factor d.

The assignment writes a denominator h; this is twice the centered derivative. **The reconstructed implementation uses 2h**, explicitly correcting the scale instead of silently claiming equivalence.

## Fairness and computational cost

The four methods use the same fixed dataset, initial coefficients, objective and `eta_k=1/k`. They have a per-method budget of 30,000 *component oracle calls*:

| Method | Oracle type | Calls/update | Updates |
| --- | --- | ---: | ---: |
| GD | component gradient | 3000 | 10 |
| SGD | component gradient | 1 | 30000 |
| ZGD | component function value | 6000 | 5 |
| ZSGD | component function value | 2 | 15000 |

The budget is never exceeded; unused calls are recorded. Component-gradient calls and component-function-value calls are **not identical wall-clock or arithmetic costs**. Full-objective evaluations for monitoring and plotting are excluded from the optimization budget and must be accounted for in any total-runtime comparison.

## Reproduce

Python 3.11, from repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python q5.py --budget 30000 --repeats 5 --seed 42 --h 0.01 --output results
```

Outputs: `results/q5.json`, `results/q5.csv`, `results/q5_convergence.png`. The JSON records seeds, budgets, per-run final objectives and the OLS minimum. The plot uses a logarithmic objective scale so unstable trajectories are not hidden.

## Interpretation

With the prescribed aggressive `1/k` schedule, the corrected zeroth-order estimators may have very high trajectory variance and severe numerical instability. This does **not** establish that zeroth-order methods are universally inferior: performance depends on the estimator, direction distribution, learning-rate schedule and oracle cost model. The five repeats are illustrative rather than a broad benchmark.

## Exploratory follow-up

See [Q5_SENSITIVITY.md](Q5_SENSITIVITY.md). That follow-up varies directions and step-size prefactors **outside the original assignment** and is not presented as part of the original university submission.

Seven Q5 tests cover the quadratic two-point identity, stochastic estimator, direction covariance, exact budget accounting, reproducibility, OLS lower bound and invalid inputs. The [Q1–Q6 verification workflow](.github/workflows/q1.yml) reruns these tests and reproduces the figures. The successful Q1–Q6 workflow execution was confirmed on commit `761d209` (run #27); check the [Actions page](https://github.com/andrea-bianco00/machine-learning-methods/actions) for newer runs.
