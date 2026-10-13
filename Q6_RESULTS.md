# Q6 — Flexible regression with zeroth-order optimization

## Assignment and reconstruction

The ENSAI assignment defines the predictor

```text
m_{beta,gamma}(x) = x^T beta + max(x^T gamma, 0)
```

and asks (1) to compare ZGD and ZSGD on this flexible model under equal function-evaluation budgets, (2) to estimate predictive risk on 1,000 new samples, and (3) to compare ZSGD on flexible and linear models under equal component-function-evaluation budgets.

This is a reproducible reconstruction, **not a claim that the original notebook's numerical results have been preserved**. The original notebook counted a full-batch objective call as one evaluation; this implementation counts its n individual contributions, which changes the number of possible updates dramatically.

## Model and experiment

- Training data: n=3,000, p=8; X entries iid N(0,1).
- DGP: Y = X_1 + max(X_2,0) + epsilon; epsilon iid N(0,0.25).
- Noise standard deviation is **0.5**, matching the original Q3–Q6 notebook. Note that the assignment's earlier Q2 wording uses noise **variance 0.5**. These are not the same experiment.
- Flexible model has 16 parameters (8 beta + 8 gamma); linear model has 8.
- Independent held-out set: 1,000 new draws, seed 123, never used to tune optimizer settings.
- Objective: mean squared error (MSE).
- Two-point estimate: [f(theta+hU)-f(theta-hU)] U/(2h), with h=0.01 and independent Rademacher U, satisfying E[UU^T]=I.
- Step size: 0.03/sqrt(k), **explicitly changed from the Q5 1/k baseline** to provide a stable illustrative Q6 experiment. This is a declared reconstruction choice, not a prescribed Q6 learning rate.
- Deterministic flexible-model initialization gamma_2=0.1 avoids the ReLU kink at zero. Linear beta initializes at zero.
- Five optimizer seeds per configuration, shared between comparisons.

## Equal-budget accounting

Each configuration receives **120,000 component-function evaluations per repeat**.

| Configuration | Calls per update | Updates |
| --- | ---: | ---: |
| Flexible ZGD | 2 x 3,000 = 6,000 | 20 |
| Flexible ZSGD | 2 | 60,000 |
| Linear ZSGD | 2 | 60,000 |

Each of the two probes of ZGD evaluates the full training objective (3,000 components); each ZSGD probe evaluates the same sampled observation at plus/minus perturbations. The budget excludes extra full-loss evaluations for diagnostics, plotting and held-out risk; it is an *optimization-oracle* budget, not a full runtime/FLOP budget.

## Verified local results

These numbers were produced locally using the exact seeds, budgets and settings above; the GitHub Actions workflow independently regenerates JSON/CSV/PNG artifacts.

| Method | Median train MSE | Median held-out MSE | Test MSE range |
| --- | ---: | ---: | ---: |
| Flexible ZGD | 0.774719 | 0.708470 | 0.645272–0.723349 |
| Flexible ZSGD | 0.265897 | **0.246118** | 0.244102–0.255931 |
| Linear ZSGD | 0.522274 | 0.520539 | 0.512942–0.524731 |

The true DGP parameter vector gives test MSE **0.236307** on these 1,000 observations; linear OLS (a gradient-based closed-form reference, not a zeroth-order method) gives test MSE **0.513187**. The irreducible population noise variance is 0.25; a finite test-set realization can have MSE below that number.

**Interpretation:** In this experiment, flexible ZSGD comes close to the true predictor's held-out risk and outperforms linear ZSGD. This is consistent with the DGP containing a ReLU component that the linear model cannot express. Flexible ZGD performs substantially worse under the same component-evaluation budget because it can take only 20 updates. These results do not prove that ZSGD is always better than ZGD, or that the selected step size is universally optimal. The flexible optimization objective is nonconvex/nonsmooth and results depend on initialization, randomness and budget.

## Reproduce

From repository root with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
python q6.py --budget 120000 --repeats 5 --step-scale 0.03 --output results
```

Outputs:
- `results/q6.json` — complete configuration, reference values and five-run summaries
- `results/q6_runs.csv` — individual train/test results and actual oracle calls
- `results/q6_trajectories.csv` — optimization checkpoints
- `results/q6_comparison.png` — training convergence and held-out risk across seeds

Eight Q6 unit tests were run successfully locally before publication. The CI status should be checked independently after publishing.

## Limitations

Only one training dataset and one independent test dataset are used, with five optimizer seeds. No confidence intervals, independent hyperparameter tuning, or wall-clock timing are reported. The fixed initialization and step-size choices influence performance. Reported training/test MSE values are not a global optimality certificate.
