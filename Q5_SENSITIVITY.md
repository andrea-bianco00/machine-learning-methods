# Q5 — Exploratory sensitivity analysis (post-assignment extension)

**This is new follow-up research, not part of the original ENSAI submission.** It examines why the corrected two-point zeroth-order algorithms can be unstable with the prescribed learning rate.

## Controlled design

Keep the same 3,000-observation Gaussian-design least-squares problem, seed 42, zero initialization and 30,000 component-function-value calls per method. For each ZGD and ZSGD configuration, use five stochastic runs. Compare:

- Gaussian directions `U ~ N(0,I)`;
- independent Rademacher directions (entries ±1);
- normalized spherical directions `sqrt(d) V/||V||` with `V ~ N(0,I)`.

Each has `E[UUᵀ]=I`, so the centered two-point estimator is unbiased for the quadratic gradient. Vary `eta_k=c/k` for `c ∈ {1, 0.1, 0.01}`. The `c=1` configuration is the assignment baseline; smaller prefactors are an **explicit sensitivity study**.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python q5_sensitivity.py --budget 30000 --repeats 5 --output results
```

Outputs: `results/q5_sensitivity.csv` (individual trajectories), `results/q5_sensitivity.json` (summary and settings), `results/q5_sensitivity.png` (six-panel log-scale objective-gap plot). The JSON contains medians, minima and maxima, not just averages.

## Interpretation

The sensitivity experiment shows that the aggressive `c=1` schedule can lead to very large objective values for some random directions, while `c=0.1` can greatly improve stability. The smaller `c=0.01` is not automatically better: it may move too slowly over the same budget.

Do not claim the best observed configuration is universally optimal. There are only five stochastic runs per setting and one fixed dataset; results are descriptive and subject to stochastic variation. The exact choice of random-direction distribution and step size affects variance and stability even when all estimators have the same expectation.

## Limitations

The budget counts component-function-value calls used for optimization, excluding diagnostics. No wall-clock normalization, tuning on independent validation data or confidence interval is provided. This is an explanatory extension, not a comprehensive optimizer benchmark. Five additional sensitivity tests cover direction covariance, quadratic finite differences, reproducibility, budget accounting and input validation. The [Q1–Q6 verification workflow](.github/workflows/q1.yml) reruns them. The successful Q1–Q6 workflow execution was confirmed on commit `761d209` (run #27); check the [Actions page](../../actions) for newer runs.
