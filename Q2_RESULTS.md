# Q2 — Revealing linear and nonlinear signals

## University question

Generate 3,000 independent observations with eight independent standard Gaussian features and `Y = X1 + max(X2, 0) + ε`, where `ε ~ N(0, 0.5)`. **0.5 is the noise variance**, not its standard deviation. Use simple graphs to show the special relationships of X1 and X2 with Y.

This is a cleaned, standalone reconstruction of the ENSAI Q2 notebook, which originally used Plotly, partial residuals and absolute correlations. The reconstruction uses Matplotlib, exact population reference curves and signed correlations.

## Mathematical reference

With `R = Y-X1`:
- `E[Y | X1=x] = x + 1/sqrt(2π)`. The intercept is positive, not zero.
- `E[R | X2=x] = max(x,0)`.
- `E[Y] = 1/sqrt(2π)`.
- `Var(Y) = 1 + (1/2 - 1/(2π)) + 0.5 = 2 - 1/(2π)`.
- `Cov(X1,Y) = 1`, `Cov(X2,Y) = 1/2`, and the other six covariances are zero.
- Population correlations are approximately 0.7370 for X1 and 0.3685 for X2.

The other covariates have **zero population correlation**, not necessarily zero sample correlation. Correlation alone cannot capture the ReLU shape.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python q2.py --n 3000 --p 8 --noise-variance 0.5 --seed 42
```

Produces `results/q2.json` and `results/q2_signals.png`, without external data or a server.

## Seed-42 reconstruction

| Statistic | Simulated | Population |
| --- | ---: | ---: |
| Mean Y | 0.4413 | 0.3989 |
| Variance Y | 1.8712 | 1.8408 |
| Corr(X1,Y) | 0.7345 | 0.7370 |
| Corr(X2,Y) | 0.3609 | 0.3685 |

The six irrelevant features' sample correlations lie approximately between -0.0272 and +0.0142. These are descriptive simulations, not causal or inferential claims.

## Verification

Five local tests passed: reproducible data generation, invalid inputs, noiseless structural identity, analytical moments and large-sample empirical agreement. GitHub Actions runs the same checks. Its status must be confirmed separately. Q3–Q6 are not yet audited.
