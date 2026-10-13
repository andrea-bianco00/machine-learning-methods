# Q3 — Linear regression: empirical and population risk

**ENSAI university assignment, question 3.** Fit the ordinary least-squares estimator using scikit-learn and print empirical risk at `n = 100, 500, 1000, 1500, 2000, 2500, 3000`. Comment on the observed behavior.

## Provenance and correction

The original notebook used the model `Y = X1 + max(X2,0) + ε` with `ε ~ N(0, 0.25)`, i.e. noise **standard deviation 0.5**. This is different from the reconstructed Q2, which specifies noise **variance 0.5**. This Q3 implementation intentionally retains the original Q3 noise setting. Both experiments are self-contained.

The original notebook labeled the squared prediction discrepancy `mean((f*(X)-f_hat(X))²)` on the training covariates as 'Bias²' and used `σ² + Bias²` as a theoretical risk. That quantity includes estimation error and is evaluated on reused covariates. It is not the pure model-approximation bias or an independent population risk. Its formula `empirical risk ≈ σ² + Bias² - Variance/n` is not a general exact identity.

## Exact analytical benchmark

For independent standard Gaussian features and the nonlinear signal `f*(X)=X1+max(X2,0)`, the best population linear predictor with intercept is

```text
f_lin*(X) = 1/sqrt(2π) + X1 + 0.5 X2.
```

Its irreducible squared approximation error is `A = 1/4 - 1/(2π) ≈ 0.09084506`. For any fitted linear coefficient vector `beta_hat` (including intercept), the **conditional population prediction risk**, given the fitted model, is exactly

```text
R(beta_hat) = noise_variance + A + ||beta_hat - beta_lin*||².
```

This follows because the features are independent, centered and have unit variance, the intercept is included, and the nonlinear approximation residual is orthogonal to every linear regressor. The final term is **estimation error**, not structural approximation error.

The limiting risk for the original Q3 noise standard deviation 0.5 is `0.25 + A ≈ 0.34084506`. It is not the Bayes risk: a correctly specified nonlinear predictor could attain 0.25.

## Reproduce

From the repository root with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python q3.py --seed 42 --noise-sd 0.5 --output results
```

The command writes `results/q3.csv`, `results/q3.json`, `results/q3_risks.png` and `results/q3_components.png`. No external data or interactive server is required. Nested prefixes of one 3,000-row generated sample match the original experiment design.

## Interpretation

- Training empirical risk is not guaranteed to decrease monotonically as sample size grows, because the data and fitted estimator change.
- The exact population risk approaches the best linear-model risk as coefficient estimation improves.
- The model-approximation term stays positive no matter how large the sample; it is the cost of omitting the ReLU nonlinearity.
- A single simulation is descriptive, not proof of asymptotic convergence. The mathematical population-risk decomposition provides the exact benchmark.

## Verification

Five Q3 unit tests cover the analytical approximation formula, seed reproducibility, input validation, population decomposition, and a large-sample sanity check. The [Q1–Q6 verification workflow](.github/workflows/q1.yml) reruns the suite and generates the figures. The successful Q1–Q6 workflow execution was confirmed on commit `761d209` (run #27); check the [Actions page](https://github.com/andrea-bianco00/machine-learning-methods/actions) for newer runs.
