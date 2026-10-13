# Methodology audit — Q1: expected excess risk

**Status:** Q1's source-level audit, corrected simulation, tests and automated reproduction are implemented. The Q1–Q7 GitHub Actions workflow passed after the repository reorganization (run #127, commit `b2de0fe`). This document distinguishes the original notebook's derivation from the quantity actually computed in its code.

## Assignment

The ENSAI computer-class Q1 asks for numerical illustrations of the scaling `sigma² (p+1) / n` in random-design linear regression with Gaussian covariates and Gaussian noise. The original notebook presents a projection-matrix derivation and an interactive Dash simulation.

## What the original derivation establishes

Let `Z` be the `n × d` design matrix with an intercept, where `d = p+1`, and assume `n >= d`, full column rank, `Y = Z theta* + epsilon`, and `E[epsilon epsilon' | Z] = sigma² I`. For ordinary least squares,

```text
theta_hat - theta* = (Z'Z)^(-1) Z' epsilon
H = Z (Z'Z)^(-1) Z'
```

Then the **conditional in-design prediction error**, averaged over the same training covariates but using an independent response/noise draw, satisfies

```text
E_epsilon[ ||Z(theta_hat - theta*)||² / n | Z ]
  = sigma² trace(H) / n
  = sigma² d / n.
```

This is an exact identity under these assumptions. The notebook's projection-matrix argument supports **this specific risk definition**.

## Critical discrepancy in the original code

The notebook's Q1 function `compute_excess_risk(theta_hat, theta_star)` returns

```python
np.sum((theta_hat - theta_star) ** 2)
```

This is **squared parameter error**, not `||Z(theta_hat - theta*)||² / n`. Its conditional expectation is

```text
sigma² trace((Z'Z)^(-1)),
```

which is generally **not** `sigma² d/n`. Thus the original plots compare nonidentical quantities and cannot verify the displayed exact formula as written.

For independent standard Gaussian features and an intercept, the distinction also affects finite-sample expectations. In particular, the unconditional expected squared parameter error need not equal the in-design excess risk.

## Implemented correction

The reconstructed [Q1 experiment](Q1_RESULTS.md) measures **in-design excess prediction risk** using `mean((Z @ (theta_hat-theta_star))**2)`, rather than treating squared coefficient error as the same quantity. The latter, `sum((theta_hat-theta_star)**2)`, is a distinct estimand and is not substituted for the former.

An optional independent-test excess prediction risk would require separately generated covariates and **must not** be assigned the same exact finite-sample formula without additional assumptions.

The Q1 Monte Carlo sweep compares the sample mean of the corrected in-design quantity with `sigma² d/n` and includes approximate uncertainty intervals while varying sample size, dimension and noise. The implementation uses seed-controlled NumPy generators and a stable least-squares solver rather than an explicit inverse. The automated tests check the corrected numerical behavior; see [Q1 results](Q1_RESULTS.md) for exact commands.

## Interpretation and caveats

- The identity requires the stated noise covariance and full-rank design.
- It concerns in-design prediction risk; training residual error is another quantity.
- The exact identity holds conditional on the design, not because Gaussian features are essential. Gaussian design is the simulation setting requested by the assignment.
- For `d >= n`, the full-rank OLS argument above does not apply as stated.
- Monte Carlo results vary with the random seed and finite number of repetitions; they should not be reported as exact theoretical values.

## Provenance and verification

This audit is based on the original `Computer Class Andrea Bianco.ipynb` (Q1 theory and code) and the ENSAI assignment. The original notebook and this subsequently reconstructed implementation are different artifacts. The corrected simulation is executable from the repository root with `python -m src.q1` and `python -m src.q1_sweep`; the [Q1–Q7 workflow](../.github/workflows/q1.yml) reproduces the outputs and uploads figures and CSV/JSON as build artifacts. The successful run #127 at commit `b2de0fe` establishes that the defined automated checks executed successfully, not that every original-notebook claim has been independently proven.
