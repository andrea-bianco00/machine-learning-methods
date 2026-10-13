# Methodology audit — Q1: expected excess risk

**Status:** source-level audit completed for Q1; corrected simulation and clean-environment execution are pending. This document distinguishes the original notebook's derivation from the quantity actually computed in its code.

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

## Correction required

The reproducible Q1 implementation must explicitly calculate:

1. **In-design excess prediction risk:** `mean((Z @ (theta_hat-theta_star))**2)`.
2. **Squared parameter estimation error:** `sum((theta_hat-theta_star)**2)`, separately labelled, if shown.
3. **Optional independent-test excess prediction risk:** evaluate on independently generated covariates; do not apply the in-design formula as an exact finite-sample identity to this different estimand.

The main figure should compare the Monte Carlo mean of item 1 against `sigma² d/n`, with uncertainty intervals, and show the effect of varying `n`, `p`, and `sigma`. The random generator should be explicit, with reproducible seeds and no global `np.random.seed` mutations.

Use `np.linalg.lstsq` or an equivalent stable solver rather than explicitly inverting `Z'Z`. Tests must check shape/rank conditions, the exact projection identity on simulated data, and Monte Carlo agreement within a justified stochastic tolerance.

## Interpretation and caveats

- The identity requires the stated noise covariance and full-rank design.
- It concerns in-design prediction risk; training residual error is another quantity.
- The exact identity holds conditional on the design, not because Gaussian features are essential. Gaussian design is the simulation setting requested by the assignment.
- For `d >= n`, the full-rank OLS argument above does not apply as stated.
- Monte Carlo results vary with the random seed and finite number of repetitions; they should not be reported as exact theoretical values.

## Provenance and next step

This audit is based on the original `Computer Class Andrea Bianco.ipynb` (Q1 theory and code) and the ENSAI assignment. The original notebook must be preserved separately; the correction belongs to the subsequently reconstructed version. No new simulation outputs have yet been claimed or published.
