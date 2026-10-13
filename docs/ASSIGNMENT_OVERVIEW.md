# Original assignment — summary

**Course:** Machine Learning, Computer Class, ENSAI (2025)  
**Source:** Original two-page course handout, *Machine learning — Computer class* (`TP_25.pdf`), supplied by the student.  
**Status:** This is an independently worded summary of the assignment, **not** a transcription, an official ENSAI publication, or the original PDF.

The original assignment required individual work submitted as a single Jupyter notebook and counted for 50% of the final course grade. It allocated five of the twenty assignment points to presentation, code quality and execution without errors. Some questions were open-ended, and originality was encouraged.

## Questions in the handout

### Q1 — Expected excess risk in linear regression

Use simulations with independent Gaussian covariates and Gaussian noise to illustrate the stated expected excess-risk expression, `σ²(p + 1)/n`, for random-design linear regression. Recall and explain the relevant course formulas carefully.

[Implementation and discussion](Q1_RESULTS.md)

### Q2 — Recognizing a nonlinear signal

Generate 3,000 observations with eight independent standard Gaussian covariates. The response combines the first covariate linearly, the positive part of the second covariate and Gaussian noise of variance 0.5. Use simple plots to show how the relationships involving the first two covariates differ from those involving the remaining variables.

[Implementation and discussion](Q2_RESULTS.md)

### Q3 — Fitting linear regression to the data

Fit a linear regression model with a standard scikit-learn implementation. Report its empirical squared-error risk for sample sizes 100, 500, 1,000, and then increasing by 500 up to 3,000. Discuss the observed pattern.

[Implementation and discussion](Q3_RESULTS.md)

### Q4 — Gradient descent and stochastic gradient descent

Implement GD and SGD for the preceding squared-error objective using basic numerical tools rather than an optimization package. Use the learning rate `1/k` and match the number of individual-observation gradient evaluations across the two methods. Provide pseudocode, repeat the experiments and plot the objective trajectories to compare deterministic GD with stochastic SGD.

[Implementation and discussion](Q4_RESULTS.md)

### Q5 — Zeroth-order optimization

Study gradient estimation from function values when analytical gradients are unavailable. Explain the finite-difference expression given in the handout, choose an appropriate random direction, and provide pseudocode. Implement full-batch zeroth-order gradient descent (ZGD) and its stochastic counterpart (ZSGD), then compare them with GD and SGD.

**Important distinction:** The handout writes a symmetric finite-difference expression with denominator `h`; the reconstruction discusses and uses the conventional centered-difference denominator `2h`. See the methodology audit for why this matters.

[Implementation and discussion](Q5_RESULTS.md) · [Methodology audit](METHODOLOGY_AUDIT.md)

### Q6 — A more flexible prediction model

Consider a predictor combining a linear term with the positive part of a second linear term. Fit it using ZGD and ZSGD under matched function-evaluation budgets. Estimate predictive risk using 1,000 additional observations generated from the same data-generating process. Compare ZSGD with the flexible predictor against ZSGD with a linear predictor, again controlling the individual-function-evaluation budget.

[Implementation and discussion](Q6_RESULTS.md)

### Q7 — Optional bonus: classification

Using the [UCI MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), train a baseline classifier, such as logistic regression, and compare it with a model trained using stochastic zeroth-order optimization. Explain the chosen procedure and its implementation.

This question is explicitly identified as a **bonus** in the original handout.

[Implementation and discussion](Q7_RESULTS.md)

## How to read this repository

This page describes **what the assignment requested**, not what was demonstrated experimentally. For the actual implementations, results, methodological changes and limitations, follow the individual reports linked above or return to the [project README](../README.md).

The original handout is **not redistributed** here because permission to publish the course material has not been established. This summary is provided for context and does not claim endorsement by ENSAI.
