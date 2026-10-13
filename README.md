# Statistical Learning and Optimization Methods

**An ENSAI university computer-class project in machine learning (2025).**

This repository documents a six-question academic assignment combining statistical learning theory, simulated regression, and first- and zeroth-order optimization. It is being reorganized from the original submitted Jupyter notebook into a readable, testable and reproducible research-style project.

> **Status: reconstruction and methodological audit in progress.** The original notebook has not yet been independently executed end-to-end in a clean environment. Results and code should not be treated as independently reproduced until that verification is complete.

## Questions addressed

| Question | Focus |
| --- | --- |
| Q1 | Gaussian-design linear regression and expected excess risk |
| Q2 | Simulated nonlinear data-generating process and feature-response relationships |
| Q3 | Ordinary least squares and empirical risk across sample sizes |
| Q4 | Full-batch gradient descent versus stochastic gradient descent |
| Q5 | Zeroth-order gradient estimation and stochastic variants |
| Q6 | Flexible nonlinear model, optimization and independent test-risk estimation |

The assignment also offered an optional Q7 classification extension, which is **not included** in the provided main notebook.

## What this project aims to demonstrate

- Careful distinction between empirical and generalization risk.
- Controlled numerical experiments, with reproducible random seeds and clear estimands.
- Correct algorithm implementations and transparent accounting of gradient/function evaluations.
- Fair comparisons across optimization methods and explicit discussion of limitations.
- Readable explanations of each university question, with code and plots that can be checked independently.

## Provenance

The source material is Andrea Bianco's ENSAI *Machine Learning — Computer Class* assignment and original 2025 notebook. The original notebook acknowledges the use of AI as a supporting tool. The repository's cleaned implementation, tests and documentation are subsequent reconstruction work; any corrections will be distinguished from the original submission.

## Reproduction status

The original notebook contains environment-specific visualization and parallel-processing code and shared state between questions. Before publishing a reproduction procedure, we will:

1. Audit the mathematics and original results question by question.
2. Extract independent Python modules and unit tests.
3. Define pinned dependencies and deterministic experiment configurations where feasible.
4. Execute the full pipeline from a clean environment and save machine-generated results.
5. Document any numerical variability and remaining limitations.

**No verified end-to-end reproduction or final benchmark results are claimed at this stage.**
