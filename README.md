# Statistical Learning and Optimization Methods

**ENSAI Machine Learning Computer Class (2025) — reproducible Python reconstruction and methodological audit**

A seven-question university project (including an optional classification bonus) exploring statistical risk, regression under model misspecification, and first- versus zeroth-order optimization. The original assignment and notebook motivated this repository; the cleaned Python implementations, corrections, tests, and sensitivity experiments are **subsequent reconstruction work**, not a verbatim copy of the original submission.

[![Q1–Q7 verification](https://github.com/andrea-bianco00/machine-learning-methods/actions/workflows/q1.yml/badge.svg)](https://github.com/andrea-bianco00/machine-learning-methods/actions/workflows/q1.yml)

## Project at a glance

| Question | Topic | Reproduction and analysis |
| --- | --- | --- |
| Q1 | OLS in-design excess prediction risk | [Results and derivation](Q1_RESULTS.md) |
| Q2 | Gaussian features and a nonlinear ReLU signal | [Results](Q2_RESULTS.md) |
| Q3 | Linear approximation and population prediction risk | [Results](Q3_RESULTS.md) |
| Q4 | Full-batch GD versus stochastic GD | [Results](Q4_RESULTS.md) |
| Q5 | Two-point zeroth-order GD and SGD | [Baseline and corrections](Q5_RESULTS.md) |
| Q5 extension | Direction distributions and step-size sensitivity | [Exploratory analysis](Q5_SENSITIVITY.md) |
| Q6 | Flexible ReLU regression, ZGD/ZSGD and independent test risk | [Results](Q6_RESULTS.md) |
| Q7 (bonus) | UCI MAGIC classification: logistic regression vs ZSGD | [Protocol and results](Q7_RESULTS.md) |

**Q1–Q6 are the core assignment; Q7 is an optional classification bonus.** The real-data Q7 experiment and its test metrics have been checked against the successful GitHub Actions run #51 artifacts. See [Q7 verified results](Q7_RESULTS.md#verified-real-data-results-github-actions-run-51).

**[View the figure index and download instructions](FIGURES_AND_RESULTS.md)** — plots, numerical artifacts, interpretation notes and full reproduction commands.

## Key findings

- **Risk definitions matter:** Q1 distinguishes in-design excess prediction risk from squared coefficient error; Q3 distinguishes training error, model-approximation error and conditional population risk.
- **Optimization budgets matter:** Q4 counts component-gradient evaluations. Q5–Q6 count component-function evaluations. These are *different oracle types*, not directly equivalent CPU time.
- **Unbiasedness is not stability:** Correctly scaled zeroth-order gradient estimates can have large variance; the Q5 extension isolates the effects of direction distribution and learning-rate prefactor.
- **Model flexibility matters:** On the simulated Q6 data with a ReLU signal, the flexible model trained by ZSGD attains lower held-out mean squared error than linear ZSGD under the specified budget. These are descriptive results for a controlled simulation, not universal performance claims.
- **Derivative-free classification is feasible:** On the real UCI MAGIC test split, standard logistic regression achieved ROC-AUC **0.8305**, versus **0.8266–0.8334** across three ZSGD seeds. At the fixed 0.5 threshold, the baseline misclassified **43.9% of hadrons** as gamma; [operating-point limitations](Q7_RESULTS.md) matter more than accuracy alone.

## Reproduce from scratch

Requires **Python 3.11**. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v

python q1.py --output results/q1.json
python q1_sweep.py --output results --repetitions 400 --seed 42
python q2.py --n 3000 --p 8 --noise-variance 0.5 --seed 42
python q3.py --seed 42 --noise-sd 0.5 --output results
python q4.py --budget 30000 --repeats 5 --data-seed 42 --sgd-seed 100 --output results
python q5.py --budget 30000 --repeats 5 --seed 42 --h 0.01 --output results
python q5_sensitivity.py --budget 30000 --repeats 5 --output results
python q6.py --budget 120000 --repeats 5 --step-scale 0.03 --output results
python q7.py --download --budget 200000 --repeats 3 --seed 42 --output results
```

Q1–Q6 use synthetic data. **Q7 uses the real UCI-hosted MAGIC dataset**, downloaded on request (or read from a local official copy); therefore Q7 requires access to the official data source. All experiments use fixed random seeds and pinned dependencies; no notebook execution order is required. Outputs are generated under `results/` (JSON/CSV/PNG). The GitHub Actions workflow [Q1–Q7 verification](.github/workflows/q1.yml) runs unit tests and the complete experiment suite on pushes and pull requests and uploads generated artifacts. **A passing workflow establishes successful execution of these tests and commands, not mathematical proof or exhaustive reproducibility across every platform.**

## Methodological transparency

The original notebook and the reconstructed experiments should not be conflated. Important choices and corrections are documented alongside each question:

1. Q1's original code measured squared parameter error, whereas the assignment's formula concerns in-design prediction risk. See [methodology audit](METHODOLOGY_AUDIT.md).
2. Q2 uses noise **variance 0.5** as specified in its assignment. Q3–Q6 preserve the original notebook's noise **standard deviation 0.5** (variance 0.25). They are intentionally different experimental settings.
3. Q5 uses the mathematically correct centered difference denominator **2h** and directions satisfying `E[UUᵀ]=I`. The handout's expression uses `h`, producing a factor-of-two discrepancy.
4. Q5's sensitivity analysis is an **explicitly labeled post-assignment extension**.
5. Q6 uses a declared step-size schedule and initialization, with independent held-out data and component-function-evaluation budgets.

The original notebook acknowledged AI assistance. This reconstructed repository has also been developed with AI assistance and reviewed through mathematical checks, unit tests, and GitHub Actions. The project should be assessed by its transparent methods and executable results, not by claims of independent authorship of every line.

## Repository structure

```text
.
├── q1.py, q1_sweep.py, q2.py, q3.py, q4.py, q5.py, q5_sensitivity.py, q6.py, q7.py
├── test_q1.py ... test_q7.py, test_q5_sensitivity.py
├── Q1_RESULTS.md ... Q7_RESULTS.md, Q5_SENSITIVITY.md
├── METHODOLOGY_AUDIT.md, FIGURES_AND_RESULTS.md
├── requirements.txt
└── .github/workflows/q1.yml
```

The current flat layout is intentional: scripts can be run directly from the repository root without package installation. A future package-style reorganization would require coordinated import, CLI, and CI updates.

## Scope and limitations

Q1–Q6 experiments are synthetic; Q7 uses an official UCI classification dataset. Experiments primarily use fixed datasets and small numbers of stochastic repetitions, and generally report training objectives or held-out MSE rather than deployment metrics. Optimization-oracle budgets exclude diagnostic full-loss evaluations and do not normalize wall-clock runtime. Hyperparameters are illustrative, not the result of a broad validation-based search. Q7's ROC-AUC values are close across models on one split; this does not establish statistical equivalence, computational parity, or deployment suitability. Consult the question-specific reports for details.
