# Statistical Learning and Optimization

*ENSAI Machine Learning Computer Class · 2025 · Python*

This project grew out of a machine learning assignment I completed at ENSAI. It starts with a simple question—how does prediction error behave in linear regression?—and gradually moves toward optimization algorithms that do not require gradients. The final, optional exercise applies one of those methods to the UCI MAGIC Gamma Telescope classification dataset.

The code here is a **later, reproducible reconstruction** of the assignment, rather than the original notebook. I revisited several formulas, implemented standalone experiments, added automated tests, and documented where the reconstructed methods differ from the original work.

[![Q1–Q7 verification](https://github.com/andrea-bianco00/machine-learning-methods/actions/workflows/q1.yml/badge.svg)](https://github.com/andrea-bianco00/machine-learning-methods/actions/workflows/q1.yml)

## The seven questions

| Question | What I investigated | What the experiment shows |
| --- | --- | --- |
| [Q1](docs/Q1_RESULTS.md) | How do sample size, dimension and noise affect OLS excess risk? | The expected **in-design** risk follows `σ²(p+1)/n`; it is not the same as squared coefficient error. |
| [Q2](docs/Q2_RESULTS.md) | Can plots reveal a nonlinear signal in Gaussian data? | `X1` has a linear effect, while `X2` contributes through a ReLU function; the other six variables have no population effect. |
| [Q3](docs/Q3_RESULTS.md) | What happens when linear regression approximates a nonlinear model? | The population risk approaches **0.340845** under the specified noise setting, including a nonzero approximation error. |
| [Q4](docs/Q4_RESULTS.md) | How do full-batch GD and SGD compare? | Both approach the OLS minimum, but take very different numbers of updates under a shared component-gradient budget. |
| [Q5](docs/Q5_RESULTS.md) | Can gradients be estimated using function values alone? | Yes, but the resulting zeroth-order methods can be unstable; [additional experiments](docs/Q5_SENSITIVITY.md) explore directions and step sizes. |
| [Q6](docs/Q6_RESULTS.md) | Does a flexible ReLU model improve prediction with zeroth-order training? | On the simulated held-out set, flexible ZSGD achieves a median MSE of about **0.2461**, versus **0.5205** for linear ZSGD. |
| [Q7, bonus](docs/Q7_RESULTS.md) | Can ZSGD train a classifier on UCI MAGIC? | The three ZSGD runs achieve test ROC-AUC **0.8266–0.8334**, close to **0.8305** for standard logistic regression. |

Each linked report explains the question, implementation, numerical results and limitations. The [figure and artifact guide](docs/FIGURES_AND_RESULTS.md) explains how to inspect the plots and download outputs from GitHub Actions.

## What I learned

The early experiments made the distinction between different notions of risk particularly important. In Q1, the assignment's theoretical expression concerns prediction error on the observed design, while the original notebook had measured parameter error. In Q3, a linear model cannot fully represent the ReLU signal, even with unlimited data.

The optimization experiments raised a different issue: **a valid gradient estimator is not necessarily a stable optimizer**. Q5 uses a correctly scaled two-point estimator, yet some learning-rate and random-direction combinations diverge. I kept those trajectories in the results instead of discarding them.

For Q7, I used the same logistic model for L-BFGS and ZSGD to make the optimization comparison easier to interpret. Their ROC-AUC scores are similar on one held-out split. That does **not** establish equal computational efficiency or general superiority of ZSGD. At a threshold of 0.5, the standard model also misclassifies about **43.9% of hadrons** as gamma, which is why the [Q7 report](docs/Q7_RESULTS.md) includes operating points with constrained false-positive rates.

## Run the experiments

Use **Python 3.11** from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v

python -m src.q1 --output results/q1.json
python -m src.q1_sweep --output results --repetitions 400 --seed 42
python -m src.q2 --n 3000 --p 8 --noise-variance 0.5 --seed 42
python -m src.q3 --seed 42 --noise-sd 0.5 --output results
python -m src.q4 --budget 30000 --repeats 5 --data-seed 42 --sgd-seed 100 --output results
python -m src.q5 --budget 30000 --repeats 5 --seed 42 --h 0.01 --output results
python -m src.q5_sensitivity --budget 30000 --repeats 5 --output results
python -m src.q6 --budget 120000 --repeats 5 --step-scale 0.03 --output results
python -m src.q7 --download --budget 200000 --repeats 3 --seed 42 --output results
```

Q1–Q6 generate synthetic data locally. Q7 downloads the official [UCI MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), or accepts a local copy with `--data`. The dataset is not committed to this repository.

Generated JSON, CSV and PNG files go into `results/`. The [GitHub Actions workflow](.github/workflows/q1.yml) reruns the tests and experiments and uploads their outputs as an artifact. A successful run confirms that the automated checks passed; it is not proof that every scientific assumption is correct.

## Repository layout

```text
src/                    Python experiments, Q1–Q7
tests/                  Unit tests and plotting smoke tests
docs/                   Results, methodological audit and figure guide
.github/workflows/      Automated tests and reproduction
README.md               Project overview
requirements.txt        Pinned dependencies
```

The `results/` directory holds generated outputs. An early Q1 JSON result is versioned in the repository; the complete current set of JSON, CSV and PNG outputs is available from successful GitHub Actions run artifacts or can be regenerated locally.

## Decisions and limitations

Several details matter when reproducing or interpreting the comparisons:

- Q2 uses noise **variance 0.5**, as stated in that question. Q3–Q6 use noise **standard deviation 0.5**, following the original notebook.
- Q5 corrects the centered finite-difference denominator to **2h** and uses random directions with covariance equal to the identity. The [methodology audit](docs/METHODOLOGY_AUDIT.md) records discrepancies with the source exercise.
- Q4 counts component-gradient calls; Q5–Q6 count component-function-value calls. These are not equivalent measures of elapsed time. Diagnostic evaluations are excluded from the optimization budgets.
- Q6 uses a declared step-size schedule different from Q5. The Q5 sensitivity study is an extension beyond the assignment.
- Q7 uses one fixed stratified split and three ZSGD seeds, without a comprehensive hyperparameter search or repeated-split uncertainty analysis. MAGIC contains simulated events, and its class proportions are not representative of operational deployment.

The original notebook acknowledged AI assistance, and AI tools were also used during this reconstruction. The code, assumptions, tests and outputs are included so the work can be inspected rather than taken on trust.
