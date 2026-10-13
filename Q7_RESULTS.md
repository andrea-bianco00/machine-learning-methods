# Q7 (bonus) — MAGIC Gamma Telescope classification

**Status:** Q7 reproduced successfully in GitHub Actions run #51. The numerical results below were verified against the uploaded `q1-q7-experiments.zip` artifact from that run, including `q7.json`, `q7_optimization.csv`, and `q7_comparison.png`. They are not hypothetical benchmark scores.

## Exact assignment scope

The optional Q7 asks to use the [UCI MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), train a baseline such as logistic regression, and compare it with a model trained using zeroth-order stochastic gradient descent (ZSGD). The implementation uses **the same logistic model family** for both approaches to isolate the optimization algorithm. A neural network is permitted by the handout but not required.

**Data source:** R. Bock, *MAGIC Gamma Telescope*, UCI Machine Learning Repository, dataset 159, DOI [10.24432/C52C8B](https://doi.org/10.24432/C52C8B). Official archive: `https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip`. License: CC BY 4.0. This project does not commit or redistribute the dataset.

UCI reports 19,020 simulated events, ten numeric features, 12,332 gamma events (`g`) and 6,688 hadron events (`h`). We map gamma to positive class 1 and hadron to negative class 0. UCI cautions that the dataset's class balance is not representative of deployment and that misidentifying background as signal matters. Consequently ROC-AUC and **gamma true-positive rates under constrained hadron false-positive rates** are reported in addition to conventional metrics.

## Reproducible procedure

1. Read `magic04.data` from the official ZIP or from a user-supplied local copy. Validate 10 numeric columns and `g/h` labels; record a SHA-256 hash of the raw data.
2. Use a seeded stratified split: 60% training, 20% validation and 20% held-out test. Fit `StandardScaler` **only on training data** and transform validation/test using that scaler.
3. Fit scikit-learn `LogisticRegression` with L-BFGS and L2 regularization. This is the conventional first-order baseline.
4. Independently fit logistic regression with our two-point **ZSGD** estimator. For one randomly selected training example and a Rademacher direction `U`, estimate the gradient by `[ell(theta+hU)-ell(theta-hU)] U/(2h)`. Since `E[UU^T]=I`, this estimates the logistic-loss gradient without evaluating derivatives. Use `h=0.01`, `eta_k=0.05/sqrt(k)`, and the same L2 penalty as the baseline. The intercept is not penalized.
5. Give each ZSGD run a fixed budget of 200,000 **component-loss evaluations**, or 100,000 two-point updates. Use three optimizer seeds and retain each run separately. Training-curve diagnostics are explicitly excluded from the optimization-oracle budget.
6. Evaluate both models on the same held-out test data, using a **fixed 0.5 classification threshold**. Report accuracy, balanced accuracy, precision, recall, F1, log-loss, ROC-AUC, confusion matrix, hadron-to-gamma false-positive rate, and maximal gamma true-positive rate at FPR caps of 1%, 2%, 5%, 10% and 20%.

Hyperparameters are **predeclared illustrative choices**, not optimized by searching the held-out test set. The validation metrics are reported separately. No claim of universally optimal tuning is made.

### Why no neural network?

The handout permits logistic regression for the proposed model. Holding the model family fixed makes the comparison scientifically interpretable: the principal experimental difference is how the parameters are optimized. A ZSGD neural network could be a future extension, but it would introduce model-capacity confounding and require additional tuning.

## Reproduce

From the repository root with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
python q7.py --download --budget 200000 --repeats 3 --seed 42 --output results
```

Or supply an official local file:

```bash
python q7.py --data /path/to/magic04.data --budget 200000 --repeats 3 --seed 42 --output results
```

Outputs:
- `results/q7.json`: dataset SHA-256, split counts, settings, baseline and per-seed ZSGD metrics
- `results/q7_optimization.csv`: full training-log-loss checkpoints, with oracle calls
- `results/q7_comparison.png`: optimization trajectories and held-out ROC-AUC comparison

All metrics must be **generated from the real UCI data**, not from the synthetic fixtures in unit tests. The automated workflow downloads the dataset during the Q7 experiment, so a temporary UCI outage can cause the real-data step to fail. The source URL and SHA-256 in the JSON permit provenance checking.

## Verified real-data results (GitHub Actions run #51)

The uploaded artifact reports **19,020 observations** (12,332 gamma and 6,688 hadron) with 11,412 training, 3,804 validation, and 3,804 held-out test observations. Its raw-data SHA-256 is `e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a`.

All figures below are from the **same held-out test set**, using the fixed 0.5 classification threshold. ZSGD results are reported individually, not selectively choosing the best run.

| Model | Accuracy | ROC-AUC | F1 (gamma) | Hadron → gamma false-positive rate |
| --- | ---: | ---: | ---: | ---: |
| Standard logistic regression | 0.783123 | 0.830492 | 0.843780 | 0.438714 |
| ZSGD seed 142 | 0.778917 | 0.826648 | 0.840508 | 0.441704 |
| ZSGD seed 143 | 0.778654 | 0.829815 | 0.840651 | 0.446188 |
| ZSGD seed 144 | 0.777340 | 0.833424 | 0.839856 | 0.449925 |
| **ZSGD median across three seeds** | **0.778654** | **0.829815** | **0.840508** | **0.446188** |

The standard logistic regression test confusion matrix (rows: true hadron, true gamma; columns: predicted hadron, predicted gamma) is `[[751,587],[238,2228]]`. Thus **587 of 1,338 hadrons** in the held-out set are incorrectly classified as gamma at the 0.5 threshold. The gamma recall is **0.903487**. At a more restrictive **5% hadron FPR cap**, the baseline achieves gamma TPR **0.305353** on the test ROC curve; this is a different operating point, not the 0.5-threshold result.

The ZSGD median ROC-AUC is **0.000676 below** the standard baseline (0.829815 vs 0.830492). One ZSGD run exceeds the baseline ROC-AUC slightly, but the three runs are not evidence of systematic superiority. The standard model is marginally better on test accuracy, F1, and hadron false-positive rate at threshold 0.5. ZSGD used exactly **200,000 component-loss calls (100,000 two-point updates)** in each run, with no unused budget.

**Interpretation:** The two approaches give similar held-out discrimination on this fixed split. This supports the feasibility of derivative-free stochastic training for a logistic classifier, not an efficiency advantage over L-BFGS. No repeated data splits, confidence intervals, runtime-matched comparisons or deployment calibration are available.

## Caveats

- The data are Monte Carlo simulations of telescope events, not actual deployment observations. Dataset class proportions are known to be nonrepresentative; raw accuracy and precision do not directly translate to operational performance.
- The 0.5 decision threshold is a reproducible baseline, **not** a scientifically calibrated detection threshold. ROC operating points are more informative for this task.
- Equal model structure and penalty do not imply equal computational cost: L-BFGS and ZSGD have different oracle types and implementations. ZSGD budget accounting excludes diagnostic evaluations.
- One fixed split and three optimizer seeds do not yield robust uncertainty quantification or guarantee generalization to other telescope settings.
- The validation split is held separate, but no hyperparameter search is conducted. This prevents test leakage while leaving performance potentially suboptimal.
