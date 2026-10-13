# Q7 (bonus) — MAGIC Gamma Telescope classification

**Status:** implementation and tests committed; **real-dataset benchmark metrics must be read from the latest successful GitHub Actions artifact**. No accuracy, ROC-AUC or confusion matrix is invented in this document.

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

## Caveats

- The data are Monte Carlo simulations of telescope events, not actual deployment observations. Dataset class proportions are known to be nonrepresentative; raw accuracy and precision do not directly translate to operational performance.
- The 0.5 decision threshold is a reproducible baseline, **not** a scientifically calibrated detection threshold. ROC operating points are more informative for this task.
- Equal model structure and penalty do not imply equal computational cost: L-BFGS and ZSGD have different oracle types and implementations. ZSGD budget accounting excludes diagnostic evaluations.
- One fixed split and three optimizer seeds do not yield robust uncertainty quantification or guarantee generalization to other telescope settings.
- The validation split is held separate, but no hyperparameter search is conducted. This prevents test leakage while leaving performance potentially suboptimal.
