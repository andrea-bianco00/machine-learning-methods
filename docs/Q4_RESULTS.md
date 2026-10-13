# Q4 — First-order optimization: GD versus SGD

## ENSAI assignment and objective

Implement gradient descent (GD) and stochastic gradient descent (SGD) **without an optimization library** on the linear least-squares objective `f(theta)=mean_i (y_i-z_i^T theta)^2`. Sample SGD indices uniformly with replacement from the 3,000 observations. Both methods use `eta_k=1/k`, where k is that algorithm's update count. Compare repeated SGD trajectories and deterministic GD using **equal numbers of component-gradient evaluations**. The intercept is included in z.

## Pseudocode

**GD**, each step costs n component-gradient evaluations:

```text
theta = 0
for k in 1,...,floor(B/n):
    g = (1/n) sum_i 2*z_i*(z_i^T*theta - y_i)
    theta = theta - g/k
```

**SGD**, each step costs one component-gradient evaluation:

```text
theta = 0
for k in 1,...,B:
    i = UniformInteger(1,n), sampled with replacement
    g = 2*z_i*(z_i^T*theta - y_i)
    theta = theta - g/k
```

The full objective is evaluated for diagnostics at matching budgets (0, n, 2n, ...). **These diagnostic evaluations are not charged to the component-gradient budget**; they are not used by either optimizer and are reported separately from optimization costs. If total wall-clock cost is the target, diagnostic evaluation costs must also be included.

## Design and local results

- Same fixed data, n=3000, p=8 plus intercept, Gaussian noise **standard deviation 0.5**, data seed 42.
- Equal per-algorithm budget B=30,000 component-gradient evaluations: 10 GD updates versus 30,000 SGD updates.
- GD is deterministic on the fixed data; SGD is repeated five times with distinct sampling seeds 100–104.
- Objective benchmark: exact least-squares minimum computed by `np.linalg.lstsq`.

| Quantity | Local result |
| --- | ---: |
| OLS minimum | 0.3460203750 |
| Final GD objective | 0.3460224578 |
| Final SGD objective (five-run mean) | 0.3461215602 |

These numbers describe **one data realization** and five stochastic index sequences. They do not establish general superiority of either method. The SGD band in the figure is ±1 *sample standard deviation across runs*, not a confidence interval. The learning-rate schedule `1/k` is applied to each algorithm's own iteration counter exactly as required, so it is not a comparison under identical learning rates as functions of gradient-evaluation budget.

## Reproduction

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.q4 --budget 30000 --repeats 5 --data-seed 42 --sgd-seed 100 --output results
```

Produces `results/q4.json`, `results/q4.csv`, and `results/q4_convergence.png`. The optimization algorithms use only NumPy; Matplotlib is used exclusively for plotting. No Dash server or multiprocessing is required.

## Validation and caveats

Six Q4 tests cover component/full gradient agreement, a central finite-difference check, budget accounting, fixed-seed reproducibility, the OLS lower bound, and input validation. The [Q1–Q6 verification workflow](../.github/workflows/q1.yml) reruns the tests and regenerates the plots. The successful Q1–Q6 workflow execution was confirmed on commit `761d209` (run #27); check the [Actions page](https://github.com/andrea-bianco00/machine-learning-methods/actions) for newer runs. All objective curves are **training objectives**, not out-of-sample risk. A full-batch gradient costs n component evaluations and a stochastic gradient costs one; the budget does not include diagnostic full-loss evaluations or Python overhead. Q5 zero-order function-evaluation budgets require a different accounting convention.
