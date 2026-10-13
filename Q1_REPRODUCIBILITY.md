# Q1 reproduction

This is a reconstructed implementation of ENSAI Q1, not the original submitted notebook. The corrected estimand is **in-design excess prediction risk**. See [METHODOLOGY_AUDIT.md](METHODOLOGY_AUDIT.md) for the mathematical distinction.

## Run

Python 3.10+ recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python q1.py --n 200 --p 8 --sigma 1 --repetitions 1000 --seed 42 --output results/q1.json
```

The experiment writes a JSON result and prints the same data. Running the command again with the same NumPy version and seed should reproduce the saved values; numerical results may differ with other library versions/platforms.

## Recorded local run

- n = 200, p = 8 (plus intercept), sigma = 1, 1000 Monte Carlo repetitions, seed = 42.
- Theory: 0.045.
- Monte Carlo mean: 0.04546727713447846.
- Monte Carlo standard error: 0.0006810498019287772.
- Approximate 95% normal Monte Carlo interval: [0.04413241952269805, 0.04680213474625886].

Four unit tests passed locally, including reproducibility, noiseless OLS and Monte Carlo agreement. These tests are evidence for Q1 only, **not** a complete independent clean-environment verification of the whole academic project.

## Limitations

The 95% interval uses a normal approximation to the Monte Carlo mean. The simulation assumes independent Gaussian features and homoscedastic Gaussian noise; it evaluates prediction error at the *training design*, not an independent test design. The code uses a stable least-squares solver.
