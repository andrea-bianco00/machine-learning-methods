import unittest
import numpy as np
from q1 import simulate_once, run_experiment


class TestQ1(unittest.TestCase):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            simulate_once(5, 4, 1.0, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            simulate_once(10, 2, -1.0, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            run_experiment(20, 3, 1.0, 1, 0)

    def test_zero_noise(self):
        output = simulate_once(50, 5, 0.0, np.random.default_rng(123))
        self.assertLess(output["in_design_excess_risk"], 1e-25)

    def test_seed_reproducible(self):
        a = run_experiment(50, 3, 1.0, 20, 99)
        b = run_experiment(50, 3, 1.0, 20, 99)
        self.assertEqual(a, b)

    def test_mc_matches_theory(self):
        result = run_experiment(100, 3, 1.0, 1500, 42)
        self.assertLess(abs(result["mean_in_design_excess_risk"] - result["theoretical_risk"]),
                        4 * result["mc_standard_error"])


if __name__ == "__main__":
    unittest.main()
