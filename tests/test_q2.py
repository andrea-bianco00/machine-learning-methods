import unittest
import numpy as np
from src.q2 import generate_data, analyze, theoretical_moments


class TestQ2(unittest.TestCase):
    def test_seed_and_shapes(self):
        a, b = generate_data()
        c, d = generate_data()
        self.assertEqual(a.shape, (3000, 8))
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)

    def test_invalid_inputs(self):
        for args in [(1, 8, 0.5), (3000, 1, 0.5), (3000, 8, -0.1), (3000, 8, float("nan"))]:
            with self.assertRaises(ValueError):
                generate_data(*args)

    def test_noiseless_structural_identity(self):
        x, y = generate_data(noise_variance=0)
        np.testing.assert_allclose(y, x[:, 0] + np.maximum(x[:, 1], 0))

    def test_population_moments(self):
        m = theoretical_moments()
        self.assertAlmostEqual(m["variance_y"], 2 - 1 / (2 * np.pi), places=12)
        self.assertAlmostEqual(m["correlation_x2_y"], 0.5 / np.sqrt(m["variance_y"]))

    def test_empirical_signal(self):
        x, y = generate_data(n=30000)
        a = analyze(x, y)
        m = theoretical_moments()
        self.assertLess(abs(a["mean_y"] - m["mean_y"]), 0.03)
        self.assertLess(abs(a["correlations"][0] - m["correlation_x1_y"]), 0.03)
        self.assertLess(abs(a["correlations"][1] - m["correlation_x2_y"]), 0.03)
        self.assertLess(max(abs(c) for c in a["correlations"][2:]), 0.03)


if __name__ == "__main__":
    unittest.main()
