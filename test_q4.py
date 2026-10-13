import unittest
import numpy as np
from q4 import make_data, loss, component_gradient, full_gradient, optimize, experiment


class TestQ4(unittest.TestCase):
    def test_gradient_identity(self):
        z, y = make_data(n=50)
        theta = np.arange(z.shape[1]) / 10
        np.testing.assert_allclose(full_gradient(z, y, theta),
            np.mean([component_gradient(a, b, theta) for a, b in zip(z, y)], axis=0), atol=1e-12)

    def test_finite_difference(self):
        z, y = make_data(n=60)
        theta = np.ones(z.shape[1]) / 5
        eps = 1e-6
        for j in range(len(theta)):
            direction = np.eye(len(theta))[j] * eps
            numerical = (loss(z, y, theta + direction) - loss(z, y, theta - direction)) / (2 * eps)
            self.assertAlmostEqual(numerical, full_gradient(z, y, theta)[j], places=7)

    def test_budget_accounting(self):
        z, y = make_data(n=30)
        _, gd = optimize(z, y, 90, "gd", checkpoint_every=30)
        _, sgd = optimize(z, y, 90, "sgd", checkpoint_every=30)
        self.assertEqual([x for x, _ in gd], [0, 30, 60, 90])
        self.assertEqual([x for x, _ in sgd], [0, 30, 60, 90])

    def test_repeatability(self):
        z, y = make_data(n=30)
        a, b = optimize(z, y, 90, "sgd", seed=13)
        c, d = optimize(z, y, 90, "sgd", seed=13)
        np.testing.assert_array_equal(a, c)
        self.assertEqual(b, d)

    def test_ols_lower_bound(self):
        rows, summary = experiment(n=50, budget=500, repeats=3)
        self.assertTrue(all(r["gd_objective"] >= summary["ols_minimum"] - 1e-10 for r in rows))
        self.assertTrue(all(r["sgd_min_objective"] >= summary["ols_minimum"] - 1e-10 for r in rows))

    def test_invalid_inputs(self):
        z, y = make_data(n=30)
        with self.assertRaises(ValueError):
            optimize(z, y, -1, "sgd")
        with self.assertRaises(ValueError):
            experiment(n=50, budget=51)


if __name__ == "__main__":
    unittest.main()
