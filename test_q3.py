import unittest
import numpy as np
from q3 import generate, approximation_error, evaluate

class TestQ3(unittest.TestCase):
    def test_population_approximation(self):
        self.assertAlmostEqual(approximation_error(), 0.25-1/(2*np.pi))
    def test_reproducibility(self):
        self.assertEqual(evaluate(), evaluate())
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate(noise_sd=-1)
        with self.assertRaises(ValueError):
            evaluate(n_values=[5])
    def test_population_decomposition(self):
        for row in evaluate():
            self.assertAlmostEqual(row["population_risk"],
                row["noise_variance"]+row["approximation_error"]+row["estimation_error"])
            self.assertGreaterEqual(row["estimation_error"], 0)
    def test_large_sample(self):
        row=evaluate(n_values=[100,10000],seed=123)[-1]
        self.assertLess(row["estimation_error"], 0.01)
        self.assertLess(abs(row["empirical_risk"]-(0.25+approximation_error())), 0.025)

if __name__=="__main__":
    unittest.main()
