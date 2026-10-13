import unittest
import numpy as np
from q5 import data, direction, objective, full_gradient, zero_gradient
from q5_sensitivity import run

class TestQ5Sensitivity(unittest.TestCase):
    def test_direction_covariances(self):
        rng=np.random.default_rng(99)
        d=9
        for name in ('gaussian','rademacher','sphere'):
            us=np.array([direction(rng,d,name) for _ in range(12000)])
            np.testing.assert_allclose(us.T@us/len(us),np.eye(d),atol=.055)
    def test_exact_directional_derivative(self):
        z,y=data(n=35)
        theta=np.linspace(-.2,.2,z.shape[1]);u=np.ones(z.shape[1])
        np.testing.assert_allclose(zero_gradient(z,y,theta,u),u*(u@full_gradient(z,y,theta)),atol=1e-10)
    def test_reproducibility(self):
        z,y=data(n=30)
        a=run(z,y,method='zsgd',distribution='rademacher',scale=.01,budget=200,seed=11)
        b=run(z,y,method='zsgd',distribution='rademacher',scale=.01,budget=200,seed=11)
        self.assertEqual(a,b)
    def test_budget_exact(self):
        z,y=data(n=30)
        for method in ('zgd','zsgd'):
            history=run(z,y,method=method,distribution='sphere',scale=.01,budget=180,seed=1)
            self.assertEqual(history[-1][0],180)
    def test_invalid(self):
        z,y=data(n=30)
        with self.assertRaises(ValueError):
            run(z,y,method='zgd',distribution='sphere',scale=0,budget=180,seed=1)
        with self.assertRaises(ValueError):
            run(z,y,method='zgd',distribution='invalid',scale=.1,budget=180,seed=1)

if __name__=='__main__': unittest.main()
