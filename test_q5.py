import unittest
import numpy as np
from q5 import (data, direction, full_gradient, component_gradient,
                zero_gradient, optimize, experiment, objective)

class TestQ5(unittest.TestCase):
    def test_quadratic_two_point_exact_directional_derivative(self):
        z, y = data(n=40)
        theta = np.arange(z.shape[1])/10
        u = np.arange(z.shape[1])+1.
        expected = u * (u @ full_gradient(z,y,theta))
        np.testing.assert_allclose(zero_gradient(z,y,theta,u,h=0.03), expected, atol=1e-10)

    def test_stochastic_two_point_identity(self):
        z,y=data(n=40)
        theta=np.arange(z.shape[1])/10
        u=np.arange(z.shape[1])+1.
        i=3
        np.testing.assert_allclose(zero_gradient(z,y,theta,u,h=0.03,stochastic_index=i),
                                   u*(u@component_gradient(z[i],y[i],theta)),atol=1e-10)

    def test_direction_covariance(self):
        rng=np.random.default_rng(11)
        for dist in ('gaussian','rademacher','sphere'):
            u=np.array([direction(rng,4,dist) for _ in range(12000)])
            np.testing.assert_allclose(u.T@u/len(u), np.eye(4), atol=0.065)

    def test_budget_exact(self):
        z,y=data(n=30)
        for method,cost in [('gd',30),('sgd',1),('zgd',60),('zsgd',2)]:
            _,hist,meta=optimize(z,y,method,125,seed=2)
            self.assertEqual(meta['oracle_calls'],(125//cost)*cost)
            self.assertEqual(hist[-1]['calls'],meta['oracle_calls'])
            self.assertLessEqual(meta['oracle_calls'],125)

    def test_reproducibility(self):
        z,y=data(n=30)
        for method in ('gd','sgd','zgd','zsgd'):
            a,h,_=optimize(z,y,method,120,seed=15)
            b,k,_=optimize(z,y,method,120,seed=15)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(h,k)

    def test_lower_bound_and_finite(self):
        z,y=data(n=30)
        opt=objective(z,y,np.linalg.lstsq(z,y,rcond=None)[0])
        for method in ('gd','sgd','zgd','zsgd'):
            _,h,_=optimize(z,y,method,180,seed=5)
            self.assertTrue(all(np.isfinite(r['objective']) and r['objective']>=opt-1e-8 for r in h))

    def test_validation(self):
        z,y=data(n=30)
        with self.assertRaises(ValueError):
            optimize(z, y, 'zsgd', -1)
        with self.assertRaises(ValueError):
            optimize(z, y, 'zsgd', 60, h=0)
        with self.assertRaises(ValueError):
            optimize(z, y, 'invalid', 60)
        with self.assertRaises(ValueError):
            experiment(repeats=1)

if __name__=='__main__':
    unittest.main()
