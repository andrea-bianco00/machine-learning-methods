import unittest
import numpy as np
from q6 import generate,predict,loss,initial_theta,estimate,optimize,direction,experiment

class TestQ6(unittest.TestCase):
    def test_dgp_true_model(self):
        x,y=generate(n=100,p=8,noise_sd=0,seed=3)
        theta=np.zeros(16);theta[0]=1;theta[9]=1
        np.testing.assert_allclose(predict(x,theta,'flexible'),y,atol=1e-12)

    def test_linear_special_case(self):
        x,y=generate(n=50)
        beta=np.arange(8)/10
        theta=np.r_[beta,np.zeros(8)]
        np.testing.assert_allclose(predict(x,theta,'flexible'),predict(x,beta,'linear'))

    def test_centered_difference_linear_exact(self):
        x,y=generate(n=50)
        beta=np.arange(8)/10
        u=np.arange(8)+1.
        g=2*x.T@(x@beta-y)/len(y)
        np.testing.assert_allclose(estimate(x,y,beta,u,'linear',h=.01),u*(u@g),rtol=1e-9,atol=1e-9)

    def test_directions_covariance(self):
        rng=np.random.default_rng(2)
        u=np.array([direction(rng,4) for _ in range(10000)])
        np.testing.assert_allclose(u.T@u/len(u),np.eye(4),atol=.04)

    def test_budget_no_overshoot(self):
        x,y=generate(n=30)
        for model,method,cost in [('linear','zsgd',2),('flexible','zsgd',2),('flexible','zgd',60)]:
            theta,h,meta=optimize(x,y,model,method,budget=125,seed=4)
            self.assertEqual(meta['oracle_calls'],125//cost*cost)
            self.assertEqual(h[-1]['calls'],meta['oracle_calls'])
            self.assertTrue(np.all(np.isfinite(theta)))

    def test_repeatability(self):
        x,y=generate(n=30)
        for method in ('zgd','zsgd'):
            a,h,_=optimize(x,y,'flexible',method,240,seed=2)
            b,k,_=optimize(x,y,'flexible',method,240,seed=2)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(h,k)

    def test_independent_test_data(self):
        x,y=generate(n=100,seed=42)
        xt,yt=generate(n=100,seed=123)
        self.assertFalse(np.array_equal(x,xt))
        self.assertFalse(np.array_equal(y,yt))

    def test_input_validation(self):
        x,y=generate(n=30)
        with self.assertRaises(ValueError): optimize(x,y,'flexible','zsgd',budget=-1)
        with self.assertRaises(ValueError): optimize(x,y,'flexible','zsgd',budget=10,h=0)
        with self.assertRaises(ValueError): optimize(x,y,'flexible','zsgd',budget=10,step_scale=0)
        with self.assertRaises(ValueError): predict(x,np.zeros(7),'linear')
        with self.assertRaises(ValueError): experiment(train_seed=1,test_seed=1)

if __name__=='__main__': unittest.main()
