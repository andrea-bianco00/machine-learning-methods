"""Unit tests use synthetic fixtures; they do not pretend to be UCI results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from q7 import (load_magic,split_and_scale,component_loss,two_point,
                zsgd,scores,metrics,experiment)


class TestQ7(unittest.TestCase):
    def fixture(self,n=90):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(n,10))
        y=np.r_[np.zeros(n//2,dtype=int),np.ones(n-n//2,dtype=int)]
        return x,y

    def test_file_labels_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"magic04.data"
            path.write_text(",".join(["1"]*10+["g"])+"\n"+",".join(["2"]*10+["h"])+"\n")
            x,y,digest=load_magic(path)
            self.assertEqual(x.shape,(2,10))
            np.testing.assert_array_equal(y,[1,0])
            self.assertEqual(len(digest),64)

    def test_reject_bad_record(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"magic04.data"
            path.write_text(",".join(["1"]*10+["unknown"])+"\n")
            with self.assertRaises(ValueError): load_magic(path)

    def test_stratified_split_and_scaler(self):
        x,y=self.fixture(100)
        (xr,yr),(xv,yv),(xt,yt)=split_and_scale(x,y)
        self.assertEqual((len(yr),len(yv),len(yt)),(60,20,20))
        self.assertAlmostEqual(float(np.max(np.abs(xr.mean(axis=0)))),0.,places=12)
        self.assertEqual(int(yr.sum()+yv.sum()+yt.sum()),50)

    def test_two_point_matches_directional_gradient_small_h(self):
        rng=np.random.default_rng(3)
        xi=rng.normal(size=10);theta=rng.normal(size=11)/5
        u=rng.choice([-1.,1.],size=11); yi=1
        score=theta[0]+xi@theta[1:]
        sigmoid=1/(1+np.exp(-score))
        g=(sigmoid-yi)*np.r_[1.,xi]
        np.testing.assert_allclose(two_point(theta,xi,yi,u,h=1e-4),
                                   u*(u@g),atol=1e-8)

    def test_oracle_budget_and_repeatability(self):
        x,y=self.fixture()
        a,ha,ma=zsgd(x,y,budget=101,seed=11,checkpoints=4)
        b,hb,mb=zsgd(x,y,budget=101,seed=11,checkpoints=4)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ha,hb)
        self.assertEqual(ma,mb)
        self.assertEqual(ma["component_loss_calls"],100)
        self.assertEqual(ma["unused_budget"],1)

    def test_metrics_perfect(self):
        y=np.array([0,0,1,1])
        p=np.array([.01,.02,.98,.99])
        m=metrics(y,p)
        self.assertEqual(m["roc_auc"],1.)
        self.assertEqual(m["fpr_hadron_to_gamma"],0.)
        self.assertEqual(m["confusion_matrix_true_rows_hadron_gamma"],[[2,0],[0,2]])

    def test_validation(self):
        x,y=self.fixture()
        with self.assertRaises(ValueError): zsgd(x,y,budget=-1)
        with self.assertRaises(ValueError): zsgd(x,y,learning_rate=0)
        with self.assertRaises(ValueError): two_point(np.zeros(11),x[0],y[0],np.ones(11),h=0)
        with self.assertRaises(ValueError): load_magic()

    def test_score_bounds(self):
        x,y=self.fixture()
        theta=np.zeros(11)
        p=scores(theta,x)
        np.testing.assert_allclose(p,.5)
        self.assertTrue(np.isfinite(component_loss(theta,x[0],y[0])))

if __name__=="__main__":
    unittest.main()
