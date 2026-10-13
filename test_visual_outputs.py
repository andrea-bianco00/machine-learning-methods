"""Regression tests for plotting outputs (synthetic fixtures only)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from q5_sensitivity import save as save_q5_sensitivity
from q7 import save as save_q7


class TestVisualOutputs(unittest.TestCase):
    def test_q5_sensitivity_plot_has_all_panels_and_exports(self):
        # Minimal controlled synthetic trajectories for every plotted combination.
        directions=['gaussian','rademacher','sphere']
        scales=[1.,.1,.01]
        rows=[]
        for method in ('zgd','zsgd'):
            for direction in directions:
                for scale in scales:
                    for run in range(2):
                        for calls in (0,6000):
                            gap=(1+run)*scale*(1+calls/6000)
                            rows.append({'method':method,'direction':direction,'scale':scale,
                                         'run':run,'component_value_calls':calls,
                                         'objective':.3+gap,'objective_gap':gap})
        metadata={'ols_minimum':.3,'budget':6000,'repeats':2,
                  'directions':directions,'scales':scales}
        with tempfile.TemporaryDirectory() as tmp:
            save_q5_sensitivity(rows,metadata,tmp)
            self.assertGreater((Path(tmp)/'q5_sensitivity.png').stat().st_size,1000)
            self.assertTrue((Path(tmp)/'q5_sensitivity.json').exists())
            self.assertEqual(len(metadata['final_summaries']),18)

    def test_q7_plot_exports_all_three_comparisons(self):
        baseline={'test':{'roc_auc':.83,'fpr_hadron_to_gamma':.44}}
        runs=[{'seed':142,'test':{'roc_auc':.826,'fpr_hadron_to_gamma':.45}},
              {'seed':143,'test':{'roc_auc':.832,'fpr_hadron_to_gamma':.43}}]
        result={'baseline':baseline,'zsgd_runs':runs}
        curves=[{'run':i,'component_loss_calls':calls,'train_logloss':loss}
                for i in range(2) for calls,loss in ((2,.69),(4,.65))]
        with tempfile.TemporaryDirectory() as tmp:
            save_q7(result,curves,tmp)
            self.assertGreater((Path(tmp)/'q7_comparison.png').stat().st_size,1000)
            self.assertTrue((Path(tmp)/'q7.json').exists())
            self.assertTrue((Path(tmp)/'q7_optimization.csv').exists())


if __name__=='__main__':
    unittest.main()
