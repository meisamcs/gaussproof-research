"""Tiny complete execution with generated input; never reported as MNIST evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import csv
import numpy as np
from gaussproof.runner import run
from gaussproof.report import export_report


class IntegrationTest(unittest.TestCase):
    def test_complete_artifacts_and_calibration_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rng=np.random.default_rng(5)
            pixels=rng.integers(0,256,(80,784))
            data=np.column_stack([np.arange(80)%10,pixels])
            path=root/'input.csv';np.savetxt(path,data,fmt='%d',delimiter=',')
            cfg=dict(seeds=[1],train_size=16,eval_size=8,calibration_size=8,shadow_pool_size=32,population_size=8,
                     steps=2,batch_sizes=[4],sigmas=[.5],clips=[1.],lr=.2,references=1,projection_dim=8,
                     sparse_penalty=.02,sparse_iterations=5,rmia_a=.5,rmia_gamma=1.,bootstrap=5,
                     permutation_repeats=5,control_repeats=1,delta=1e-5,threads=2)
            rows=run(cfg,path,root/'out',log=lambda *a,**kw:None)
            self.assertEqual(len(rows),9)
            complete=json.loads((root/'out/completion.json').read_text())
            self.assertEqual(complete['status'],'complete')
            condition=root/'out/seed1_sigma0.5_clip1_B4'
            with (condition/'scores.csv').open() as f:scores=list(csv.DictReader(f))
            eval_ids={int(r['record_id']) for r in scores if r['role']=='evaluation'}
            fits=json.loads((condition/'attack_fits.json').read_text())
            self.assertFalse(eval_ids & set(fits['hybrid_fit_ids']))
            self.assertFalse(eval_ids & set(fits['threshold_ids']))
            self.assertFalse(set(fits['hybrid_fit_ids']) & set(fits['threshold_ids']))
            self.assertEqual(len(eval_ids),16)
            self.assertTrue((root/'out/figures/attack_comparison.pdf').is_file())
            self.assertTrue((condition/'figures/roc.png').is_file())
            export_report(root/'out',root/'report')
            self.assertTrue((root/'report/summary.csv').is_file())
            self.assertFalse(list((root/'report').rglob('*.pt')))
            self.assertFalse(list((root/'report').rglob('scores.csv')))
            self.assertFalse(list((root/'report').rglob('splits*.json')))
            with self.assertRaises(FileExistsError):run(cfg,path,root/'out',log=lambda *a,**kw:None)
