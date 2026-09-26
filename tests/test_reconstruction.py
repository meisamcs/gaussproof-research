import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from gaussproof.reconstruction_prior import GradientPrior,exact_bank
from gaussproof.reconstruction import split_roles,run_reconstruction
from gaussproof.reconstruction_report import export_report


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_gaussian_posterior_matches_dense_covariance(self):
        rng=np.random.default_rng(1)
        g=rng.normal(size=(300,8)).astype('float32');labels=np.arange(300)%10
        prior=GradientPrior().fit(g,labels,rank=4)
        obs=rng.normal(size=(3,8)).astype('float32');lab=np.array([1,4,9]);noise=.3
        basis=prior.basis.numpy();variance=prior.variance.numpy();floor=prior.floor
        cov=(basis*variance)@basis.T+floor*(np.eye(8)-basis@basis.T)
        means=prior.means.numpy()[lab]
        expected=means+(obs-means)@np.linalg.solve(cov+noise*np.eye(8),cov)
        np.testing.assert_allclose(prior.estimate(obs,lab,noise),expected,atol=2e-6)
        np.testing.assert_allclose(prior.estimate(obs,lab,1e12),means,atol=1e-6)
        np.testing.assert_array_equal(prior.estimate(obs,lab,noise,method='prior_only'),means)

    def test_likelihood_uses_candidate_norm_and_sample_mean_sufficiency(self):
        bank=np.array([[1.,0.],[3.,0.]])
        out=exact_bank([[1.,0.]],bank,[0],[0,0],.5)
        self.assertEqual(out['alignment'][0],1)
        self.assertEqual(out['likelihood'][0],0)
        releases=np.array([[.8,.1],[1.1,-.2],[1.2,.1]])
        mean=exact_bank(releases.mean(0)[None],bank,[0],[0,0],.5/3)['posterior']
        logits=-np.sum((releases[:,None]-bank[None])**2,axis=(0,2))/(2*.5)
        w=np.exp(logits-logits.max());w/=w.sum()
        np.testing.assert_allclose(mean[0],w@bank)

    def test_splits_balanced_and_disjoint(self):
        cfg=dict(model_train_size=20,prior_size=40,calibration_size=20,target_size=30,distractor_size=20)
        roles=split_roles(300,cfg,2,np.arange(300)%10)
        allids=np.concatenate(list(roles.values()))
        self.assertEqual(len(allids),len(set(allids)))
        np.testing.assert_array_equal(np.bincount(roles['target']%10),np.full(10,3))

    def test_complete_reconstruction_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(17)
            data=np.column_stack([np.arange(600)%10,rng.integers(0,256,(600,784))])
            path=root/'data.csv';np.savetxt(path,data,fmt='%d',delimiter=',')
            cfg=dict(seeds=[17],model_train_size=100,prior_size=200,calibration_size=100,target_size=20,
                distractor_size=20,checkpoint_steps=[2],public_batch_size=10,public_lr=.05,clip=1.,
                sigmas=[1.],repeats=[2],batch_size=2,rank=4,denoiser_steps=2,inversion_sigma=1.,
                inversion_repeats=2,inversion_targets=1,inversion_steps=2,inversion_restarts=1,
                bootstrap=10,delta=1e-5,threads=2)
            rows=run_reconstruction(cfg,path,root/'run',log=lambda *a,**kw:None)
            self.assertEqual(len(rows),300)
            self.assertTrue(all(np.isfinite(r['squared_l2']) for r in rows))
            completion=json.loads((root/'run/completion.json').read_text())
            self.assertEqual(completion['image_measurements'],6)
            self.assertTrue((root/'run/figures/reconstruction_gallery.pdf').is_file())
            export_report(root/'run',root/'report')
            self.assertFalse(list((root/'report').rglob('*.pt')))
            self.assertFalse(list((root/'report').rglob('*records.csv')))
            self.assertTrue((root/'report/manifest.json').is_file())
            with self.assertRaises(FileExistsError):run_reconstruction(cfg,path,root/'run')
