import unittest
import importlib.util
import numpy as np
import torch
OPTIONAL = all(importlib.util.find_spec(m) for m in ["diffusers", "opacus"])
if OPTIONAL:
    from gaussproof.trajectory_dp import histories, trajectory, epsilon, net_input
from gaussproof.models import initialize

@unittest.skipUnless(OPTIONAL, "Optional trajectory dependencies not installed")
class TrajectoryTests(unittest.TestCase):
    def test_history_causal(self):
        a=np.arange(640,dtype=np.float32).reshape(10,64)
        h,v=histories(a,8)
        np.testing.assert_array_equal(h[0],0)
        np.testing.assert_array_equal(h[8],a[1:8])
        b=a.copy();b[8:]=999
        np.testing.assert_array_equal(histories(b,8)[0][:9],h[:9])
        self.assertEqual(v[3].sum(),3)

    def test_full_history_features(self):
        a=np.arange(64*64,dtype=np.float32).reshape(64,64)
        h,v=histories(a,64)
        np.testing.assert_array_equal(h[63],a[:63])
        features=net_input(torch.tensor(a),torch.tensor(h),torch.tensor(v),torch.ones(64))
        self.assertEqual(tuple(features.shape),(64,128,8,8))
        masked=net_input(torch.tensor(a),torch.tensor(h),torch.tensor(v),torch.zeros(64))
        self.assertEqual(float(masked[:,0].abs().sum()),0.)
        torch.testing.assert_close(masked[:,1:127],features[:,1:127])

    def test_accountant_noise_monotonicity(self):
        cfg=dict(steps=64,batch_size=32,population=2000,delta=1e-5)
        self.assertGreater(epsilon(cfg,1),epsilon(cfg,4))
        self.assertGreater(epsilon(cfg,4),epsilon(cfg,16))

    def test_actual_release_and_update(self):
        torch.set_num_threads(1);torch.manual_seed(22)
        x=torch.rand(50,1,28,28);labels=torch.arange(50)%10
        cfg=dict(steps=2,clip=.1,batch_size=8)
        initial=initialize(10)
        data,_,model=trajectory(initial,x,labels,np.arange(30),np.arange(30,40),np.arange(40,50),cfg,4,11)
        noise=(data['observed']-data['clean'])/(4*.1/8)
        self.assertLess(abs(noise.std()-1),.03)
        self.assertLess(abs(noise.mean()),.03)
        initial_flat=torch.cat([p.detach().flatten() for p in initial.parameters()]).numpy()
        final_flat=torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
        np.testing.assert_allclose(final_flat,initial_flat-.5*data['observed'].sum(0),atol=1e-7)

if __name__=='__main__':unittest.main()
