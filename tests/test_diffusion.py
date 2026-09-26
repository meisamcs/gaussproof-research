import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec('diffusers'),'Optional diffusion dependencies not installed')
class DiffusionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_official_noise_conversion_recovers_known_clean_gradient(self):
        from gaussproof.diffusion_denoiser import scheduler,reconstruct_x0
        torch.manual_seed(10);clean=torch.randn(3,2,8,8);noise=torch.randn_like(clean)
        for prediction in ['epsilon','sample','v_prediction']:
            s=scheduler(prediction)
            for t in [0,100,198]:
                ts=torch.full((3,),t);xt=s.add_noise(clean,noise,ts)
                target=noise if prediction=='epsilon' else clean if prediction=='sample' else s.get_velocity(clean,noise,ts)
                actual=reconstruct_x0(s,target,t,xt)
                torch.testing.assert_close(actual,clean,atol=4e-4,rtol=4e-4)

    def test_clipping_precedes_average_and_noise_loss_identity(self):
        from gaussproof.diffusion_lab import GradientSpace
        raw=np.array([[[10.,0.]],[[0.,1.]]],dtype=np.float32)
        np.testing.assert_allclose(GradientSpace.unit(raw,1).mean(0),[[.5,.5]])
        g=np.array([.2,.4]);noise=np.array([1.,-1.]);tau=10.;y=g+tau*noise
        prediction=y/tau;recovered=y-tau*prediction
        np.testing.assert_allclose(np.sum((g-recovered)**2),tau**2*np.sum((noise-prediction)**2))
        np.testing.assert_allclose(recovered,0,atol=1e-12)

    def test_tiny_training_conditioning_and_saved_upstream_model(self):
        from gaussproof.diffusion_lab import GradientSpace,calibration_set
        from gaussproof.diffusion_denoiser import DiffusionDenoiser
        from diffusers import UNet2DModel
        cfg=dict(rank=64,clips=[.1,1.],batch_sizes=[1,8],bank_size=16,train_batch=4,
            diffusion_steps=200,inference_steps=100,training_steps=2,learning_rate=.0005,
            calibration_batches=2,absolute_noise=[.1],threads=1)
        rng=np.random.default_rng(2);raw=rng.normal(size=(120,2,80)).astype(np.float32);labels=np.arange(120)%10
        space=GradientSpace(raw,labels,cfg,1)
        model=DiffusionDenoiser(cfg,'epsilon',2)
        calibration=calibration_set(space,raw[:20],labels[:20],model,cfg,3)
        h=model.train(space.sampler(4),calibration,seed=5,log=lambda *a,**kw:None)
        self.assertTrue(np.isfinite(h[-1]['calibration_x0_mse']))
        xt,t,clean,hist,clips,batches,bank=calibration[0]
        for mode in ['single','trajectory','trajectory_bank']:
            prediction=model.predict(xt,t,hist,clips,batches,bank,mode)
            self.assertEqual(prediction.shape,clean.shape);self.assertTrue(torch.isfinite(prediction).all())
        prediction=model.predict(xt,t,hist,clips,batches,bank,iterative=True)
        self.assertTrue(torch.isfinite(prediction).all())
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder));loaded=UNet2DModel.from_pretrained(Path(folder)/'unet',low_cpu_mem_usage=False)
            for key,value in model.model.state_dict().items():torch.testing.assert_close(loaded.state_dict()[key],value)
            from scripts.denoise_gradients import denoise
            root=Path(folder)/'lab';(root/'seed17').mkdir(parents=True)
            (root/'config.json').write_text(json.dumps(cfg));space.save(root/'seed17/gradient_space.npz')
            model.save(root/'seed17/epsilon')
            observations=.1*space.unit(raw[:2],.1)+.1*rng.standard_normal(raw[:2].shape)
            estimate,metadata=denoise(root,17,observations,np.eye(10)[labels[:2]],.1,1,.1,prediction='epsilon')
            self.assertEqual(estimate.shape,observations.shape);self.assertTrue(np.isfinite(estimate).all())
            self.assertEqual(metadata['prediction_type'],'epsilon')
