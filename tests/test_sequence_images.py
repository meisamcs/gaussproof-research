import importlib.util
import unittest
import torch

AVAILABLE=all(importlib.util.find_spec(p) for p in ['diffusers','opacus'])

@unittest.skipUnless(AVAILABLE,'Optional sequence dependencies are not installed')
class SequenceImageTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_batch_permutation_invariance(self):
        from gaussproof.sequence_images import batch_mse
        truth=torch.rand(3,2,28,28)
        self.assertEqual(float(batch_mse(truth.flip(1),truth)),0)
        self.assertGreater(float(batch_mse(torch.zeros_like(truth),truth)),0)

    def test_bidirectional_uses_later_releases(self):
        from gaussproof.sequence_images import SequenceLSTM
        torch.manual_seed(11);net=SequenceLSTM(8).eval()
        y=torch.randn(2,4,8);c=torch.randn_like(y);changed=y.clone();changed[:,3]+=10
        with torch.no_grad():a=net(y,c);b=net(changed,c)
        self.assertGreater(float((a[:,0]-b[:,0]).abs().max()),1e-5)

    def test_full_diffusion_sampler_and_generator(self):
        from gaussproof.sequence_images import TemporalDiffusion,diffusion_sample,BatchGenerator,batch_mse
        from diffusers import DDPMScheduler
        model=TemporalDiffusion(8,4)
        scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',prediction_type='v_prediction',clip_sample=False)
        data={'y':torch.randn(2,4,8),'context':torch.randn(2,4,8)}
        cfg={'diffusion_sampling_steps':5,'posterior_draws':2}
        samples=diffusion_sample(model,scheduler,data,cfg,7)
        self.assertEqual(tuple(samples.shape),(2,2,4,8));self.assertTrue(torch.isfinite(samples).all())
        torch.testing.assert_close(samples,diffusion_sample(model,scheduler,data,cfg,7))
        generator=BatchGenerator(8);images=generator(samples[0].flatten(0,1),data['context'].flatten(0,1))
        self.assertEqual(tuple(images.shape),(8,2,28,28))
        loss=batch_mse(images,torch.rand_like(images));loss.backward()
        self.assertTrue(all(p.grad is not None for p in generator.parameters()))

if __name__=='__main__':unittest.main()
