import unittest,importlib.util
import torch
AVAILABLE=all(importlib.util.find_spec(p) for p in ['diffusers','opacus'])

@unittest.skipUnless(AVAILABLE,'Optional trajectory dependencies are not installed')
class AblationTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_current_and_past_cannot_read_forbidden_releases(self):
  from gaussproof.sequence_ablation import probe_inputs
  y=torch.randn(2,5,8);c=torch.randn_like(y);s=torch.tensor([0,1]);t=torch.tensor([2,2])
  for mode in ['current','past']:
   changed=y.clone();changed[:,3:]+=100
   if mode=='current':changed[:,:2]+=100
   torch.testing.assert_close(probe_inputs(y,c,s,t,mode),probe_inputs(changed,c,s,t,mode))
 def test_other_checkpoint_context_never_accessed(self):
  from gaussproof.sequence_ablation import probe_inputs
  y=torch.randn(2,5,8);c=torch.randn_like(y);changed=c.clone();changed[:,[0,1,3,4]]+=100
  s=torch.tensor([0,1]);t=torch.tensor([2,2])
  for mode in ['current','past','full']:
   torch.testing.assert_close(probe_inputs(y,c,s,t,mode),probe_inputs(y,changed,s,t,mode))
 def test_shuffle_keeps_current_and_preserves_other_values(self):
  from gaussproof.sequence_ablation import probe_inputs
  y=torch.arange(40).reshape(1,5,8).float();c=torch.zeros_like(y);s=torch.tensor([0]);t=torch.tensor([2])
  original=probe_inputs(y,c,s,t,'full');shuffled=probe_inputs(y,c,s,t,'shuffled',torch.Generator().manual_seed(8))
  torch.testing.assert_close(original[:,2],shuffled[:,2])
  torch.testing.assert_close(original[:,:,:8].sort(dim=1).values,shuffled[:,:,:8].sort(dim=1).values)
  torch.testing.assert_close(original[:,:,8:],shuffled[:,:,8:])
 def test_parameters_and_training_step(self):
  from gaussproof.sequence_ablation import MatchedReconstructor,probe_inputs
  net=MatchedReconstructor(8);y=torch.randn(2,5,8);c=torch.randn_like(y);t=torch.tensor([2,3]);s=torch.tensor([0,1])
  for mode in ['current','past','full','shuffled']:
   net.zero_grad();out=net(probe_inputs(y,c,s,t,mode),t);out.square().mean().backward()
   self.assertTrue(all(p.grad is not None for p in net.parameters()))
