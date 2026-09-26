import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from gaussproof.attacks import LogisticAttack, rmia, trajectory_scores
from gaussproof.data import make_splits, load_mnist
from gaussproof.metrics import auc, roc, threshold_at_fpr, evaluate
from gaussproof.models import initialize, per_record_gradients, clip_gradients, train, projection_indices, conservative_epsilon
from gaussproof.validation import synthetic_controls


class MetricsTests(unittest.TestCase):
    def test_perfect_reversed_and_tied(self):
        y=np.array([0,0,1,1])
        self.assertEqual(auc([0,0,1,1],y),1)
        self.assertEqual(auc([1,1,0,0],y),0)
        self.assertEqual(auc([1,1,1,1],y),.5)
        f,t,_=roc([1,1,1,1],y)
        np.testing.assert_array_equal(f,[0,1]);np.testing.assert_array_equal(t,[0,1])

    def test_low_fpr_no_tie_splitting(self):
        s=np.ones(200); y=np.r_[np.zeros(100),np.ones(100)]
        self.assertEqual(threshold_at_fpr(s,y,.01),np.inf)
        m=evaluate(s,y,s,y,bootstrap=10)
        self.assertEqual(m['tpr_at_1pct_fpr'],0)
        self.assertEqual(m['balanced_accuracy'],.5)

    def test_auc_matches_pairwise_definition(self):
        rng=np.random.default_rng(9)
        s=rng.integers(0,5,30);y=np.r_[np.ones(10),np.zeros(20)]
        d=s[:10,None]-s[None,10:]
        self.assertAlmostEqual(auc(s,y),np.mean((d>0)+.5*(d==0)))

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError):auc([np.nan,1],[0,1])
        with self.assertRaises(ValueError):auc([0,1],[1,1])


class AttackTests(unittest.TestCase):
    def test_rmia_matches_explicit_population_test(self):
        x=np.array([.1,.5,.9]);refs=np.array([[.1,.2,.3],[.2,.3,.4]])
        z=np.array([.2,.7]);zrefs=np.array([[.3,.5],[.2,.4]])
        den=lambda r:.75*r.mean(0)+.25
        manual=((x/den(refs))[:,None]>(z/den(zrefs))[None,:]).mean(1)
        np.testing.assert_array_equal(rmia(x,refs,z,zrefs),manual)

    def test_logistic_heldout_signal(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(400,3));y=(x[:,0]>0).astype(int)
        model=LogisticAttack().fit(x[:200],y[:200])
        self.assertGreater(auc(model.score(x[200:]),y[200:]),.98)

    def test_sparse_recovery_and_candidate_permutation(self):
        bank=np.eye(10);d=np.repeat(bank[None],10,axis=0);obs=np.zeros((10,10));obs[:,:3]=1
        scores=trajectory_scores(d,obs,.1,40)
        self.assertGreater(auc(scores['gaussproof'],np.r_[np.ones(3),np.zeros(7)]),.99)
        order=np.random.default_rng(1).permutation(10)
        perm=trajectory_scores(d[:,order],obs,.1,40)
        for k in scores:np.testing.assert_allclose(perm[k],scores[k][order],atol=1e-10)

    def test_controls(self):
        rows=synthetic_controls(repeats=10)
        for attack in ['rero','gaussproof']:
            get=lambda control:np.mean([r['auc'] for r in rows if r['attack']==attack and r['control']==control])
            self.assertTrue(.35<get('pure_noise')<.65)
            self.assertGreater(get('structured'),.8)
            self.assertLess(get('nonmember_only'),.2)
            self.assertLess(get('high_noise'),get('structured'))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_vmap_matches_individual_autograd(self):
        model=initialize(1);x=torch.rand(3,1,28,28);y=torch.tensor([1,2,3])
        actual=per_record_gradients(model,x,y,microbatch=2)
        expected=[]
        for xx,yy in zip(x,y):
            loss=torch.nn.functional.cross_entropy(model(xx[None]),yy[None])
            grads=torch.autograd.grad(loss,tuple(model.parameters()))
            expected.append(torch.cat([g.flatten() for g in grads]))
        torch.testing.assert_close(actual,torch.stack(expected),rtol=2e-5,atol=1e-6)
        clipped=clip_gradients(actual,.2)
        self.assertLessEqual(float(clipped.norm(dim=1).max()),.200001)

    def test_saved_release_is_actual_update(self):
        model=initialize(3);x=torch.rand(8,1,28,28);y=torch.arange(8)%10
        before=torch.cat([p.detach().flatten() for p in model.parameters()])
        coords=projection_indices(model,20,4)
        trace=train(model,x,y,np.arange(8),1,4,1.,.5,.2,5,np.arange(8),coords)
        after=torch.cat([p.detach().flatten() for p in model.parameters()])
        np.testing.assert_allclose((before-after)[coords].numpy(),trace['releases'][0]*.2/4,atol=1e-7)

    def test_reproducible_training(self):
        torch.manual_seed(8);x=torch.rand(8,1,28,28);y=torch.arange(8)%10
        a,b=initialize(7),initialize(7)
        train(a,x,y,np.arange(8),2,4,1.,.5,.2,9)
        train(b,x,y,np.arange(8),2,4,1.,.5,.2,9)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_accountant_monotonic(self):
        self.assertIsNone(conservative_epsilon(10,0,1e-5))
        self.assertGreater(conservative_epsilon(10,1,1e-5),conservative_epsilon(10,2,1e-5))


class DataTests(unittest.TestCase):
    def test_fixed_disjoint_roles(self):
        cfg=dict(train_size=100,eval_size=30,calibration_size=20,shadow_pool_size=200,population_size=50)
        a=make_splits(500,cfg,17);b=make_splits(500,cfg,17)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        groups=['train','eval_nonmember','cal_nonmember','shadow_pool','population']
        for i,g in enumerate(groups):
            for h in groups[i+1:]:self.assertEqual(len(np.intersect1d(a[g],a[h])),0)
        self.assertEqual(len(np.intersect1d(a['eval_member'],a['cal_member'])),0)

    def test_csv_header_and_pixel_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'digits.csv';a=np.zeros((2,785));a[:,0]=[3,4];a[0,1+28]=255
            np.savetxt(p,a,delimiter=',',header=','.join(['label']+[f'p{i}' for i in range(784)]),comments='')
            x,y=load_mnist(p)
            self.assertEqual(x[0,0,1,0],1);self.assertEqual(y.tolist(),[3,4])


if __name__=='__main__':unittest.main()
