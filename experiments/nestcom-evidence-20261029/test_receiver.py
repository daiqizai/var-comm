"""CPU tests of intervention boundaries; synthetic values, no scientific samples."""
import unittest,types,contextlib,numpy as np,torch
from receiver_worker import ada_tail,coarse,SIZES
class FakeAda:
 def __init__(self):
  self.hist=[];self.torch=types.SimpleNamespace(random=types.SimpleNamespace(fork_rng=lambda **k:contextlib.nullcontext()),manual_seed=lambda s:None,cuda=types.SimpleNamespace(manual_seed_all=lambda s:None),split=torch.split)
 def start(self,c):self.previous=None;self.path=[];self.c=c
 def logits(self,i):
  self.path.append(self.previous);z=torch.zeros(2,1,4);z[0,0,(int(self.previous or 0)+1)%4]=2;return z
 def tail(self,z):return torch.tensor([3])
 def render(self,t):self.hist.append(self.path.copy());return np.zeros((3,256,256),np.float32)
class T:
 def __init__(self,a):self.a=np.asarray(a);self.shape=self.a.shape
 def cpu(self):return self
 def numpy(self):return self.a
 def __getitem__(self,i):return T(self.a[i])
 def clamp(self,a,b):return T(np.clip(self.a,a,b))
class FakePrior:
 def __init__(self,*a):self.fhat=T(np.zeros((1,1,16,16),np.float32));self.adv=[];FakePrior.last=self
 def advance(self,t,k):self.adv.append((k,np.array(t).copy()));self.fhat=T(self.fhat.a+float(k+1))
 def close(self):self.closed=True
class ReceiverTests(unittest.TestCase):
 def test_greedy_rebuilds_own_context(self):
  n=FakeAda();prefix=np.array([1,2],np.int64)
  _,sample,_=ada_tail(n,prefix,9,1,False);_,greedy,c=ada_tail(n,prefix,9,1,True)
  self.assertTrue(np.array_equal(greedy[:2],prefix));self.assertEqual(greedy[2:6].tolist(),[3,0,1,2]);self.assertEqual(sample[2:6].tolist(),[3]*4);self.assertEqual(n.hist[-1][4],0);self.assertEqual(c['prior_forwards'],256)
 def test_coarse_keeps_full_grid_and_original_indices(self):
  n=types.SimpleNamespace(receiver=types.SimpleNamespace(_Prior=FakePrior),loaded=dict(vae=None,var=None,device=None,decoder=lambda x:T(np.zeros((1,3,256,256),np.float32))))
  a=np.arange(sum(s*s for s in SIZES[:3]),dtype=np.int64);_,proof=coarse(n,a,3)
  self.assertEqual([k for k,_ in FakePrior.last.adv],[0,1,2]);self.assertTrue(np.array_equal(np.concatenate([a for _,a in FakePrior.last.adv]),a));self.assertEqual(proof['S'],10);self.assertEqual(proof['latent_shape'],[1,1,16,16]);self.assertEqual(proof['unreceived_scale_updates'],0);self.assertEqual(proof['prior_forwards'],0)
if __name__=='__main__':unittest.main()
