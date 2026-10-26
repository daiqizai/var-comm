"""Synthetic guards only: no real weights, source assets, channel or metrics."""
import unittest,tempfile
from pathlib import Path
import numpy as np
from common import grids,OFFSETS,SIZES,Ledger
from b_ablation import merge_boundary,ablate
class Contracts(unittest.TestCase):
 def test_nested(self):
  g=grids();self.assertEqual([len(g[str(d)])for d in(1,2,4,8)],[7,13,25,49])
  for a,b in[(1,2),(2,4),(4,8)]:self.assertLess(set(g[str(a)]),set(g[str(b)]))
 def test_missing_exact24(self):self.assertEqual(len(set(grids()['8'])-set(grids()['4'])),24)
 def test_maximum_same(self):
  for v in grids().values():self.assertEqual(max(OFFSETS[m]+k for m,k in v),680)
 def test_actual_received_values_only(self):
  p=np.array([2,3,4]);r=np.array([9,8]);a=merge_boundary(p,r,2)
  np.testing.assert_array_equal(a,[9,8,4]);np.testing.assert_array_equal(p,[2,3,4])
 def test_ledger_failed_is_not_retryable(self):
  with tempfile.TemporaryDirectory()as d:
   l=Ledger(Path(d)/'a.sqlite',{'x':1})
   with self.assertRaises(ZeroDivisionError):l.call('x','one',lambda:1/0)
   self.assertEqual(l.summary()['unresolved'],1)
   with self.assertRaises(AssertionError):l.call('x','one',lambda:2)
 def test_ledger_cap(self):
  with tempfile.TemporaryDirectory()as d:
   l=Ledger(Path(d)/'a.sqlite',{'x':1});self.assertEqual(l.call('x','one',lambda:7),7)
   with self.assertRaises(AssertionError):l.call('x','two',lambda:8)
 def toy(self):
  import torch,types
  class Q:
   def embedding(self,idx):return idx.float()[...,None]
   def get_next_autoregressive_input(self,s,n,f,e):
    f=f+e.mean().reshape(1,1,1,1)/(s+2);return f,f
  q=Q();vae=types.SimpleNamespace(quantize=q);var=types.SimpleNamespace(Cvae=1)
  class Prior:
   def __init__(self,*args):self.device=torch.device('cpu');self.fhat=torch.zeros(1,1,1,1)
   def logits(self,s):
    logits=torch.zeros(1,SIZES[s]**2,16);j=int(self.fhat.item()*7)%8
    logits[:,:,j]=1;return logits
   def advance(self,a,s):self.fhat,_=q.get_next_autoregressive_input(s,10,self.fhat,torch.as_tensor(a).float())
   def close(self):pass
  decoder=lambda f:torch.sigmoid(f/10).expand(1,3,256,256)
  return types.SimpleNamespace(torch=torch,receiver=types.SimpleNamespace(_Prior=Prior),loaded=dict(vae=vae,var=var,device='cpu',decoder=decoder))
 def reference(self,n,a,m,K,overwrite):
  p=n.receiver._Prior();off=0;tokens=[]
  for s,size in enumerate(SIZES):
   v=p.logits(s)[0].argmax(-1).numpy();count=size*size if s<m else K if s==m else 0
   if count and(s<m or overwrite):v[:count]=a[off:off+count]
   off+=count;p.advance(v,s);tokens.append(v.copy())
  return n.loaded['decoder'](p.fhat)[0].numpy(),tokens
 def test_A0_trajectory_and_direct_B0(self):
  n=self.toy();m,K=7,25;a=np.full(OFFSETS[m]+K,9,dtype=np.int64)
  images,tr=ablate(n,a,m,K);ref,tt=self.reference(n,a,m,K,False)
  np.testing.assert_array_equal(images['A0'],ref)
  for s in range(10):np.testing.assert_array_equal(tr[s]['context_tokens'],tt[s])
  self.assertGreater(float(images['B0'].mean()),float(images['A0'].mean()))
  np.testing.assert_array_equal(tr[m]['latent_tokens'][:K],a[-K:])
  _,changed=self.reference(n,a,m,K,True)
  self.assertTrue(any(not np.array_equal(tt[s],changed[s])for s in range(m+1,10)))
 def test_final_scale_B0_equals_C0(self):
  n=self.toy();m,K=9,64;a=np.full(OFFSETS[m]+K,9,dtype=np.int64)
  images,_=ablate(n,a,m,K);c0,_=self.reference(n,a,m,K,True)
  np.testing.assert_array_equal(images['B0'],c0)
 def test_K0_must_reuse_not_reconstruct(self):
  with self.assertRaises(AssertionError):ablate(self.toy(),np.zeros(OFFSETS[8],dtype=np.int64),8,0)
if __name__=='__main__':unittest.main()
