"""CPU engineering checks; these do not establish real image quality."""
import unittest,math
import torch
from types import SimpleNamespace
from probe import noise,distance
class ProbeEngineeringTests(unittest.TestCase):
 def test_real_complex_snr_convention(self):
  for db in (-5,-2,0,1,4,7,13):
   gamma=10**(db/10);eta=10**(-db/20)
   self.assertAlmostEqual(eta*eta,1/gamma,places=12)
   self.assertAlmostEqual(2/(2*eta*eta),gamma,places=12)
 def test_source_noise_replay_and_pairing(self):
  a=noise('calibration-source-a',4101)
  self.assertTrue(torch.equal(a,noise('calibration-source-a',4101)))
  self.assertFalse(torch.equal(a,noise('calibration-source-b',4101)))
  self.assertFalse(torch.equal(a,noise('calibration-source-a',4102)))
  self.assertEqual(a.shape,(1,32,16,16))
 def test_distance_is_sum_not_mean(self):
  g=torch.Generator().manual_seed(17)
  q=SimpleNamespace(embedding=torch.nn.Embedding(4096,32))
  with torch.no_grad():q.embedding.weight.copy_(torch.randn(4096,32,generator=g))
  x=torch.randn((1,32,16,16),generator=g)
  d,z=distance(q,x,0)
  ref=(q.embedding.weight-z[0]).square().sum(-1)
  torch.testing.assert_close(d[0,0],ref,rtol=1e-5,atol=1e-5)
 def test_zero_noise_prior_need_not_reproduce_nearest(self):
  # Nonzero residual variance remains in the supplied likelihood at eta=0.
  distances=torch.tensor([0.,.2])
  logprior=torch.tensor([.01,.99]).log()
  self.assertEqual(int(distances.argmin()),0)
  self.assertEqual(int((logprior-distances/(2*.1)).argmax()),1)
if __name__=='__main__':unittest.main()
