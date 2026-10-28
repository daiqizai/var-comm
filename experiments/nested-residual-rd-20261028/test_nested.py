import unittest
import torch
from residual_model import ResidualCodec
from nested_entropy import *
class NestedTest(unittest.TestCase):
    def test_quantization(self):
        for order in('scalar','progressive'):
            m=ResidualCodec(order);u=torch.linspace(-1,1,64)[None].requires_grad_()
            for j in(0,1,16,63,64,65,128,255,256):
                q,d,v=m.quantize(u,j);self.assertEqual(int(d.sum()),j)
                expected=np.where(d.numpy()>0,-1+(2*(v.numpy()>>(4-d.numpy()))+1)/(2.**d.numpy()),0)
                np.testing.assert_array_equal(q.detach().numpy(),expected.astype('float32'))
            q,d,_=m.quantize(u,1);q.sum().backward();self.assertEqual(int(torch.count_nonzero(u.grad)),1)
    def test_both_gradients(self):
        torch.manual_seed(20261028);torch.set_num_threads(2)
        m=ResidualCodec('progressive');full=torch.randn(2,32,16,16);coarse=torch.randn_like(full)
        pred,rate=m(full,coarse,17)
        gd=torch.autograd.grad((pred-full).square().mean(),list(m.analysis.parameters()),retain_graph=True)
        gr=torch.autograd.grad(rate.mean(),list(m.analysis.parameters()))
        for gradients in(gd,gr):self.assertTrue(all(torch.isfinite(x).all()for x in gradients));self.assertGreater(sum(float(x.abs().sum())for x in gradients),0)
        self.assertLess(sum(p.numel()for p in m.parameters()),5_000_000)
        self.assertTrue(torch.equal(m(full,coarse,0)[0],coarse))
    def test_cdf(self):
        for d in range(4):
            for prefix in range(2**d):
                c=binary_cdf(np.ones(16)/16,prefix,d);np.testing.assert_array_equal(c,[[0,TOTAL//2,TOTAL]])
if __name__=='__main__':unittest.main()
