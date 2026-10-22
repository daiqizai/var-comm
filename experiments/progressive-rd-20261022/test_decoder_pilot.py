"""Synthetic CPU checks only; no pretrained network, source or GPU access."""
import unittest,torch
from decoder_model import ProfileDecoder

class TestPilot(unittest.TestCase):
    def test_zero_init_preserves_outputs_and_capacity(self):
        torch.manual_seed(7);z=torch.randn(4,32,4,4);e=torch.tensor([0,2,5,7])
        a=ProfileDecoder(torch.nn.Identity(),False);b=ProfileDecoder(torch.nn.Identity(),True)
        self.assertTrue(torch.equal(a(z,e),z));self.assertTrue(torch.equal(b(z,e),z))
        self.assertEqual(sum(p.numel()for p in a.parameters()),576)
        self.assertEqual(sum(p.numel()for p in a.parameters()),sum(p.numel()for p in b.parameters()))
        self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k])for k in a.state_dict()))

    def test_public_endpoint_controls_conditioning(self):
        z=torch.ones(2,32,2,2);e=torch.tensor([0,7])
        a=ProfileDecoder(torch.nn.Identity(),False);b=ProfileDecoder(torch.nn.Identity(),True)
        with torch.no_grad():
            a.profile_affine.weight[:,7]=1;b.profile_affine.weight[:,7]=1
        self.assertTrue(torch.equal(a(z,e)[0],a(z,e)[1]))
        self.assertFalse(torch.equal(b(z,e)[0],b(z,e)[1]))

    def test_finite_gradients_both_controls(self):
        for conditional in [False,True]:
            m=ProfileDecoder(torch.nn.Conv2d(32,3,1),conditional)
            loss=m(torch.randn(4,32,4,4),torch.tensor([0,2,5,7])).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in m.parameters()))

if __name__=='__main__':unittest.main()
