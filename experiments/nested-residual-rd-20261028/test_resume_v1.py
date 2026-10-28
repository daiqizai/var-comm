"""CPU-only state persistence check; no source images or scientific training."""
import unittest,tempfile,pathlib,torch
from train_residual_v1 import atomic_torch,schedule
class ResumeTests(unittest.TestCase):
    def test_optimizer_rng_and_schedule(self):
        torch.manual_seed(17);m=torch.nn.Linear(3,2);o=torch.optim.AdamW(m.parameters(),lr=1e-4)
        def step(model,optimizer):
            optimizer.zero_grad();x=torch.rand(4,3);model(x).square().mean().backward();optimizer.step();return x
        step(m,o)
        with tempfile.TemporaryDirectory()as d:
            p=pathlib.Path(d)/'latest.pt';atomic_torch(p,dict(model=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()))
            expected_input=step(m,o);expected={k:v.clone()for k,v in m.state_dict().items()}
            q=torch.load(p,weights_only=False);m2=torch.nn.Linear(3,2);o2=torch.optim.AdamW(m2.parameters(),lr=1e-4);m2.load_state_dict(q['model']);o2.load_state_dict(q['optimizer']);torch.set_rng_state(q['rng']);actual_input=step(m2,o2)
            self.assertTrue(torch.equal(expected_input,actual_input))
            for k,v in expected.items():self.assertTrue(torch.equal(v,m2.state_dict()[k]))
        a,j=schedule(2026102801,19968,5000);b,k=schedule(2026102801,19968,5000)
        self.assertTrue((a==b).all()and(j==k).all());self.assertEqual(a.shape,(5000,4));self.assertTrue(((j>=1)&(j<=256)).all())
if __name__=='__main__':unittest.main()
