"""CPU tensor tests on the isolated native Torch runtime, no GPU workload."""
import collections
import importlib.util
import unittest

TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH,'Torch not installed in local test runtime')
class TensorTests(unittest.TestCase):
    def test_unique_gather_exact_derivative_and_duplicate_rejection(self):
        import torch
        from swin_model import _UniqueChannelSelect
        generator = torch.Generator().manual_seed(3401)
        x = torch.randn((2, 7, 19), generator=generator, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([[0, 2, 8, 18], [1, 4, 7, 15]])
        weights = torch.randn((2, 7, 4), generator=generator, dtype=torch.float64)
        expected = x.gather(2, indices[:, None, :].expand(-1, 7, -1))
        actual = _UniqueChannelSelect.apply(x, indices)
        self.assertTrue(torch.equal(expected, actual))
        a = torch.autograd.grad((actual * weights).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((expected * weights).sum(), x)[0]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.autograd.gradcheck(lambda v: _UniqueChannelSelect.apply(v, indices), (x,)))
        with self.assertRaises(RuntimeError):
            _UniqueChannelSelect.apply(x, torch.tensor([[1, 1], [2, 3]]))

    def test_pack_roundtrip_independent_frame_power_and_gradient(self):
        import torch
        from swin_model import pack_data,unpack_data
        raw=torch.arange(2*256*320,dtype=torch.float32).reshape(2,256,320)/10000+1
        raw.requires_grad_()
        mask=torch.zeros_like(raw); mask[0,:,[1,4,7,9,11,300]]=1; mask[1,:,[0,3,6,8,99,319]]=1
        feature=raw*mask
        iq,power,indices=pack_data(feature,mask)
        self.assertEqual(tuple(iq.shape),(2,768,2))
        self.assertFalse(torch.equal(power[:1],power[1:]))
        torch.testing.assert_close(iq.square().sum((1,2)),torch.full((2,),1536.),rtol=1e-6,atol=.001)
        recovered=unpack_data(iq,power,indices)
        torch.testing.assert_close(recovered,feature,rtol=1e-6,atol=1e-6)
        selected=feature.gather(2,indices[:,None,:].expand(-1,256,-1)).flatten(1)/power.sqrt()[:,None]
        self.assertTrue(torch.equal(iq[:,:,0],selected[:,:768]))
        self.assertTrue(torch.equal(iq[:,:,1],selected[:,768:]))
        recovered.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all()); self.assertTrue((raw.grad != 0).any())

    def test_rejected_context_and_non_channel_mask(self):
        import torch
        from swin_model import pack_data,unpack_data
        feature=torch.ones((1,256,320)); mask=torch.ones_like(feature); mask[0,0,1]=0
        with self.assertRaises(RuntimeError): pack_data(feature,mask)
        with self.assertRaises(RuntimeError): unpack_data(torch.zeros((1,768,2)),torch.tensor([-1.]),torch.arange(6)[None])
        with self.assertRaises(RuntimeError): unpack_data(torch.zeros((1,768,2)),torch.tensor([1.]),torch.zeros((1,6),dtype=torch.long))

    def test_checkpoint_plain_dict_ordereddict_equality(self):
        import torch
        from swin_train import cpu_tree,exact_tree
        original=collections.OrderedDict(a=torch.tensor([1.]),b={'nested':[torch.tensor(2.)]})
        self.assertTrue(exact_tree(cpu_tree(original),original))
        changed=cpu_tree(original); changed['a'][0]=2.
        self.assertFalse(exact_tree(changed,original))

    def test_completed_calibration_resume_is_idempotent(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock
        import swin_train as t
        from swin_state import identity,write
        from swin_protocol import RATES,CAL_SNRS
        class Model:
            def eval(self): pass
            def train(self): pass
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder); checkpoint=output/'model.pt'; checkpoint.write_bytes(b'ENGINEERING_ONLY')
            ids=[f'cal{i}' for i in range(1000)]
            for N in RATES:
                for snr in CAL_SNRS:
                    rows=[dict(source_index=i,source_id=s,mse=.1,header_accepted=True) for i,s in enumerate(ids)]
                    cell=dict(status='COMPLETE',context=dict(registration_sha256='fixture',checkpoint_sha256=t.sha(checkpoint),
                        step=0,N=N,snr=snr,seed=4101,source_ids=ids),rows=rows,mean_mse=.1,header_failures=0,synthetic=True)
                    cell['payload_sha256']=identity(cell)
                    write(output/'calibration/000000'/f'N{N}_snr{snr}_seed4101.json',cell)
            with mock.patch.object(t,'rng_state',return_value={}),mock.patch.object(t,'restore_rng'):
                first=t.calibrate(Model(),SimpleNamespace(ids=ids),{},output,0,checkpoint,'fixture',[False])
                second=t.calibrate(Model(),SimpleNamespace(ids=ids),{},output,0,checkpoint,'fixture',[False])
            self.assertEqual(first,second)


if __name__=='__main__': unittest.main()
