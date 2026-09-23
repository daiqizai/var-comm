import copy
import torch
import pytest
from short_prefix.models import matched_models,prefix_latent
from short_prefix.protocol import SIZES

@pytest.mark.parametrize('m',[6,7,8])
def test_pair_shape_energy_and_information_interface(m):
    pair=matched_models(m,torch.ones(32),42)
    A,B=pair.values()
    assert sum(p.numel() for p in A.parameters())==sum(p.numel() for p in B.parameters())
    assert all(torch.equal(v,B.state_dict()[k]) for k,v in A.state_dict().items())
    f=torch.randn(2,32,16,16);base=torch.randn_like(f);wave=A.transmit(f,base)
    torch.testing.assert_close(wave.square().sum((1,2)),torch.full((2,),2.*A.uses),atol=.01,rtol=1e-5)
    b={'F':f,'B_TX':base,'B_RX':base,'G_RX':base+1,'label':torch.tensor([0,999]),'snr':torch.tensor([1.,7.]),'status':torch.tensor([[1.,0.,m],[1.,1.,m]]),'noise':torch.zeros_like(wave)}
    z,_=A(b);p,_=B(b)
    torch.testing.assert_close(z,base+1);torch.testing.assert_close(p,base)
    # Populated Adam state must remain independent after subsequent A update.
    oa=torch.optim.AdamW(A.parameters());ob=torch.optim.AdamW(B.parameters())
    for model,opt in ((A,oa),(B,ob)):
        opt.zero_grad();model(b)[0].square().mean().backward();opt.step()
    before=copy.deepcopy(B.state_dict());ostate=copy.deepcopy(ob.state_dict())
    oa.zero_grad();A(b)[0].square().mean().backward();oa.step()
    assert all(torch.equal(v,B.state_dict()[k]) for k,v in before.items())
    for k,v in ostate['state'].items():
        for field,value in v.items():assert torch.equal(value,ob.state_dict()['state'][k][field])

def test_prefix_keeps_official_ten_scale_mapping():
    class Quant:
        v_patch_nums=SIZES
        embedding=torch.nn.Embedding(4096,32)
        calls=[]
        def get_next_autoregressive_input(self,i,total,latent,e):
            self.calls.append((i,total));return latent+torch.nn.functional.interpolate(e,size=(16,16)),None
    class VAE:quantize=Quant()
    vae=VAE();scales=[torch.zeros(1,s*s,dtype=torch.long) for s in SIZES]
    assert prefix_latent(vae,scales,6).shape==(1,32,16,16)
    assert vae.quantize.calls==[(i,10) for i in range(6)]
