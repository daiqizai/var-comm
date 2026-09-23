import torch
from latent_research.models import PureContinuous,light_warm_start
from latent_enhancement_b.model import CommunicationArm

def test_pure_exact_budget_input_gradient_and_rx_boundary():
    torch.manual_seed(1);m=PureContinuous(torch.ones(32),width=8,blocks=1)
    f=torch.randn(2,32,16,16,requires_grad=True);snr=torch.tensor([1.,13.]);z,w=m(f,snr,torch.zeros(2,4084,2))
    assert w.shape==(2,4084,2)
    torch.testing.assert_close(w.square().sum((1,2)),torch.full((2,),8168.),rtol=1e-5,atol=.01)
    z.square().mean().backward();assert torch.isfinite(f.grad).all() and f.grad.abs().sum()>0
    assert torch.equal(z,m.receive(w,snr))

def test_light_reparameterization_preserves_forward_if_condition_unchanged():
    torch.manual_seed(2);parent=CommunicationArm((32,16,16),1024,torch.ones(32),width=8,blocks=1);light=light_warm_start(parent)
    f,b=torch.randn(2,32,16,16),torch.randn(2,32,16,16)
    torch.testing.assert_close(parent.encoder(f-b,b),light.encoder(f,b),atol=3e-6,rtol=3e-5)

def test_light_tx_waveform_does_not_read_receiver_or_completed_tx_state():
    from latent_research.models import forward_arm
    torch.manual_seed(3);model=light_warm_start(CommunicationArm((32,16,16),1024,torch.ones(32),width=8,blocks=1))
    b={'F':torch.randn(2,32,16,16),'F_prefix':torch.randn(2,32,16,16),'Fb_TX':torch.randn(2,32,16,16),'Fb_RX':torch.randn(2,32,16,16),'rx_status':torch.tensor([[1.,1.,8.],[1.,0.,8.]]),'snr_db':torch.tensor([1.,13.]),'standard_noise':torch.randn(2,1024,2)}
    _,a=forward_arm('light_tx',model,b)
    changed={**b,'Fb_TX':torch.randn_like(b['Fb_TX'])*100,'Fb_RX':torch.randn_like(b['Fb_RX']),'rx_status':torch.zeros_like(b['rx_status'])}
    _,c=forward_arm('light_tx',model,changed)
    assert torch.equal(a,c)
