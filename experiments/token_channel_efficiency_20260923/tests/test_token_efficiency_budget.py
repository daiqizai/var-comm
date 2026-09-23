import pytest,torch
from token_efficiency.models import BudgetContinuous
from token_efficiency.source import freeze_targets
from short_prefix.train import forward

@pytest.mark.parametrize('N',[2048,3060])
def test_real_length_training_gradient_receiver_and_noise(N):
    torch.manual_seed(23);m=BudgetContinuous(torch.ones(32),N,width=8,blocks=1)
    F=torch.randn(2,32,16,16,requires_grad=True);snr=torch.tensor([1.,7.]);noise=torch.randn(2,N,2)
    z,wave=forward(m,{'F':F,'snr':snr,'noise':noise})
    assert wave.shape==(2,N,2)
    torch.testing.assert_close(wave.square().sum((1,2)),torch.full((2,),2.*N),atol=.005,rtol=1e-5)
    torch.testing.assert_close(z,m.receive(wave+noise*10**(-snr[:,None,None]/20),snr),atol=0,rtol=0)
    z.square().mean().backward();assert F.grad.abs().sum()>0 and m.encode[-1].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):m.receive(torch.ones(2,4084,2),snr)
    with pytest.raises(ValueError):m(F,snr,torch.full((2,N,2),float('nan')))

def test_thresholds_require_calibration_full_unique_population():
    rows=[{'population':'calibration','method':method,'source_id':str(i),'psnr_db':20+i/1000,'lpips_alex':.1+i/10000,'dino_cosine':.9} for method in ('Dc_Fq','Dc_VAR_m8','Dc_prefix_m6') for i in range(1000)]
    result=freeze_targets(rows,'abc');assert set(result['targets'])=={'high','balanced','coarse'}
    with pytest.raises(ValueError):freeze_targets(rows[:-1],'abc')
    rows[0]['population']='development'
    with pytest.raises(ValueError):freeze_targets(rows,'abc')
