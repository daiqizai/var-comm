"""Synthetic interface tests only; never formal real-image quality acceptance."""
import json
from types import SimpleNamespace
import numpy as np,pytest,torch
from token_efficiency import execution as ex
from token_efficiency.models import BudgetContinuous
from latent_enhancement.runtime import digest

class Encoder(torch.nn.Module):
    def forward(self,x):return torch.nn.functional.adaptive_avg_pool2d(x[:,:1],(16,16)).expand(-1,32,-1,-1)
class Decoder(torch.nn.Module):
    def forward(self,x):return torch.nn.functional.interpolate(x[:,:3].sigmoid(),(256,256))

@pytest.mark.parametrize('N',[2048,3060])
def test_continuous_online_uses_same_model_rx_and_one_noise(N):
    torch.manual_seed(18);model=BudgetContinuous(torch.ones(32),N,width=8,blocks=1).eval()
    vae=SimpleNamespace(encoder=Encoder(),quant_conv=torch.nn.Identity())
    record={'pixels':np.full((3,256,256),127,dtype=np.uint8),'image_id':'engineering-fixture','class_index':None}
    cell=ex.Cell('continuous',N);image,event=ex.execute(record,cell,7,2001,vae,None,Decoder(),'cpu',model)
    replay,rx=ex.receive(event['observation'],7,cell,vae,None,Decoder(),'cpu',model)
    np.testing.assert_array_equal(image,replay)
    np.testing.assert_array_equal(event['observation'],ex.apply_channel(event['waveform'],7,record['image_id'],2001,cell))
    assert event['ledger']['N']==N and event['ledger']['N_header']==0 and event['rx']['decoded_label'] is None
    assert event['total_ms']==event['tx_ms']+event['rx_ms']
    assert event['waveform_sha256']!=event['observation_sha256']

def test_timing_channel_outside_rx_and_no_tx_state_passed(monkeypatch):
    events=[];clock=iter([0.,1.,10.,12.]);cell=ex.Cell('raw',2048,6)
    monkeypatch.setattr(ex.time,'perf_counter',lambda:next(clock))
    monkeypatch.setattr(ex,'synchronize',lambda d:events.append('sync'))
    def tx(*args):events.append('TX');return np.ones((2048,2)),{'N':2048}
    def ch(*args):events.append('noise');return np.zeros((2048,2))
    def rx(observed,snr,rxcell,vae,var,decoder,device,model):
        events.append('RX');assert np.array_equal(observed,np.zeros((2048,2)));assert rxcell==cell
        return np.full((3,256,256),.5),{'header_ok':False}
    monkeypatch.setattr(ex,'transmit',tx);monkeypatch.setattr(ex,'apply_channel',ch);monkeypatch.setattr(ex,'receive',rx)
    _,rec=ex.execute({'pixels':None,'image_id':'fixture','class_index':99},cell,1,2001,None,None,None,'cpu')
    assert rec['tx_ms']==1000 and rec['rx_ms']==2000 and rec['total_ms']==3000
    assert events==['sync','TX','sync','noise','sync','RX','sync']

def test_illegal_inputs_not_implicitly_cast():
    with pytest.raises(ValueError):ex.source_pixels(np.full((3,256,256),.5))
    with pytest.raises(ValueError):ex.Cell('continuous',3060,m=8)
    with pytest.raises(ValueError):ex.Cell('raw',3060,m=11)
    with pytest.raises(ValueError):ex.receive(np.zeros((1,2)),7,ex.Cell('continuous',3060),None,None,None,'cpu')

def test_selected_checkpoint_uses_recorded_step_and_sha(tmp_path):
    model=BudgetContinuous(torch.ones(32),2048);name='P2048';registration={'N':2048,'bindings':{},'decoder_sha256':'decoder'}
    reg=tmp_path/'registration.json';reg.write_text(json.dumps(registration));regsha=digest(reg)
    checkpoint=tmp_path/'actual_selected_02500.pt';torch.save({'registration_sha256':regsha,'models':{name+'.'+k:v for k,v in model.state_dict().items()},'state':{'step':2500,'updates':{name:2500}}},checkpoint)
    selected=tmp_path/'selected_P2048.json';record={'registration_sha256':regsha,'N':2048,'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'step':2500,'total_updates':2500,'arm_key':name};selected.write_text(json.dumps(record))
    loaded,identity=ex.load_selected_budget(selected,torch.ones(32),'cpu');assert identity['step']==2500
    for key,value in model.state_dict().items():assert torch.equal(value,loaded.state_dict()[key])
    record['step']=10000;selected.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='step'):ex.load_selected_budget(selected,torch.ones(32),'cpu')
    record['step']=2500;record['checkpoint_sha256']='0'*64;selected.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='SHA'):ex.load_selected_budget(selected,torch.ones(32),'cpu')
