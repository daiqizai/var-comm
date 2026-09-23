import pytest
from token_efficiency import thermal_guard as g

def row(temp=70,hw=False,sw=False):
    return {'temperature':temp,'hardware_thermal_slowdown':hw,'software_thermal_slowdown':sw}

def test_software_throttle_is_not_hidden_by_temperature():
    r=g.parse_thermal('83, Not Active, Active\n')
    assert g.hot(r) and not g.cool(r)
    assert g.hot(row(86)) and g.hot(row(hw=True))

@pytest.mark.parametrize('text',['70, Not Active','70, Not Active, N/A','130, Not Active, Not Active','bad, Active, Active'])
def test_incomplete_sensor_is_rejected(text):
    with pytest.raises(ValueError):g.parse_thermal(text)

def test_only_consecutive_samples_pause_and_resume():
    w=g.Window()
    assert w.add(row(sw=True))==(False,False)
    assert w.add(row(sw=True))==(False,False)
    w.add(row())
    assert w.add(row(sw=True))==(False,False)
    assert w.add(row(sw=True))==(False,False)
    assert w.add(row(86))==(True,False)
    for _ in range(5):assert w.add(row())==(False,False)
    assert w.add(row())==(False,True)
    assert w.add(row(76))==(False,False)

def test_pid_reuse_and_wrong_command_cannot_be_signaled(monkeypatch):
    prior={'pid':123,'start_ticks':'100','cmdline':'python3 -u -m token_efficiency.coordinator_v2 ','state':'S'}
    sent=[];monkeypatch.setattr(g.os,'kill',lambda *x:sent.append(x))
    monkeypatch.setattr(g,'proc_identity',lambda _:dict(prior,start_ticks='101'))
    with pytest.raises(RuntimeError,match='identity changed'):g.signal_verified(prior)
    wrong=dict(prior,cmdline='unrelated task')
    monkeypatch.setattr(g,'proc_identity',lambda _:wrong)
    with pytest.raises(RuntimeError,match='owned budget'):g.signal_verified(wrong)
    assert sent==[]
    monkeypatch.setattr(g,'proc_identity',lambda _:prior)
    assert g.signal_verified(prior) and sent==[(123,g.signal.SIGTERM)]
    monkeypatch.setattr(g,'proc_identity',lambda _:None)
    assert g.signal_verified(prior) is False

def test_resume_requires_safe_checkpoint_and_no_surviving_child(tmp_path,monkeypatch):
    from latent_enhancement.runtime import write_json,digest
    monkeypatch.setattr(g,'OUT',tmp_path);monkeypatch.setattr(g,'active_owned_queue',lambda:[])
    write_json(tmp_path/'status.json',{'status':'PAUSED_SAFE','command':['token_efficiency.budget_train','--N','2048']})
    cp=tmp_path/'weights';cp.write_bytes(b'synthetic-checkpoint')
    p=tmp_path/'training/P2048_seed2026092304/latest.json';r={'path':str(cp),'sha256':digest(cp),'reason':'safe_pause'};write_json(p,r)
    assert g.pause_receipt()['checkpoints']['P2048']==r
    write_json(p,{**r,'reason':'regular'})
    with pytest.raises(RuntimeError,match='safe checkpoint'):g.pause_receipt()
    write_json(p,r);cp.write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='checkpoint hash'):g.pause_receipt()
    r['sha256']=digest(cp);write_json(p,r);monkeypatch.setattr(g,'active_owned_queue',lambda:[{'pid':123}])
    with pytest.raises(RuntimeError,match='still alive'):g.pause_receipt()
