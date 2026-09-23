import pytest
from latent_enhancement.runtime import write_json,digest
from token_efficiency import microbatch_runtime as m

def make_execution(tmp_path):
    source=tmp_path/'source.py';source.write_text('unchanged')
    path=tmp_path/'execution.json';r={'historical_registration_sha256':'reg','actual_microbatch':8,'spec':{'version':'v1'},'bindings':{str(source):digest(source)}};write_json(path,r)
    e={'path':str(path),'sha256':digest(path),'microbatch':8,'version':'v1'}
    return e,source

def test_selected_requires_same_verified_runtime_as_checkpoint(tmp_path):
    e,source=make_execution(tmp_path);payload={'execution_identity':e,'registration_sha256':'reg'}
    m.verify_selected_execution({'execution_identity':e},payload)
    with pytest.raises(RuntimeError,match='lineage'):m.verify_selected_execution({},payload)
    with pytest.raises(RuntimeError,match='historical'):m.verify_selected_execution({'execution_identity':e},{**payload,'registration_sha256':'other'})
    source.write_text('changed')
    with pytest.raises(RuntimeError):m.verify_selected_execution({'execution_identity':e},payload)
    m.verify_selected_execution({}, {'registration_sha256':'old'})

def test_initial_migration_only_accepts_exact_checkpoint(tmp_path):
    e,_=make_execution(tmp_path);cp=tmp_path/'checkpoint';cp.write_bytes(b'synthetic')
    spec={'initial_resume':{'step':8862,'sha256':digest(cp)}};payload={'state':{'step':8862}}
    m.validate_execution_checkpoint(payload,e,spec,cp)
    with pytest.raises(RuntimeError,match='initial'):m.validate_execution_checkpoint({'state':{'step':8861}},e,spec,cp)
    cp.write_bytes(b'wrong')
    with pytest.raises(RuntimeError,match='initial'):m.validate_execution_checkpoint(payload,e,spec,cp)
    m.validate_execution_checkpoint({'execution_identity':e},e,spec,cp)
    with pytest.raises(RuntimeError,match='version'):m.validate_execution_checkpoint({'execution_identity':dict(e,microbatch=16)},e,spec,cp)

def test_execution_receipt_hash_is_enforced(tmp_path):
    e,_=make_execution(tmp_path)
    with pytest.raises(RuntimeError,match='hash'):m.verify_execution(dict(e,sha256='wrong'))
    with pytest.raises(RuntimeError,match='context'):m.verify_execution(dict(e,microbatch=16))

def test_new_guard_recognizes_new_controller_only(monkeypatch):
    from token_efficiency import thermal_guard_v2 as g
    r={'pid':123,'start_ticks':'1','cmdline':'python -u -m token_efficiency.coordinator_v3 ','state':'S'}
    sent=[];monkeypatch.setattr(g,'proc_identity',lambda _:r);monkeypatch.setattr(g.os,'kill',lambda *v:sent.append(v))
    assert g.signal_verified(r) and len(sent)==1
    wrong=dict(r,cmdline='python -m token_efficiency.coordinator_v2 ');monkeypatch.setattr(g,'proc_identity',lambda _:wrong)
    with pytest.raises(RuntimeError,match='owned'):g.signal_verified(wrong)
