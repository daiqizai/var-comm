"""Synthetic lifecycle tests; no signals or real training processes are used."""
import json
import pytest
from token_efficiency import retired_gate as g
from latent_enhancement.runtime import digest

@pytest.fixture
def gate(tmp_path,monkeypatch):
    out=tmp_path/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923';out.mkdir(parents=True)
    old=tmp_path/'outputs/SHORT-PREFIX-20260923';old.mkdir()
    source=tmp_path/'source.txt';source.write_text('unchanged')
    (old/'queue_identity.json').write_text(json.dumps({str(source):digest(source)}))
    receipt=out/'retirement.json';receipt.write_text(json.dumps({'status':'VERIFIED_PREDECESSOR_EXITED_WITHOUT_M6_TRAINING','held_controller_pid':123,'held_controller_start_ticks':'456','short_prefix_release_allowed':False,'evidence_bindings':{str(source):digest(source)}}))
    monkeypatch.setattr(g,'ROOT',tmp_path);monkeypatch.setattr(g,'OUT',out);monkeypatch.setattr(g,'proc_identity',lambda pid:None);monkeypatch.setattr(g,'live_short_prefix',lambda:[])
    return {'predecessor_mode':'RETIRED_VERIFIED','held_controller_pid':123,'held_controller_start_ticks':'456','retirement_receipt':str(receipt),'retirement_receipt_sha256':digest(receipt)}

def test_retired_predecessor_and_recycled_pid_are_distinguished(gate,monkeypatch):
    assert not g.check_retired(gate)['short_prefix_release_allowed']
    monkeypatch.setattr(g,'proc_identity',lambda pid:{'start_ticks':'999','state':'R'})
    assert not g.check_retired(gate)['short_prefix_release_allowed']
    monkeypatch.setattr(g,'proc_identity',lambda pid:{'start_ticks':'456','state':'T'})
    with pytest.raises(RuntimeError,match='still alive'):g.check_retired(gate)

def test_live_predecessor_work_refused(gate,monkeypatch):
    monkeypatch.setattr(g,'live_short_prefix',lambda:[{'pid':321}])
    with pytest.raises(RuntimeError,match='already active'):g.check_retired(gate)

def test_altered_receipt_or_source_refused(gate,tmp_path):
    old=gate['retirement_receipt_sha256'];gate['retirement_receipt_sha256']='bad'
    with pytest.raises(RuntimeError,match='hash'):g.check_retired(gate)
    gate['retirement_receipt_sha256']=old;(tmp_path/'source.txt').write_text('changed')
    with pytest.raises(RuntimeError,match='changed'):g.check_retired(gate)

def test_predecessor_lock_prevents_duplicate_controller(tmp_path):
    path=tmp_path/'controller.lock';first=g.acquire_predecessor_lock(path)
    try:
        with pytest.raises(BlockingIOError):g.acquire_predecessor_lock(path)
    finally:first.close()
    second=g.acquire_predecessor_lock(path);second.close()
