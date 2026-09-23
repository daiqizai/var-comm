import json
import torch
import pytest
from latent_mechanisms.linear_measurement import correction_coordinates, validate_coverage
from latent_followup.policy_development import align_complete_rows, _aggregate_source_snr
from latent_followup.run_identity import selection_record, checked_checkpoint, load_resume

def test_failed_norm_preserves_source_and_residual_base():
    base=torch.tensor([[1.,2.,3.,4.]])
    for method in ('source','residual'):
        delta=correction_coordinates(torch.zeros(1,2),base[:,:2],torch.tensor([False]),method)
        got=base.clone();got[:,:2]+=.6*delta
        assert torch.equal(got,base)
    # Accepted zero norm is a legitimate zero measurement, not packet failure.
    delta=correction_coordinates(torch.zeros(1,2),base[:,:2],torch.tensor([True]),'source')
    assert torch.equal(delta,-base[:,:2])
    # Accepted corrupt code must use the RX observation, not oracle TX truth.
    assert torch.equal(correction_coordinates(torch.tensor([[9.,8.]]),base[:,:2],torch.tensor([True]),'source'),torch.tensor([[8.,6.]]))

def test_completion_requires_every_unique_key():
    rows=[dict(source_index=0,method=m,snr_db=1,seed=n) for m in ('source','residual') for n in (1,2,3)]
    validate_coverage(rows,1,[1],[1,2,3])
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(RuntimeError):validate_coverage(bad,1,[1],[1,2,3])

def test_pair_rejects_duplicate_changed_seed_and_preprocessing():
    rows=[dict(source_index=0,snr_db=1,seed=n,preprocessing_id='p',psnr_db=20,lpips_alex=.1,dino_cosine=.8) for n in (1,2,3)]
    a,b=align_complete_rows(rows,list(reversed(rows)))
    assert _aggregate_source_snr(a)==_aggregate_source_snr(b)
    for bad in (rows+[rows[0]],rows[:2]+[{**rows[2],'seed':4}],rows[:2]+[{**rows[2],'preprocessing_id':'q'}]):
        with pytest.raises(RuntimeError):align_complete_rows(rows,bad)

def test_selected_and_resume_reject_changed_identity(tmp_path):
    p=tmp_path/'c.pt';torch.save({'control':{},'registration_sha256':'a'},p)
    r=selection_record('original_residual_control',{'step':4000},p,40000)
    assert r['arm_key']=='control' and r['total_step']==44000
    assert checked_checkpoint(r,tmp_path)==p
    with pytest.raises(RuntimeError,match='registration'):load_resume(r,tmp_path,'b')
    p.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='SHA'):checked_checkpoint(r,tmp_path)

def test_registration_rejects_changed_config_and_unbound_latest(tmp_path,monkeypatch):
    import latent_followup.run_identity as module
    exp=tmp_path/'exp';exp.mkdir();config=exp/'config.json';config.write_text('{}')
    gate=tmp_path/'gate.json';gate.write_text('{}');parent=tmp_path/'parent.json';parent.write_text('{}')
    b=tmp_path/'B';(b/'rx_cache').mkdir(parents=True);(b/'rx_cache/completion.json').write_text('{}')
    monkeypatch.setattr(module,'CONFIG',config);monkeypatch.setattr(module,'EXPERIMENT',exp);monkeypatch.setattr(module,'OUT_B',b);monkeypatch.setattr(module,'decoder_gate_path',lambda:gate)
    output=tmp_path/'run';sha=module.register_run(output,config,[parent]);assert sha==module.register_run(output,config,[parent])
    config.write_text('{"noise_seed":7}')
    with pytest.raises(RuntimeError):module.register_run(output,config,[parent])
    old=tmp_path/'old';old.mkdir();(old/'latest.json').write_text('{}')
    with pytest.raises(RuntimeError,match='unbound'):module.register_run(old,config,[parent])
