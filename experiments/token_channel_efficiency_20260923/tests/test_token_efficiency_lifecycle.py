import json
from pathlib import Path
import numpy as np,pytest
from token_efficiency.budget_lifecycle import should_extend
from token_efficiency import budget_lifecycle as lifecycle
from token_efficiency import C_lifecycle as cl
from token_efficiency.C_report import cross_noise_diagnostic,grouped_pair
from token_efficiency.common import read,register
from latent_enhancement.runtime import digest,write_json

def test_extension_requires_both_registered_improvements():
    assert should_extend([1.,.99,.98])[0]
    assert not should_extend([1.,.99,.991])[0]
    assert not should_extend([1.,1.,.9])[0]
    for values in ([1,0,.1],[1,float('nan'),.5],[1,.5]):
        with pytest.raises(ValueError):should_extend(values)

def build_budget(tmp_path,monkeypatch,values):
    monkeypatch.setattr(lifecycle,'OUT',tmp_path);monkeypatch.setattr(lifecycle,'checked_checkpoint',lambda rec,root:tmp_path/'weights_not_loaded')
    folder=tmp_path/'training/P2048_seed2026092304';write_json(folder/'latest.json',{'step':20000,'path':'not_a_quality_test','sha256':'x'})
    write_json(folder/'selected_P2048.json',{'step':17500})
    for s,v in zip((15000,17500,20000),values):
        p=folder/'calibration'/f'full_{s:05d}.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic CPU lifecycle fixture, no model or metrics\n')
        write_json(p.with_suffix('.json'),{'step':s,'rows':15000,'sha256':digest(p),'summary':{'P2048':v}})
    return folder

def test_extension_plan_keeps_original_parent_on_interrupted_resume(tmp_path,monkeypatch):
    folder=build_budget(tmp_path,monkeypatch,[1,.99,.98]);a=lifecycle.next_budget_action(2048);plan=read(a['plan'])
    assert a['kind']=='extend' and a['until']==30000 and plan['microbatch']==8
    write_json(folder/'latest.json',{'step':23456,'sha256':'later-resume'})
    assert lifecycle.next_budget_action(2048)==a and read(a['plan'])==plan

def test_rule_stop_requires_complete_hashed_calibration(tmp_path,monkeypatch):
    folder=build_budget(tmp_path,monkeypatch,[1,.99,.991]);a=lifecycle.next_budget_action(2048)
    assert a['kind']=='final' and read(a['path'])['convergence_claimed'] is False
    (folder/'calibration/full_20000.csv').write_text('tampered')
    with pytest.raises(RuntimeError,match='receipt'):lifecycle.next_budget_action(2048)

def test_matched_pair_extends_together_if_either_arm_improves(tmp_path,monkeypatch):
    monkeypatch.setattr(cl,'checked_checkpoint',lambda *a:None)
    write_json(tmp_path/'latest.json',{'step':20000});write_json(tmp_path/'completion.json',{'state':{'step':20000,'updates':{'H6-V':20000,'H6-P':20000}}})
    for s,v in zip((15000,17500,20000),(1,.99,.98)):
        p=tmp_path/'calibration'/f'full_{s:05d}.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU fixture')
        write_json(p.with_suffix('.json'),{'step':s,'rows':30000,'sha256':digest(p),'summary':{'H6-V':v,'H6-P':1}})
    assert cl.next_group_action(tmp_path)=={'kind':'extend','until':30000}
    decision=read(tmp_path/'delivery_decisions/at_20000.json');assert decision['paired_arms_extend_together'] and not decision['arms']['H6-P']['extend']

def test_cross_noise_choice_excludes_held_noise_and_retains_negative_gain():
    u=np.zeros((2,2,1,3));u[0]=[1,1,100];u[1]=[2,2,0]
    rows=cross_noise_diagnostic(u);held=[r for r in rows if r['held_noise_index']==2]
    assert all(r['source_action']==0 and r['source_utility']==100 for r in held)
    with pytest.raises(ValueError):cross_noise_diagnostic(np.zeros((2,2,1,2)))

def test_identity_registration_never_overwrites(tmp_path):
    p=tmp_path/'identity.json';register(p,{'selected_sha':'old'});before=p.read_bytes()
    with pytest.raises(RuntimeError):register(p,{'selected_sha':'changed'})
    assert p.read_bytes()==before
