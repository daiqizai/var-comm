import pytest
from artifact_bridge import combined_cost_rule,ready_actual,quality_bundle,freeze_shortlist
from uep_common import write,sha,csv_write,read

def test_parallel_cost_uses_critical_path():
    r=combined_cost_rule(30,20,10,5,1)
    assert r['estimated_total_hours']==46 and not r['screen300']
    assert combined_cost_rule(34,20,10,5,1)['screen300']

def test_gate_requires_all_actual_snrs():
    cells=[dict(N=1024,snr_db=s,ready_for_actual_link=True) for s in (4,7,10,13)]
    assert ready_actual({'cells':cells})
    assert not ready_actual({'cells':cells[:-1]})
    cells[0]['ready_for_actual_link']=False
    assert not ready_actual({'cells':cells})

def test_reject_nonfinite_cost():
    with pytest.raises(RuntimeError):combined_cost_rule(float('nan'),1,2,3)

def test_screen_bridge_retains_full_calibration_order(tmp_path):
    ids=['source%d'%i for i in range(1000)]
    write(tmp_path/'registration.json',dict(source_ids=ids,source_count=300))
    rows=[dict(source_id=s,state_id='gray',receiver='gray',population='calibration',dinov2_vitl14_cosine=.2) for s in ids[:300]]
    csv_write(tmp_path/'quality_per_source.csv',rows)
    write(tmp_path/'completion.json',dict(status='SOURCE_QUALITY_COMPLETE',population='calibration',development_read=False,
        source_ids=ids[:300],sources=300,rows=300,registration_sha256=sha(tmp_path/'registration.json'),
        outputs={str(tmp_path/'quality_per_source.csv'):sha(tmp_path/'quality_per_source.csv')}))
    value=quality_bundle(tmp_path,tmp_path/'bundle.json')
    assert value['source_ids']==ids[:300] and value['calibration_source_ids']==ids

def test_shortlist_neighbour_retains_full_matched_control_pool(tmp_path):
    rows=[dict(stable_id='a',N=1024,G=2,m=6,K=3,j=4),dict(stable_id='b',N=1024,G=2,m=7,K=7,j=5),
          dict(stable_id='c',N=1024,G=2,m=7,K=7,j=5),dict(stable_id='excluded',N=1024,G=2,m=7,K=9,j=5)]
    choice=lambda r:dict(stable_id=r['stable_id'],profile=r)
    screen=dict(stage='screen',source_count=300,development_read=False,synthetic=False,cells=[dict(
        independent_optima={'B3':{'top3':[choice(rows[0])]}},strong_baseline=choice(rows[0]),matched_to_B3={},
        frozen_distinct_validation_candidate=dict(status='FROZEN',selected=choice(rows[1])))])
    write(tmp_path/'profiles.json',rows);write(tmp_path/'screen.json',screen)
    freeze_shortlist(tmp_path/'profiles.json',tmp_path/'screen.json',tmp_path/'shortlist.json')
    assert {r['stable_id'] for r in read(tmp_path/'shortlist.json')}=={'a','b','c'}
