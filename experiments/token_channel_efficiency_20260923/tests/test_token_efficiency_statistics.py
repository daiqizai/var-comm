"""Synthetic statistics fixtures, not scientific measurements."""
import copy
import numpy as np,pytest
from token_efficiency.statistics import FrameTable,freeze_digital_policy,validate_policy_on_development,paired,minimum_tested_uses,target_rates

SOURCES={'a':'rgb-a','b':'rgb-b','c':'rgb-c'};SNRS=[1,7];SEEDS=[1,2,3]
METHODS=['raw2048-low','raw2048-high','raw3060','P2048','P3060','qam2048']

def fixture(population='calibration'):
    rows=[]
    for method in METHODS:
        N=3060 if '3060' in method else 2048;pure=method.startswith('P');qam=method.startswith('qam')
        quality={'raw2048-low':(23.,.18),'raw2048-high':(25.,.12),'raw3060':(28.,.06),'P2048':(26.,.11),'P3060':(27.,.08),'qam2048':(28.,.07)}[method]
        for i,source in enumerate(SOURCES):
            for snr in SNRS:
                for seed in SEEDS:
                    psnr=quality[0]+(i-1)*.6+(seed-2)*.2;lp=quality[1]+(i-1)*.01
                    rows.append({'method':method,'source_id':source,'population':population,'preprocessing_id':SOURCES[source],'snr_db':snr,'noise_seed':seed,'run_id':population+'-fixture','context_sha256':'context-'+method,'family':'continuous' if pure else 'raw','N':N,'mcs':'continuous' if pure else '16QAM' if qam else 'QPSK','energy_constraint':'fixed_constellation_average_2_per_symbol' if qam else 'per_frame_2N','noise_namespace':method,'N_header':0 if pure else 70,'N_data':0 if pure else N-70,'N_continuous':N if pure else 0,'E':float(2*N*(.8+.1*seed)) if qam else 2*N,'mse':10**(-psnr/10),'psnr_db':psnr,'lpips_alex':lp,'dino_cosine':.9,'header_ok':'not_applicable' if pure else seed!=1,'body_crc_ok':'not_applicable' if pure else seed!=1,'source_overflow_erasure':False})
    return rows

def table(rows=None,population='calibration'):
    return FrameTable(fixture(population) if rows is None else rows,population,SOURCES,SNRS,SEEDS,METHODS)

@pytest.mark.parametrize('kind',['duplicate','missing','nan','scope','preprocessing','energy','resource','qam_constant','population','noise'])
def test_grid_scope_and_illegal_data_rejected(kind):
    rows=fixture()
    if kind=='duplicate':rows[-1]=dict(rows[0])
    if kind=='missing':rows.pop()
    if kind=='nan':rows[0]['psnr_db']=float('nan')
    if kind=='scope':rows[0]['context_sha256']='different'
    if kind=='preprocessing':rows[0]['preprocessing_id']='wrong'
    if kind=='energy':rows[0]['E']=4090
    if kind=='resource':rows[0]['N_data']-=1
    if kind=='qam_constant':rows[-1]['energy_constraint']='per_frame_2N'
    if kind=='population':rows[0]['population']='development'
    if kind=='noise':rows[0]['noise_seed']=99
    with pytest.raises(ValueError):table(rows)

def test_source_pairing_and_mean_identity():
    t=table();r=paired(t,'raw3060','raw2048-high',1,repeats=10000)
    result=r['metrics']['psnr_db'];assert result['difference']==pytest.approx(3.)
    assert result['difference']==pytest.approx(result['mean_left']-result['mean_right'])
    np.testing.assert_allclose(result['ci95'],[3.,3.]);assert r['sources']==3
    assert r['same_noise_observations_claimed'] is False and r['training_seed_variation_included'] is False
    assert r['left_context']['N']==3060 and r['right_context']['N']==2048

def test_calibration_policy_frozen_before_development():
    cal=table();policy=freeze_digital_policy(cal)
    assert all(c['method']=='raw2048-high' for c in policy['choices'] if c['N']==2048 and c['mcs']=='QPSK')
    dev=table(population='development');validate_policy_on_development(dev,policy)
    with pytest.raises(ValueError):freeze_digital_policy(dev)
    changed=copy.deepcopy(policy);changed['choices'][0]['context_sha256']='wrong'
    with pytest.raises(ValueError):validate_policy_on_development(dev,changed)
    changed=copy.deepcopy(policy);changed['choices'].pop()
    with pytest.raises(ValueError):validate_policy_on_development(dev,changed)

def test_minimum_tested_N_preserves_negative_and_unmet_targets():
    policy=freeze_digital_policy(table());dev=table(population='development')
    targets={'status':'FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT','source_csv_sha256':'calibration-source','targets':{'balanced':{'psnr_min':25.5,'lpips_max':.115},'unreachable':{'psnr_min':40,'lpips_max':.001}}}
    rows=minimum_tested_uses(dev,policy,targets,['P2048','P3060'])
    for r in rows:
        if r['target']=='unreachable':assert r['saving'] is None and r['status']=='NOT_REACHED_IN_MEASURED_GRID'
        elif r['mcs']=='QPSK':assert r['digital_N_required']==3060 and r['continuous_N_required']==2048 and r['saving']==pytest.approx(1-3060/2048)
        else:assert r['digital_N_required']==2048 and not r['strict_per_frame_equal_energy_protocol']
        assert r['interpolation_used'] is False
    rate=target_rates(dev,'P2048',1,{'psnr_min':25.9,'lpips_max':.12})
    assert rate['mean_target_met'] and 0<rate['frame_joint_success_rate']<1

def test_failure_frames_count_and_actual_qam_energy_distribution():
    t=table();assert len(t.frames)==len(METHODS)*3*2*3
    s=next(r for r in t.summary() if r['method']=='qam2048' and r['snr_db']==1)
    assert s['E_min']<s['E_mean']<s['E_max'] and s['energy_constraint']=='fixed_constellation_average_2_per_symbol'
