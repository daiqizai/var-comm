"""Register the user-selected immutable continuation after CPU admission."""
import argparse
from pathlib import Path
from audit_selected_runtime import ROSTER
from history_common import read, write, sha, source_bindings
from history_controller import validate_queue


def register(root, audit_path):
    root=Path(root).resolve(); here=Path(__file__).resolve().parent
    out=root/'outputs/HISTORICAL-METRICS-20261003'
    audit_path=Path(audit_path).resolve(); audit=read(audit_path)
    if audit.get('status')!='PASS' or audit.get('GPU') is not False:
        raise RuntimeError('Complete CPU original-table audit is required')
    expected={study:count for _,study,count in ROSTER}
    if {r['study']:r.get('frames') for r in audit['studies']} != expected:
        raise RuntimeError('Constructor audit scope differs')
    audited={Path(p).name:h for p,h in audit['source_bindings'].items()}
    for p in here.glob('historical_*.py'):
        if audited.get(p.name)!=sha(p):
            raise RuntimeError('Adapter differs from the CPU-admitted source: '+p.name)
    caches=out/'selected_cache_admission.json'; cache=read(caches)
    cache_manifest=out/'cached_studies.json'
    if (cache['status']!='SELECTED_CACHE_CPU_ADMISSION_PASS'
            or cache['manifest_sha256']!=sha(cache_manifest)):
        raise RuntimeError('Selected saved RGB admission differs')
    counts=dict(required=46800,analysis_supplements=9900,total=56700)
    required=[study for _,study,_ in ROSTER[:8]]
    optional=[study for _,study,_ in ROSTER[8:]]
    coverage=dict(status='REGISTERED_SELECTED_SCOPE',user_scope_date='2026-10-03',
        selected_studies={study:dict(adapter=adapter,frames=count,sources=100,
            phase='required' if study in required else 'analysis_supplements') for adapter,study,count in ROSTER},
        counts=counts, original_six_studies='existing independent pipeline; preserve all 237000 rows',
        selection_before_inference=True,training_updates=0,policy_selection_updates=0,
        optional_user_decision='必补完成后补这三项',
        excluded=['7','10','12-16','18','19','20','22','new low-bandwidth external training'],
        N2048='Existing final paid-class D_C policies included; current-protocol D_U remains unmeasured',
        no_channel=dict(outputs=18,sources=100,rows=1800,noise_repeats=0,
                        old_scalar_replay_rows=1700,new_deterministic_m8_U_rows=100),
        original_scalar_cache_crosscheck=read(out/'optional8_main_cache_crosscheck.json'),
        classification='D_U main; paid true-class D_C reference explicitly marked',
        seed_lineage='P4084 selected27500 inherited 10k/initialization2026092303; second seed fresh2026092404 separately',
        KID='DEFERRED_HOLDOUT',FID='NOT_EVALUATED')
    coverage_path=out/'coverage_manifest.json'
    if coverage_path.exists() and read(coverage_path)!=coverage:
        raise RuntimeError('Existing selected coverage differs')
    write(coverage_path,coverage)
    contrasts=[]
    def pair(name,a,b,fields=('N','snr_db','decoder')):
        contrasts.append(dict(name=name,A=a,B=b,match_fields=list(fields)))
    for n in (2048,3060,4084):
        p_study='FINAL_P2048_P3060' if n!=4084 else 'FINAL_P4084_SELECTED_SEEDS'
        p_method='P'+str(n) if n!=4084 else 'P4084_N4084_seed2026092304'
        for family in ('raw','arithmetic'):
            pair(f'paid_C_{family}_QPSK_vs_P_N{n}',
                dict(study='FINAL_DIGITAL_QPSK',method=f'final_{family}_QPSK_N{n}_Dc'),
                dict(study=p_study,method=p_method))
    for method in ('raw_adaptive','arithmetic_adaptive'):
        pair(method+'_vs_perceptual_deepjscc_D0_historical',
             dict(study='LEGACY_N3060_FINAL',method=method),
             dict(study='LEGACY_N3060_FINAL',method='perceptual_deepjscc'),('N','snr_db'))
    for m in range(4,9):
        for condition in ('U','C'):
            pair(f'no_channel_{condition}_vs_prefix_m{m}',
                dict(study='SELECTED_NOISELESS_REFERENCES',method=f'Dc_{condition}_m{m}'),
                dict(study='SELECTED_NOISELESS_REFERENCES',method=f'Dc_prefix_m{m}'))
    for study,base in [('OPTIONAL_RX_STEP2_A','B2'),('OPTIONAL_RX_STEP2_B','P_low')]:
        for method in ('A1_policy','A2_policy','V_policy'):
            pair(study+'_'+method+'_vs_base',dict(study=study,method=method),dict(study=study,method=base))
    pair('H6_V_vs_P_N4084_13dB',
         dict(study='OPTIONAL_H6_13DB',method='H6-V_N4084_seed2026092304'),
         dict(study='OPTIONAL_H6_13DB',method='H6-P_N4084_seed2026092304'))
    for method in ('original_1024_Dc','calibration_frozen_resource_lookup'):
        pair(method+'_vs_original_pure10k',dict(study='OPTIONAL_PHASE2_MAIN',method=method),
             dict(study='OPTIONAL_PHASE2_MAIN',method='pure_continuous'))
    shared=root/'experiments/unified-metrics-20261002'
    proof_paths=[audit_path,caches,cache_manifest,out/'optional8_main_cache_crosscheck.json']
    queue=dict(status='REGISTERED',scope='user selected required then analysis supplements',
        training_updates=0,policy_selection_updates=0,source_bindings=source_bindings(here),
        shared_metric_bindings={str(shared/name):sha(shared/name)
            for name in ('runner.py','replay.py','metric_models.py','batch_speed.py','publish.py')},
        input_proof_bindings={str(p):sha(p) for p in proof_paths},
        coverage_manifest=dict(path=str(coverage_path),sha256=sha(coverage_path)),
        jobs=[dict(adapter=adapter,study=study) for adapter,study,_ in ROSTER],
        execution_groups=dict(required=required,analysis_supplements=optional),
        contrasts=contrasts,retain_float_reconstructions=True,
        full_history_claim=False,stop_after_scope=True)
    path=out/'queue_registration.json'
    if path.exists() and read(path)!=queue:
        raise RuntimeError('Existing historical queue differs; never overwrite active registration')
    write(path,queue)
    validate_queue(root,here,path)
    print('REGISTERED',len(queue['jobs']),'studies;',sum(expected.values()),'rows;',sha(path),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True)
    a=p.parse_args();register(a.root,a.audit)
