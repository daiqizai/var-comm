"""Read-only scientific admission and explicit finite EMAIN registration."""
import argparse
import subprocess
from common import *

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'registration.json').exists(),'Existing registration must be reused')
    sources=read(OUT/'e0_reuse/source_records.json');configs=read(OUT/'e0_reuse/frozen_configs.json')
    timing=read(ROOT/'results/baseline_strength_20261011/timing/six_method_v1/registration.json')
    v=read(checked(timing['visual_request']));p=read(checked(timing['phy_request']))
    kodak=read(ROOT/'results/generalization_kodak_final_20261013/registration.json')
    import ep_plan
    expected={4:'m7_K25_q2_r2-3',10:'m8_K126_q4_r2-3',19:'m9_K192_q6_r5-6'}
    policies=configs['policies'];candidates={x['candidate_id']:x for x in ep_plan.candidates()}
    for s,cid in expected.items():
        require(policies[str(s)]['candidate_id']==cid,'Original calibration winner differs')
        # Preserve the exact scientific candidate from the frozen constructor.
        require(all(policies[str(s)].get(k)==value for k,value in candidates[cid].items()),'Policy mathematical fields differ')
        policies[str(s)]=candidates[cid]
    files=[Path(__file__).with_name(n+'.py') for n in ('common','prepare','owner','source','phy','visual','score')]+list(ES.glob('*.py'))
    files+=list(TS.glob('t1_*.py'))+[TS/'t2_ledger.py']
    code={str(x):sha(x) for x in files}
    r=dict(schema='ENTROPY_MAIN_COMMON500_V1',status='REGISTERED_NOT_EXECUTED',created_unix=time.time(),
        source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        sources=sources['records'],source_manifest=descriptor(OUT/'e0_reuse/source_records.json'),
        frozen_policy_binding=descriptor(OUT/'e0_reuse/frozen_configs.json'),policies=policies,SNRs=list(SNRS),noise_seeds=list(SEEDS),
        logical_frames=4500,N=1024,training_updates=0,E4_allowed=False,engineering_frames_new=0,
        existing_H800_qualification_reused=True,visual_request=timing['visual_request'],phy_request=timing['phy_request'],
        native_identity=v['visual_identity'],metric_config=kodak['metric_config'],code_bindings=code,
        method='EC_VAR_PARTIAL',population='original_common500_frozen_policy_extension_not_first_blind_test',
        source_encoder_new_calls=0,source_entropy_TX_cap=500,source_RX_cap=4500,reconstruction_cap=4500,metric_cells_cap=18000,packet_decode_cap=9000,
        noise_rule='t1_phy.standard_noise(source_id,noise_seed)',counter_rule='(six_snr_index*500+source_index)*3+(seed-6201)',
        deadline_unix=time.time()+24*3600,stage_max_seconds=6*3600,target_gpu=0,
        output_root=str(ART),quality_shared_gpu=True,timing_shared_authorized=True,
        model_policy_changed=False,automatic_science_retry=False,old_host_bits_used=False,
        exact_received_input_reuse=True,receiver_truth_argument=False)
    require(len(sources['records'])==500,'Original500 source count')
    for rec in sources['records']:checked(rec['archive'])
    write(OUT/'registration.json',r)
    print(json.dumps(dict(status='REGISTERED_NOT_EXECUTED',registration=descriptor(OUT/'registration.json'),logical_frames=4500)))

if __name__=='__main__':prepare()
