"""Metadata-only separate BPG request; preserves the live neural request."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SUP=ROOT/'.research/main_raw64_20261008_paper_supplement'
ACTUAL=ROOT/'.research/main_raw64_20261009_remaining/a3_bpg_memory_v1'
REMOTE='/home/liulu/projects/VAR_COMM'
OUT=REMOTE+'/outputs/PAPER-SUPPLEMENT-20261008'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,value):
    with Path(p).open('x',encoding='utf8',newline='\n')as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def prepare(destination):
    destination=Path(destination)
    if destination.exists():raise RuntimeError('Preserve prior materials; choose a new destination')
    base_path=SUP/'a3_fixed16/materials_v1/request.json';r=read(base_path)
    assert sha(base_path)=='4bcc1dd7d8dcc84376f0d9de827ef89db5ae5133924e60fb6d63f861e73f2cb7'
    # Keep the already running neural executable bytes unchanged.
    for name,entry in r['a3_sources'].items():assert sha(Path(__file__).parent/name)==entry['sha256']
    report=read(SUP/'bpg_delivery_actual/actual_report_request.json')
    original=SUP/'bpg_audit/actual/bpg_metadata_r2/request.json'
    freeze_path=SUP/'bpg_audit/actual/freeze.json';freeze=read(freeze_path)
    assert sha(original)==report['bpg_request']['sha256']==freeze['request_sha256']
    assert sha(freeze_path)==report['bpg_freeze']['sha256']
    assert freeze['status']=='BPG_POLICIES_FROZEN_ON_CALIBRATION_ONLY_V1' and freeze['holdout_used_for_selection']is False
    parity=read(ACTUAL/'qualification/completion.json');q=read(ACTUAL/'qualification/request.json')
    build=read(ACTUAL/'build/build_receipt.json')
    assert parity['status']=='COMPLETE' and parity['source_count']==20 and parity['case_count']==40
    assert parity['complete_byte_identity_all']and parity['exact_uint8_pixel_identity_all']
    assert parity['request_sha256']==sha(ACTUAL/'qualification/request.json')
    assert parity['library']==build['library']and parity['library_sha256']==build['library_sha256']
    assert q['bindings'][OUT+'/a3_bpg_memory_v1/build/build_receipt.json']==sha(ACTUAL/'build/build_receipt.json')
    memory=ROOT/'experiments/paper_supplement_20261008/a3_bpg/memory_codec.py'
    memory_remote=REMOTE+'/experiments/paper_supplement_20261008/a3_bpg/memory_codec.py'
    assert q['bindings'][memory_remote]==sha(memory)
    phy_registration=read(SUP/'bpg_phy_runtime_v1/materials_v1/phy_registration.json')
    phy_remote=OUT+'/bpg_phy_runtime_v1/bpg_phy.py'
    assert sha(SUP/'bpg_phy_runtime_v1/bpg_phy.py')==phy_registration['source_bindings'][phy_remote]
    endpoint=ROOT/'experiments/paper_supplement_20261008/a3_bpg/timing_endpoint.py'
    runner=Path(__file__).parent/'benchmark_bpg.py'
    r.update(schema='A3_FIXED16_BPG_TIMING_V1',methods=['BPG_LDPC_native256'],bpg_packet_cap=384,
        packet_caps={'BPG_LDPC_native256':384},total_independent_packet_cap=384,
        bpg_out=OUT+'/a3_timing_v1/results/BPG_LDPC_native256',
        bpg_endpoint=dict(path=OUT+'/a3_timing_v1/runtime/bpg_timing_endpoint.py',sha256=sha(endpoint)),
        bpg_runner=dict(path=OUT+'/a3_timing_v1/runtime/benchmark_bpg.py',sha256=sha(runner)),
        bpg=dict(original_request=report['bpg_request'],freeze=report['bpg_freeze'],
            parity_completion=dict(path=OUT+'/a3_bpg_memory_v1/qualification/completion.json',sha256=sha(ACTUAL/'qualification/completion.json')),
            build_receipt=dict(path=OUT+'/a3_bpg_memory_v1/build/build_receipt.json',sha256=sha(ACTUAL/'build/build_receipt.json')),
            phy_module=dict(path=phy_remote,sha256=phy_registration['source_bindings'][phy_remote]),
            memory_module=dict(path=memory_remote,sha256=sha(memory))),
        bpg_display_protocol=dict(source_population='original_fixed16_development',noise_seed=2001,
            noise_namespace='A3_DEVELOPMENT_TIMING',save_first_measured_repetition=1,
            source_unfit_display='gray explicitly labelled no received frame',strategy_selection=False),
        neural_base_request_sha256=sha(base_path))
    destination.mkdir(parents=True)
    save(destination/'request.json',r)
    uploads={str(runner):r['bpg_runner']['path'],str(endpoint):r['bpg_endpoint']['path'],
             str(destination.resolve()/'request.json'):OUT+'/a3_timing_v1/materials_bpg/request.json'}
    status=dict(status='PREPARED_NOT_RUN',request_sha256=sha(destination/'request.json'),uploads=uploads,
        fixed_sources=16,SNRs=[7,13,19],source_attempts=192,maximum_actual_link_frames=192,
        maximum_new_paid_PHY_packets=384,new_qualification_PHY_packets=0,display_archives_planned=48,
        source_unfit_RX_and_E2E_are_NA=True,neural_request_unchanged=sha(base_path)==r['neural_base_request_sha256'],
        actual_codec_parity=dict(status='COMPLETE',sources=20,cases=40,bytes_exact=True,pixels_exact=True),
        actual_timing_frames_run_by_preparation=0,SSH_opened=False)
    save(destination/'local_preparation.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=SUP/'a3_fixed16/materials_bpg_v1')
    a=p.parse_args();print(json.dumps(prepare(a.out),indent=2))
