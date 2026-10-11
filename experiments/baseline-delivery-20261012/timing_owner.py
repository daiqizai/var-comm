"""Once-only fixed16 timing owner, with a strict pre-model resource gate.

prepare/readiness do not import a model, construct a decoder, or consume a case.
run returns WAIT when occupied. An admitted attempt has no automatic retry.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from timing_contract import FIXED16, SNRS, METHODS, fixed_cases, CaseReservations, exclusive_snapshot

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/baseline_strength_20261011'
OUT=BASE/'timing/six_method_v1'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
RUN=TMP/'six_method_timing_v1'
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
PY=RT/'envs/frozen_visual_py310/bin/python'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not (OUT/'registration.json').exists(), 'Already registered'
    old_path=TMP/'timing_original_t4_request.json';old=read(old_path)
    assert sha(old_path)==read(TMP/'timing_original_t4_completion.json')['request_sha256']=='548bec34f1ef532db53389bc3edfadac6a3689f5d13010751c9d4e0c443d00c1'
    assert old['source_indices']==list(FIXED16) and old['snrs']==list(SNRS)
    assert old['warmup_per_source_snr']==old['measured_per_source_snr']==3
    dev=TMP/'timing_original_development_plan.py'
    assert sha(dev)==old['legacy']['bindings'][old['legacy']['development_plan_module']]
    neural_path=ROOT/'results/wcl_evidence_closure_20261009/T4_resources_cost/external_timing_reuse_v1/original_receipts/neural_request.json'
    assert read(neural_path)['timing_protocol']['P_Swin_seed']==2001
    phy_path=RT/'qualification/h800_ep_new100_phy_v1_attempt1/registered/request.json'
    visual_path=RT/'qualification/h800_ep_new100_visual_v3_attempt1/registered/request.json'
    phy=read(phy_path);policies={m:{} for m in METHODS[:4]}
    for frame in phy['frames']:
        dest=policies[frame['method']];key=str(frame['snr_db']);p=frame['policy']
        if key in dest:assert dest[key]==p
        else:dest[key]=p
    for method in METHODS[:2]:
        for snr in SNRS:
            prior=old['raw_points'][method.replace('RAW_','RAW64_')][str(snr)]
            assert all(prior[k]==v for k,v in policies[method][str(snr)].items())
            policies[method][str(snr)]=prior
    sources=read(BASE/'timing/fixed16_source_bindings_20261012.json')
    assert sources['source_indices']==list(FIXED16)
    chain_path=BASE/'automatic_followthrough_v1/registration.json';chain=read(chain_path)
    inputs={str(p):sha(p) for p in [old_path,TMP/'timing_original_t4_completion.json',dev,
                                   neural_path,phy_path,visual_path,chain_path,
                                   BASE/'timing/fixed16_source_bindings_20261012.json']}
    for row in sources['records']:
        for key in ['archive','checkpoint']:
            item=row[key];assert sha(item['path'])==item['sha256'];inputs[item['path']]=item['sha256']
    for branch in ['P','Swin']:
        cp=chain['selections'][branch]['checkpoint'];assert sha(cp['path'])==cp['sha256'];inputs[cp['path']]=cp['sha256']
    code={str(HERE/n):sha(HERE/n) for n in ['timing_owner.py','timing_contract.py','timing_endpoints.py','timing_runtime.py']}
    code.update(chain['code_bindings'])
    for name,digest in read(visual_path)['tool_bindings'].items():
        path=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'/name
        assert sha(path)==digest;code[str(path)]=digest
    for p,h in code.items():assert sha(p)==h
    private=TMP/'timing_private_sionna_v1'
    for name in ['binding.json','dependency_binding.json']:
        p=private/name;inputs[str(p)]=sha(p)
        for row in read(p)['entries'].values():
            assert sha(row['copy'])==row['sha256'];inputs[row['copy']]=row['sha256']
    assert read(private/'import_probe_with_dependencies.json')['exit_code']==0
    native_source=ROOT/'src/var_comm/token_trellis.cpp'
    native_dir=TMP/'native_build/outputs/build'/sha(native_source)[:16]
    native_header=dict(source=pin(native_source),descriptor=pin(native_dir/'build.json'),library=pin(native_dir/'token_trellis.so'))
    for item in native_header.values():inputs[item['path']]=item['sha256']
    assert read(native_dir/'build.json')['library_sha256']==native_header['library']['sha256']
    reg=dict(native_header=native_header,schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V1',status='REGISTERED_NOT_EXECUTED',
        input_sha256=inputs,code_sha256=code,policies=policies,records=sources['records'],
        visual_request=pin(visual_path),phy_request=pin(phy_path),development_module=pin(dev),
        P_checkpoint=chain['selections']['P']['checkpoint'],Swin_checkpoint=chain['selections']['Swin']['checkpoint'],
        methods=list(METHODS),SNRs=list(SNRS),fixed_indices=list(FIXED16),warmups=3,repeats=3,
        total_frames=1728,measured_frames=864,digital_packet_cap=2304,batch_size=1,
        CPU_slots=[25,29,31,35,39,43],threads=6,interop_threads=2,max_seconds=14400,
        raw_noise='original development namespace, seed6201 and original development_slot counter',
        entropy_noise='original T1 full-frame source seed6201 and original SNR/index counter',
        continuous_noise_seed=2001,precision='FP32, TF32 disabled, deterministic',
        original_environments_modified=False,automatic_retries=False,
        qualification='Pending actual fixed warmup repeatability; historical PHY identity is not new timing completion',
        model_or_packet_calls_during_preparation=0)
    write(OUT/'registration.json',reg)
    print('REGISTERED_NO_MODEL_OR_PACKET_CALLS')


def verify():
    reg=read(OUT/'registration.json')
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items():assert sha(p)==h,p
    return reg


def readiness():
    assert not (RUN/'failure.json').exists(), 'Existing failure requires manual audit'
    snapshot=exclusive_snapshot(ROOT)
    write(OUT/'resource_wait.json',dict(status='RESOURCE_READY_NOT_STARTED' if snapshot['admitted'] else 'WAIT_EXCLUSIVE_WINDOW',**snapshot))
    return snapshot


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(), 'Existing attempt: do not retry'
    if not readiness()['admitted']:
        print('WAIT_EXCLUSIVE_WINDOW_NO_SCIENTIFIC_CALLS');return
    reg=verify();RUN.mkdir(exist_ok=False)
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
                  argv=Path('/proc/self/cmdline').read_bytes().decode().split('\0')[:-1],wait_only=True)
    write(OUT/'launch.json',dict(owner=identity,registration_sha256=sha(OUT/'registration.json'),started_unix=time.time()))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',
        MPLCONFIGDIR=str(TMP/'matplotlib'),XDG_CACHE_HOME=str(TMP/'cache'),TMPDIR=str(RUN))
    with (RUN/'worker.log').open('xb') as log:
        child=subprocess.Popen([str(PY),str(Path(__file__)),'_worker'],cwd=ROOT,env=env,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'worker_launch.json',dict(pid=child.pid,owner_pid=os.getpid()))
        try:code=child.wait(timeout=reg['max_seconds']+1800)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
            write(OUT/'actual_wait.json',dict(actual_child_waited=True,exit_code=code,timed_out=True));raise
    write(OUT/'actual_wait.json',dict(actual_child_waited=True,exit_code=code,timed_out=False))
    assert code==0,'Timing stopped; preserve failure and do not repeat completed calls'
    assert read(OUT/'completion.json')['status']=='COMPLETE_FIXED16_SIX_METHOD_TIMING'


def worker():
    import numpy as np
    import timing_runtime as runtime
    import timing_endpoints as endpoints
    reg=verify();owner=read(OUT/'launch.json')['owner']
    assert owner['pid']==os.getppid()
    assert Path(f"/proc/{owner['pid']}/stat").read_text().split()[21]==owner['start_ticks']
    assert Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().split('\0')[:-1]==owner['argv']
    observer=dict(pid=owner['pid'],identity_verified=True,wait_only=True)
    assert set(reg['CPU_slots']).issubset(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(reg['CPU_slots']));os.nice(10)
    started=time.monotonic();stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def fast_boundary():
        assert not stopped and os.getppid()==owner['pid']
        assert time.monotonic()-started < reg['max_seconds']
    def boundary():
        fast_boundary()
        assert not (OUT/'STOP').exists() and not (ROOT/'STOP').exists()
    boundary();assert exclusive_snapshot(ROOT,observer)['admitted']
    reservations=CaseReservations(RUN/'case_reservations.json')
    system=runtime.build(reg,reservations,RUN,fast_boundary)
    records={row['source_index']:row for row in reg['records']};pixels={}
    for i,row in records.items():
        with np.load(row['archive']['path'],allow_pickle=False) as z:a=z['pixels'].copy()
        assert a.dtype==np.uint8 and a.shape==(3,256,256)
        assert hashlib.sha256(a.tobytes()).hexdigest()==row['preprocessing_id'];pixels[i]=a
    first={};results=[]
    for case in fixed_cases():
        boundary();snapshot=exclusive_snapshot(ROOT,observer)
        assert snapshot['admitted'],'Exclusive window lost; no automatic replay'
        write(OUT/'running_status.json',dict(status='TIMING',case=case,completed=len(results),resources=snapshot))
        for p,h in reg['code_sha256'].items():assert sha(p)==h
        i=case['source_index'];s=case['snr_db'];method=case['method'];key=(method,i,s)
        system.backend.source_index=i;system.charges.events=[]
        counter=(system.development.frame_counter(reg['policies'][method][str(s)]['development_slot'],i,6201)
                 if method.startswith('RAW_') else ([1,4,7,10,13,19].index(s)*1000+i)*3)
        identity,timings=endpoints.fresh_case(system.endpoints[method],pixels[i],case,
                                             records[i]['source_id'],counter,reservations)
        if 'fresh_tokens_sha256' in identity:
            assert identity['fresh_tokens_sha256']==records[i]['tokens_sha256'],'Fresh tokenizer differs from frozen source tokens'
        if method=='P100000':assert abs(identity['actual_energy']-2048)<.02
        if key in first:assert first[key]==identity,'Warmup/repeat wave, RX or output identity differs'
        else:first[key]=identity
        results.append(dict(case=case,identity=identity,timings=timings))
    system.backend.close();system.resolver.reverify();system.environment.reverify()
    for p,h in system.audit_hashes.items():assert sha(p)==h
    verify();boundary();assert exclusive_snapshot(ROOT,observer)['admitted']
    assert reservations.state['status']=='COMPLETE_ALL_FIXED_TIMING_CASES' and len(results)==1728
    columns=['method','source_index','snr_db','repeat','phase','TX_seconds','RX_seconds',
             'channel_seconds','software_e2e_excluding_channel_seconds','receiver_status','header_accepted']
    with (OUT/'timing_per_call.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for row in results:
            c=row['case'];v=row['timings'];a=row['identity']
            w.writerow(dict(**c,TX_seconds=v['tx_seconds'],RX_seconds=v['rx_seconds'],
                channel_seconds=v['channel_seconds'],software_e2e_excluding_channel_seconds=v['outer_seconds']-v['channel_seconds'],
                receiver_status=a['status'],header_accepted=a['header_accepted']))
    summaries=[]
    for method in METHODS:
        for snr in SNRS:
            selected=[r for r in results if r['case']['method']==method and r['case']['snr_db']==snr and r['case']['phase']=='measurement']
            assert len(selected)==48
            for component in ['tx_seconds','rx_seconds','software_e2e_excluding_channel_seconds']:
                values=np.array([r['timings'][component] if component in r['timings'] else r['timings']['outer_seconds']-r['timings']['channel_seconds'] for r in selected])
                summaries.append(dict(method=method,snr_db=snr,component=component,source_count=16,repeats=3,
                    calls=48,mean_seconds=float(values.mean()),median_seconds=float(np.median(values)),
                    p95_seconds=float(np.quantile(values,.95)),min_seconds=float(values.min()),max_seconds=float(values.max())))
    with (OUT/'timing_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    write(OUT/'completion.json',dict(status='COMPLETE_FIXED16_SIX_METHOD_TIMING',frames=1728,measured_frames=864,
        packet_calls=reservations.state['packet_decodes'],source_calls=system.source_calls.counts,
        original_intervals_recomputed=0,training_updates=0,case_journal=pin(RUN/'case_reservations.json'),
        output_sha256={p.name:sha(p) for p in OUT.glob('*.csv')},qualification='Three fixed fresh warmups plus three identity-matched measured repetitions per source/SNR/method',
        cross_host_bitwise_equivalence_claimed=False,airtime_measured=False,energy_consumption_measured=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','readiness','run','_worker'])
    args=parser.parse_args()
    try:
        {'prepare':prepare,'readiness':readiness,'run':run,'_worker':worker}[args.command]()
    except BaseException:
        if args.command in ('run','_worker') and (OUT/'launch.json').exists():
            write(RUN/('worker_failure.json' if args.command=='_worker' else 'failure.json'),
                  dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
