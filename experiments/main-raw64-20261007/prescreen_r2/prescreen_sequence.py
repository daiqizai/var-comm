"""Bounded tests -> read-only waiter -> prescreen, with waited parent exit."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

N=Path('/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007')
O=N.parent/'CONTENT-REAL-64QAM-20261006'
D=N/'prescreen_sequence_r2'
P=N/'prescreen_runtime_r2'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(s[19]),uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def verify(pins):
    for p,h in pins.items():assert sha(p)==h,p

def recovery_gate(inputs):
    r=N/'prescreen_identity_recovery_r2'
    for name in ('failure_archive_receipt.json','diagnosis.json'):
        p=str(r/name);assert p in inputs and sha(p)==inputs[p]
    archive=read(r/'failure_archive_receipt.json');diag=read(r/'diagnosis.json')
    assert diag['status']=='PRESCREEN_PREWORKER_IDENTITY_SCHEMA_FAILURE_DIAGNOSED'
    assert diag['archive_sha256']==sha(r/'failure_archive_receipt.json')
    assert diag['new_packet_calls']==0 and diag['prescreen_worker_launched'] is False
    assert archive['originals_retained'] is True and archive['original_owner_success'] is False
    for row in archive['items']:
        assert sha(row['original'])==sha(row['archive'])==row['sha256']
    for pid in archive['processes_exited']:assert not Path('/proc',str(pid)).exists()
def call(name,argv,env):
    identity=None;error=None;log=D/(name+'.log')
    with log.open('xb') as f:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=P)
        try:
            save(D/(name+'_pending.json'),dict(pid=child.pid,argv=argv))
            until=time.monotonic()+10
            while True:
                identity=ident(child.pid)
                if identity['argv']==argv:break
                assert child.poll() is None and time.monotonic()<until,'child admission'
                time.sleep(.01)
            save(D/(name+'_launch.json'),dict(identity=identity,expected_argv=argv))
        except BaseException:error=traceback.format_exc()
        finally:rc=child.wait()
    save(D/(name+'_exit.json'),dict(identity=identity,expected_argv=argv,exit_code=rc,process_waited=True,log=str(log),log_sha256=sha(log),capture_error=error))
    assert error is None and rc==0,(name,error,rc)

def main():
    D.mkdir(exist_ok=False);inputs=read(N/'prescreen_inputs_r2.json');verify(inputs);recovery_gate(inputs)
    save(D/'registration.json',dict(status='READ_ONLY_SEQUENCE_BOUND',source_bindings=inputs,
        scope=['16_CPU_tests','actual_original_normal_identity_contract','actual_clean_scalar_contract','wait_original_CPU_normal_exit','registered_six_SNR_shortlist'],new_PHY=0,GPU=False,automatic_calibration=False))
    save(D/'identity.json',ident(os.getpid()))
    os.sched_setaffinity(0,{28,29});os.setpriority(os.PRIO_PROCESS,0,15)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    call('qualification',[sys.executable,'-B','-m','unittest','-v','test_raw64_prescreen','test_raw64_selection'],env)
    sys.path.insert(0,str(P));import raw64_prescreen as stage
    actual_exit=read(N/'cpu_sequence_v1/owner_exit.json')
    actual_done=read(N/'cpu_execution_v1/completion.json')
    assert stage.same_process(actual_exit['identity'],actual_done['owner_identity'])
    assert actual_exit['exit_code']==0 and actual_exit['process_waited'] is True
    assert sha(actual_exit['log'])==actual_exit['log_sha256']
    save(D/'actual_identity_qualification.json',dict(status='ORIGINAL_NORMAL_CPU_IDENTITY_FIELDS_VERIFIED',
        owner_exit_sha256=sha(N/'cpu_sequence_v1/owner_exit.json'),owner_completion_sha256=sha(N/'cpu_execution_v1/completion.json'),
        compared_fields=['pid','start_ticks','uid','argv'],original_observation_fields_preserved=True))
    old_config=O/'MAIN/prescreen_execution_r1/config.json'
    paths=[N/'SCIENCE_REGISTRATION_V1.json',N/'assemble_v1/completion.json',N/'prepared_v1/cpu_plan_v1.json',
        N/'cpu_execution_v1/registration.json',N/'cpu_execution_v1/config.json',
        old_config,O/'MAIN/prescreen_execution_r1/execution_registration.json',O/'MAIN/prescreen_execution_r1/completion.json',
        O/'H/source200/completion.json',O/'H/clean-quality200/completion.json',
        N/'clean_missing_v1/completion.json',N/'clean_missing_execution_v1/config.json',N/'clean_missing_execution_v1/registration.json',
        N/'clean_sequence_v1/completion.json',N/'clean_sequence_v1/worker_exit.json',
        D/'qualification_exit.json',D/'qualification.log',D/'actual_identity_qualification.json']
    oldreg=read(O/'MAIN/prescreen_execution_r1/execution_registration.json')
    assert oldreg['config_sha256']==sha(old_config)
    request=dict(new_root=str(N),old_root=str(O),old_clean_compatibility_config=str(old_config),
        out=str(N/'prescreen_r2'),affinity=[28,29],deadline_unix=1791651239.4938014,
        bindings={**inputs,**{str(p):sha(p) for p in paths}},scientific_selection_rule='ROOT_SCIENCE_REGISTRATION_V1',
        new_PHY_calls=0,GPU=False,automatic_calibration=False)
    # Actual archived scalar input schemas/identities are checked now, without
    # computing proxy quality or selecting an action while the table is partial.
    ids,gray,clean=stage.clean_data(request)
    assert len(ids)==len(gray)==len(clean)==200 and all(len(x)==396 for x in clean.values())
    save(D/'actual_clean_qualification.json',dict(status='ACTUAL_CLEAN_SCALAR_CONTRACT_PASS',source_count=200,
        states_per_source=396,source_images_read=False,proxy_ranked=False,PHY=0,GPU=False))
    request['bindings'][str(D/'actual_clean_qualification.json')]=sha(D/'actual_clean_qualification.json')
    path=P/'request.json';save(path,request)
    call('owner',[sys.executable,'-B',str(P/'raw64_prescreen.py'),'--request',str(path)],env)
    done=N/'prescreen_r2/completion.json'
    assert read(done)['status']=='RAW64_PRESCREEN_NORMAL_COMPLETE'
    verify(inputs)
    save(D/'completion.json',dict(status='RAW64_PRESCREEN_SEQUENCE_NORMAL_COMPLETE',original_owner_success=True,
        owner_completion_sha256=sha(done),owner_exit_sha256=sha(D/'owner_exit.json'),automatic_calibration=False,new_PHY=0,GPU=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if D.exists() and not (D/'failure.json').exists():save(D/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()))
        raise
