"""One-shot B handoff after verified calibration exit. Never resumes old queues."""
import os,json,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=Path(__file__).parent;R=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929/revision_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=p.joinpath('stat').read_text().split(') ')[1].split()
    return dict(pid=pid,start_ticks=s[19],state=s[0],cmdline=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
def main():
    lock=(R/'B_handoff.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pin=read(R/'B_handoff_registration.json')
    for p,d in pin['bindings'].items():assert sha(p)==d,p
    while True:
        for f in ['B_calibration_failure.json','B_calibration_supervisor_failure.json']:
            assert not (R/f).exists(),f
        current=identity(pin['calibration_process']['pid'])
        if current is not None:assert current['start_ticks']==pin['calibration_process']['start_ticks'] and current['cmdline']==pin['calibration_process']['cmdline']
        completed=(R/'B_calibration_completion.json').exists() and (R/'B_calibration_supervisor_completion.json').exists()
        if completed and (current is None or current['state']=='Z'):break
        if current is None and not completed:raise RuntimeError('calibration stopped before successful completion')
        write(R/'B_handoff_status.json',dict(status='WAITING_FOR_CALIBRATION_COMPLETE_AND_EXIT',time=time.time(),pid=os.getpid()))
        time.sleep(10)
    for p,d in pin['bindings'].items():assert sha(p)==d,p
    cal=read(R/'B_calibration_completion.json');frozen=read(R/'B_frozen_config.json')
    assert cal['status']=='REAL_V3_B_CALIBRATION_COMPLETE' and cal['frozen_config_sha256']==sha(R/'B_frozen_config.json')
    assert frozen['development_read'] is False and read(R/'limit_check.json')['passed']
    assert read(R/'A_completion.json')['source_A_approximate_check_passed']
    bindings=dict(read(R/'B_calibration_registration.json')['bindings'])
    for p in [E/'rx_v3_evaluate.py',E/'rx_v3_b_helpers.py',R/'B_frozen_config.json',R/'B_calibration_completion.json',R/'B_selection.json',R/'quality_identity.json',R/'A_population.json']:
        bindings[str(p)]=sha(p)
    for p,d in bindings.items():assert sha(p)==d,p
    reg=R/'B_evaluation_registration.json';assert not reg.exists() and not (R/'B_evaluation_launch_receipt.json').exists()
    write(reg,dict(stage='B_evaluation',bindings=bindings,frozen_before_development=True,time=time.time(),original_training_remains_stopped=True))
    cmd=['python3','-u',str(E/'run_v3_stage.py'),'--stage','B_evaluation']
    with (R/'B_evaluation_console.log').open('ab') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.2);proc=identity(p.pid);assert proc and os.getsid(p.pid)==p.pid
    write(R/'B_evaluation_launch_receipt.json',dict(time=time.time(),process=proc,session_id=p.pid,command=cmd,registration_sha256=sha(reg)))
    write(R/'B_handoff_completion.json',dict(status='B_EVALUATION_LAUNCHED_AFTER_CALIBRATION_EXIT',process=proc,time=time.time(),original_training_remains_stopped=True))
if __name__=='__main__':
    try:main()
    except Exception:write(R/'B_handoff_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
