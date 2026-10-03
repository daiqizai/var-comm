"""Wait for the authorized external computation, render and normally publish."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R=Path('/home/liulu/projects/VAR_COMM')
X=R/'outputs/EXTERNAL-COMPARISON-20261004'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def verify(bindings):
    if not isinstance(bindings,dict):raise RuntimeError('Missing delivery bindings')
    for p,h in bindings.items():
        if sha(p)!=h:raise RuntimeError('Delivery input changed: '+p)

def controller_complete(done,registration):
    if (done.get('status')!='EXTERNAL_TRAINING_AND_EVALUATION_COMPLETE'
            or done.get('config_sha256')!=registration['controller_config_sha256']
            or set(done.get('stages',{}))!={'qualification','hifi_preflight','training','hifi_qualification','reconstruct','score'}):
        raise RuntimeError('Full external pipeline has not completed')
    if sha(X/'controller_config.json')!=registration['controller_config_sha256']:
        raise RuntimeError('Controller configuration changed')
    for proof in done['stages'].values():
        verify({proof['path']:proof['sha256']})
        if read(proof['path']).get('status')!=proof['status']:raise RuntimeError('Controller stage completion differs')

def run():
    import fcntl
    out=X/'delivery';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registration=read(out/'registration.json')
    verify(registration['bindings'])
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    while not (X/'controller/completion.json').exists():
        if stop[0]:write(out/'status.json',dict(status='PAUSED'));return 75
        if (X/'controller/failure.json').exists():raise RuntimeError('Upstream stopped for review; no automatic retry')
        status=read(X/'controller/status.json')
        if status['config_sha256']!=registration['controller_config_sha256']:raise RuntimeError('Controller identity changed')
        if status['status']=='PAUSED':write(out/'status.json',dict(status='UPSTREAM_PAUSED'));return 75
        write(out/'status.json',dict(status='WAITING_FOR_REGISTERED_EXTERNAL_EVALUATION',upstream=status['status'],time=time.time()))
        time.sleep(30)
    done=read(X/'controller/completion.json')
    controller_complete(done,registration)
    verify(registration['bindings'])
    if stop[0]:return 75
    for phase,command in [('report',registration['report_command']),('publish',registration['publish_command'])]:
        if stop[0]:return 75
        verify(registration['bindings'])
        write(out/'status.json',dict(status='RUNNING',phase=phase,time=time.time()))
        with (out/(phase+'.log')).open('ab') as log:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            proc=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            # CPU report/publication finish atomically before honoring a stop;
            # never leave an unknown Git transaction by killing this child.
            result=proc.wait()
        if result:raise RuntimeError(phase+' failed; review required')
    pub=read(X/'results_publication.json')
    if (pub.get('status')!='PUSHED' or pub.get('phase')!='results' or pub.get('checks')!='PASS'
            or pub.get('full_plan_complete') is not False or pub['commit']!=pub['remote_commit']):
        raise RuntimeError('Publication not verified')
    verify(pub['inputs']);verify(pub['published_files'])
    report=read(X/'evaluation/report_completion.json')
    if (report.get('status')!='EXTERNAL_TWO_METHOD_REPORT_COMPLETE' or report.get('full_plan_complete') is not False
            or report.get('missing_matched_controls')!=['P','D_U','M1']):
        raise RuntimeError('Two-method report identity differs')
    receipt=dict(status='EXTERNAL_BASELINES_REPORTED_AND_PUSHED',commit=pub['commit'],full_plan_complete=False,
        pending='Integrate matched P/D_U/M1 controls and independently registered N2048 controls',time=time.time(),
        input_bindings={**registration['bindings'],str(X/'controller/completion.json'):sha(X/'controller/completion.json'),
            str(X/'evaluation/report_completion.json'):sha(X/'evaluation/report_completion.json'),
            str(X/'results_publication.json'):sha(X/'results_publication.json')})
    write(out/'completion.json',receipt);write(out/'status.json',receipt);return 0

if __name__=='__main__':
    try:sys.exit(run())
    except Exception as e:
        write(X/'delivery/failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(e),time=time.time()));raise
