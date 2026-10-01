"""One detached, locked owner; M1 delivery strictly precedes all M2 experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
def read(p):return json.loads(Path(p).read_text())
def write(p,j):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_qualification(publication,dependencies):
    receipt_path=OUT/'qualification.json';receipt=read(receipt_path)
    if receipt.get('status')!='REAL_WEIGHT_QUALIFICATION_PASS' or receipt.get('training_updates')!=0 or receipt.get('development_read') is not False:
        raise RuntimeError('Real calibration-only qualification not passed')
    report_path=Path(receipt['qualification_path'])
    if not report_path.is_absolute():report_path=ROOT/report_path
    if sha(report_path)!=receipt.get('qualification_sha256'):raise RuntimeError('Qualification report SHA changed')
    report=read(report_path)
    if report.get('status')!='REAL_WEIGHT_QUALIFICATION_PASS' or report.get('training_updates')!=0 or report.get('development_read') is not False or report.get('calibration_sources')!=4:
        raise RuntimeError('Wrong qualification scope or status')
    qualified=report.get('qualification_source_bindings')
    if not isinstance(qualified,dict) or not qualified or qualified!=receipt.get('source_bindings'):
        raise RuntimeError('Qualification receipt/report source bindings differ')
    final=dict(dependencies)
    for path,h in publication['source_bindings'].items():
        if path in final and final[path]!=h:raise RuntimeError('Final source/dependency bindings disagree')
        final[path]=h
        if qualified.get(path)!=h:raise RuntimeError('Final execution source was not qualified: '+path)
    for path,h in qualified.items():
        if final.get(path)!=h or sha(path)!=h:raise RuntimeError('Qualified input differs from final source/dependency bindings: '+path)
    return {str(receipt_path):sha(receipt_path),str(report_path):sha(report_path)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pub=read(OUT/'source_publication.json')
    if pub['status']!='PUSHED' or pub['commit']!=pub['remote_commit']:raise RuntimeError('Checked source must be pushed first')
    bindings=pub['source_bindings'];stages=read(HERE/'stages.json')
    dependencies=read(OUT/'dependency_bindings.json')
    qualification_inputs=verify_qualification(pub,dependencies)
    engineering_path=OUT/'engineering_probe.json';engineering=read(engineering_path)
    if engineering.get('status')!='REAL_RUNNER_ENGINEERING_PASS' or engineering.get('frames')!=925 or engineering.get('online_cached_tokens_equal') is not True or engineering.get('scientific_result') is not False or engineering.get('development_read') is not False:
        raise RuntimeError('Final runner/online-source engineering qualification missing')
    if engineering.get('source_bindings')!=read(OUT/'qualification.json')['source_bindings']:
        raise RuntimeError('Runner probe and model qualification did not test identical final sources')
    qualification_inputs[str(engineering_path)]=sha(engineering_path)
    dependency_receipt_sha=sha(OUT/'dependency_bindings.json')
    def verify():
        if sha(OUT/'dependency_bindings.json')!=dependency_receipt_sha:raise RuntimeError('Dependency binding receipt changed')
        for path,h in qualification_inputs.items():
            if sha(path)!=h:raise RuntimeError('Qualification receipt/report changed: '+path)
        for path,h in bindings.items():
            if sha(path)!=h:raise RuntimeError('Active source changed: '+path)
        for path,h in dependencies.items():
            if sha(path)!=h:raise RuntimeError('Frozen dependency changed: '+path)
    def status(**kw):write(OUT/'supervisor_status.json',dict(pid=os.getpid(),time=time.time(),**kw))
    write(OUT/'supervisor_registration.json',dict(bindings=bindings,dependencies=dependencies,stages=stages,
        qualification_inputs=qualification_inputs,source_commit=pub['commit']))
    try:
        for st in stages:
            verify();rp=ROOT/st['receipt']
            if rp.exists() and read(rp).get('status') in st['statuses']:continue
            attempt=0
            while True:
                verify();attempt+=1;log=OUT/f'{st["name"]}_{time.time_ns()}.log'
                with log.open('a') as f:
                    worker=subprocess.Popen([sys.executable,'-u',str(HERE/st['script']),*st.get('args',[])],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                    status(status='RUNNING',stage=st['name'],worker_pid=worker.pid,log=str(log),attempt=attempt)
                    code=worker.wait()
                if code==75:
                    status(status='WAITING_FOR_SAFE_RESOURCE',stage=st['name'],log=str(log));time.sleep(30);continue
                if code!=0 or not rp.exists() or read(rp).get('status') not in st['statuses']:
                    raise RuntimeError(f'{st["name"]} failed, exit={code}, log={log}')
                break
        verify();status(status='COMPLETE',stopped_after_authorized_scope=True)
    except Exception as e:
        status(status='FAILED',error=str(e));raise

if __name__=='__main__':main()
