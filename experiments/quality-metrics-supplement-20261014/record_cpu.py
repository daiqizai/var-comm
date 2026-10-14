"""Bounded, one-attempt CPU metadata and byte-restoration steps."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/quality-metrics-supplement-20261014'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
    a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=int(a[19]))
def main(a):
    assert sha(__file__)==a.expected_entry
    names={'tests':'test_hifi_score.py','materialize':'model_assets.py','register':'hifi_score.py','finalize':'hifi_score.py'}
    code=E/names[a.role];assert sha(code)==a.expected_child
    d=T/('cpu_'+a.role+'_v1');assert not d.exists();d.mkdir()
    argv=[CPY,'-B',str(code)]
    if a.role=='materialize':argv+=['materialize','--archive-sha256',a.archive_sha256]
    elif a.role=='register':argv+=['register']
    elif a.role=='finalize':argv+=['finalize','--expected-registration',a.expected_registration,'--actual-wait',str(T/'hifi_score_dispatch_v1/actual_wait.json'),'--actual-wait-sha256',a.actual_wait_sha256]
    else:argv+=['-v']
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    common=dict(argv=argv,environment=dict(CUDA_VISIBLE_DEVICES=''),parent_process=identity(os.getpid()),code=pin(code),entry=pin(Path(__file__)),max_seconds=900,automatic_retry=False)
    save(d/'prepared.json',common);start=time.monotonic();timed_out=terminated=killed=False;error=None
    with (d/'stdout.txt').open('xb') as out,(d/'stderr.txt').open('xb') as err:
        child=subprocess.Popen(argv,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        process=identity(child.pid);save(d/'launch.json',dict(common,process=process))
        try:rc=child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out=terminated=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=15)
            except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=15)
    wait=dict(common,process=process,actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=error,stdout=pin(d/'stdout.txt'),stderr=pin(d/'stderr.txt'))
    save(d/'actual_wait.json',wait)
    if rc!=0:save(d/'failure.json',dict(returncode=rc,automatic_retry=False))
    print(json.dumps(dict(returncode=rc,actual_wait=pin(d/'actual_wait.json'),stdout=(d/'stdout.txt').read_text()[:1200],stderr=(d/'stderr.txt').read_text()[-1800:]),indent=2),flush=True)
    if rc!=0:raise SystemExit(rc)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('role',choices=('tests','materialize','register','finalize'));a.add_argument('--expected-entry',required=True);a.add_argument('--expected-child',required=True);a.add_argument('--archive-sha256');a.add_argument('--expected-registration');a.add_argument('--actual-wait-sha256');main(a.parse_args())
