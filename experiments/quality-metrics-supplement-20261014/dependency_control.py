from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/quality-metrics-supplement-20261014'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
RT='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def ident(pid):return dict(pid=pid,start_ticks=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
p=argparse.ArgumentParser();p.add_argument('role',choices=('setup','qualify'));p.add_argument('--expected-child',required=True);a=p.parse_args()
code=E/('materialize_dependencies.py' if a.role=='setup' else 'import_dependency_qualification.py');assert sha(code)==a.expected_child
py=RT+('/envs/frozen_ldpc_py311/bin/python' if a.role=='setup' else '/envs/frozen_visual_py310/bin/python')
d=T/('dependency_'+a.role+'_control_v1');assert not d.exists();d.mkdir();argv=[py,'-B',str(code)];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
common=dict(argv=argv,parent_process=ident(os.getpid()),child_code=pin(code),max_seconds=300,CUDA_VISIBLE_DEVICES='',no_model_construction_or_score=True)
save(d/'prepared.json',common);start=time.monotonic();terminated=killed=timed_out=False
with (d/'stdout.txt').open('xb') as out,(d/'stderr.txt').open('xb') as err:
    child=subprocess.Popen(argv,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True);process=ident(child.pid);save(d/'launch.json',dict(common,process=process))
    try:rc=child.wait(timeout=300)
    except subprocess.TimeoutExpired:
        terminated=timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:rc=child.wait(timeout=15)
        except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=15)
save(d/'actual_wait.json',dict(common,process=process,actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(d/'stdout.txt'),stderr=pin(d/'stderr.txt')))
print(json.dumps(dict(returncode=rc,actual_wait=pin(d/'actual_wait.json'),stdout=(d/'stdout.txt').read_text()[-2200:],stderr=(d/'stderr.txt').read_text()[-2500:])),flush=True)
if rc:save(d/'failure.json',dict(returncode=rc,automatic_retry=False));raise SystemExit(rc)
