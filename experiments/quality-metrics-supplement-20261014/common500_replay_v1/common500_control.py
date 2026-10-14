"""Finite real-wait control for the explicitly authorized pixel replay."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/quality-metrics-supplement-20261014/common500_replay_v1'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
REG=R/'results/quality_metrics_supplement_20261014_v1/common500_replay_v1/registration.json'
RT='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010';CPY=RT+'/envs/frozen_ldpc_py311/bin/python';PY=RT+'/envs/frozen_visual_py310/bin/python'
CODE_SHA='44810f911d6d7fb463984420ba67171f27f821e6137051b4d661ccad27b7fa33'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):return dict(pid=pid,start_ticks=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))
def main(a):
    assert sha(__file__)==a.expected_entry and sha(E/'common500_replay.py')==CODE_SHA
    if a.role=='dispatch':
        assert sha(REG)==a.expected_registration and not (T/'common500_replay_v1').exists()
        for name in ('tests','register'):
            w=json.loads((T/('common500_replay_'+name+'_control_v1')/'actual_wait.json').read_text());assert w['actual_wait'] is True and type(w['returncode']) is int and w['returncode']==0
        I=T/'common500_replay_initial_control_v1';assert not I.exists() and not (T/'common500_replay_dispatch_v1').exists();I.mkdir()
        argv=[CPY,'-B',str(Path(__file__).resolve()),'supervise','--expected-entry',a.expected_entry,'--expected-registration',a.expected_registration]
        common=dict(argv=argv,parent_process=ident(os.getpid()),registration=pin(REG),entry=pin(Path(__file__)),automatic_retry=False)
        save(I/'prepared.json',common)
        with (I/'stdout.txt').open('xb') as out,(I/'stderr.txt').open('xb') as err:child=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        save(I/'launch.json',dict(common,process=ident(child.pid)));print(json.dumps(dict(supervisor=ident(child.pid),initial_launch=pin(I/'launch.json'))));return
    d=T/('common500_replay_dispatch_v1' if a.role=='supervise' else 'common500_replay_'+a.role+'_control_v1');assert not d.exists();d.mkdir()
    if a.role=='tests':argv=[CPY,'-B',str(E/'common500_replay_tests.py'),'-v'];limit=120;gpu=''
    elif a.role=='register':argv=[CPY,'-B',str(E/'common500_replay.py'),'register'];limit=600;gpu=''
    else:assert sha(REG)==a.expected_registration;argv=[PY,'-B',str(E/'common500_replay.py'),'run','--expected-registration',a.expected_registration];limit=7200;gpu='0'
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,BASELINE_GPU_ID='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',PYTHONUNBUFFERED='1')
    common=dict(argv=argv,parent_process=ident(os.getpid()),source=pin(E/'common500_replay.py'),entry=pin(Path(__file__)),max_seconds=limit,CUDA_VISIBLE_DEVICES=gpu,registration_sha256=a.expected_registration,automatic_retry=False)
    save(d/'prepared.json',common);start=time.monotonic();timed_out=terminated=killed=False
    with (d/'stdout.txt').open('xb') as out,(d/'stderr.txt').open('xb') as err:
        child=subprocess.Popen(argv,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True);process=ident(child.pid);save(d/'launch.json',dict(common,process=process))
        try:rc=child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out=terminated=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=15)
            except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=15)
    w=dict(common,process=process,actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(d/'stdout.txt'),stderr=pin(d/'stderr.txt'));save(d/'actual_wait.json',w)
    if rc:save(d/'failure.json',dict(returncode=rc,automatic_retry=False))
    print(json.dumps(dict(actual_wait=pin(d/'actual_wait.json'),returncode=rc,stdout=(d/'stdout.txt').read_text()[-1800:],stderr=(d/'stderr.txt').read_text()[-1800:])),flush=True)
    if rc:raise SystemExit(rc)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=('tests','register','dispatch','supervise'));p.add_argument('--expected-entry',required=True);p.add_argument('--expected-registration');main(p.parse_args())
