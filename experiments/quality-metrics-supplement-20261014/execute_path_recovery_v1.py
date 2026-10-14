"""One detached final16 metric supervisor, with durable real child.wait."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
E=R/'experiments/quality-metrics-supplement-20261014'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
REG=R/'results/quality_metrics_supplement_20261014_v1/hifi_fixed16/path_recovery_registration.json'
D=T/'hifi_score_path_recovery_dispatch_v1';I=T/'hifi_score_path_recovery_initial_control_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
    s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=int(s[19]))
def main(a):
    assert os.name=='posix' and sha(REG)==a.expected_registration and sha(__file__)==a.expected_entry
    if a.role=='dispatch':
        assert not D.exists() and not I.exists() and not (T/'hifi_score_path_recovery_v1').exists()
        I.mkdir();argv=[CPY,'-B',str(Path(__file__).resolve()),'supervise','--expected-registration',a.expected_registration,'--expected-entry',a.expected_entry]
        prepared=dict(argv=argv,parent_process=identity(os.getpid()),registration_sha256=a.expected_registration,entry_sha256=a.expected_entry,automatic_retry=False)
        save(I/'prepared.json',prepared)
        with (I/'stdout.txt').open('xb') as out,(I/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
        save(I/'launch.json',dict(prepared,process=identity(child.pid)));print(json.dumps(dict(supervisor=identity(child.pid),initial_launch=str(I/'launch.json'))));return
    assert not D.exists();D.mkdir();argv=[PY,'-B',str(E/'hifi_score_path_recovery_v1.py'),'run','--expected-registration',a.expected_registration]
    reg=json.loads(REG.read_text());assert sha(E/'hifi_score_path_recovery_v1.py')==reg['code']['sha256']
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',MKL_CBWR='COMPATIBLE')
    common=dict(argv=argv,parent_process=identity(os.getpid()),registration_sha256=a.expected_registration,entry_sha256=a.expected_entry,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','MKL_CBWR']},max_seconds=7200,automatic_retry=False)
    save(D/'prepared.json',common);start=time.monotonic();error=None;terminated=killed=timed_out=False
    with (D/'stdout.txt').open('xb') as out,(D/'stderr.txt').open('xb') as err:
        child=subprocess.Popen(argv,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        process=identity(child.pid);save(D/'launch.json',dict(common,process=process))
        try:code=child.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            timed_out=terminated=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
        except BaseException as exc:
            error=repr(exc);terminated=True;os.killpg(child.pid,signal.SIGTERM);code=child.wait(timeout=15)
    value=dict(common,process=process,actual_wait=True,returncode=code,elapsed_seconds=time.monotonic()-start,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=error,
        stdout=dict(path=str(D/'stdout.txt'),bytes=(D/'stdout.txt').stat().st_size,sha256=sha(D/'stdout.txt')),stderr=dict(path=str(D/'stderr.txt'),bytes=(D/'stderr.txt').stat().st_size,sha256=sha(D/'stderr.txt')))
    save(D/'actual_wait.json',value)
    if type(code) is not int or code!=0 or error or timed_out:
        save(D/'failure.json',dict(returncode=code,automatic_retry=False));raise RuntimeError('Metric child stopped; preserve and diagnose')
    save(D/'completion.json',dict(status='ACTUAL_FIXED16_METRIC_CHILD_EXIT_ZERO',actual_wait_sha256=sha(D/'actual_wait.json'),parent_must_be_gone_before_finalize=True))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('role',choices=('dispatch','supervise'));a.add_argument('--expected-registration',required=True);a.add_argument('--expected-entry',required=True)
    main(a.parse_args())
