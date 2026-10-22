"""Only two proved-never-launched jobs; original worker/math/budget remain frozen."""
import pathlib,sys
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');sys.path.insert(0,str(R/'experiments/progressive-rd-scaleup-20261022'))
from rdscale_common import *
from rdscale_resource_v2 import admitted
import subprocess,signal,concurrent.futures,argparse,traceback
NAMES=('multiprefix_full_anchor_s2026102201','multiprefix_unconditional_s2026102202')
def eligible(name,reg):
 assert name in NAMES and name in reg['jobs']
 d=L/'dispatch_v1'/name
 assert (d/'failure.json').exists() and not(d/'launch.json').exists()and not(d/'actual_wait.json').exists()
 failure=read(d/'failure.json');assert "'97'"in failure['error']and 'admitted(int(fields[1]),int(fields[2]))'in failure['traceback']
 assert not(L/'training_v1'/name).exists()
 return pin(d/'failure.json')
def main(regsha,authsha):
 assert sha(O/'registration.json')==regsha=='86119c506d6c82d43d75028ac5810cf12fea372488ff1d33b23bfe8c368ce4ac'
 authpath=O/'unlaunched_training_recovery_preparation_v1.json';assert sha(authpath)==authsha;auth=read(authpath);reg=read(O/'registration.json')
 for d in reg['sources']:checked(d)
 for d in auth['sources']:checked(d)
 assert auth['jobs']==list(NAMES)and auth['extra_scientific_budget']==0
 for name in NAMES:assert eligible(name,reg)==auth['failures'][name]
 out=L/'unlaunched_training_recovery_v1';out.mkdir(exist_ok=False);active={};stop=[False]
 def stop_owned(*_):
  stop[0]=True
  for pid,ident in list(active.items()):
   cur=process(pid)
   if cur and cur['start_ticks']==ident['start_ticks']:
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
 signal.signal(signal.SIGTERM,stop_owned);signal.signal(signal.SIGINT,stop_owned)
 def run(name):
  d=out/name;d.mkdir(exist_ok=False);p=None;ident=None
  try:
   resource=reg['jobs'][name]['resource'];admissions=[];started=time.perf_counter()
   while True:
    assert not stop[0]and time.time()+reg['training_walltime_seconds']+180<reg['absolute_deadline_unix']
    f=[x.strip()for x in subprocess.check_output(['nvidia-smi','-i',str(resource['index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
    assert f[0]==resource['uuid'];snapshot=dict(unix=time.time(),uuid=f[0],free_mib=int(f[1]),utilization_percent=int(f[2]));admissions.append(snapshot)
    if admitted(snapshot['free_mib'],snapshot['utilization_percent']):break
    assert time.perf_counter()-started<300,'No admission within300s; no scientific launch';time.sleep(1)
   eligible(name,reg);save(d/'admission.json',dict(observations=admissions,admitted=snapshot,unchanged_admission=dict(min_free_mib=60000,max_utilization_percent=80)))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(resource['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
   argv=['taskset','-c',','.join(map(str,resource['cpu_affinity'])),str(PY),'-B',str(E/'train.py'),'--registration-sha256',regsha,'--job',name];t=time.perf_counter()
   with(d/'stdout.log').open('xb')as so,(d/'stderr.log').open('xb')as se:
    p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True);ident=process(p.pid);assert ident;active[p.pid]=ident
    save(d/'launch.json',dict(argv=argv,process=ident,parent_process=process(os.getpid()),resource=resource,started_unix=time.time(),recovery_authorization=pin(authpath),same_registered_job=True,prior_scientific_attempts=0))
    term=False;killed=False
    try:rc=p.wait(timeout=reg['training_walltime_seconds']+180)
    except subprocess.TimeoutExpired:
     cur=process(p.pid);assert cur and cur['start_ticks']==ident['start_ticks'];os.killpg(p.pid,signal.SIGTERM);term=True
     try:rc=p.wait(timeout=90)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
    active.pop(p.pid,None)
   v=dict(returncode=rc,actual_wait=True,process=ident,elapsed_seconds=time.perf_counter()-t,terminated=term,killed=killed,timed_out=term,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'))
   save(d/'actual_wait.json',v);return v
  except BaseException as e:
   if p and p.poll()is None:
    cur=process(p.pid)
    if cur and cur['start_ticks']==ident['start_ticks']:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=120)
   save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),prior_scientific_attempts=0 if p is None else 1));return dict(error=repr(e))
 with concurrent.futures.ThreadPoolExecutor(2)as pool:results=list(pool.map(run,NAMES))
 save(out/'completion.json',dict(results=results,all_exit0=all(x.get('returncode')==0 for x in results),registration_sha256=regsha,authorization_sha256=authsha,parent_process=process(os.getpid())))
 assert all(x.get('returncode')==0 for x in results),'Unlaunched recovery failed; no automatic retry'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);p.add_argument('--authorization-sha256',required=True);a=p.parse_args();main(a.registration_sha256,a.authorization_sha256)
