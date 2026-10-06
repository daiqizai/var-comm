"""Fixed clean-reference completion; 200 existing calibration sources, no PHY."""
import json,os,subprocess,time,traceback,sys,hashlib
from pathlib import Path
R=Path('/home/liulu/projects/VAR_COMM');N=R/'outputs/MAIN-RAW64-20261007';D=N/'clean_sequence_v1';T=N/'clean_missing_runtime_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]),uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def call(stage,argv):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0' if stage=='worker' else '',PYTHONPATH=str(T))
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='6' if stage=='worker' else '2'
 def resources():os.sched_setaffinity(0,set(range(4,10)) if stage=='worker' else {16,17});os.setpriority(os.PRIO_PROCESS,0,15)
 c=None;ident=None;error=None;log=D/(stage+'.log');start=time.time()
 with log.open('xb') as f:
  try:
   c=subprocess.Popen(argv,cwd=R,env=env,preexec_fn=resources,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   until=time.monotonic()+10
   while True:
    ident=identity(c.pid)
    if ident['argv']==argv:break
    if c.poll() is not None or time.monotonic()>until:raise RuntimeError('Identity admission failed')
    time.sleep(.01)
   save(D/(stage+'_launch.json'),dict(identity=ident,expected_argv=argv,started_unix=start))
  except BaseException:
   error=traceback.format_exc()
   if c is not None and stage=='worker':
    stop=N/'clean_missing_v1/STOP'
    if stop.parent.exists() and not stop.exists():
     try:save(stop,dict(reason='Outer admission failed'))
     except BaseException:error+='\nSTOP_WRITE_FAILED\n'+traceback.format_exc()
  finally:
   if c is not None:rc=c.wait()
  if c is None:raise RuntimeError(error)
 save(D/(stage+'_exit.json'),dict(identity=ident,exit_code=rc,process_waited=True,log=str(log),log_sha256=sha(log),elapsed_seconds=time.time()-start))
 if error or rc:raise RuntimeError(error or f'{stage} exited {rc}')
def main():
 D.mkdir();index=read(T/'LOCAL_PREPARED_DELIVERY_INDEX.json');pins=read(N/'clean_sequence_inputs.json')
 for p,h in pins.items():assert sha(p)==h,p
 save(D/'registration.json',dict(scope=['qualify','request','register','worker'],bindings=pins,new_PHY=0,images=200,automatic_successor=False));save(D/'identity.json',identity(os.getpid()))
 for stage in ['qualify','request','register','worker']:call(stage,index['commands'][stage])
 done=N/'clean_missing_v1/completion.json';x=read(done);assert x['images']==200 and x['source_count']==200 and x['new_packet_decodes']==0
 for p,h in x['outputs'].items():assert sha(p)==h,p
 for p,h in pins.items():assert sha(p)==h,p
 save(D/'completion.json',dict(status='RAW64_CLEAN200_SEQUENCE_COMPLETE',original_worker_exit_zero=True,scientific_completion_sha256=sha(done),worker_exit_sha256=sha(D/'worker_exit.json'),new_PHY=0,automatic_successor=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if D.exists() and not (D/'failure.json').exists():save(D/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()))
  raise
