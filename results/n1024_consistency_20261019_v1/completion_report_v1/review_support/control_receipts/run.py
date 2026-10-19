
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
C=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/n1024_consistency_20261019_v1/completion_report_v1_control');R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/n1024-consistency-20261019/completion_report_v1')
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(a[19]),state=a[0],host=os.uname().nodename)
parent=ident(os.getpid());argv=['/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python','-B',str(E/'report.py'),'--expected-registration','4789b2595190262a5970f985d25f420663d8225321830a770c4400fc82487670']
save(C/'supervisor.json',dict(process=parent,parent_process=ident(os.getppid()),argv=['/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python','-B',str(C/'run.py')],source=pin(C/'run.py')))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(C/'matplotlib_cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
timed=terminated=killed=False;child=None;identity=None;start=time.monotonic()
try:
 for name,(size,sha) in {'report.py': (21646, '6ba046f5e41c9fbbd0d8b647fd172f9f9911ad40743332ece636b48ef7a3290b'), 'README.md': (5990, 'eaf48cbf27317e7acabb9f8fd1d6d551c9f363ca01aff9f95166a6e461bf6a51')}.items():
  d=pin(E/name);assert d['bytes']==size and d['sha256']==sha
 with (C/'stdout.txt').open('xb') as so,(C/'stderr.txt').open('xb') as se:
  child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);identity=ident(child.pid)
  save(C/'launch.json',dict(process=identity,parent_process=parent,argv=argv,timeout_seconds=600,started_unix=time.time(),science_calls_authorized=0))
  try:rc=child.wait(timeout=600)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=30)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
 wait=dict(actual_wait=True,returncode=rc,process=identity,parent_process=parent,argv=argv,seconds=time.monotonic()-start,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt'))
 save(C/'actual_wait.json',wait);assert type(rc) is int and rc==0 and not timed
 save(C/'completion.json',dict(status='COMPLETE_REPORT_EXPORT',actual_wait=pin(C/'actual_wait.json')))
 print(json.dumps(dict(status='COMPLETE_REPORT_EXPORT',wait=wait)))
except BaseException:save(C/'failure.json',dict(traceback=traceback.format_exc(),process=identity,automatic_retry=False));raise
