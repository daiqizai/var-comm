"""One immutable role dispatch, durable stdout/stderr and actual child wait."""
from common import *
import subprocess,signal,traceback,fcntl
def run(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 stage=r['stages'][role];out=L/(role+'_control');out.mkdir(parents=True,exist_ok=False)
 child=None;terminated=killed=timed_out=False;metadata_error=None;beg=time.monotonic()
 lock=(L/(role+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  env=os.environ.copy();env.update(stage['environment']);env['PYTHONDONTWRITEBYTECODE']='1'
  if stage.get('gpu'):
   g=stage['gpu'];rows=subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
   assert rows[0].strip().lower().lstrip('0')==g['pci'].lower().lstrip('0')
   assert int(rows[1])>=24576 and int(rows[2])<=80,rows
  cmd=stage['argv']+['--registration',str(path),'--expected-registration',expected]
  with (out/'stdout.log').open('xb')as so,(out/'stderr.log').open('xb')as se:
   child=subprocess.Popen(cmd,stdout=so,stderr=se,env=env,start_new_session=True,cwd=R)
   ci=ident(child.pid);assert ci is not None
   save(out/'launch.json',dict(process=ci,parent_process=ident(os.getpid()),argv=cmd,registration=pin(path),environment=stage['environment']))
   try:rc=child.wait(timeout=stage['max_seconds']+120)
   except subprocess.TimeoutExpired:
    timed_out=terminated=True;os.killpg(child.pid,signal.SIGTERM)
    try:rc=child.wait(timeout=30)
    except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
  save(out/'actual_wait.json',dict(actual_wait=True,returncode=int(rc),process=ci,parent_process=ident(os.getpid()),argv=cmd,elapsed_seconds=time.monotonic()-beg,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=metadata_error,registration=pin(path)))
  if rc!=0:raise RuntimeError('CHILD_ACTUAL_EXIT_'+str(rc))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),child_created=child is not None,automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1],sys.argv[2],sys.argv[3])
