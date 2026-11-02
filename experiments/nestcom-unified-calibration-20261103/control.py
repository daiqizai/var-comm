"""Real child wait, exact process identity, no retries."""
from study_common import *
import signal
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(x)for x in r['code']]
 c=Path(r['control']);c.mkdir(exist_ok=False);owner=ident(os.getpid());save(c/'owner.json',dict(owner=owner,registration=pin(request)))
 start=time.time()
 with(c/'worker.log').open('xb')as log:
  p=subprocess.Popen(r['command'],cwd=R,env=dict(os.environ,**r['environment']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=ident(p.pid);save(c/'child.json',dict(child=child,started_unix=start,command=r['command']))
  timed_out=False
  try:rc=p.wait(timeout=r['hard_seconds'])
  except subprocess.TimeoutExpired:
   timed_out=True;n=ident(p.pid);assert n and n['start_ticks']==child['start_ticks'];os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
  save(c/'actual_wait.json',dict(returncode=rc,child=child,owner=owner,final_identity=ident(p.pid),started_unix=start,finished_unix=time.time(),elapsed_seconds=time.time()-start,gpu=r['gpu'],timed_out=timed_out,automatic_retry=False))
 sys.exit(0 if rc==0 else 1)
if __name__=='__main__':main(*sys.argv[1:])
