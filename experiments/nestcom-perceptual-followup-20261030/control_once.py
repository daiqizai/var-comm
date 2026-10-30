"""One registered child, bounded execution, preserved actual wait and failure."""
from followup_common import *
import subprocess,signal
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(x)for x in r['code']]
 c=Path(r['control']);c.mkdir(exist_ok=False);save(c/'owner.json',dict(owner=ident(os.getpid()),registration=pin(request)))
 start=time.time()
 with(c/'worker.log').open('xb')as log:
  p=subprocess.Popen(r['command'],cwd=R,env=dict(os.environ,**r['environment']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=ident(p.pid);save(c/'child.json',dict(child=child,started_unix=start,command=r['command']))
  try:rc=p.wait(timeout=r['hard_seconds'])
  except subprocess.TimeoutExpired:
   p.send_signal(signal.SIGTERM)
   try:rc=p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();rc=p.wait()
  save(c/'actual_wait.json',dict(returncode=rc,child=child,final_identity=ident(p.pid),started_unix=start,finished_unix=time.time(),device_seconds=time.time()-start if r['gpu']else 0,CPU_process_wall_seconds=time.time()-start if not r['gpu']else 0,automatic_retry=False))
 sys.exit(0 if rc==0 else 1)
if __name__=='__main__':main(*sys.argv[1:])
