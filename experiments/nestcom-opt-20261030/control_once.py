"""One exclusive actual child; immutable registrations and real wait receipts."""
from opt_common import *
import subprocess,signal,traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(d)for d in r['code']]
 out=Path(r['control']);out.mkdir(exist_ok=False)
 save(out/'owner.json',dict(owner=ident(os.getpid()),registration=pin(request)))
 env=dict(os.environ,**r['environment']);start=time.time()
 with(out/'worker.log').open('xb')as log:
  p=subprocess.Popen(r['command'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=ident(p.pid);save(out/'child.json',dict(child=child,command=r['command'],started_unix=start))
  try:rc=p.wait(timeout=r['hard_seconds'])
  except subprocess.TimeoutExpired:
   p.send_signal(signal.SIGTERM)
   try:rc=p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();rc=p.wait()
  save(out/'actual_wait.json',dict(returncode=rc,child=child,final_identity=ident(p.pid),started_unix=start,finished_unix=time.time(),device_seconds=time.time()-start if r['gpu']else 0,automatic_retry=False))
 sys.exit(0 if rc==0 else 1)
if __name__=='__main__':main(*sys.argv[1:])
