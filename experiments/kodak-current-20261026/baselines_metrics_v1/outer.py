"""Persistent supervisor. Local SSH parent records this supervisor's actual exit."""
from common import *
import argparse,subprocess,sys,traceback,signal
def main(role,expected):
 assert sha(O/'metric_registration.json')==expected;reg=read(O/'metric_registration.json');rr=reg['resources'][role]
 for d in reg['source_files']:checked(d)
 out=L/(role+'_initial_control_v1');out.mkdir();started=time.monotonic();terminated=killed=timed_out=False
 argv=[rr['python'],'-B',str(E/'owner.py'),role,'--registration',expected]
 try:
  with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
   c=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True);cp=process(c.pid)
   save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),child_process=cp,argv=argv,registration=desc(O/'metric_registration.json')))
   try:rc=c.wait(timeout=reg['max_seconds']+300)
   except subprocess.TimeoutExpired:
    timed_out=terminated=True;os.killpg(c.pid,signal.SIGTERM)
    try:rc=c.wait(timeout=30)
    except subprocess.TimeoutExpired:killed=True;os.killpg(c.pid,signal.SIGKILL);rc=c.wait()
  save(out/'actual_wait.json',dict(returncode=int(rc),process=cp,parent_process=process(os.getpid()),elapsed_seconds=time.monotonic()-started,terminated=terminated,killed=killed,timed_out=timed_out,metadata_error=None))
  if rc or timed_out:raise RuntimeError('Owner unsuccessful')
  save(out/'completion.json',dict(status='COMPLETE',actual_wait=desc(out/'actual_wait.json')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['three_metrics','clip_metrics']);p.add_argument('--registration',required=True);a=p.parse_args();main(a.role,a.registration)
