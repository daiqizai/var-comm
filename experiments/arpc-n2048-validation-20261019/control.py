"""Single-shot stage supervisor. No implicit retries or ungated successors."""
from shared import *
import argparse,fcntl
def main():
 a=argparse.ArgumentParser();a.add_argument('stage');a.add_argument('--expected',required=True);s=a.parse_args();r=registration(s.expected)
 assert s.stage in r['stage_seconds'];phase,role=s.stage.split('_',1)
 if phase=='formal':assert read(O/'qualification_review.json')['status']=='PASS'
 root=L/'controls'/s.stage;root.mkdir(exist_ok=False)
 put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv,registration_sha256=s.expected,started_unix=time.time()))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 if role in ['tx','rx','audit']:
  argv=[str(CPY),'-B',str(E/'phy_worker.py'),role,phase]
  if role!='rx':argv+=['--expected',s.expected]
  previous={'rx':'tx','audit':'rx'}.get(role)
  if previous:
   w=read(L/'controls'/(phase+'_'+previous)/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])
 elif role=='nest':
  argv=[str(PY),'-B',str(E/'nest_rx.py'),str(L/phase/'nest/public.json')]
  env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
 else:
  assert role=='arpc'
  argv=['/usr/bin/python3','-B',str(OLD_E/'a3_rx_v1/native.py'),'rx','--public',str(L/phase/'arpc/public.json')]
  env.update(CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(OLD_L/'environment_v1')+':'+str(OLD_E),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  wait_child(root,argv,env,r['stage_seconds'][s.stage]);put(root/'completion.json',dict(status='COMPLETE',actual_wait=pin(root/'actual_wait.json')))
 except BaseException:put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
