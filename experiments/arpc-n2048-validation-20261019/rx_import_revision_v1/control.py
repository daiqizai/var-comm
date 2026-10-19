"""Single authorized recovery wait; original control/failed attempt unchanged."""
import sys,fcntl
from pathlib import Path
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-n2048-validation-20261019');sys.path.insert(0,str(E))
from shared import *
def main(auth,expected):
 assert sha(auth)==expected;a=read(auth);assert a['human_approved_four_repeated_decoder_calls']
 for d in a['bindings']:verify(d)
 assert pin(__file__) in a['bindings']
 root=L/'controls/qualification_rx_import_recovery_v1';root.mkdir()
 lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv,authorization=pin(auth)))
 try:
  wait_child(root,[str(CPY),'-B',str(E/'rx_import_revision_v1/receiver.py'),auth,expected],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),600)
  put(root/'completion.json',dict(status='COMPLETE',actual_wait=pin(root/'actual_wait.json')))
 except BaseException:put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
