"""Exclusive finite child wait; no retries or automatic successor stages."""
from p1_common import *
import subprocess,signal,traceback
def supervise(role,expected):
 r=registration(expected);d=L/(role+'_control');assert(d/'dispatch.json').is_file()
 argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/('p1_'+role+'.py')),expected]
 parent=ident(os.getpid());start=time.monotonic();terminated=False;killed=False;timed_out=False
 with(d/'stdout.txt').open('xb')as so,(d/'stderr.txt').open('xb')as se:
  p=subprocess.Popen(argv,cwd=E,stdout=so,stderr=se,start_new_session=True);child=ident(p.pid)
  save(d/'launch.json',dict(argv=argv,process=child,parent_process=parent,registration_sha256=expected))
  try:rc=p.wait(timeout=r['max_seconds']+300)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM);terminated=True
   try:rc=p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait(timeout=30)
 save(d/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=child,parent_process=parent,terminated=terminated,killed=killed,timed_out=timed_out,seconds=time.monotonic()-start))
 return rc
def dispatch(role,expected):
 registration(expected);assert role=='link';d=L/(role+'_control');d.mkdir(exist_ok=False)
 save(d/'dispatch.json',dict(role=role,expected_registration=expected,unix=time.time()))
 with(d/'supervisor.stdout').open('xb')as so,(d/'supervisor.stderr').open('xb')as se:
  p=subprocess.Popen([sys.executable,'-B',str(E/'p1_control.py'),'supervise',role,expected],cwd=E,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
  save(d/'supervisor.json',ident(p.pid))
 print(json.dumps(dict(control=str(d),supervisor=read(d/'supervisor.json'))))
if __name__=='__main__':
 mode,role,expected=sys.argv[1:]
 if mode=='dispatch':dispatch(role,expected)
 elif mode=='supervise':sys.exit(supervise(role,expected))
 else:raise ValueError(mode)
