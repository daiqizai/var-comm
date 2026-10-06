"""Finite qualification-tests, registered CPU launch, and original waited exit."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
N=Path('/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007');P=N/'prepared_v1';D=N/'cpu_sequence_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(t[19]),uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def call(label,argv,env):
 log=D/(label+'.log');started=time.time();c=None;identity=None;error=None
 with log.open('xb') as f:
  try:
   c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=P/'runner',env=env)
   until=time.monotonic()+10
   while True:
    identity=ident(c.pid)
    if identity['argv']==argv:break
    if c.poll() is not None or time.monotonic()>until:raise RuntimeError('Identity admission failed')
    time.sleep(.01)
   save(D/(label+'_launch.json'),dict(identity=identity,expected_argv=argv,started_unix=started))
  except BaseException:
   error=traceback.format_exc()
   if c is not None and label=='owner' and not (N/'STOP').exists():
    try:save(N/'STOP',dict(reason='Outer owner admission failure',traceback=error))
    except BaseException:error+='\nSTOP_WRITE_FAILED\n'+traceback.format_exc()
  finally:
   if c is not None:rc=c.wait()
  if c is None:raise RuntimeError(error)
 save(D/(label+'_exit.json'),dict(identity=identity,expected_argv=argv,exit_code=rc,process_waited=True,log=str(log),log_sha256=sha(log),elapsed_seconds=time.time()-started))
 if error:raise RuntimeError(error)
 if rc:raise RuntimeError(label+' exited '+str(rc))
def main():
 D.mkdir();manifest=read(N/'cpu_sequence_inputs.json')
 for p,h in manifest.items():assert sha(p)==h,p
 save(D/'registration.json',dict(source_bindings=manifest,scope=['pure_tests','prepare_register','CPU_qualification','CPU_proxy','merge'],automatic_calibration=False))
 save(D/'identity.json',ident(os.getpid()))
 cfg=read(N/'plan_config_v1.json');python=cfg['python'];env=os.environ.copy()
 env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MAIN64_TEST_ASSETS=str(N/'plan_config_v1.json'),MAIN64_TEST_CATALOGUE=str(N/'assemble_v1/completion.json'),PYTHONPATH=os.pathsep.join([str(P/'runner'),str(P/'phy'),str(N.parent.parent/'src')]))
 call('pure_tests',[python,'-B','-m','unittest','-v','test_main_raw64_runner','test_main_raw64_plan_only','test_main_raw64_packet_adapter'],env)
 save(P/'cpu_cpu_qualification.json',dict(status='RAW64_CPU_LOCAL_CONTRACT_TESTS_PASS_IN_ACTUAL_LDPC_ENVIRONMENT',exit_receipt_sha256=sha(D/'pure_tests_exit.json'),log_sha256=sha(D/'pure_tests.log'),bindings=manifest,new_decoder_calls=0,GPU_used=False))
 call('prepare',[python,'-B',str(N/'prepare_cpu.py')],env)
 reg=N/'cpu_execution_v1/registration.json'
 call('owner',[python,'-B',str(P/'runner/main_raw64_cpu_runner.py'),'--registration',str(reg),'--run'],env)
 done=N/'cpu_execution_v1/completion.json';assert read(done)['status']=='MAIN_RAW64_QUALIFICATION_PROXY_NORMALLY_COMPLETE_V1'
 for p,h in manifest.items():assert sha(p)==h,p
 save(D/'completion.json',dict(status='RAW64_CPU_SEQUENCE_COMPLETE',original_owner_success=True,owner_completion_sha256=sha(done),exit_receipt_sha256=sha(D/'owner_exit.json'),automatic_successor_started=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if D.exists() and not (D/'failure.json').exists():save(D/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()))
  raise
