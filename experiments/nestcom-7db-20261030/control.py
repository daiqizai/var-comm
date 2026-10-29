"""One actual wait; no retry. Sum reserved GPU time cannot exceed 12 hours."""
from g import *
import signal
def main(request,expected):
 assert sha(request)==expected;r=read(request);o=L/(r['role']+'_control');o.mkdir()
 save(o/'started.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  for d in r['code']:checked(d)
  for d in r.get('dependencies',[]):
   v=cr(d);assert v['returncode']==0 and gone(v['process'])
  env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',CUDA_VISIBLE_DEVICES=str(r.get('gpu_index',1))if r['gpu']else'',CUDA_DEVICE_ORDER='PCI_BUS_ID');env.pop('PYTHONPATH',None)
  if r['gpu']:
   import sqlite3
   db=sqlite3.connect(L/'gpu_reservations.sqlite',timeout=30);db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE IF NOT EXISTS reservations(role TEXT PRIMARY KEY,maximum_seconds REAL)');db.commit();db.execute('BEGIN IMMEDIATE')
   spent=db.execute('SELECT COALESCE(SUM(maximum_seconds),0) FROM reservations').fetchone()[0];assert spent+r['max_seconds']<=43200
   db.execute('INSERT INTO reservations VALUES(?,?)',(r['role'],r['max_seconds']));db.commit();db.close()
   a=subprocess.check_output(['nvidia-smi','-i',str(r.get('gpu_index',1)),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
   assert a[0]==r['gpu_uuid'] and int(a[1])>=45000 and int(a[2])<=80,a
  argv=r['argv']+[request,expected];beg=time.monotonic()
  with(o/'stdout.log').open('xb')as so,(o/'stderr.log').open('xb')as se:
   p=subprocess.Popen(argv,stdout=so,stderr=se,env=env,start_new_session=True);v=ident(p.pid);save(o/'child.json',dict(process=v,argv=argv));timeout=False
   try:rc=p.wait(timeout=r['max_seconds'])
   except subprocess.TimeoutExpired:
    timeout=True;os.killpg(p.pid,signal.SIGTERM)
    try:rc=p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
  save(o/'actual_wait.json',dict(returncode=rc,process=v,child_gone=gone(v),timed_out=timeout,seconds=time.monotonic()-beg));assert rc==0 and not timeout
 except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
