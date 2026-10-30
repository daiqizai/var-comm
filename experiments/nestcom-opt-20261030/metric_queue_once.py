"""Finite metric-only successor queue; zero scientific retries, original caps."""
from opt_common import *
from launch_metric_stage import main as launch_metric
import subprocess,traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 beg=time.monotonic();completed=[]
 def wait_boundary():
  assert not(L/'STOP').exists()and time.monotonic()-beg<r['seconds'],'Finite queue stopped; no deadline reset'
  time.sleep(5)
 def closed(role):
  p=L/(role+'_control')/'actual_wait.json'
  if not p.exists():return False
  w=read(p);assert w['returncode']==0,('Unknown failed dependency, no retry',role,w)
  v=ident(w['child']['pid']);assert v is None or v['start_ticks']!=w['child']['start_ticks']
  return True
 try:
  for item in r['stages']:
   role=item['role']+'_'+item['mode'];assert not(D/(role+'_registration.json')).exists()and not(L/role).exists(),('Already launched; do not duplicate',role)
   while not all(closed(d)for d in item['dependencies'])or not(L/(item['role']+'_metric_plan/completion.json')).exists():wait_boundary()
   while True:
    active=[]
    for p in L.glob('*_control/child.json'):
     if(p.parent/'actual_wait.json').exists():continue
     c=read(p);v=ident(c['child']['pid']);assert v is not None and v['start_ticks']==c['child']['start_ticks']
     if cr(read(p.parent/'owner.json')['registration'])['gpu']:active.append(c['child']['pid'])
    gs=[v.split(', ')for v in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
    g=next(g for g in gs if g[0]==r['metric_gpu_uuid'])
    if len(active)<2 and int(g[1])>=60000 and int(g[2])<=50:break
    wait_boundary()
   # launch_metric has a second atomic admission check. A race is retained as
   # a zero-call queue failure; this controller never retries a launcher.
   launch_metric(item['role'],item['mode'],item['route'],tuple(item['cache_roles']))
   while not closed(role):wait_boundary()
   c=read(L/role/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
   completed.append(dict(role=role,completion=pin(L/role/'completion.json'),wait=pin(L/(role+'_control')/'actual_wait.json')))
   save(out/(role+'_closed.json'),completed[-1]);print('CLOSED',role,flush=True)
  save(out/'completion.json',dict(status='COMPLETE',stages=completed,science_retries=0,seconds=time.monotonic()-beg))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),completed=completed,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
