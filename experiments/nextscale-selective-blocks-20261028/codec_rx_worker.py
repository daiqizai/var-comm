"""Independent clean-wire source proof. No source pixels, truth tokens or TX CDF access."""
from block_common import *
import traceback,tempfile

def main(path,expected):
 assert sha(path)==expected;r=read(path);rp=pin(path)
 for d in r['code']:checked(d)
 jobs=cr(r['inputs']);routes=cr(r['model_routes'])
 assert len({j['key']for j in jobs})==len(jobs)
 for j in jobs:
  assert set(j)=={'key','projection'}and j['key']==digest(j['projection'])
  assert set(j['projection'])=={'m','K','C','A','B'}
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'records').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);beg=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=rp))
 try:
  sys.path[:0]=[str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path']]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs',Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu')])
  from kn_native import load
  from same_scale_codec import SameScaleCodec
  def boundary():assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));codec=SameScaleCodec(R,native=n);rows=[]
  for j in jobs:
   boundary();p=j['projection']
   def receive():
    v=codec.receive(**p)
    v.update(key=j['key'],source_truth_used=False,TX_CDF_used=False)
    save(out/'records'/(j['key']+'.json'),v);return v
   rows.append(ledger.call('independent_decode',j['key'],receive))
   if len(rows)%16==0:print('SOURCE_RX_SAVED',len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE_INDEPENDENT_CODEC_RX',request=rp,outputs=pin(out/'outputs.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-beg,render_calls=0,source_truth_used=False))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
