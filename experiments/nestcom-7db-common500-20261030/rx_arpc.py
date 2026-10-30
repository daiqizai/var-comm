"""Independent ARPC receiver; accepts only actual decoded public inputs."""
from g import *
def main(request,expected):
 r,o=initialize(request,expected);jobs=cr(r['inputs']);cfg=cr(r['timing_config']);beg=time.monotonic();ledger=Ledger(o/'calls.sqlite',dict(receive_render=len(jobs)))
 try:
  for j in jobs:assert j['key']==digest(j['projection']) and set(j['projection'])=={'codec','actual'} and j['projection']['codec']=='ARPC'
  sys.path.insert(0,str(R/'experiments/five-method-timing-20261020/library_recovery_v1'))
  from adapter import adapted_worker
  import native
  w=adapted_worker();blocked=w.firewall(cfg,'ARPC',o);c=native.arpc(cfg,o,'rx');(o/'images').mkdir();rows=[]
  for j in jobs:
   assert time.monotonic()-beg<r['max_seconds'] and not(L/'STOP').exists()
   def compute():
    image,status=w.native_rx(j['projection']['actual'],cfg,'ARPC',c,w.Clock(c.torch));p=o/'images'/(j['key']+'.npz')
    assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all()
    with p.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
    v=dict(key=j['key'],image=pin(p),pixel_sha256=asha(image),status=status,source_truth_used=False);save(p.with_suffix('.json'),v);return v
   rows.append(ledger.call('receive_render',j['key'],compute))
   if len(rows)%25==0:print('ARPC_RX',len(rows),len(jobs),flush=True)
  assert not blocked and ledger.summary()['unresolved']==0;save(o/'outputs.json',rows);save(o/'completion.json',dict(status='COMPLETE_RX',outputs=pin(o/'outputs.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-beg,source_truth_used=False))
 except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary()));raise
if __name__=='__main__':main(*sys.argv[1:])
