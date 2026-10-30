"""Materialize exact known receiver inputs without any model invocation."""
from opt_common import *
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']];jobs=cr(r['jobs']);reuse=cr(r['reuse'])
 out=Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',dict(model_load=0,source_decode=0,render=0));rows=[]
 try:
  for j in jobs:
   key=j['key'];p=j['projection']
   if p['status']!='SOURCE_INPUT':
    image=np.full((3,256,256),.5,np.float32);tokens=np.empty(0,np.int64)
    rows.append(dict(key=key,variant='original',image=None,pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='FIXED_GRAY',reason=p['reason'],origin='FIXED_GRAY'));continue
   v=reuse[key];assert v['status']=='ACCEPTED_SOURCE'
   with np.load(checked(v['image']),allow_pickle=False)as z:image=z['image'].copy();tokens=z['received_tokens'].copy()
   assert asha(image)==v['pixel_sha256']and asha(tokens)==v['received_tokens_sha256']and tokens.shape==(OFFSETS[p['m']]+p['K'],)
   rows.append(dict(v,variant='original',origin='EXACT_CACHED_RECEIVER'))
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',unique_inputs=len(jobs),reused=sum(x['origin']=='EXACT_CACHED_RECEIVER'for x in rows),gray=sum(x['status']=='FIXED_GRAY'for x in rows),ledger=ledger.summary(),new_models=0,new_PHY=0,registration=pin(request)))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
