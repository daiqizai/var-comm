"""Join native output identities to the frozen ordered source manifest."""
from opt_common import *
def main(role):
 out=L/(role+'_metric_plan');out.mkdir(exist_ok=False)
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 assert read(L/role/'completion.json')['status']=='COMPLETE'
 outputs=read(L/role/'outputs.json');bykey={}
 for v in outputs:bykey.setdefault(v['key'],[]).append(v)
 sources=read(L/'inputs/fit100.json');sm={s['source_id']:s for s in sources};logical=read(L/'fit_rx_plan/logical.json');pairs={};rows=[]
 for f in logical:
  s=sm[f['source_id']];assert s['reference_sha256']==f['source_float_sha256']
  assert f['received_key']in bykey
  for v in bykey[f['received_key']]:
   k=digest([s['reference_sha256'],v['pixel_sha256']]);pairs.setdefault(k,dict(pair_key=k,source_id=s['source_id'],source_index=s['source_index'],image=v['image'],pixel_sha256=v['pixel_sha256']))
   rows.append(dict(f,variant=v['variant'],pair_key=k,receiver_status=v['status'],generation_seed=v.get('generation_seed'),temperature=v.get('temperature',0),output_origin=v['origin']))
 save(out/'plan.json',dict(sources=[dict(source_id=s['source_id'],source_index=s['source_index'],archive=s['archive'],reference_sha256=s['reference_sha256'])for s in sources],pairs=list(pairs.values())))
 save(out/'logical.json',rows);save(out/'completion.json',dict(status='PREPARED',logical_rows=len(rows),unique_pairs=len(pairs),receiver_outputs=pin(L/role/'outputs.json'),receiver_wait=pin(L/(role+'_control')/'actual_wait.json')))
 print(role,len(rows),len(pairs))
if __name__=='__main__':main(*sys.argv[1:])
