"""Logical grid/TX choices map to a deterministic actual received output."""
from opt_common import *

def main(role,rx_plan,rx_outputs):
 out=L/(role+'_metric_plan');out.mkdir(exist_ok=False);outputs=read(L/rx_outputs/'outputs.json');bykey={v['key']:v for v in outputs}
 w=read(L/(rx_outputs+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 sources=read(L/'inputs/fit100.json');sm={s['source_id']:s for s in sources};rows=[];pairs={}
 for f in read(L/rx_plan/'logical.json'):
  s=sm[f['source_id']];assert s['reference_sha256']==f['source_float_sha256'];v=bykey[f['received_key']]
  k=digest([s['reference_sha256'],v['pixel_sha256']]);pairs.setdefault(k,dict(pair_key=k,source_id=s['source_id'],source_index=s['source_index'],image=v['image'],pixel_sha256=v['pixel_sha256']))
  rows.append(dict(f,pair_key=k,receiver_status=v['status'],generation_seed=None,temperature=0,output_origin=v['origin']))
 save(out/'plan.json',dict(sources=[dict(source_id=s['source_id'],source_index=s['source_index'],archive=s['archive'],reference_sha256=s['reference_sha256'])for s in sources],pairs=list(pairs.values())))
 save(out/'logical.json',rows);save(out/'completion.json',dict(status='PREPARED',logical_rows=len(rows),unique_pairs=len(pairs),receiver_outputs=pin(L/rx_outputs/'outputs.json'),receiver_wait=pin(L/(rx_outputs+'_control')/'actual_wait.json')))
 print(role,len(rows),len(pairs))
if __name__=='__main__':main(*sys.argv[1:])
