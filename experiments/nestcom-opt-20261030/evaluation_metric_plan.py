"""Native-pair metrics for an explicitly frozen source population and RX mapping."""
from opt_common import *

def main(request):
 r=read(request);sources=cr(r['sources']);sm={s['source_id']:s for s in sources};pairs={};rows=[];receipts=[]
 for segment in r['segments']:
  role=segment['receiver_role'];w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  c=read(L/role/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
  outputs=read(L/role/'outputs.json');bykey={}
  for v in outputs:bykey.setdefault(v['key'],{})[v.get('variant','original')]=v
  for f in cr(segment['logical']):
   s=sm[f['source_id']];assert s['reference_sha256']==f['source_float_sha256']and s['preprocessing_id']==f['source_pixels_sha256']
   variants=segment.get('receiver_variants',['original'])
   for variant in variants:
    v=bykey[f['received_key']][variant];key=digest([s['reference_sha256'],v['pixel_sha256']])
    pairs.setdefault(key,dict(pair_key=key,source_id=s['source_id'],source_index=s['source_index'],image=v['image'],pixel_sha256=v['pixel_sha256']))
    rows.append(dict(f,variant=f['variant']if variant=='original'else variant,pair_key=key,receiver_status=v['status'],generation_seed=v.get('generation_seed'),temperature=v.get('temperature',0),output_origin=v['origin']))
  receipts.append(pin(L/(role+'_control')/'actual_wait.json'))
 out=L/(r['role']+'_metric_plan');out.mkdir(exist_ok=False)
 save(out/'plan.json',dict(sources=[dict(source_id=s['source_id'],source_index=s['source_index'],archive=s['archive'],reference_sha256=s['reference_sha256'])for s in sources],pairs=list(pairs.values())))
 save(out/'logical.json',rows);save(out/'completion.json',dict(status='PREPARED',logical_rows=len(rows),unique_pairs=len(pairs),registration=pin(request),actual_receiver_waits=receipts))
 print(r['role'],len(rows),len(pairs))

if __name__=='__main__':main(*sys.argv[1:])
