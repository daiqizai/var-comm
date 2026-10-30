"""Projection-only RX plan, exact completed current and qualified historical reuse."""
from opt_common import *

def main(request):
 r=read(request);jobs={};frames={};logical=[];reuse={};lineage=[]
 for role in r['frame_roles']:
  w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  assert read(L/role/'completion.json')['ledger']['unresolved']==0
  for v in read(L/role/'rx_inputs.json'):
   if v['key']in jobs:assert jobs[v['key']]==v
   jobs[v['key']]=v
  for v in read(L/role/'frames.json'):frames[v['physical_key']]=v
 for f in r['logical']:
  for row in cr(f):logical.append(dict(row,received_key=frames[row['physical_key']]['received_key']))
 def accept(row):
  key=row['key']
  if key not in jobs:return False
  if row['image']is None:assert jobs[key]['projection']['status']!='SOURCE_INPUT';return False
  with np.load(checked(row['image']),allow_pickle=False)as z:a=z['image'].copy();t=z['received_tokens'].copy()
  p=jobs[key]['projection'];assert p['status']=='SOURCE_INPUT'and t.shape==(OFFSETS[p['m']]+p['K'],)
  assert asha(t)==row['received_tokens_sha256']and asha(a)==row['pixel_sha256']
  if key in reuse:assert reuse[key]['pixel_sha256']==row['pixel_sha256']and reuse[key]['received_tokens_sha256']==row['received_tokens_sha256'];return False
  reuse[key]=row;return True
 for role in r.get('current_original_roles',[]):
  w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  count=0
  for row in read(L/role/'outputs.json'):
   if row.get('variant','original')=='original':count+=accept(row)
  if count:lineage.append(dict(index=pin(L/role/'outputs.json'),wait=pin(L/(role+'_control')/'actual_wait.json'),reused=count))
 if r.get('historical_n512',False):
  root=B/'nestcom_n512_20261028_v1';identity=read(L/'p0_replay/actual_native_identity.json')
  for phase in ('pilot','full','formal'):
   for i in range(4):
    name=f'{phase}_rx_nest_{i:02d}';p=root/name;index=p/'outputs.json';ni=p/'actual_native_identity.json';wa=root/'controls'/name/'actual_wait.json'
    if not(index.exists()and ni.exists()and wa.exists()):continue
    old=read(ni);assert all(old[k]==identity[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling'])and old['GPU_name']==identity['GPU_name']
    assert read(wa)['returncode']==0;count=0
    for row in read(index):
     if row.get('codec')!='NEST'or row['key']not in jobs or row['key']in reuse:continue
     assert row['status']=='ACCEPTED_SOURCE'and row['canonical']and not row['source_truth_used']and not row['TX_CDF_used']
     count+=accept(row)
    if count:lineage.append(dict(index=pin(index),identity=pin(ni),wait=pin(wa),reused=count))

 if r.get('historical_seven',False):
  root=B/'nestcom_7db_common500_20261030_v1';old=read(root/'rx_nest/actual_native_identity.json');identity=read(L/'p0_replay/actual_native_identity.json')
  assert all(old[k]==identity[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name'])
  wa=root/'rx_nest_control/actual_wait.json';assert read(wa)['returncode']==0
  count=0
  for row in read(root/'rx_nest/outputs.json'):
   if row['key']not in jobs or row['key']in reuse or row['status']!='ACCEPTED_SOURCE':continue
   assert row['canonical']and not row['source_truth_used']and not row['TX_CDF_used'];count+=accept(row)
  if count:lineage.append(dict(index=pin(root/'rx_nest/outputs.json'),identity=pin(root/'rx_nest/actual_native_identity.json'),wait=pin(wa),reused=count))
 out=L/r['output'];out.mkdir(exist_ok=False);save(out/'inputs.json',list(jobs.values()));save(out/'reuse.json',reuse);save(out/'logical.json',logical)
 save(out/'completion.json',dict(status='PREPARED',registration=pin(request),logical=len(logical),unique_inputs=len(jobs),reused=len(reuse),new_decode=sum(v['projection']['status']=='SOURCE_INPUT'and k not in reuse for k,v in jobs.items()),lineage=lineage,source_truth_in_RX_jobs=False))
 print({k:v for k,v in read(out/'completion.json').items()if k!='lineage'})

if __name__=='__main__':main(*sys.argv[1:])
