"""Exact input/model identities for legal reconstruction reuse; no models run."""
from followup_common import *
from opt_common import OFFSETS

def wait_for(root,role):
 choices=[root/(role+'_control/actual_wait.json'),root/(role+'_owner_control/actual_wait.json'),root/'controls'/role/'actual_wait.json']
 p=next((p for p in choices if p.exists()),None)
 if p is None:return None
 w=read(p);assert w['returncode']==0
 for k in ['child','process','owner']:
  v=w.get(k)
  if isinstance(v,dict)and'pid'in v:
   now=ident(v['pid']);assert now is None or now['start_ticks']!=v['start_ticks']
 return pin(p)

def main():
 closed('E2_prepare_metadata_v2');plan=L/'E2_prepare_metadata_v2';out=L/'E2_receiver_plan';out.mkdir(exist_ok=False)
 assert read(plan/'generation_seed_registration.json')==read(L/'E2_prepare/generation_seed_registration.json')
 nj={j['key']:j for j in read(plan/'nest_jobs.json')};aj={j['key']:j for j in read(plan/'ada_jobs.json')};reuse={};ar={};lineage=[]
 donor=read(OL/'final_rx/actual_native_identity.json')
 candidates=[(OL,'final_rx'),(B/'nestcom_7db_common500_20261030_v1','rx_nest')]
 candidates += [(B/'nestcom_n512_20261028_v1',f'{phase}_rx_nest_{i:02d}')for phase in ['formal','full','pilot']for i in range(4)]
 for root,role in candidates:
  index=root/role/'outputs.json';ip=root/role/'actual_native_identity.json'
  if not(index.exists()and ip.exists()):continue
  identity=read(ip)
  if not all(identity[k]==donor[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name']):continue
  wait=wait_for(root,role)
  if wait is None:continue
  used=0
  for row in read(index):
   key=row['key']
   if key not in nj or key in reuse or row.get('status')!='ACCEPTED_SOURCE':continue
   p=nj[key]['projection'];assert key==digest(p)and p['status']=='SOURCE_INPUT'
   with np.load(checked(row['image']),allow_pickle=False)as z:a=z['image'];tokens=z['received_tokens']
   assert asha(a)==row['pixel_sha256']and asha(tokens)==row['received_tokens_sha256']and len(tokens)==OFFSETS[p['m']]+p['K']
   reuse[key]=row;used+=1
  if used:lineage.append(dict(codec='NEST',index=pin(index),identity=pin(ip),wait=wait,reused=used))
 aroot=B/'adatoken_response_20261026_v1';ad=read(B/'nestcom_7db_common500_20261030_v1/rx_ada/numeric_identity.json')
 # Original C production records carry the actual decoded arithmetic hash,
 # class, canonical status and received token hash. At full length the seed
 # cannot influence any generated token because no generation remains.
 target={}
 for j in aj.values():
  p=j['projection']
  if p.get('m')==256:target.setdefault((asha(np.array(p['bits'],np.uint8)),p['received_class']),[]).append(j['key'])
 for i in range(2):
  role=f'c_predclass_common500_rx_s{i:02d}';index=aroot/role/'outputs.json';ip=aroot/role/'numeric_identity.json'
  if not(index.exists()and ip.exists()):continue
  identity=read(ip);assert all(identity[k]==ad[k]for k in ad if k!='pci')
  wait=wait_for(aroot,role)
  if wait is None:lineage.append(dict(codec='ADA',role=role,status='REUSE_NOT_ADMITTED_NO_LOCATED_ACTUAL_WAIT'));continue
  used=0
  for row in read(index):
   k=(row['payload_sha256'],row['received_class'])
   if row['prefix']!=256 or k not in target:continue
   assert row['status']=='ACCEPTED_SOURCE'and not row['source_truth_used']and not row['TX_CDF_used']
   with np.load(checked(row['image']),allow_pickle=False)as z:assert asha(z['image'])==row['pixel_sha256']and asha(z['received_tokens'])==row['received_tokens_sha256']and len(z['received_tokens'])==256
   for key in target[k]:
    ar[key]=dict(row,key=key);used+=1
  if used:lineage.append(dict(codec='ADA',index=pin(index),identity=pin(ip),wait=wait,reused=used))
 save(out/'nest_reuse.json',reuse);save(out/'nest_jobs.json',list(nj.values()));save(out/'ada_all_reuse.json',ar)
 ordered=sorted(aj.values(),key=lambda j:j['key']);shards=[]
 for i in range(4):
  js=ordered[i::4];re={j['key']:ar[j['key']]for j in js if j['key']in ar}
  save(out/f'ada_jobs_{i:02d}.json',js);save(out/f'ada_reuse_{i:02d}.json',re)
  shards.append(dict(shard=i,unique_inputs=len(js),reuse=len(re),new=len(js)-len(re)))
 save(out/'completion.json',dict(status='PREPARED',nest_unique=len(nj),nest_reuse=len(reuse),nest_new=len(nj)-len(reuse),ada_unique=len(aj),ada_reuse=len(ar),ada_new=len(aj)-len(ar),shards=shards,lineage=lineage,new_models=0,new_PHY=0))
 print({k:v for k,v in read(out/'completion.json').items()if k!='lineage'})
if __name__=='__main__':main()
