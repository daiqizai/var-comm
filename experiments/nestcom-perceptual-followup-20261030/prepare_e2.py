"""Plan the six preregistered information budgets from frozen source streams."""
from followup_common import *
from opt_common import OFFSETS
from information_packet import catalogue,transfer,generation_seed
from grid_codec import endpoints
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(x)for x in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic()
 try:
  design=cr(r['protocol'])['design']['E2_information_budget'];sources=cr(r['sources']);streams=cr(r['token_stream_index']);byid={v['source_id']:v for v in streams}
  assert len(sources)==len(byid)==500
  cat=catalogue();save(out/'information_catalogue.json',cat)
  seed_rule=dict(namespace='NEST_FOLLOWUP_E2_GENERATION_V1',formula="little-endian first 8 SHA256 bytes of namespace|full_source_id|anchor modulo 2**31",anchors=design['Ada_generation_seed_anchors'],budget_independent=True,seeds={s['source_id']:[generation_seed(s['source_id'],a)for a in design['Ada_generation_seed_anchors']]for s in sources})
  save(out/'generation_seed_registration.json',seed_rule)
  logical=[];nest={};ada={};proofs=[]
  for s in sources:
   assert time.monotonic()-start<r['seconds']and not(L/'STOP').exists()
   v=byid[s['source_id']];assert v['source_index']==s['source_index']and v['source_pixels_sha256']==s['preprocessing_id']and v['original_all25_exact']
   with np.load(checked(v['streams']),allow_pickle=False)as z:ns={k:z[k].copy()for k in z.files}
   with np.load(checked(s['ada_streams']),allow_pickle=False)as z:ads={k:z[k].copy()for k in z.files}
   assert len(ns)==651 and len(ads)==256
   for k,a in ns.items():assert len(a)==v['lengths'][k]and asha(a)==v['array_sha256'][k]
   for i in range(1,257):
    a=ads[f'p{i:03d}'];assert len(a)==s['ada_lengths'][f'p{i:03d}']and asha(a)==s['ada_payload_hashes'][f'p{i}']
   for budget in design['B_bits']:
    for method in design['methods']:
     if method=='Ada':candidates=list(range(256,0,-1));bank=ads;capacity=budget-67
     else:candidates=sorted([e for e in endpoints('token')if method!='WHOLE'or e[1]==0],key=lambda e:OFFSETS[e[0]]+e[1],reverse=True);bank=ns;capacity=budget-57
     checks=0;selected=None
     for endpoint in candidates:
      checks+=1;k=f'p{endpoint:03d}'if method=='Ada'else f'm{endpoint[0]}_K{endpoint[1]}';bits=bank[k]
      if len(bits)<=capacity:selected=endpoint;break
     if selected is None:bits=np.zeros(2,np.uint8);profile=cat[-1]
     elif method=='Ada':profile=cat[651+selected-1]
     else:profile=next(p for p in cat if p['codec']=='NEST'and(p['m'],p['K'])==selected)
     projection,proof=transfer(bits,profile,budget,int(s['predicted_class'])if method=='Ada'else None)
     assert proof['padding_bits']>=0;source_key=digest(projection);repeats=range(3)if method=='Ada'else range(1)
     proofs.append(dict(source_id=s['source_id'],method=method,B_bits=budget,length_checks=checks,selected=selected,source_key=source_key,**proof))
     for repeat in repeats:
      seed=seed_rule['seeds'][s['source_id']][repeat]if method=='Ada'else None
      key=digest(dict(projection=projection,generation_seed=seed))if method=='Ada'else source_key
      j=dict(key=key,projection=projection)
      if method=='Ada':j.update(generation_seed=seed,source_key=source_key)
      target=ada if method=='Ada'else nest
      if key in target:assert target[key]==j
      target[key]=j
      logical.append(dict(source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],reference_sha256=s['reference_sha256'],method=method,B_bits=budget,generation_repeat=repeat,generation_seed=seed,key=key,source_key=source_key,endpoint=[profile['m'],profile['K']],source_bits=len(bits),total_overhead=67 if method=='Ada'else 57,padding_bits=proof['padding_bits'],length_checks=checks,receiver_status=projection['status']))
   if(s['source_index']+1)%50==0:print('E2_PLAN',s['source_index']+1,flush=True)
  assert len(logical)==15000 and len(proofs)==9000
  save(out/'nest_jobs.json',list(nest.values()));save(out/'ada_jobs.json',list(ada.values()));save(out/'logical.json',logical);save(out/'packet_proofs.json',proofs)
  csvout(D/'e2/selected_prefixes.csv',proofs)
  save(out/'completion.json',dict(status='COMPLETE',sources=500,logical=15000,units=9000,nest_unique=len(nest),ada_unique=len(ada),new_model_calls=0,new_PHY=0,new_source_traversals=0,source_streams_reused=1000,seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
