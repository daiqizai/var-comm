"""Post-output truth audit and exact prior RX cache inventory; no neural calls."""
from block_common import *
from protocol import CONFIGS,OFFSETS,endpoints,state_identity

def closed(root,role,control):
 w=read(control/'actual_wait.json');assert w['actual_wait']and w['returncode']==0 and not w['timed_out']and gone(w['process'])and gone(w['owner'])
 c=read(root/role/'completion.json');assert c['ledger']['unresolved']==0
 checked(c['request']);return c

def main():
 out=L/'p1_audit_prepare';out.mkdir(exist_ok=False)
 sources=read(O/'source_manifest.json');streamrows=read(O/'p1_source_delivery.json')['records'];streams={s['source_id']:s for s in streamrows}
 rx={};receipts=[]
 for i in range(2):
  role=f'p1_codec_rx_{i:02d}';c=closed(L,role,L/(role+'_control'));receipts.append(pin(L/(role+'_control')/'actual_wait.json'))
  for r in cr(c['outputs']):assert r['key']not in rx;rx[r['key']]=r
 events=read(O/'p1_codec_events.json');assert len(events)==384
 events_by={(e['source_id'],e['m'],e['K'],e['proof']):rx[e['key']]for e in events}
 routes=cr(read(O/'scope.json')['model_routes']);route=dict(runtime=routes['native_qualification']['numeric_settings']['runtime'],numeric=routes['numeric_flags'],weights=[p['sha256']for p in routes['files']],generation='class1000_greedy_no_CFG_FP32_no_grad_TF32off',device_type='PRO6000_same_type')
 lengths=[];state_events=[];selective_jobs={};truth_audit=[]
 for s in sources:
  with np.load(checked(s['archive']),allow_pickle=False)as z:truth=z['tokens'].copy()
  for m,K in CONFIGS:
   a=events_by[s['source_id'],m,K,'COMPLETE'];b=events_by[s['source_id'],m,K,'DELETE_A'];c=events_by[s['source_id'],m,K,'DELETE_C']
   assert all(a['status_'+q]=='ACCEPTED'for q in('C','A','B'))
   assert b['status_A']=='MISSING'and b['status_B']=='ACCEPTED'and b['status_C']=='ACCEPTED'
   assert a['C']==b['C']==truth[:OFFSETS[m]].tolist()
   assert a['A']==truth[OFFSETS[m]:OFFSETS[m]+K//2].tolist()
   assert a['B']==b['B']==truth[OFFSETS[m]+K//2:OFFSETS[m]+K].tolist()
   assert c['C']is None and c['A']is None and c['B']is None and c['trace']==[]
   assert c['status_B']=='CONDITION_UNAVAILABLE'
   assert a['trace']==b['trace']and all(not r['source_truth_used']and not r['TX_CDF_used']for r in(a,b,c))
   txtrace=cr(streams[s['source_id']]['trace']);assert a['trace']==txtrace[:m+1]
   for label,pos,v in [('C',[],[]),('CA',list(range(K//2)),a['A']),('CB',list(range(K//2,K)),b['B']),('CAB',list(range(K)),a['A']+a['B'])]:
    state=state_identity(m,a['C'],pos,v,s['preprocessing_id'],route);key=digest(state)
    state_events.append(dict(source_id=s['source_id'],source_index=s['source_index'],m=m,K=K,state=label,key=key))
    if label in('CB','CAB'):selective_jobs[key]=dict(key=key,state=state,role='MECHANISM'if label=='CB'else'ALIGNMENT')
   meta=streams[s['source_id']]['lengths'];C=meta['C'+str(m)];A=meta[f'm{m}_K{K}_A'];Bv=meta[f'm{m}_K{K}_B'];single_key=f'm{m+1}_K0'if K in(25,36)else f'm{m}_K{K}';single=s['nest_lengths'][single_key]
   lengths.append(dict(source_id=s['source_id'],source_index=s['source_index'],m=m,K=K,C_bits=C['bits'],A_bits=A['bits'],B_bits=Bv['bits'],independent_sum=C['bits']+A['bits']+Bv['bits'],single_bits=single,AC_difference=C['bits']+A['bits']+Bv['bits']-single,fixed_extra_syntax=58,total_extra_bits=58+C['bits']+A['bits']+Bv['bits']-single))
   truth_audit.append(dict(source_id=s['source_id'],m=m,K=K,complete_exact=True,delete_A_B_exact=True,delete_C_no_context=True,RX_saved_before_audit=True))
 save(out/'codec_truth_audit.json',truth_audit);save(O/'codec_source_proof.json',dict(status='SOURCE_INDEPENDENCE_PASS_RENDER_ALIGNMENT_PENDING',complete=128,delete_A=128,missing_C_no_model=128,source_truth_used_in_RX=False,proof_receipts=receipts,audit=pin(out/'codec_truth_audit.json')))
 csvsave(O/'source_lengths.csv',lengths);save(O/'clean_state_events.json',state_events);save(out/'selective_jobs.json',list(selective_jobs.values()))
 old=B/'nestcom_n512_20261028_v1';cache={};origins={}
 roles=['route_existing25_rx']+[f'{phase}_rx_nest_{i:02d}'for phase in('pilot','full')for i in range(4)]
 for role in roles:
  c=closed(old,role,old/'controls'/role);req=cr(c['request']);native=read(old/role/'actual_native_identity.json')
  assert native['runtime']==route['runtime']and native['numeric_flags']==route['numeric']and native['GPU_name']=='NVIDIA RPBZZZ6'and native['sampling']=='unconditional_class1000_greedy_no_CFG'
  projections={j['key']:j['projection']for j in cr(req['inputs'])}
  for rec in cr(c['outputs']):
   assert rec['key']==digest(projections[rec['key']])
   if rec['status']!='ACCEPTED_SOURCE':continue
   assert rec['canonical']and not rec['source_truth_used']and not rec['TX_CDF_used']
   if rec['key']in cache:assert cache[rec['key']]['record']['pixel_sha256']==rec['pixel_sha256']
   else:cache[rec['key']]=dict(record=rec,projection=projections[rec['key']],origin=role)
  origins[role]=dict(completion=pin(old/role/'completion.json'),wait=pin(old/'controls'/role/'actual_wait.json'),native=pin(old/role/'actual_native_identity.json'))
 new=[];reuse=[];single_events=[]
 for s in sources:
  with np.load(checked(s['nest_streams']),allow_pickle=False)as z:
   for m,K in endpoints():
    projection=dict(status='SOURCE_INPUT',codec='NEST',m=m,K=K,family='EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',bits=z[f'm{m}_K{K}'].tolist());key=digest(projection)
    event=dict(source_id=s['source_id'],source_index=s['source_index'],m=m,K=K,wire_key=key,preprocessing=s['preprocessing_id']);single_events.append(event)
    if key in cache:
     oldrec=cache[key];assert oldrec['projection']==projection;record=oldrec['record']
     with np.load(checked(record['image']),allow_pickle=False)as zi:tokens=zi['received_tokens'].copy();im=zi['image'].copy()
     assert asha(im)==record['pixel_sha256']and asha(tokens)==record['received_tokens_sha256']
     state=state_identity(m,tokens[:OFFSETS[m]].tolist(),list(range(K)),tokens[OFFSETS[m]:].tolist(),s['preprocessing_id'],route)
     reuse.append(dict(**event,key=digest(state),state=state,image=record['image'],pixel_sha256=record['pixel_sha256'],origin=oldrec['origin'],old_record=record))
    else:new.append(dict(wire_key=key,projection=projection,preprocessing=s['preprocessing_id']))
 assert len(new)+len(reuse)==800
 save(out/'single_jobs.json',new);save(out/'single_reuse.json',reuse);save(O/'single_state_events.json',single_events)
 save(O/'reuse_inventory.json',dict(status='EXACT_PRIOR_RX_ADMITTED',origins=origins,reused_single_states=len(reuse),new_single_states=len(new),single_reuse=pin(out/'single_reuse.json'),route=route,normal_state_limit=928,extra_alignment_render_pairs=128,alignment_note='Predeclared technical full-CAB equivalence comparison; not extra nominal clean states or wireless events. Each new renderer call occurs once.'))
 save(out/'completion.json',dict(status='COMPLETE_CPU_AUDIT_PLAN',single_new=len(new),single_reused=len(reuse),selective_new=len(selective_jobs),source_proof=pin(O/'codec_source_proof.json'),route=route))
 print(json.dumps(dict(single_new=len(new),single_reused=len(reuse),selective_new=len(selective_jobs),source_proofs=384)))
if __name__=='__main__':main()
