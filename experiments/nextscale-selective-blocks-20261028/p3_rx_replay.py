"""Exact prior independent-RX memoization, then same-waveform ALL/SEQ/SEL replay."""
from block_common import *
from protocol import OFFSETS,state_identity,replay_states
from audit_prepare_p1 import closed
import collections,traceback

def make_cache():
 """Read only completed independent RX inputs/outputs, never TX source archives."""
 routes=read(O/'reuse_inventory.json')['route'];cache={};provenance=[]
 def put(projection,value,evidence):
  key=digest(projection)
  if key in cache:assert cache[key]['projection']==projection and cache[key]['value']==value
  else:cache[key]=dict(projection=projection,value=value,evidence=evidence)
 for role in ['p1_codec_rx_00','p1_codec_rx_01']:
  c=closed(L,role,L/(role+'_control'));r=cr(c['request']);inputs={j['key']:j['projection']for j in cr(r['inputs'])};proof=pin(L/role/'completion.json');provenance.append(proof)
  for row in cr(c['outputs']):
   p=inputs[row['key']];assert digest(p)==row['key']and not row['source_truth_used']and not row['TX_CDF_used']
   if row['status_C']!='ACCEPTED':continue
   assert len(row['C'])==OFFSETS[p['m']]
   put(dict(type='C',m=p['m'],bits=p['C']),row['C'],dict(completion=proof,key=row['key']))
   for label,pos in [('A',list(range(p['K']//2))),('B',list(range(p['K']//2,p['K'])))]:
    if row['status_'+label]!='ACCEPTED':continue
    put(dict(type='BOUNDARY',m=p['m'],C_bits=p['C'],positions=pos,bits=p[label]),row[label],dict(completion=proof,key=row['key'],block=label))
 normal=read(L/'p1_image_audit/normal_native_states.json');bywire={r['wire_key']:r for r in normal if r.get('wire_key')};assert len(bywire)==800
 found=set()
 # Direct P1 wire projections for the 553 newly completed continuous states.
 for role in ['p1_images_single_00','p1_images_single_01']:
  c=closed(L,role,L/(role+'_control'));req=cr(c['request']);proof=pin(L/role/'completion.json');provenance.append(proof)
  for j in cr(req['inputs']):
   p=j['projection'];key=digest(p);assert key==j['wire_key'];v=bywire[key]['state']
   put(p,dict(m=v['m'],coarse=v['coarse'],positions=v['positions'],values=v['values']),dict(completion=proof,key=key));found.add(key)
 # The 247 old entries are admitted only through original RX job projections.
 old=B/'nestcom_n512_20261028_v1';inv=read(O/'reuse_inventory.json')
 for role,origin in inv['origins'].items():
  c=cr(origin['completion']);req=cr(c['request']);provenance.append(origin['completion'])
  for j in cr(req['inputs']):
   key=j['key']
   if key not in bywire or key in found:continue
   p=j['projection'];assert key==digest(p);v=bywire[key]['state']
   assert bywire[key]['old_record']['canonical']and not bywire[key]['old_record']['source_truth_used']
   put(p,dict(m=v['m'],coarse=v['coarse'],positions=v['positions'],values=v['values']),dict(completion=origin['completion'],key=key));found.add(key)
 assert found==set(bywire)
 preprocess={r['state']['preprocessing']for r in normal};assert len(preprocess)==1
 native={r['key']:dict(state=r['state'],image=r['image'],pixel_sha256=r['pixel_sha256'])for r in normal}
 assert len(native)==928
 for key,r in native.items():assert key==digest(r['state'])and r['state']['route']==routes
 return dict(cache=cache,native=native,route=routes,preprocessing=next(iter(preprocess)),provenance=provenance)

def replay(v,bundle):
 """Only actual parsed RX blocks, public profile, and previously verified RX cache."""
 used=[];missing=[];decisions={}
 def take(projection):
  key=digest(projection);row=bundle['cache'].get(key)
  if row is None:missing.append(dict(key=key,projection=projection));return None
  assert row['projection']==projection;used.append(key);return row['value']
 def finish(states,kind):
  result={}
  for mode,state in states.items():
   if state is None:result[mode]=dict(state='GRAY',state_key=None,pixel_sha256=None);continue
   label,m,C,pos,values=state
   identv=state_identity(m,C,pos,values,bundle['preprocessing'],bundle['route']);key=digest(identv);n=bundle['native'].get(key)
   if n is None:missing.append(dict(type='NATIVE_STATE',key=key,state=identv));result[mode]=dict(state=label,state_key=key,pending=True);continue
   assert n['state']==identv
   result[mode]=dict(state=label,state_key=key,pixel_sha256=n['pixel_sha256'])
  return dict(kind=kind,modes=result,source_status=decisions,reused_decode_keys=used,missing=missing,TX_truth_used=False)
 if not v['header']['header_ok']:
  decisions['header']='REJECTED';return finish(dict(ALL=None,SEQ=None,SEL=None),'UNKNOWN')
 p=v['profile'];kind=p['kind'];decisions['header']='ACCEPTED'
 if p['null']:
  decisions['C']='PAID_NULL';return finish(dict(ALL=None,SEQ=None,SEL=None),kind)
 blocks=v['blocks'];c=blocks[0];decisions['C']=c['status']
 if not c['accepted']:
  decisions.update(A='CONDITION_UNAVAILABLE',B='CONDITION_UNAVAILABLE');return finish(dict(ALL=None,SEQ=None,SEL=None),kind)
 if kind=='SINGLE':
  projection=dict(status='SOURCE_INPUT',codec='NEST',m=p['m'],K=p['K'],family='EC_VAR_PARTIAL'if p['K']else'EC_VAR_WHOLE',bits=c['bits'])
  s=take(projection)
  if s is None:return dict(kind=kind,missing=missing,reused_decode_keys=used,TX_truth_used=False)
  decisions['C']='ACCEPTED_CANONICAL_CACHE'
  item=('SINGLE',s['m'],s['coarse'],s['positions'],s['values']);return finish(dict(ALL=item,SEQ=item,SEL=item),kind)
 C=take(dict(type='C',m=p['m'],bits=c['bits']))
 if C is None:return dict(kind=kind,missing=missing,reused_decode_keys=used,TX_truth_used=False)
 decisions['C']='ACCEPTED_CANONICAL_CACHE';tokens={}
 for gi,label in [(1,'A'),(2,'B')]:
  d=blocks[gi];decisions[label]=d['status']
  if p['K']==0:decisions[label]='DISABLED_PAID_EMPTY';continue
  if not d['accepted']:continue
  pos=p['positions_'+label];values=take(dict(type='BOUNDARY',m=p['m'],C_bits=c['bits'],positions=pos,bits=d['bits']))
  if values is not None:assert len(values)==len(pos);tokens[label]=values;decisions[label]='ACCEPTED_CANONICAL_CACHE'
 if missing:return dict(kind=kind,missing=missing,reused_decode_keys=used,TX_truth_used=False)
 labels=replay_states(True,'A'in tokens,'B'in tokens,p['K']>0);states={}
 for mode,label in labels.items():
  if label=='GRAY':states[mode]=None;continue
  pos=[];vals=[]
  for block in ('A','B'):
   if block in label:pos+=p['positions_'+block];vals+=tokens[block]
  states[mode]=(label,p['m'],C,pos,vals)
 return finish(states,kind)

def main():
 out=L/'p3_replay';out.mkdir(exist_ok=False)
 try:
  bundle=make_cache();save(out/'receiver_cache.json',bundle)
  rows=[];wire={};missing={};receipts=[]
  for i in range(4):
   role=f'p3_phy_{i:02d}';c=closed(L,role,L/(role+'_control'));receipts.append(pin(L/role/'completion.json'))
   for event in cr(c['events']):
    path=event['rx_record']['path']
    if path not in wire:wire[path]=cr(event['rx_record'])
    v=wire[path]['events'][event['rx_index']]
    assert not v['TX_truth_used'];result=replay(v,bundle)
    for m in result['missing']:missing[m['key']]=m
    rows.append(dict(event_id=event['event_id'],result=result))
  # Persist source-decoded states BEFORE joining source IDs or TX audit classifications.
  save(out/'rx_replay.json',rows);save(out/'missing.json',list(missing.values()))
  status='COMPLETE_EXACT_RX_REPLAY'if not missing else'PENDING_NATIVE_RX_CACHE_MISSES'
  save(out/'completion.json',dict(status=status,physical_frames=len(rows),replay=pin(out/'rx_replay.json'),receiver_cache=pin(out/'receiver_cache.json'),missing=pin(out/'missing.json'),missing_count=len(missing),receipts=receipts,independent_RX_cache_entries=len(bundle['cache']),unique_reused_decode_keys=len({k for r in rows for k in r['result']['reused_decode_keys']}),source_model_new_calls=0,render_new_calls=0,TX_truth_used=False))
  print(json.dumps(dict(status=status,frames=len(rows),missing=len(missing),unique_decode_keys=len({k for r in rows for k in r['result']['reused_decode_keys']}))))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
