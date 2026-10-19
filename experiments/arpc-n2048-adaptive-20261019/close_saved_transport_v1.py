"""Metadata-only closure of saved packet results; never transmits or decodes."""
from shared import *
import pair_protocol as p
import numpy as np
C=L/'qualification_rx_closure_v1'
def close():
 w=read(L/'controls/qualification_rx/actual_wait.json');launch=read(L/'controls/qualification_rx/launch.json')
 assert w['returncode']==1 and gone(w['process']) and gone(launch['parent_process'])
 failed=L/'qualification/rx';incoming=L/'qualification/received';C.mkdir()
 evidence=[pin(failed/'failure.json'),pin(L/'controls/qualification_rx/actual_wait.json'),pin(L/'qualification_constructor_diagnostic_v1/diagnosis.json')]
 diagnosis=read(L/'qualification_constructor_diagnostic_v1/diagnosis.json')
 expected=str(p.OLD_PATH.parent/'__pycache__/pair_protocol.cpython-311.pyc')
 assert [x['path'] for x in diagnosis['blocked']]==[expected] and diagnosis['new_packet_calls']==diagnosis['model_calls']==0
 # Load exact allowed source before the audit hook, eliminating optional .pyc lookup.
 p.legacy();directory=p.directory()
 sys.path.insert(0,str(OLD_E));from a1_native import install_rx_boundary
 blocked=install_rx_boundary([E,OLD_E,p.OLD_PATH,incoming,failed,C,RT/'envs/frozen_ldpc_py311'])
 db=sqlite3.connect('file:'+str(failed/'calls.sqlite')+'?mode=ro',uri=True)
 rows=db.execute('SELECT kind,key,status,error FROM calls ORDER BY kind,key').fetchall();db.close()
 assert len(rows)==96 and all(s=='COMPLETE' and err is None for _,_,s,err in rows)
 assert {k for k,_,_,_ in rows}=={'header','body'}
 jobs=read(incoming/'jobs.json');assert len(jobs)==24
 results=[];sources={};checkpoints=[]
 for j in jobs:
  path=failed/(j['key']+'.json');x=read(path);assert x['key']==j['key'];verify(j['observation'])
  for i,part in enumerate(x['received']):
   for kind in ('header','body'):
    cp=failed/'physical_checkpoints'/(j['key']+'_'+str(i)+'_'+kind+'.json')
    assert read(cp)==part[kind];checkpoints.append(pin(cp))
   q=part['profile'];assert q==directory[str(q['profile_id'])]
   b=part['body'];bits=b['decoded_bits']
   arr=np.asarray(bits,np.uint8)
   assert hashlib.sha256(str(arr.dtype).encode()+str(arr.shape).encode()+arr.tobytes()).hexdigest()==b['hard_bits_sha256']
   try:
    data=p.parse_body(bits,q);assert b['payload_hex']==data.hex() and b['status']=='FRAGMENT_PARSED'
   except p.Reject:assert b['status']=='BODY_REJECT'
  if x['status']=='PAIR_ASSEMBLED':
   v=p.reassemble([(bytes.fromhex(t['body']['payload_hex']),t['profile']) for t in x['received']]);data=v.pop('data');key=p.input_key(v['codec'],v['endpoint'],v['bit_length'],data)
   assert v==x['actual_metadata'] and key==x['source_input_key'] and hashlib.sha256(data).hexdigest()==x['source_object_sha256']
   ip=failed/'source_inputs'/(key+'.bin');assert ip.read_bytes()==data
   sources[key]=dict(key=key,codec=v['codec'],endpoint=v['endpoint'],bit_length=v['bit_length'],input=pin(ip),generation_seed=0)
  else:
   assert x['status']=='PAIR_REJECT'
   try:p.reassemble([(bytes.fromhex(t['body']['payload_hex']),t['profile']) for t in x['received']])
   except (p.Reject,KeyError):pass
   else:raise AssertionError('saved rejection is inconsistent')
  results.append(pin(path))
 assert not blocked
 put(C/'source_jobs.json',list(sources.values()))
 put(C/'completion.json',dict(status='SAVED_PACKET_OUTPUTS_CLOSED_METADATA_ONLY',original_RX_returncode=1,original_failure_preserved=True,original_optional_cache_denial_reproduced=expected,original_blocked_list_not_saved=True,original_packet_calls_complete=96,new_transmit_calls=0,new_decoder_calls=0,new_model_calls=0,blocked_reads=blocked,logical=24,outputs=results,source_jobs=pin(C/'source_jobs.json'),physical_checkpoints=checkpoints,ledger=pin(failed/'calls.sqlite'),failure_evidence=evidence))
def audit(expected):
 registration(expected);w=read(L/'controls/qualification_close_saved/actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 c=read(C/'completion.json');txs=read(L/'qualification/tx/records.json');rows=[]
 for t,d in zip(txs,c['outputs']):
  x=read(verify(d));assert x['key']==t['key'] and x['observed_sha256']==t['observed_sha256'];accepted=x['status']=='PAIR_ASSEMBLED'
  exact=accepted and x['source_object_sha256']==t['source_object_sha256'] and x['source_input_key']==t['source_input_key']
  if t['case']=='clean':assert exact
  elif t['case'] in ('whole_corrupt','profile_mismatch'):assert not accepted
  rows.append(dict(key=x['key'],codec=t['codec'],source_id=t['source_id'],endpoint=t['endpoint'],status=x['status'],accepted=accepted,exact_post_exit=exact,accepted_wrong=accepted and not exact,source_input_key=x['source_input_key'],case=t['case']))
 out=O/'qualification';out.mkdir();put(out/'logical_frames.json',rows)
 put(out/'phy_completion.json',dict(status='QUALIFICATION_TRANSPORT_CLOSED_NATIVE_REVIEW_PENDING',logical=24,subframes=48,header_decodes=48,body_decodes=48,accepted=sum(x['accepted'] for x in rows),accepted_wrong=sum(x['accepted_wrong'] for x in rows),original_RX_returncode=1,closure=pin(C/'completion.json'),new_packet_calls_in_recovery=0,TX_truth_only_post_RX_exit=True,outputs=[pin(out/'logical_frames.json'),pin(C/'source_jobs.json')]))
if __name__=='__main__':
 try:close() if sys.argv[1]=='close' else audit(sys.argv[2])
 except BaseException:
  if C.exists() and not(C/'failure.json').exists():put(C/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False))
  raise
