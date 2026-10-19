"""Proposed receiver-only continuation. Requires an explicit new authorization.

No TX replay. Preserve original failure and reuse q0000/q0001. q0002 has four
completed decoder calls without a persisted result, and must be disclosed as
four repeated calls before continuation. Original bound source is untouched.
"""
import sys
from pathlib import Path
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-n2048-validation-20261019')
sys.path.insert(0,str(E))
from shared import *
import pair_protocol as p

def environment():
 # This path was omitted by the original RX entry. Resolve it before any packet.
 sys.path.insert(0,str(OLD_E));sys.path.insert(0,str(OLD_E/'upstream'))
 import wire
 assert Path(wire.__file__).resolve()==OLD_E/'wire.py'
 return wire

def run(authorization,expected):
 auth=read(authorization);assert sha(authorization)==expected
 assert auth['schema']=='N2048_RX_IMPORT_CONTINUATION_AUTHORIZATION_V1'
 assert auth['human_approved_four_repeated_decoder_calls'] is True
 assert auth['qualification_decoder_cap']==100 and auth['total_decoder_cap']==7300
 assert auth['reused_completed_keys']==['q0000','q0001'] and auth['repeated_decoder_keys']==['q0002_0','q0002_1']
 assert auth['new_transmissions']==0 and auth['new_TX_model_passes']==0
 for d in auth['bindings']:verify(d)
 assert pin(__file__) in auth['bindings']
 oldwait=read(L/'controls/qualification_rx/actual_wait.json');assert oldwait['returncode']==1 and gone(oldwait['process'])
 assert gone(read(L/'controls/qualification_rx/supervisor.json')['process'])
 # All module qualification happens before any new scientific reservation.
 wire=environment()
 import numpy as np
 from pair_link import Runtime,ep,legacy
 from a1_native import install_rx_boundary
 target=L/'qualification/rx_import_recovery_v1';target.mkdir();cache(target);incoming=L/'qualification/received'
 reuse={d['key']:d for d in auth['completed_RX_records']};assert set(reuse)==set(auth['reused_completed_keys'])
 allowed=[E,OLD_E,legacy.OLD,R/'experiments/wcl-evidence-closure-20261009/scripts',R/'experiments/prior-aware-grouped-mcs-v1',R/'experiments/main-raw64-20261007',R/'src',R/'outputs/build',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',RT/'envs/frozen_ldpc_py311',RT/'envs/frozen_python_base_v1/lib/python3.11',RT/'envs/frozen_python_base_v1/lib/python311.zip',RT/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',incoming,target]
 allowed += [Path(x['record']['path']) for x in reuse.values()] + [Path(x['source_input']['path']) for x in reuse.values() if x.get('source_input')]
 blocked=install_rx_boundary(allowed);rt=Runtime();assert rt.catalogue==read(incoming/'catalogue.json');jobs=read(incoming/'jobs.json');assert len(jobs)==24
 ledger=Ledger(target/'calls.sqlite',dict(header=44,body=44));outputs=[];sourcejobs={};wires=target/'source_inputs';wires.mkdir();checkpoints=target/'physical_checkpoints';checkpoints.mkdir()
 for j in jobs:
  if j['key'] in reuse:
   old=reuse[j['key']];result=read(verify(old['record']));assert result['key']==j['key']
   if old.get('source_input'):
    data=verify(old['source_input']).read_bytes();key=result['source_input_key'];m=result['actual_metadata'];assert p.input_key(m['codec'],m['bit_length'],data)==key
    ip=wires/(key+'.bin');ip.open('xb').write(data);sourcejobs[key]=dict(key=key,codec=m['codec'],endpoint=m['endpoint'],bit_length=m['bit_length'],input=pin(ip),generation_seed=0)
  else:
   verify(j['observation']);y=np.load(j['observation']['path'],allow_pickle=False)
   rs=[rt.receive(y[i*1024:(i+1)*1024],float(j['snr_db']),j['counters'][i],ledger,j['key']+'_'+str(i)) for i in (0,1)]
   put(checkpoints/(j['key']+'.json'),dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y)))
   result=dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y),source_truth_used=False,status='PAIR_REJECT',source_input_key=None)
   try:
    if any(x['status']!='FRAGMENT_PARSED' for x in rs):raise p.Reject('incomplete or null pair')
    obj=p.reassemble([(bytes.fromhex(x['body']['payload_hex']),x['profile']) for x in rs]);data=obj.pop('data');key=p.input_key(obj['codec'],obj['bit_length'],data)
    if obj['codec']==1:
     try:wire.unpack(data,7)
     except wire.SourceParseError as ex:raise p.Reject(str(ex))
    result.update(status='PAIR_ASSEMBLED',source_input_key=key,actual_metadata=obj,source_object_sha256=hashlib.sha256(data).hexdigest())
    if key not in sourcejobs:
     ip=wires/(key+'.bin');ip.open('xb').write(data);sourcejobs[key]=dict(key=key,codec=obj['codec'],endpoint=obj['endpoint'],bit_length=obj['bit_length'],input=pin(ip),generation_seed=0)
   except p.Reject as ex:result['reason']=str(ex)
  out=target/(j['key']+'.json');put(out,result);outputs.append(pin(out))
 assert not blocked and ledger.summary()['unresolved']==0
 put(target/'source_jobs.json',list(sourcejobs.values()))
 put(target/'completion.json',dict(status='COMPLETE',logical=24,outputs=outputs,source_jobs=pin(target/'source_jobs.json'),ledger=ledger.summary(),blocked_reads=blocked,original_attempt=auth['original_attempt'],old_decoder_calls=12,maximum_combined_decoder_calls=100,completed_pairs_reused=2,repeated_decoder_calls=4,new_transmissions=0))
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
