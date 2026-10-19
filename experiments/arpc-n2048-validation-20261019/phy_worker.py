"""Independent TX observation writer, RX decoder and post-exit truth auditor."""
from shared import *
import numpy as np
import pair_protocol as p
def tx(phase,expected):
 r=registration(expected);target=L/phase/'tx';target.mkdir();cache(target);incoming=L/phase/'received';incoming.mkdir()
 from pair_link import Runtime,noise,ep
 rt=Runtime();assert rt.catalogue['backend_identity']==r['backend_identity']
 jobs=r['qualification_jobs'] if phase=='qualification' else r['formal_jobs']
 if phase=='formal':assert read(O/'qualification_review.json')['status']=='PASS'
 ledger=Ledger(target/'calls.sqlite',dict(transmit=2*len(jobs)));pub=[];records=[]
 put(incoming/'catalogue.json',rt.catalogue)
 for j in jobs:
  asset=j['asset'];verify(asset['wire'])
  if j['codec']==0:
   with np.load(asset['wire']['path'],allow_pickle=False) as z:bits=z['m10_K0'].copy()
   assert bits.dtype==np.uint8 and len(bits)==asset['payload_bits'] and hashlib.sha256(bits.tobytes()).hexdigest()==asset['raw_bit_bytes_sha256'];data=p.pack_bits(bits)
  else:data=Path(asset['wire']['path']).read_bytes()
  length=asset['payload_bits'];p.validate_object(data,length,j['codec']);fragments=p.fragment(data,length,j['codec'],j['pair_id'])
  if j['case']=='corrupt':
   b=bytearray(fragments[1]);b[-1]^=128;fragments[1]=bytes(b)
  frames=[];receipts=[]
  for index,fragment in enumerate(fragments):
   pid=2*j['codec']+index;counter=2*j['pair_id']+index
   frame,receipt=ledger.call('transmit',j['key']+'_'+str(index),lambda:rt.transmit(fragment,pid,counter));frames.append(frame);receipts.append(receipt)
  frame=np.concatenate(frames);z=noise(j['source_id'],j['noise_seed']) if phase=='formal' else np.zeros((2048,2),np.float64)
  observed=frame+z*10**(-j['snr_db']/20);path=incoming/(j['key']+'.npy')
  with path.open('xb') as f:np.save(f,observed,allow_pickle=False)
  pub.append(dict(key=j['key'],snr_db=j['snr_db'],counters=[2*j['pair_id'],2*j['pair_id']+1],observation=pin(path)))
  records.append(dict(key=j['key'],source_id=j['source_id'],codec=j['codec'],case=j['case'],bit_length=length,source_object_sha256=hashlib.sha256(data).hexdigest(),source_input_key=p.input_key(j['codec'],length,data),transmissions=receipts,noise_sha256=ep.array_sha(z),observed_sha256=ep.array_sha(observed)))
 put(incoming/'jobs.json',pub);put(target/'records.json',records)
 assert ledger.summary()['unresolved']==0
 put(target/'completion.json',dict(status='COMPLETE',logical=len(jobs),subframes=2*len(jobs),ledger=ledger.summary(),records=pin(target/'records.json'),jobs=pin(incoming/'jobs.json')))
def rx(phase):
 target=L/phase/'rx';target.mkdir();cache(target);incoming=L/phase/'received'
 from pair_link import Runtime,ep,legacy
 sys.path.insert(0,str(OLD_E));from a1_native import install_rx_boundary
 allow=[E,OLD_E,legacy.OLD,R/'experiments/wcl-evidence-closure-20261009/scripts',R/'experiments/prior-aware-grouped-mcs-v1',R/'experiments/main-raw64-20261007',R/'src',R/'outputs/build',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',RT/'envs/frozen_ldpc_py311',RT/'envs/frozen_python_base_v1/lib/python3.11',RT/'envs/frozen_python_base_v1/lib/python311.zip',RT/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',incoming,target]
 blocked=install_rx_boundary(allow);rt=Runtime();assert rt.catalogue==read(incoming/'catalogue.json');jobs=read(incoming/'jobs.json')
 count=24 if phase=='qualification' else 1800;assert len(jobs)==count
 ledger=Ledger(target/'calls.sqlite',dict(header=2*count,body=2*count));outputs=[];sourcejobs={};wires=target/'source_inputs';wires.mkdir()
 for j in jobs:
  verify(j['observation']);y=np.load(j['observation']['path'],allow_pickle=False)
  rs=[rt.receive(y[i*1024:(i+1)*1024],float(j['snr_db']),j['counters'][i],ledger,j['key']+'_'+str(i)) for i in (0,1)]
  result=dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y),source_truth_used=False,status='PAIR_REJECT',source_input_key=None)
  try:
   if any(x['status']!='FRAGMENT_PARSED' for x in rs):raise p.Reject('incomplete or null pair')
   obj=p.reassemble([(bytes.fromhex(x['body']['payload_hex']),x['profile']) for x in rs]);data=obj.pop('data');key=p.input_key(obj['codec'],obj['bit_length'],data)
   # Actual wire syntax is checked before scheduling any neural decoder.
   if obj['codec']==1:
    import wire
    try:wire.unpack(data,7)
    except wire.SourceParseError as ex:raise p.Reject(str(ex))
   result.update(status='PAIR_ASSEMBLED',source_input_key=key,actual_metadata=obj,source_object_sha256=hashlib.sha256(data).hexdigest())
   if key not in sourcejobs:
    ip=wires/(key+'.bin');ip.open('xb').write(data);sourcejobs[key]=dict(key=key,codec=obj['codec'],endpoint=obj['endpoint'],bit_length=obj['bit_length'],input=pin(ip),generation_seed=0)
  except p.Reject as e:result['reason']=str(e)
  out=target/(j['key']+'.json');put(out,result);outputs.append(pin(out))
  if len(outputs)%50==0:print(json.dumps(dict(completed=len(outputs),total=count)),flush=True)
 assert not blocked and ledger.summary()['unresolved']==0
 put(target/'source_jobs.json',list(sourcejobs.values()));put(target/'completion.json',dict(status='COMPLETE',logical=count,outputs=outputs,source_jobs=pin(target/'source_jobs.json'),ledger=ledger.summary(),blocked_reads=blocked))
def audit(phase,expected):
 r=registration(expected);target=L/phase;txc=read(target/'tx/completion.json');rxc=read(target/'rx/completion.json');txs=read(verify(txc['records']));assert txc['logical']==rxc['logical']
 rows=[]
 for t,d in zip(txs,rxc['outputs']):
  x=read(verify(d));assert x['key']==t['key'] and x['observed_sha256']==t['observed_sha256'];accepted=x['status']=='PAIR_ASSEMBLED'
  exact=accepted and x['source_object_sha256']==t['source_object_sha256'] and x['source_input_key']==t['source_input_key']
  rows.append(dict(key=x['key'],codec=t['codec'],source_id=t['source_id'],status=x['status'],accepted=accepted,exact_post_exit=exact,accepted_wrong=accepted and not exact,source_input_key=x['source_input_key'],case=t['case']))
  if phase=='qualification':assert (t['case']=='clean' and exact) or (t['case']=='corrupt' and not accepted)
 out=O/phase;out.mkdir(exist_ok=False);put(out/'logical_frames.json',rows)
 put(out/'phy_completion.json',dict(status='PASS_QUALIFICATION_PHY' if phase=='qualification' else 'FORMAL_PHY_COMPLETE',logical=len(rows),subframes=txc['subframes'],accepted=sum(x['accepted'] for x in rows),accepted_wrong=sum(x['accepted_wrong'] for x in rows),new_TX_probability_passes=0,TX_truth_only_post_RX_exit=True,outputs=[pin(out/'logical_frames.json'),pin(target/'rx/source_jobs.json')]))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('role',choices=['tx','rx','audit']);a.add_argument('phase',choices=['qualification','formal']);a.add_argument('--expected');s=a.parse_args()
 if s.role=='rx':assert s.expected is None;rx(s.phase)
 else:globals()[s.role](s.phase,s.expected)
