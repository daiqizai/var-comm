"""Separate actual TX, RX-only packet decoding, and post-exit truth audit."""
from shared import *
import numpy as np
import pair_protocol as p
def tx(phase,expected):
 r=registration(expected);jobs=r['qualification_jobs'];assert phase=='qualification' and len(jobs)==24
 target=L/phase/'tx';target.mkdir();cache(target);incoming=L/phase/'received';incoming.mkdir()
 from pair_link import Runtime,noise,ep
 rt=Runtime();assert rt.catalogue['backend_identity']==r['backend_identity']
 ledger=Ledger(target/'calls.sqlite',dict(transmit=48));pub=[];records=[]
 put(incoming/'catalogue.json',rt.catalogue)
 for j in jobs:
  asset=j['asset'];verify(asset['wire'])
  if j['codec']==0:
   with np.load(asset['wire']['path'],allow_pickle=False)as z:bits=z[asset['array_key']].copy()
   assert hashlib.sha256(bits.tobytes()).hexdigest()==asset['bits_sha256'];data=p.pack_bits(bits)
  else:data=Path(asset['wire']['path']).read_bytes()
  length=asset['bit_length'];assert hashlib.sha256(data).hexdigest()==asset['object_sha256']
  fragments=p.fragment(data,length,j['codec'],asset['endpoint'],j['pair_id'],j['layout_id'])
  if j['case']=='whole_corrupt':
   fragment,q=fragments[1];fragments[1]=(fragment[:-1]+bytes([fragment[-1]^128]),q)
  if j['case']=='profile_mismatch':
   fragment,q=fragments[1];wrong=(asset['endpoint']+1)%(25 if j['codec']==0 else 7)
   if j['codec']==1:wrong=asset['endpoint']%7+1
   fragments[1]=(fragment,p.profile(j['codec'],wrong,j['layout_id'],1))
  frames=[];receipts=[]
  for index,(fragment,q) in enumerate(fragments):
   frame,receipt=ledger.call('transmit',j['key']+'_'+str(index),lambda:rt.transmit(fragment,q['profile_id'],2*j['pair_id']+index));frames.append(frame);receipts.append(receipt)
  frame=np.concatenate(frames);z=noise(j['source_id'],j['noise_seed']) if j['case']=='awgn' else np.zeros((2048,2),np.float64)
  observed=frame+z*10**(-j['snr_db']/20);path=incoming/(j['key']+'.npy')
  with path.open('xb')as f:np.save(f,observed,allow_pickle=False)
  pub.append(dict(key=j['key'],snr_db=j['snr_db'],counters=[2*j['pair_id'],2*j['pair_id']+1],observation=pin(path)))
  records.append(dict(key=j['key'],source_id=j['source_id'],codec=j['codec'],endpoint=asset['endpoint'],case=j['case'],bit_length=length,source_object_sha256=hashlib.sha256(data).hexdigest(),source_input_key=p.input_key(j['codec'],asset['endpoint'],length,data),transmissions=receipts,noise_sha256=ep.array_sha(z),observed_sha256=ep.array_sha(observed)))
 put(incoming/'jobs.json',pub);put(target/'records.json',records)
 put(target/'completion.json',dict(status='COMPLETE',logical=24,subframes=48,ledger=ledger.summary(),records=pin(target/'records.json'),jobs=pin(incoming/'jobs.json')))
def rx(phase):
 assert phase=='qualification';target=L/phase/'rx';target.mkdir();cache(target);incoming=L/phase/'received'
 # Resolve the original source-container implementation before any decoder call.
 sys.path.insert(0,str(OLD_E));sys.path.insert(0,str(OLD_E/'upstream'));import wire
 from pair_link import Runtime,ep,legacy
 from a1_native import install_rx_boundary
 allow=[E,OLD_E,p.OLD_PATH,legacy.OLD,R/'experiments/wcl-evidence-closure-20261009/scripts',R/'experiments/prior-aware-grouped-mcs-v1',R/'experiments/main-raw64-20261007',R/'src',R/'outputs/build',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',RT/'envs/frozen_ldpc_py311',RT/'envs/frozen_python_base_v1/lib/python3.11',RT/'envs/frozen_python_base_v1/lib/python311.zip',RT/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',incoming,target]
 blocked=install_rx_boundary(allow);rt=Runtime();assert rt.catalogue==read(incoming/'catalogue.json');jobs=read(incoming/'jobs.json');assert len(jobs)==24
 rawledger=Ledger(target/'calls.sqlite',dict(header=48,body=48));checkpoints=target/'physical_checkpoints';checkpoints.mkdir()
 class Recorded:
  def call(self,kind,key,fn):
   def saved():
    v=fn();put(checkpoints/(key+'_'+kind+'.json'),v);return v
   return rawledger.call(kind,key,saved)
 ledger=Recorded();outputs=[];sourcejobs={};wires=target/'source_inputs';wires.mkdir()
 for j in jobs:
  verify(j['observation']);y=np.load(j['observation']['path'],allow_pickle=False)
  rs=[rt.receive(y[i*1024:(i+1)*1024],float(j['snr_db']),j['counters'][i],ledger,j['key']+'_'+str(i)) for i in (0,1)]
  result=dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y),source_truth_used=False,status='PAIR_REJECT',source_input_key=None)
  try:
   if any(x['status']!='FRAGMENT_PARSED' for x in rs):raise p.Reject('incomplete or null pair')
   obj=p.reassemble([(bytes.fromhex(x['body']['payload_hex']),x['profile']) for x in rs]);data=obj.pop('data');key=p.input_key(obj['codec'],obj['endpoint'],obj['bit_length'],data)
   if obj['codec']==1:
    try:wire.unpack(data,obj['endpoint'])
    except wire.SourceParseError as ex:raise p.Reject(str(ex)) from ex
   result.update(status='PAIR_ASSEMBLED',source_input_key=key,actual_metadata=obj,source_object_sha256=hashlib.sha256(data).hexdigest())
   if key not in sourcejobs:
    ip=wires/(key+'.bin')
    with ip.open('xb')as f:f.write(data)
    sourcejobs[key]=dict(key=key,codec=obj['codec'],endpoint=obj['endpoint'],bit_length=obj['bit_length'],input=pin(ip),generation_seed=0)
  except p.Reject as ex:result['reason']=str(ex)
  out=target/(j['key']+'.json');put(out,result);outputs.append(pin(out))
 assert not blocked and rawledger.summary()['unresolved']==0
 put(target/'source_jobs.json',list(sourcejobs.values()));put(target/'completion.json',dict(status='COMPLETE',logical=24,outputs=outputs,source_jobs=pin(target/'source_jobs.json'),ledger=rawledger.summary(),blocked_reads=blocked))
def audit(phase,expected):
 r=registration(expected);target=L/phase;txc=read(target/'tx/completion.json');rxc=read(target/'rx/completion.json');txs=read(verify(txc['records']));assert txc['logical']==rxc['logical']==24
 w=read(L/'controls/qualification_rx/actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 rows=[]
 for t,d in zip(txs,rxc['outputs']):
  x=read(verify(d));assert x['key']==t['key'] and x['observed_sha256']==t['observed_sha256'];accepted=x['status']=='PAIR_ASSEMBLED'
  exact=accepted and x['source_object_sha256']==t['source_object_sha256'] and x['source_input_key']==t['source_input_key']
  rows.append(dict(key=x['key'],codec=t['codec'],source_id=t['source_id'],endpoint=t['endpoint'],status=x['status'],accepted=accepted,exact_post_exit=exact,accepted_wrong=accepted and not exact,source_input_key=x['source_input_key'],case=t['case']))
  if t['case']=='clean':assert exact
  elif t['case'] in ('whole_corrupt','profile_mismatch'):assert not accepted
  else:assert t['case']=='awgn'
 out=O/phase;out.mkdir();put(out/'logical_frames.json',rows)
 put(out/'phy_completion.json',dict(status='QUALIFICATION_TRANSPORT_CLOSED_NATIVE_REVIEW_PENDING',logical=24,subframes=48,accepted=sum(x['accepted'] for x in rows),accepted_wrong=sum(x['accepted_wrong'] for x in rows),zero_source_model_calls=True,TX_truth_only_post_RX_exit=True,outputs=[pin(out/'logical_frames.json'),pin(target/'rx/source_jobs.json')]))
if __name__=='__main__':
 try:
  role,phase=sys.argv[1:3]
  if role=='rx':assert len(sys.argv)==3;rx(phase)
  else:globals()[role](phase,sys.argv[3])
 except BaseException:
  target=L/phase/role
  if target.exists() and not(target/'failure.json').exists():put(target/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False))
  raise
