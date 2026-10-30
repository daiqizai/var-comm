"""Frozen BPG/JSCC with three registered outcomes; isolated receive processes."""
from g import *
import base64,tempfile,shutil
from contextlib import contextmanager
def main(path,expected):
 r,o=initialize(path,expected);cfg=cr(r['timing_config']);policy=cr(r['policy']);method=r['method'];phase=r['phase'];jobs=cr(r['inputs']);cfg['_tmpfs']=str(o/'tmp')
 cfg['resource']=dict(cfg['resource'],gpu_index=r.get('gpu_index',1),gpu_uuid=r.get('gpu_uuid'),gpu_pci=r.get('pci'))
 assert method in ('BPG','P100000','Swin200000');assert len(jobs)==(500 if phase=='tx' or method=='BPG'and phase=='phy'else 1500)
 beg=time.monotonic();ledger=Ledger(o/'calls.sqlite',dict(tx=500,channel=1500,phy=1500,header=1500,body=1500,rx=1500));rows=[]
 try:
  sys.path.insert(0,str(R/'experiments/five-method-timing-20261020/library_recovery_v1'))
  from adapter import adapted_worker
  import native
  w=adapted_worker();blocked=[];c=None;calls=native.Calls()
  if phase=='rx':
   for j in jobs:
    if method!='BPG':
     with np.load(checked(j['observation']),allow_pickle=False)as z:j['_observed']=z['observed'].copy()
   blocked=w.firewall(cfg,method,o)
  if method!='BPG':c,calls=native.continuous(cfg,o,method)
  elif phase=='phy':c=w.phy_runtime(cfg,'BPG',o)
  else:
   import bpg
   bpg.legacy()
   if phase=='tx':
    shim=B/'five_method_timing_20261020_v1/bpg_threads_qualification_v1/affinity_cpu_count.so';assert sha(shim)=='953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'
    def codec_environment(argv):
     if argv[0]!=cfg['bpg_config']['bpgenc']['path']:return None
     assert not os.environ.get('LD_PRELOAD') and len(os.sched_getaffinity(0))==6
     return dict(os.environ,LD_PRELOAD=str(shim))
    bpg.codec_environment=codec_environment;bpg.run=native.replace_function(bpg,'run',[('subprocess.run(argv, capture_output=True, timeout=120)','subprocess.run(argv, capture_output=True, timeout=120, env=codec_environment(argv))')])
   @contextmanager
   def workspace(root):
    p=Path(tempfile.mkdtemp(dir=root))
    try:yield str(p)
    except BaseException:raise
    else:shutil.rmtree(p)
   bpg.workspace=workspace
  class Packets:
   def decode(self,phase,key,kind,request,fn):return ledger.call(kind,str(key),fn)
   def call(self,event,request,fn):return ledger.call(event['kind'],str(event['event_id']),fn)
  received={}
  for j in jobs:
   assert time.monotonic()-beg<r['max_seconds'] and not(L/'STOP').exists()
   sid=j['source_id'];idx=j['source_index'];meter=w.Clock(c.t if method!='BPG'else None)
   if phase=='tx':
    def encode():
     with np.load(checked(j['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
     assert hashlib.sha256(pixels.tobytes()).hexdigest()==j['preprocessing_id']
     if method=='BPG':
      pid=policy['methods']['BPG']['profile_id'];p=next(x for x in cfg['bpg_config']['catalogue']if x['profile_id']==pid)
      payload,evidence=bpg.encode(pixels,cfg['bpg_config'],p['capacity_bytes'],cfg['_tmpfs']);assert evidence['candidate_encodes']<=208
      v=dict(source_id=sid,source_index=idx,payload_b64=None if payload is None else base64.b64encode(payload).decode(),profile_id=pid,evidence=evidence)
      save(o/('source_%04d.json'%idx),v);return v
     wave,witness=c.encode(pixels,7,26000000+16384*idx,meter);assert wave.shape==(1024,2)
     return wave
    tx=ledger.call('tx',sid,encode)
    if method=='BPG':rows.append(tx);continue
    for rep in range(3):
     key='s7_i%04d_r%d'%(idx,rep+1);seed=2001+rep
     def channel():
      if method=='P100000':noise=c.seeded_noise('VAR-CONTINUOUS-1024|'+sid,seed,(1024,2)).astype(np.float32);obs=tx+noise*np.float32(10**(-7/20))
      else:noise=c.protocol.standard_noise(sid,seed,1024,7);obs=tx+noise/10**(7/20)
      assert obs.shape==(1024,2)and np.isfinite(obs).all();p=o/(key+'.npz')
      with p.open('xb')as f:np.savez(f,observed=obs);f.flush();os.fsync(f.fileno())
      v=dict(key=key,source_id=sid,source_index=idx,method=method,N=1024,snr_db=7,repeat=rep+1,noise_seed=seed,counter=26000000+16384*idx+4096*rep,observation=pin(p),observation_sha256=asha(obs),waveform_sha256=asha(tx),noise_sha256=asha(noise),energy=float(np.square(tx.astype(np.float64)).sum()))
      save(o/(key+'.json'),v);return v
     rows.append(ledger.call('channel',key,channel))
   elif phase=='phy':
    for rep in range(3):
     key='s7_i%04d_r%d'%(idx,rep+1);counter=26000000+16384*idx+4096*rep;seed=2001+rep
     def physical():
      if j['payload_b64']is None:actual=dict(accepted=False,status='SOURCE_UNFIT',payload_b64=None);evidence=dict(no_frame=True)
      else:
       p=next(p for p in cfg['bpg_config']['catalogue']if p['profile_id']==j['profile_id']);payload=base64.b64decode(j['payload_b64']);assert len(payload)<=p['capacity_bytes']
       wave=c.transmit(payload,p,counter);obs=c.noisy(wave,'COMMON500_7DB',7,idx,seed);rx=c.receive(obs,7,counter,Packets(),'formal_7db',key);body=rx['body'];accepted=bool(rx['header']['header_ok']and body and body['crc_accepted']and body['parser_accepted'])
       actual=dict(accepted=accepted,status=rx['status'],payload_b64=body['payload_b64']if accepted else None)
       assert wave.shape==obs.shape==(1024,2);evidence=dict(no_frame=False,profile=p,source_bytes=len(payload),waveform_sha256=asha(wave),observation_sha256=asha(obs),energy=float(np.square(wave.astype(np.float64)).sum()),header=rx['header'],body=body)
      v=dict(key=key,source_id=sid,source_index=idx,method=method,N=1024,snr_db=7,repeat=rep+1,noise_seed=seed,counter=counter,actual=actual,evidence=evidence);save(o/(key+'.json'),v);return v
     rows.append(ledger.call('phy',key,physical))
   else:
    key=j['key'];counter=j['counter'];reusekey=digest(j['actual'])if method=='BPG'else j['observation_sha256']
    if reusekey in received:record=dict(received[reusekey],origin='EXACT_RECEIVED_INPUT_REUSE')
    else:
     def receive():
      if method=='BPG':image,status=w.native_rx(j['actual'],cfg,'BPG',None,meter)
      else:image,actual=c.receive(j['_observed'],7,counter,meter,key);status=actual['status']
      assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
      p=o/(reusekey+'.npz')
      with p.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
      return dict(image=pin(p),pixel_sha256=asha(image),status=status,source_truth_used=False,origin='NEW_RX')
     record=ledger.call('rx',reusekey,receive);received[reusekey]=record
    v=dict(record,key=key,source_id=sid,source_index=idx,method=method,N=1024,snr_db=7,repeat=j['repeat'],noise_seed=j['noise_seed']);save(o/(key+'.json'),v);rows.append(v)
   if (idx+1)%50==0:print(method,phase,idx+1,len(rows),flush=True)
  assert not blocked and ledger.summary()['unresolved']==0
  save(o/'outputs.json',rows);save(o/'completion.json',dict(status='COMPLETE',outputs=pin(o/'outputs.json'),count=len(rows),ledger=ledger.summary(),calls=calls.counts,seconds=time.monotonic()-beg))
 except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary()));raise
if __name__=='__main__':main(*sys.argv[1:])
