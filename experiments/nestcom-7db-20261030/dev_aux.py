"""Two fixed development events for continuous JSCC and adaptive BPG.
TX, physical reception and RX use independent registered processes.
"""
from g import *
import base64

def main(path,expected):
 r,o=initialize(path,expected);cfg=cr(r['timing_config']);policy=cr(r['policy']);method=r['method'];phase=r['phase'];jobs=cr(r['inputs']);assert len(jobs)==2
 assert r['snr_db']==7 and policy['status']=='FROZEN_BEFORE_DEVELOPMENT'
 cfg['_tmpfs']=str(o/'tmp');beg=time.monotonic();ledger=Ledger(o/'calls.sqlite',{phase:2});rows=[]
 try:
  sys.path.insert(0,str(R/'experiments/five-method-timing-20261020/library_recovery_v1'))
  from adapter import adapted_worker
  import native
  w=adapted_worker();blocked=[];c=None;calls=native.Calls()
  if phase=='rx':
   # Read only the registered received object; no TX source/tokens enter RX.
   for j in jobs:
    if method!='BPG':
     with np.load(checked(j['observation']),allow_pickle=False)as z:j['_observed']=z['observed'].copy()
   blocked=w.firewall(cfg,method,o)
  if method in ('P100000','Swin200000'):c,calls=native.continuous(cfg,o,method)
  elif phase=='phy':c=w.phy_runtime(cfg,'BPG',o)
  else:
   import bpg
   bpg.legacy()
   if phase=='tx':
    shim=B/'five_method_timing_20261020_v1/bpg_threads_qualification_v1/affinity_cpu_count.so'
    assert sha(shim)=='953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'
    def codec_environment(argv):
     if argv[0]!=cfg['bpg_config']['bpgenc']['path']:return None
     assert not os.environ.get('LD_PRELOAD') and sorted(os.sched_getaffinity(0))==list(range(6))
     return dict(os.environ,LD_PRELOAD=str(shim))
    bpg.codec_environment=codec_environment
    bpg.run=native.replace_function(bpg,'run',[('subprocess.run(argv, capture_output=True, timeout=120)','subprocess.run(argv, capture_output=True, timeout=120, env=codec_environment(argv))')])
   # Failed CLI evidence stays under this task, never the historical timing run.
   from contextlib import contextmanager
   import tempfile,shutil
   @contextmanager
   def workspace(root):
    p=Path(tempfile.mkdtemp(dir=root))
    try:yield str(p)
    except BaseException:raise
    else:shutil.rmtree(p)
   bpg.workspace=workspace
  for j in jobs:
   assert time.monotonic()-beg<r['max_seconds'] and not(L/'STOP').exists()
   sid=j['source_id'];idx=j['source_index'];key='s7_i%04d'%idx;counter=24000000+idx
   meter=w.Clock(c.t if method in ('P100000','Swin200000')else None)
   def compute():
    if phase=='tx':
     with np.load(checked(j['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
     assert hashlib.sha256(pixels.tobytes()).hexdigest()==j['preprocessing_id']
     if method=='BPG':
      pid=policy['methods']['BPG']['profile_id'];p=next(x for x in cfg['bpg_config']['catalogue']if x['profile_id']==pid)
      payload,evidence=bpg.encode(pixels,cfg['bpg_config'],p['capacity_bytes'],cfg['_tmpfs'])
      assert evidence['candidate_encodes']<=208
      result=dict(payload_b64=None if payload is None else base64.b64encode(payload).decode(),profile_id=pid,evidence=evidence)
     else:
      wave,witness=c.encode(pixels,7,counter,meter);observed=c.noise(wave,sid,7)
      assert wave.shape==observed.shape==(1024,2)
      p=o/(key+'.npz')
      with p.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
      result=dict(observation=pin(p),waveform_sha256=asha(wave),observation_sha256=asha(observed),energy=float(np.square(wave.astype(np.float64)).sum()),noise_seed=2001)
    elif phase=='phy':
     actual,evidence=w.phy_call(j,cfg,'BPG',dict(snr_db=7,counter=counter,source_id=sid,source_index=idx,key=key),c,calls,meter)
     assert not evidence.get('no_frame',False),'No BPG fit: preserve diagnostic; do not invent a frame'
     result=dict(actual=actual,evidence=evidence,noise_seed=2001)
    else:
     if method=='BPG':image,status=w.native_rx(j['actual'],cfg,'BPG',None,meter)
     else:image,actual=c.receive(j['_observed'],7,counter,meter,key);status=actual['status']
     assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
     p=o/(key+'.npz')
     with p.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
     result=dict(image=pin(p),pixel_sha256=asha(image),status=status,source_truth_used=False)
    result.update(source_id=sid,source_index=idx,method=method,N=1024,snr_db=7,counter=counter)
    save(o/(key+'.json'),result);return result
   rows.append(ledger.call(phase,sid,compute));print(method,phase,idx,flush=True)
  assert not blocked and ledger.summary()['unresolved']==0
  save(o/'outputs.json',rows);save(o/'completion.json',dict(status='COMPLETE',outputs=pin(o/'outputs.json'),count=2,ledger=ledger.summary(),calls=calls.counts,seconds=time.monotonic()-beg))
 except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary()));raise
if __name__=='__main__':main(*sys.argv[1:])
