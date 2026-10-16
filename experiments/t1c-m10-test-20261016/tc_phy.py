"""Only changed 19 dB test transmissions; unchanged events get byte-level waveform proof."""
from tc_common import *
import numpy as np,traceback
import tb_plan as plan
from tc_protocol import validate_grid
def run(expected):
 r=registration('link',expected);out=T/'phy';out.mkdir();start=time.monotonic();ledger=None
 try:
  import ep_phy as phy
  from t2_ledger import Ledger as PacketLedger
  rt=phy.create_runtime(R,r['phy_qualification']['path']);cat=checkread(r['catalogue']);assert plan.extend_catalogue(rt.catalogue)==cat and rt.backend.identity==r['backend_identity']
  rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};assert not rt.torch.cuda.is_initialized()
  ledger=PacketLedger(out/'packets.sqlite',expected,r['caps']['packets']);sources=checkread(r['sources']);rows=checkread(r['frames']);cache={};logical=[];projection={};num=0;proofs=[]
  assert validate_grid(rows)==r['exact_grid']
  (out/'physical').mkdir();(out/'observations').mkdir();current=-1
  for row in rows:
   assert time.monotonic()-start<r['max_seconds'] and not(T/'STOP').exists();j=row['source_global_index'];key=row['event_key']
   if j!=current:
    checked(sources[j]['current_streams'])
    with np.load(sources[j]['current_streams']['path'],allow_pickle=False) as z:streams={k:z[k].copy() for k in z.files}
    current=j
   if key not in cache:
    bits=streams[f'm{row["m"]}_K{row["K"]}'];assert len(bits)==row['arithmetic_bits'] and asha(bits)==row['payload_sha256']
    wave,tx=rt.transmit(row['profile_id'],bits,row['public_frame_counter']);standard=phy.standard_noise(row['source_id'],row['noise_seed']);noise=standard*10**(-19/20);observed=wave+noise
    if row['reuse']:
     old=checkread(row['execution_reference']['physical_frame']);ident=old.get('observation_identity',old);oldtx=old['transmission']
     assert tx['transmitted_frame_sha256']==oldtx['transmitted_frame_sha256'] and asha(observed)==ident['observation_sha256']
     if 'noise_sha256' in ident:assert asha(noise)==ident['noise_sha256']
     if 'standard_noise_sha256' in ident:assert asha(standard)==ident['standard_noise_sha256']
     assert plan.directory_unchanged_on_old_event(old['actual_RX']['header']);pd=row['execution_reference']['physical_frame'];rx=old['actual_RX'];origin='EXACT_OLD_WAVE_OBSERVATION_AND_RECEIVE_DOMAIN'
     proofs.append(dict(frame_index=row['frame_index'],old_physical=pd,waveform_sha256=asha(wave),observation_sha256=asha(observed),paid_profile_key=tx['profile_key'],header_directory_equivalent=True))
    else:
     obs=out/'observations'/f'{num:04d}.npz'
     with obs.open('xb') as f:np.savez(f,observed=observed)
     rx=rt.receive(observed,19,row['public_frame_counter'],rt.profiles,ledger,f'T1C/{num:04d}',phase='T1C_19dB_test_only')
     p=out/'physical'/f'{num:04d}.json';save(p,dict(schema='T1C_ACTUAL_CHANGED_EVENT_V1',event={k:v for k,v in row.items() if k!='legacy'},transmission=tx,payload_sha256=asha(bits),standard_noise_sha256=asha(standard),noise_sha256=asha(noise),waveform_sha256=asha(wave),observation_sha256=asha(observed),observation=pin(obs),actual_RX=rx,catalogue=r['catalogue'],source_truth_supplied_to_RX=False));pd=pin(p);origin='ACTUAL_NEW';num+=1
     sk=source_key(rx);projection.setdefault(sk,dict(source_input_key=sk,actual_RX=rx))
    cache[key]=dict(physical_frame=pd,source_input_key=source_key(rx),origin=origin)
   event=cache[key]
   # A shared new event may serve both families. An unchanged logical output always retains its own historical image.
   logical.append(dict(event=row,**event))
   if len(logical)%100==0:progress(out,dict(logical=len(logical),new_physical=num,packets=ledger.snapshot()))
  assert len(logical)==3600 and num==r['caps']['physical'] and ledger.snapshot()['unresolved']==0
  save(out/'logical_frames.json',logical);save(out/'rx_projection.json',list(projection.values()));save(out/'old_event_waveform_proofs.json',proofs)
  save(out/'completion.json',dict(status='COMPLETE_T1C_PHY',registration_sha256=expected,logical_frames=3600,new_physical=num,new_packets=ledger.snapshot(),unique_new_RX_inputs=len(projection),logical=pin(out/'logical_frames.json'),rx_projection=pin(out/'rx_projection.json'),reuse_proofs=pin(out/'old_event_waveform_proofs.json'),seconds=time.monotonic()-start,CUDA_initialized=rt.torch.cuda.is_initialized()))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.snapshot(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()
if __name__=='__main__':run(sys.argv[1])
