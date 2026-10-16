"""Bounded new single956 calibration events; identical old PHY math is reused."""
from tb_common import *
import tb_plan as plan
import numpy as np,traceback
def run(expected):
 r,out=stage_start('phy',expected);ledger=None;start=time.monotonic()
 try:
  import ep_phy as phy
  from t2_ledger import Ledger as PacketLedger
  inherit=checkread(r['inheritance']);oldreq=checkread(inherit['requests']['full_phy']);cat=checkread(r['catalogue'])
  rt=phy.create_runtime(R,oldreq['phy_identity']['qualification']['path'])
  assert rt.backend.identity==oldreq['phy_identity']['backend_identity'] and plan.extend_catalogue(rt.catalogue)==cat
  # Append paid IDs only. Original Backend/transmit/receive functions are untouched.
  rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
  assert not rt.torch.cuda.is_initialized()
  ledger=PacketLedger(out/'packets.sqlite',expected,r['caps']['packets'])
  rows=checkread(r['frames']);sources=checkread(r['sources']);images=checkread(r['old_images'])
  (out/'physical').mkdir();(out/'observations').mkdir();cache={};logical=[];rxinputs={};numnew=0;counts={}
  cur=-1;streams=None
  for ix,row in enumerate(rows):
   assert time.monotonic()-start<r['seconds']['phy'] and not(T/'STOP').exists()
   i=row['source_index'];key=row['event_lookup']
   if i!=cur:
    d=sources[i]['current_streams'];checked(d)
    with np.load(d['path'],allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
    cur=i
   if key not in cache:
    bits=streams[f'm{row["m"]}_K{row["K"]}'];assert len(bits)==row['arithmetic_bits'] and asha(bits)==row['payload_sha256']
    wave,tx=rt.transmit(row['profile_id'],bits,row['public_frame_counter'])
    noise=phy.standard_noise(row['source_id'],row['noise_seed'])*10**(-row['snr_db']/20);observed=wave+noise
    if row['old_physical']:
     packet=checkread(row['old_physical']);assert packet['transmission']==tx and packet['noise_sha256']==asha(noise) and packet['observation_sha256']==asha(observed)
     assert plan.directory_unchanged_on_old_event(packet['actual_RX']['header']);rx=packet['actual_RX'];desc=row['old_physical'];origin='CLOSED_EXACT_EVENT_WITH_DIRECTORY_PROOF'
    else:
     obs=out/'observations'/f'{numnew:05d}.npz'
     with obs.open('xb')as f:np.savez(f,observed=observed)
     rx=rt.receive(observed,row['snr_db'],row['public_frame_counter'],rt.profiles,ledger,f'T1B/{numnew:05d}',phase='T1B_original_cal1000')
     packet=dict(schema='T1B_ACTUAL_SINGLE956_EVENT_V1',logical_event=row,transmission=tx,payload_sha256=asha(bits),noise_sha256=asha(noise),observation_sha256=asha(observed),observation=pin(obs),actual_RX=rx,full_public_receive_catalogue=True,source_truth_supplied_to_RX=False,catalogue= r['catalogue'])
     p=out/'physical'/f'{numnew:05d}.json';save(p,packet);desc=pin(p);origin='ACTUAL_NEW';numnew+=1
    sk=source_key(rx);cache[key]=dict(physical_frame=desc,source_input_key=sk,origin=origin)
    if sk not in rxinputs:
     # RX projection has no TX endpoint, truth, source image/token or TX CDF.
     rxinputs[sk]=dict(source_input_key=sk,actual_RX=rx,reuse=images.get(sk))
   item=cache[key];logical.append(dict(frame_index=ix,event=row,**item));counts[item['origin']]=counts.get(item['origin'],0)+1
   if(ix+1)%300==0:progress(out,dict(logical=ix+1,new_physical=numnew,packets=ledger.snapshot()))
  assert numnew==r['caps']['physical'] and len(logical)==36000 and ledger.snapshot()['unresolved']==0
  save(out/'logical_frames.json',logical);save(out/'rx_projection.json',list(rxinputs.values()))
  save(out/'completion.json',dict(status='COMPLETE_T1B_PHY',registration_sha256=expected,logical=36000,actual_new_physical=numnew,reuse_counts=counts,packets=ledger.snapshot(),new_unique_RX_inputs=sum(x['reuse']is None for x in rxinputs.values()),RX_inputs=len(rxinputs),logical_frames=pin(out/'logical_frames.json'),rx_projection=pin(out/'rx_projection.json'),seconds=time.monotonic()-start,CUDA_initialized=rt.torch.cuda.is_initialized()))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.snapshot(),automatic_retry=False));raise
 finally:
  if ledger is not None:ledger.close()
if __name__=='__main__':run(sys.argv[1])
