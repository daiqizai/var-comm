"""Current paid N1024 PHY; families share standard noise, not observations."""
from kn_common import *
import numpy as np,traceback
def run(expected):
 r,out=start('phy',expected);beg=time.monotonic();packet_ledger=None
 try:
  source_done=closed('source',expected);sources=cr(source_done['sources'])
  import ep_phy as phy
  import tb_plan
  from t2_ledger import Ledger as PacketLedger
  from kn_native import choose
  rt=phy.create_runtime(R,checked(r['phy_qualification']));cat=cr(r['catalogue'])
  assert tb_plan.extend_catalogue(rt.catalogue)==cat and rt.backend.identity==r['backend_identity']
  rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};assert not rt.torch.cuda.is_initialized()
  profiles={(p['m'],p['K'],p['q'],p['nominal_rate']):p for p in cat['profiles']if p['family']!='EC_STATIC_WHOLE'}
  packet_ledger=PacketLedger(out/'packets.sqlite',expected,864);logical=[];projection={};events={};physical=0
  (out/'physical').mkdir();(out/'observations').mkdir()
  for s in sources:
   with np.load(checked(s['streams']),allow_pickle=False)as z:streams={(int(k.split('_')[0][1:]),int(k.split('_')[1][1:])):z[k].copy()for k in z.files}
   i=s['source_index'];sid=s['source_id']
   for family in ('WHOLE','PARTIAL'):
    for snr in (4,10,19):
     ep,cap,target=choose(streams,family,snr);bits=streams[ep];q={4:2,10:4,19:6}[snr];rate='5/6'if snr==19 else'2/3';p=profiles[ep[0],ep[1],q,rate]
     assert p['k']-29==cap
     for seed in (2001,2002,2003):
      assert time.monotonic()-beg<r['max_seconds']['phy']and not(L/'STOP').exists()
      fc=counter(i,snr,seed);ek=digest(dict(source_id=sid,source_index=i,snr_db=snr,noise_seed=seed,counter=fc,profile_id=p['profile_id'],payload_sha256=asha(bits)))
      if ek not in events:
       wave,tx=rt.transmit(p['profile_id'],bits,fc);standard=phy.standard_noise(sid,seed);noise=standard*10**(-snr/20);observed=wave+noise
       obs=out/'observations'/f'{physical:04d}.npz'
       with obs.open('xb')as f:np.savez(f,observed=observed)
       rx=rt.receive(observed,snr,fc,rt.profiles,packet_ledger,f'KODAK_CURRENT/{physical:04d}',phase='KODAK_CURRENT_TEST_ONLY')
       pd=out/'physical'/f'{physical:04d}.json';save(pd,dict(event_key=ek,source_id=sid,source_index=i,snr_db=snr,noise_seed=seed,public_frame_counter=fc,transmission=tx,standard_noise_sha256=asha(standard),noise_sha256=asha(noise),waveform_sha256=asha(wave),observation_sha256=asha(observed),observation=pin(obs),actual_RX=rx,source_truth_supplied_to_RX=False))
       sk=source_key(rx);projection.setdefault(sk,dict(source_input_key=sk,actual_RX=rx));events[ek]=dict(physical_frame=pin(pd),source_input_key=sk,rx_status=rx['status']);physical+=1
      logical.append(dict(method='EC_VAR_PARTIAL'if family=='PARTIAL'else'EC_VAR_WHOLE',family=family,source_index=i,source_id=sid,snr_db=snr,noise_seed=seed,public_frame_counter=fc,event_key=ek,m=ep[0],K=ep[1],terminated_arithmetic_bits=len(bits),source_capacity_bits=cap,utilization=len(bits)/cap,**events[ek]))
   print('PHY_SOURCE_COMPLETE',i+1,24,'PHYSICAL',physical,'LOGICAL',len(logical),flush=True)
  assert len(logical)==432 and physical<=432 and packet_ledger.snapshot()['unresolved']==0
  save(out/'logical_frames.json',logical);save(out/'rx_projection.json',list(projection.values()))
  save(out/'completion.json',dict(status='COMPLETE_PHY432_LOGICAL',registration_sha256=expected,logical_frames=432,new_physical=physical,new_packets=packet_ledger.snapshot(),unique_RX_inputs=len(projection),logical=pin(out/'logical_frames.json'),rx_projection=pin(out/'rx_projection.json'),seconds=time.monotonic()-beg,CUDA_initialized=rt.torch.cuda.is_initialized(),standard_noise_shared_by_family=True))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if packet_ledger is None else packet_ledger.snapshot(),automatic_retry=False));raise
 finally:
  if packet_ledger is not None:packet_ledger.close()
if __name__=='__main__':run(sys.argv[1])
