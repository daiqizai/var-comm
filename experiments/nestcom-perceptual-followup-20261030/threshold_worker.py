"""Actual 30,000 fixed-policy frames; original waveforms/counters, new AWGN only."""
from followup_common import *
from frame_worker import Packets
import traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];donors=cr(r['donors']);seedreg=cr(r['noise_registration'])
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'frames').mkdir();os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',dict(frame_tx=len(donors),header=30000,body=30000))
 try:
  sys.path[:0]=[str(R/'experiments/nestcom-n512-20261028'),str(R/'experiments/adatoken-response-20261026')]
  import c_predclass_frame as cp
  from n512_phy import received_projection
  phy=cp.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL)
  rt=phy.create_runtime(R,str(checked(r['qualification'])));cats={k:cr(v)for k,v in r['catalogues'].items()};waves={};proof=[]
  # Verify every original waveform and scrambler identity before any new RX.
  for d in donors:
   rt.catalogue=cats[d['method']];rt.profiles={str(p['profile_id']):p for p in rt.catalogue['profiles']};assert rt.profiles[str(d['profile']['profile_id'])]==d['profile']
   b=np.array(d['payload'],np.uint8);assert asha(b)==d['payload_sha256']
   wave,tx=ledger.call('frame_tx',d['key'],lambda:rt.transmit(d['profile']['profile_id'],b,d['counter']))
   assert tx==d['transmission']and tx['session']==d['session']and tx['public_frame_counter']==d['counter']
   assert wave.shape==(1024,2)and asha(wave)==tx['transmitted_frame_sha256'];waves[d['key']]=wave
   proof.append(dict(donor_key=d['key'],original_frame=d['frame'],counter=d['counter'],session=d['session'],waveform_sha256=asha(wave),exact_transmission_metadata=True))
  save(out/'waveform_qualification.json',proof);rows=[];inputs={};noise_hashes={}
  for d in donors:
   rt.catalogue=cats[d['method']];rt.profiles={str(p['profile_id']):p for p in rt.catalogue['profiles']};wave=waves[d['key']]
   for snr in r['actual_snr_db']:
    for repeat,seed in enumerate(seedreg['seeds']):
     if time.monotonic()-beg>=r['seconds']or(L/'STOP').exists():raise BudgetStop('E6 stage boundary')
     key=digest(dict(donor_key=d['key'],actual_snr_db=snr,noise_seed=seed));noise=phy.standard_noise(d['source_id'],seed);nh=asha(noise);nk=(d['source_id'],seed)
     if nk in noise_hashes:assert noise_hashes[nk]==nh
     else:noise_hashes[nk]=nh
     observed=wave+noise*10**(-snr/20)
     # Receiver gets only symbols, true variance, public counter and catalogue.
     rx=rt.receive(observed,snr,d['counter'],rt.profiles,Packets(ledger),key,phase='FOLLOWUP_E6')
     p=received_projection(rx);rk=digest(dict(projection=p,generation_seed=d['generation_seed']))if d['method']=='Ada'else digest(p)
     job=dict(key=rk,projection=p)
     if d['method']=='Ada':job['generation_seed']=d['generation_seed']
     inputs.setdefault(rk,job);fp=out/'frames'/(key+'.json');save(fp,dict(key=key,donor_key=d['key'],actual_RX=rx,projection=p,received_key=rk,actual_snr_db=snr,noise_seed=seed,noise_sha256=nh,observed_sha256=asha(observed),counter=d['counter'],session=d['session'],waveform_sha256=asha(wave),source_truth_supplied_to_RX=False))
     rows.append(dict(key=key,donor_key=d['key'],method=d['method'],source_id=d['source_id'],source_index=d['source_index'],reference_sha256=d['reference_sha256'],N=1024,design_snr_db=10,snr_db=snr,channel_repeat=repeat,noise_seed=seed,received_key=rk,frame=pin(fp),wire_status=rx['status'],projection_status=p['status'],generation_seed=d['generation_seed']))
   if len(rows)%1000==0:print('E6_ACTUAL_FRAMES',len(rows),30000,flush=True)
  assert len(rows)==30000 and ledger.summary()['unresolved']==0
  save(out/'frames.json',rows);save(out/'rx_inputs.json',list(inputs.values()));save(out/'completion.json',dict(status='COMPLETE',actual_frames=30000,distinct_standard_draws=len(noise_hashes),waveform_donors=len(donors),ledger=ledger.summary(),seconds=time.monotonic()-beg,new_models=0,CUDA_initialized=rt.torch.cuda.is_initialized(),new_noise_changes_counter=False))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
