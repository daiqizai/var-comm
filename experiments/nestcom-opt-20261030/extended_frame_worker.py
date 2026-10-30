"""Fixed additive catalogue, unchanged real header/body channel engine."""
from opt_common import *
from frame_worker import Packets
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 jobs=cr(r['jobs']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'frames').mkdir();os.sched_setaffinity(0,set(r['cpu_affinity']))
 ledger=Ledger(out/'calls.sqlite',dict(header=len(jobs),body=len(jobs)));start=time.monotonic();N=r['N'];save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));import n512_phy as n5
  if N==512:phy,rt,oldcat=n5.runtime()
  else:
   sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));import c_predclass_frame as cp
   phy=cp.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL)
   rt=phy.create_runtime(R,str(checked(r['qualification'])))
  cat=cr(r['catalogue']);rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
  assert len(rt.profiles)==len(cat['profiles'])and max(map(int,rt.profiles))<4096
  rows=[];inputs={}
  for j in jobs:
   assert time.monotonic()-start<r['seconds']and not(L/'STOP').exists()
   b=np.asarray(j['payload'],np.uint8);assert asha(b)==j['payload_sha256']and rt.profiles[str(j['profile_id'])]==j['profile']
   wave,tx=phy.transmit_frame(rt,b,j['profile_id'],j['counter'],n5.WIRE)if N==512 else rt.transmit(j['profile_id'],b,j['counter'])
   noise=phy.standard_noise(j['source_id'],j['noise_seed']);observed=wave+noise*10**(-j['snr_db']/20);assert wave.shape==observed.shape==(N,2)
   old=cr(j['old'])if j.get('old')else None
   # A larger directory can change an UNKNOWN_PROFILE rejection. Reuse only
   # a CRC-valid decoded paid ID with the exact same assigned meaning.
   old_profile=old['actual_RX']['rx_profile']if old is not None else None
   can_reuse=old is not None and old['actual_RX']['header']['header_ok']and old_profile is not None and all(rt.profiles[str(old_profile['profile_id'])][k]==v for k,v in old_profile.items())
   if can_reuse:
    assert old['transmission']==tx and old['noise_sha256']==asha(noise)and old['observed_sha256']==asha(observed)
    rx=old['actual_RX'];origin='EXACT_WAVEFORM_NOISE_DECODED_PROFILE'
   else:
    if j['snr_db']==7:
     sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'));from public_counter_rx import receive
     rx=receive(phy,rt,observed,7,j['public_base'],Packets(ledger),j['key'],phase='OPT_EXTENDED')
    elif N==512:
     assert cr(r['n512_qualification'])['status']=='COMPLETE'
     rx=phy.receive_frame(rt,observed,j['snr_db'],j['counter'],Packets(ledger),j['key'],phase='OPT_EXTENDED',session=n5.WIRE)
    else:rx=rt.receive(observed,j['snr_db'],j['counter'],rt.profiles,Packets(ledger),j['key'],phase='OPT_EXTENDED')
    origin='ACTUAL_NEW'
   projection=n5.received_projection(rx);rk=digest(projection);inputs.setdefault(rk,dict(key=rk,projection=projection))
   path=out/'frames'/(j['key']+'.json');save(path,dict(key=j['key'],actual_RX=rx,projection=projection,received_key=rk,transmission=tx,noise_sha256=asha(noise),observed_sha256=asha(observed),origin=origin,old=j.get('old'),catalogue=r['catalogue'],TX_truth_supplied_to_RX=False))
   rows.append(dict(physical_key=j['key'],received_key=rk,frame=pin(path),origin=origin))
   if len(rows)%50==0:print('FRAME_EXTENDED',N,len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'frames.json',rows);save(out/'rx_inputs.json',list(inputs.values()));save(out/'completion.json',dict(status='COMPLETE',physical=len(rows),new=sum(v['origin']=='ACTUAL_NEW'for v in rows),reused=sum(v['origin']!='ACTUAL_NEW'for v in rows),ledger=ledger.summary(),seconds=time.monotonic()-start,CUDA_initialized=rt.torch.cuda.is_initialized()))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
