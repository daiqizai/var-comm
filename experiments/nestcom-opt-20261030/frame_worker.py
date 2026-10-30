"""Actual unchanged LDPC/channel engine; immutable once-only packet ledger."""
from opt_common import *
import traceback,types
class Packets:
 def __init__(self,l):self.l=l
 def decode(self,phase,key,kind,request,fn):return self.l.call(kind,str(key),fn)
 def call(self,kind,key,fn):
  if isinstance(kind,dict):kind,key=kind['kind'],kind['event_id']
  return self.l.call(kind,str(key),fn)
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 jobs=cr(r['jobs']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'frames').mkdir()
 os.sched_setaffinity(0,set(r['cpu_affinity']));N=r['N'];ledger=Ledger(out/'calls.sqlite',dict(header=len(jobs),body=len(jobs)));beg=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));import n512_phy as n5
  if N==512:
   phy,rt,cat=n5.runtime();assert cat==cr(r['catalogue'])
  else:
   sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));import c_predclass_frame as cp
   phy=cp.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL)
   rt=phy.create_runtime(R,str(checked(r['qualification'])));cat=cr(r['catalogue'])
   sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
   from tb_plan import extend_catalogue
   assert extend_catalogue(rt.catalogue)==cat;rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
   cat7=cr(r['catalogue7']);assert cat7['profiles'][:372]==cat['profiles']
  rows=[];inputs={};seen={}
  for j in jobs:
   assert time.monotonic()-beg<r['seconds']and not(L/'STOP').exists()
   if N==1024:
    rt.catalogue=cat7 if j['snr_db']==7 else cat;rt.profiles={str(p['profile_id']):p for p in rt.catalogue['profiles']}
   b=np.asarray(j['payload'],np.uint8);assert asha(b)==j['payload_sha256']and rt.profiles[str(j['profile_id'])]==j['profile']
   wave,tx=phy.transmit_frame(rt,b,j['profile_id'],j['counter'],n5.WIRE)if N==512 else rt.transmit(j['profile_id'],b,j['counter'])
   noise=phy.standard_noise(j['source_id'],j['noise_seed']);scaled=noise*10**(-j['snr_db']/20);observed=wave+scaled
   assert wave.shape==observed.shape==(N,2)
   if j['old']:
    old=cr(j['old']);assert old['transmission']==tx
    if N==512:assert old['noise_sha256']==asha(noise)and old['observed_sha256']==asha(observed)
    else:
     assert old['noise_sha256']==asha(scaled)and old['observation_sha256']==asha(observed)
     from tb_plan import directory_unchanged_on_old_event
     assert directory_unchanged_on_old_event(old['actual_RX']['header'])
    rx=old['actual_RX'];origin='EXACT_OLD_WAVEFORM_NOISE_PROTOCOL'
   else:
    if j['snr_db']==7:
     sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'));from public_counter_rx import receive
     rx=receive(phy,rt,observed,7,j['public_base'],Packets(ledger),j['key'],phase='OPT_FIT')
    else:rx=rt.receive(observed,j['snr_db'],j['counter'],rt.profiles,Packets(ledger),j['key'],phase='OPT_FIT')
    origin='ACTUAL_NEW'
   projection=n5.received_projection(rx);rk=digest(projection)
   inputs.setdefault(rk,dict(key=rk,projection=projection))
   p=out/'frames'/(j['key']+'.json');save(p,dict(key=j['key'],actual_RX=rx,projection=projection,received_key=rk,transmission=tx,noise_sha256=asha(noise),observed_sha256=asha(observed),origin=origin,old=j['old'],TX_truth_supplied_to_RX=False))
   rows.append(dict(physical_key=j['key'],received_key=rk,frame=pin(p),origin=origin))
   if len(rows)%50==0:print('FRAME',N,len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'frames.json',rows);save(out/'rx_inputs.json',list(inputs.values()));save(out/'completion.json',dict(status='COMPLETE',physical=len(rows),new=sum(x['origin']=='ACTUAL_NEW'for x in rows),reused=sum(x['origin']!='ACTUAL_NEW'for x in rows),ledger=ledger.summary(),seconds=time.monotonic()-beg,CUDA_initialized=rt.torch.cuda.is_initialized()))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
