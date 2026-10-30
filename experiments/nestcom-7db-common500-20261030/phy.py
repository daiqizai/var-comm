"""Actual complete N1024 frames from cached terminated streams, no model calls."""
from g import *
import base64,types
def runtime(r,method):
 if method=='ARPC':
  sys.path.insert(0,str(R/'experiments/arpc-external-baseline-20261019/phy_v1'))
  import link
  return link,link.Runtime()
 sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
 import c_predclass_frame as c
 phy=c.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],c.FAMILY,c.NULL)
 rt=phy.create_runtime(R,str(checked(r['phy_qualification'])))
 # All five layouts are existing qualified N1024 layouts. Add only paid Ada IDs.
 import fixed_frame
 fixed_frame.MCS=MCS
 cat=c.catalogue(rt.catalogue)
 rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
 return phy,rt
class Packets:
 def __init__(self,l):self.l=l
 def decode(self,phase,key,kind,request,fn):return self.l.call(kind,str(key),fn)
 def call(self,kind,key,fn):
  if isinstance(kind,dict):kind,key=kind['kind'],kind['event_id']
  return self.l.call(kind,str(key),fn)
def main(request,expected):
 r,o=initialize(request,expected);method=r['method'];sources=cr(r['sources']);stage=r['stage'];ledger=Ledger(o/'calls.sqlite',dict(header=len(sources)*3,body=len(sources)*3))
 try:
  phy,rt=runtime(r,method);save(o/'catalogue.json',rt.catalogue)
  sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'));from public_counter_rx import receive
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'));from tb_plan import fallback
  jobs={};frames=[];beg=time.monotonic()
  assert stage=='formal' and method in ('NEST','ADA','ARPC');assert len(sources)==500
  for s in sources:
   if method in ('NEST','WHOLE','ADA'):
    with np.load(checked(s['ada_streams' if method=='ADA'else'nest_streams']),allow_pickle=False)as z:streams={k:z[k].copy() for k in z.files}
   for rep in range(3):
    q,rate=r['frozen_mcs'];seed=(2001 if method=='ADA'else 6201)+rep
    assert time.monotonic()-beg<r['max_seconds'] and not(L/'STOP').exists()
    cid=f'q{q}_r{rate.replace("/","-")}';key=digest([stage,method,s['source_id'],7,cid,rep]);base=26000000+16384*s['source_index']+4096*rep
    if method=='ARPC':
     template=rt.profile(7,q,rate);eligible=[v for v in s['arpc_streams']if v['payload_bits']<=template['source_capacity_bits']];end=max([v['endpoint']for v in eligible],default=0);p=rt.profile(end,q,rate)
     payload=checked(next(v['wire']for v in eligible if v['endpoint']==end)).read_bytes() if end else b''
     counter=base;wave,tx=rt.transmit(payload,p['profile_id'],counter)
     noise=phy.ep.standard_noise(s['source_id'],seed);obs=wave+noise*10**(-7/20)
     rx=rt.receive(obs,7,counter,Packets(ledger),key);body=rx['body'];accepted=bool(rx['header']['header_ok']and body and body['crc_accepted']and body['parser_accepted'])
     actual=dict(accepted=accepted,status=rx['status'],profile=rx['profile'],payload_b64=base64.b64encode(body['payload']).decode()if accepted else None)
     if body and isinstance(body.get('payload'),bytes):body['payload_b64']=base64.b64encode(body.pop('payload')).decode()
     projection=dict(codec='ARPC',actual=actual);bits=len(payload)*8;no_fit=end==0
    else:
     if method=='ADA':
      import c_predclass_frame as c
      p,payload,attempts=c.choose(streams,list(rt.profiles.values()),q,rate,s['predicted_class']);no_fit=p['family']==c.NULL
     else:
      options=[p for p in rt.profiles.values()if p['q']==q and p['nominal_rate']==rate];cap=options[0]['source_capacity_bits'];end=next((e for e in fallback('PARTIAL'if method=='NEST'else'WHOLE') if len(streams['m%d_K%d'%e])<=min(cap,8191)),None)
      if end is None:raise RuntimeError('No source endpoint fits; preserve event, do not invent N1024 NULL')
      p=next(p for p in options if (p['m'],p['K'],p['family'])==(*end,'EC_VAR_PARTIAL'if end[1]else'EC_VAR_WHOLE'));payload=streams['m%d_K%d'%end];no_fit=False
     counter=base+p['profile_id'];wave,tx=rt.transmit(p['profile_id'],payload,counter)
     noise=phy.standard_noise(s['source_id'],seed);obs=wave+noise*10**(-7/20)
     rx=receive(phy,rt,obs,7,base,Packets(ledger),key,phase='formal_common500_7db');body=rx['body'];accepted=bool(rx['header']['header_ok']and body and body['crc_accepted']and body['parser_accepted'])
     if method=='ADA':
      v=c.source_projection(rx);projection=dict(status='SOURCE_INPUT',codec='ADA',m=v['prefix'],K=0,received_class=v['received_class'],bits=v['bits'])if v['status']=='SOURCE_INPUT'else None
     else:projection=dict(status='SOURCE_INPUT',codec='NEST',m=rx['rx_profile']['m'],K=rx['rx_profile']['K'],family=rx['rx_profile']['family'],bits=body['payload'])if accepted else None
     bits=len(payload)
    assert wave.shape==obs.shape==(1024,2)
    rxkey=digest(projection)if projection is not None else None
    if rxkey:jobs.setdefault(rxkey,dict(key=rxkey,projection=projection))
    f=dict(key=key,repeat=rep+1,method=method,source_id=s['source_id'],source_index=s['source_index'],snr_db=7,N=1024,candidate_id=cid,q=q,rate=rate,profile_id=p['profile_id'],endpoint=[p.get('m',p.get('endpoint')),p.get('K',0)],source_bits=bits,capacity_bits=p['source_capacity_bits'],no_fit=no_fit,accepted=accepted,rx_key=rxkey,public_base=base,counter=counter,noise_seed=seed,noise_sha256=asha(noise),waveform_sha256=asha(wave),observation_sha256=asha(obs),energy=float(np.square(wave.astype(np.float64)).sum()),transmission=tx,actual_RX=rx,source_truth_supplied_to_RX=False)
    save(o/(key+'.json'),f);frames.append(f)
   if len(frames)%100==0:print('PHY',method,len(frames),flush=True)
  assert ledger.summary()['unresolved']==0;save(o/'frames.json',frames);save(o/'inputs.json',list(jobs.values()))
  save(o/'completion.json',dict(status='COMPLETE',frames=pin(o/'frames.json'),inputs=pin(o/'inputs.json'),physical_frames=len(frames),unique_received_inputs=len(jobs),ledger=ledger.summary(),seconds=time.monotonic()-beg))
 except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary()));raise
if __name__=='__main__':main(*sys.argv[1:])
