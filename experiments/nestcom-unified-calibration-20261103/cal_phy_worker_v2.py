"""Actual protected-header/body receptions with a public pre-frozen directory."""
from study_common import *
from frame_worker import Packets
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 jobs=cr(r['jobs']);choices=cr(r['choices']);pub=cr(r['directories']);reuse=cr(r['reuse'])
 cats={k:cr(v)for k,v in pub['directory'].items()};out=Path(r['output']);out.mkdir(exist_ok=False);(out/'frames').mkdir();os.sched_setaffinity(0,set(r['cpu_affinity']))
 ledger=Ledger(out/'calls.sqlite',dict(header=len(jobs),body=len(jobs)));start=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));import c_predclass_frame as cp
  sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));from n512_phy import received_projection
  phy=cp.isolated_engine();phy.pack_body.__globals__['FAMILIES']=('EC_STATIC_WHOLE','EC_VAR_WHOLE','EC_VAR_PARTIAL','ADA_PREDICTED_CLASS_PREFIX','EXPLICIT_NULL_FIXEDFRAME')
  rt=phy.create_runtime(R,str(checked(pub['qualification'])));rows=[];inputs={};new=0
  for j in jobs:
   assert time.monotonic()-start<r['seconds']and not(L/'STOP').exists()
   cat=cats[j['catalogue_key']];assert pub['directory'][j['catalogue_key']]['sha256']==j['catalogue_sha256']
   rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};c=choices[j['choice_id']]
   b=np.asarray(c['payload'],np.uint8);assert asha(b)==j['payload_sha256']and c['profile_id']==j['profile_id']
   if j['key']in reuse:
    prior=reuse[j['key']];record=cr(prior['frame']);assert record['identity']==j or all(record['identity'][k]==v for k,v in j.items()if k!='choice_id')
    row=dict(prior,origin='EXACT_THIS_STUDY_FRAME');projection=record['projection']
   else:
    wave,tx=rt.transmit(j['profile_id'],b,j['counter']);noise=phy.standard_noise(j['source_id'],j['noise_seed']);observed=wave+noise*10**(-j['snr_db']/20);assert observed.shape==(1024,2)
    rx=rt.receive(observed,j['snr_db'],j['counter'],rt.profiles,Packets(ledger),j['key'],phase='UNIFIED_CALIBRATION')
    projection=received_projection(rx);key=digest(projection);path=out/'frames'/(j['key']+'.json')
    save(path,dict(identity=j,catalogue=pub['directory'][j['catalogue_key']],actual_RX=rx,projection=projection,received_key=key,transmission=tx,noise_sha256=asha(noise),observation_sha256=asha(observed),TX_truth_supplied_to_RX=False))
    row=dict(physical_key=j['key'],received_key=key,frame=pin(path),origin='ACTUAL_NEW');new+=1
   inputs.setdefault(row['received_key'],dict(key=row['received_key'],projection=projection));rows.append(row)
   if len(rows)%100==0:print('UNIFIED_PHY',len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'frames.json',rows);save(out/'rx_inputs.json',list(inputs.values()));save(out/'completion.json',dict(status='COMPLETE',physical=len(rows),new=new,reused=len(rows)-new,ledger=ledger.summary(),seconds=time.monotonic()-start,CUDA_initialized=rt.torch.cuda.is_initialized()))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
