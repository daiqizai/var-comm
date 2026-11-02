"""72 no-noise engineering frame checks, not image-quality observations."""
from study_common import *
from frame_worker import Packets
import traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']];pub=cr(r['directories']);out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 ledger=Ledger(out/'calls.sqlite',dict(header=72,body=72));start=time.monotonic()
 try:
  sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));import c_predclass_frame as cp
  phy=cp.isolated_engine();phy.pack_body.__globals__['FAMILIES']=('EC_STATIC_WHOLE','EC_VAR_WHOLE','EC_VAR_PARTIAL','ADA_PREDICTED_CLASS_PREFIX','EXPLICIT_NULL_FIXEDFRAME');rt=phy.create_runtime(R,str(checked(pub['qualification'])));rows=[]
  for key,desc in pub['directory'].items():
   cat=cr(desc);rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};assert len(rt.profiles)==len(cat['profiles'])and max(map(int,rt.profiles))<4096
   for p in [cat['profiles'][0],cat['profiles'][-1]]:
    assert time.monotonic()-start<r['seconds'];payload=np.zeros(12 if p['class_bits']else 2,np.uint8);counter=len(rows)
    wave,meta=rt.transmit(p['profile_id'],payload,counter);rx=rt.receive(wave,30,counter,rt.profiles,Packets(ledger),key+'_'+str(counter),phase='DIRECTORY_ENGINEERING')
    assert wave.shape==(1024,2)and meta['header_symbols']==68 and rx['header']['header_ok']and rx['rx_profile']['profile_id']==p['profile_id']and rx['body']['crc_accepted']and rx['body']['parser_accepted']and np.array_equal(rx['body']['payload'],payload)
    rows.append(dict(catalogue=key,profile_id=p['profile_id'],family=p['family'],q=p['q'],rate=p['nominal_rate'],m=p['m'],K=p['K'],header_bits=12,N=1024,actual_RX=rx))
  save(out/'frames.json',rows);save(out/'completion.json',dict(status='COMPLETE',engineering_frames=len(rows),image_observations=0,ledger=ledger.summary(),seconds=time.monotonic()-start,all_directories_fixed_before_evaluation=True))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
