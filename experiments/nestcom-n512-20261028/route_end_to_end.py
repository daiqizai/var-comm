"""Two new valid same-route NeST frames; reuse identical independent RX outputs."""
from n512_phy import *
import traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(d)for d in r['code']];out=Path(r['output']);out.mkdir(exist_ok=False);(out/'packets').mkdir();ledger=Ledger(out/'calls.sqlite',dict(header=2,body=2));pl=PacketLedger(ledger,out/'packets')
 try:
  phy,rt,cat=runtime();assert cat==cr(r['catalogue']);s=cr(r['source']);rows=[];reuse={v['key']:v for v in cr(r['independent_RX'])}
  with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
  for q,rate in [(2,'1/2'),(6,'5/6')]:
   p,b,nofit=choose(streams,cat['profiles'],dict(method='NEST',target=[10,0],q=q,rate=rate));assert not nofit
   tx,meta=phy.transmit_frame(rt,b,p['profile_id'],81000000+len(rows),WIRE);rx=phy.receive_frame(rt,tx.copy(),30,81000000+len(rows),pl,str(len(rows)),'new_route_noiseless',WIRE)
   projection=received_projection(rx);key=digest(projection);assert key in reuse and reuse[key]['status']=='ACCEPTED_SOURCE'
   assert projection['bits']==b.tolist() and meta['total_symbols']==512
   rows.append(dict(profile=p,transmission=meta,actual_RX=rx,received_key=key,independent_RX=reuse[key],source_truth_used_by_RX=False))
  save(out/'frames.json',rows);save(out/'completion.json',dict(status='N512_SAME_ROUTE_END_TO_END_PASS',frames=pin(out/'frames.json'),ledger=ledger.summary(),new_PHysical_frames=2,new_source_calls=0,independent_RX_exact_reuse=2,request=pin(request)))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
