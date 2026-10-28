"""New 512-symbol sends/AWGN/actual CPU decodes; existing exact N512 reuse only."""
from n512_phy import *
import traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request)
 for d in r['code']:checked(d)
 protocol=cr(r['protocol']);cat=cr(protocol['catalogue']);jobs=cr(r['jobs']);choices=cr(r['choices']);reuse=cr(r['reuse']) if r.get('reuse') else {}
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'packets').mkdir();(out/'frames').mkdir();os.sched_setaffinity(0,set(r['cpu_affinity']))
 ledger=Ledger(out/'calls.sqlite',dict(header=len(jobs),body=len(jobs)));pl=PacketLedger(ledger,out/'packets');beg=time.monotonic();rows=[]
 try:
  phy,rt,current=runtime();assert current==cat
  for j in jobs:
   assert time.monotonic()-beg<r['max_seconds'] and not (L/'STOP').exists();key=j['key'];choice=choices[j['choice_id']]
   if key in reuse:
    old=cr(reuse[key]);assert old['physical_identity']==j['key'] and old['protocol_sha256']==r['protocol']['sha256'];rows.append(reuse[key]);continue
   bits=np.array(choice['payload'],np.uint8);assert asha(bits)==j['payload_sha256']
   frame,tx=phy.transmit_frame(rt,bits,j['profile_id'],j['counter'],WIRE)
   noise=phy.standard_noise(j['source_id'],j['noise_seed']);y=frame+noise*10**(-j['snr_db']/20)
   assert frame.shape==y.shape==noise.shape==(512,2)
   rx=phy.receive_frame(rt,y,j['snr_db'],j['counter'],pl,key,r['stage'],WIRE)
   projection=received_projection(rx);projection_key=digest(projection)
   path=out/'frames'/(key+'.json');save(path,dict(physical_identity=key,protocol_sha256=r['protocol']['sha256'],job=j,choice=choice,transmission=tx,noise_sha256=asha(noise),observed_sha256=asha(y),actual_RX=rx,projection=projection,received_key=projection_key,TX_truth_supplied_to_RX=False))
   rows.append(pin(path))
   if len(rows)%100==0:print('PHYSICAL_COMMITTED',len(rows),len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'frames.json',rows);save(out/'completion.json',dict(status='COMPLETE_PHY_SHARD',request=pin(request),frames=pin(out/'frames.json'),count=len(rows),reused=sum(j['key']in reuse for j in jobs),ledger=ledger.summary(),seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
