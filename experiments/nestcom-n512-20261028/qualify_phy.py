"""Once-only real CPU LDPC/QAM/header checks, not simulated metadata PASS."""
from n512_phy import *
import traceback
def main(expected):
 rp=O/'phy_qualification_registration.json';assert sha(rp)==expected;r=read(rp)
 for p in r['code']:checked(p)
 out=L/'phy_qualification';out.mkdir(exist_ok=False);(out/'packets').mkdir();(out/'frames').mkdir();beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);pl=PacketLedger(ledger,out/'packets')
 try:
  phy,rt,cat=runtime();save(O/'public_catalogue.json',cat)
  assert cat['profiles'] and all(0<=p['profile_id']<4096 for p in cat['profiles'])
  # Qualify every paid profile header against the complete catalogue.
  for p in cat['profiles']:
   h=rt.header.transmit(p['profile_id']);v=phy.receive_header(rt,h,30,pl,'header_id_'+str(p['profile_id']),'qualification')
   assert v['header_ok'] and v['profile_id']==p['profile_id']
  sources=cr(r['cal_sources'])[:2];frames=[];inputs={}
  def packet(s,p,bits,snr,noisy,case,tx_no_fit=False):
   assert time.monotonic()-beg<r['max_seconds'];counter=80000000+len(frames)
   tx,meta=phy.transmit_frame(rt,bits,p['profile_id'],counter,WIRE)
   assert tx.shape==(512,2) and meta['body_symbols']==444 and meta['total_symbols']==512 and meta['header_energy']==136 and meta['rho']==meta['E_frame']/1024
   standard=phy.standard_noise(s['source_id'],4101);y=tx+standard*10**(-snr/20) if noisy else tx.copy()
   key=digest(dict(case=case,index=len(frames),payload=asha(bits),profile=p['profile_id'],snr=snr,noisy=noisy))
   rx=phy.receive_frame(rt,y,snr,counter,pl,key,'qualification',WIRE)
   if not noisy:assert rx['header']['header_ok'] and rx['body']['parser_accepted'] and rx['body']['payload']==bits.tolist() and rx['rx_profile']['profile_id']==p['profile_id']
   projection=received_projection(rx);rk=digest(projection);inputs[rk]=dict(key=rk,projection=projection)
   record=dict(key=key,source_id=s['source_id'],source_index=s['source_index'],case=case,noisy=noisy,snr_db=snr,tx_no_fit=tx_no_fit,tx_profile=p['profile_id'],payload_sha256=asha(bits),received_key=rk,transmission=meta,standard_noise_sha256=asha(standard),observed_sha256=asha(y),actual_RX=rx)
   save(out/'frames'/(key+'.json'),record);frames.append(record)
  s=sources[0]
  with np.load(checked(s['nest_streams']),allow_pickle=False)as z:n={k:z[k].copy() for k in z.files}
  with np.load(checked(s['ada_streams']),allow_pickle=False)as z:a={k:z[k].copy() for k in z.files}
  for layout in cat['layouts']:
   q,rate=layout['q'],layout['rate']
   for method,target,stream in [('NEST',[4,0],n),('NEST',[10,0],n),('ADA',[1,0],a),('ADA',[256,0],a)]:
    p,b,no_fit=choose(stream,cat['profiles'],dict(method=method,target=target,q=q,rate=rate),s['predicted_class']);packet(s,p,b,30,False,f'{method}_{target}_{q}_{rate}',no_fit)
   p=next(p for p in cat['profiles'] if p['family']==NULL and (p['q'],p['nominal_rate'])==(q,rate));packet(s,p,np.zeros(2,np.uint8),30,False,f'NULL_{q}_{rate}',True)
  for s in sources:
   for snr,q,rate in [(4,2,'1/2'),(10,4,'2/3'),(19,6,'5/6')]:
    if not any(v['q']==q and v['rate']==rate for v in cat['layouts']):continue
    for method,field,target in [('NEST','nest_streams',[10,0]),('ADA','ada_streams',[256,0])]:
     with np.load(checked(s[field]),allow_pickle=False)as z:stream={k:z[k].copy() for k in z.files}
     p,b,no_fit=choose(stream,cat['profiles'],dict(method=method,target=target,q=q,rate=rate),s['predicted_class']);packet(s,p,b,snr,True,f'NOISY_{method}_{s["source_index"]}_{snr}',no_fit)
  save(out/'frames.json',frames);save(out/'received_only.json',list(inputs.values()));assert ledger.summary()['unresolved']==0
  save(out/'completion.json',dict(status='COMPLETE_REAL_N512_PHY',registration=pin(rp),catalogue=pin(O/'public_catalogue.json'),frames=pin(out/'frames.json'),received_only=pin(out/'received_only.json'),frame_count=len(frames),header_ID_count=len(cat['profiles']),layouts=len(cat['layouts']),rejected_layouts=cat['rejected_layouts'],ledger=ledger.summary(),seconds=time.monotonic()-beg,independent_source_RX_required_before_pilot=True))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(sys.argv[1])
