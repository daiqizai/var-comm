"""One real noiseless roundtrip per new geometry, plus four header integration cases."""
from block_common import *
from protocol import geometry,RATES
from phy_runtime import runtime,rng,counter,receive,save_npz
import traceback

def main(path,expected):
 assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']]
 assert cr(r['quality_gate'])['status']=='PASS'
 out=Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',r['caps']);start=time.monotonic();os.sched_setaffinity(0,set(r['cpu_affinity']))
 try:
  b,phy,header,packet=runtime();from profiles import UnsupportedConfiguration
  rows=[]
  for N in [119,179,239,478,598,718,956]:
   for ri,rate in enumerate(RATES):
    assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
    g=geometry(N,rate);key=f'N{N}_r{rate.replace("/","-")}'
    def construct():
     try:return dict(status='CONSTRUCTIBLE',backend=b.plan(g['k'],g['n'],2))
     except UnsupportedConfiguration as e:return dict(status='EXCLUDED_UNSUPPORTED',reason=str(e))
    c=ledger.call('construct',key,construct)
    if c['status']!='CONSTRUCTIBLE':rows.append(dict(key=key,geometry=g,**c));continue
    length=[0,2,g['capacity']//2,g['capacity']][ri];bits=rng('P2_QUAL_PAYLOAD',key).integers(0,2,length,dtype=np.uint8)
    info=packet.pack_group(bits,g['k'],explicit_empty=length==0)[None]
    ctr=[counter('P2_QUAL',key)];session='CAB_P2_QUAL';mask=b.torch.as_tensor(phy.mask_batch(g['n'],ctr,0,session))
    code=ledger.call('encode',key,lambda:b.encode(info,g['n'],2));wave=phy.modulate_torch((code+mask)%2,2).cpu().numpy()
    assert wave.shape==(1,N,2)and np.all((wave**2).sum(-1)==2)
    hard,decisions=ledger.call('decode',key,lambda:receive(b,phy,packet,wave,g,30,ctr,0,[length==0],session))
    p=out/(key+'.npz');save_npz(p,received=wave,hard=hard)
    # Saved observations/decisions precede the separate comparison with TX bits.
    save(out/(key+'_rx.json'),dict(statuses=[d['status']for d in decisions],accepted=[bool(d['accepted'])for d in decisions],observation=pin(p)))
    exact=bool(np.array_equal(hard,info));assert exact and decisions[0]['accepted']
    rows.append(dict(key=key,geometry=g,**c,roundtrip_exact=exact,payload_length=length,energy_per_symbol=2,received_record=pin(out/(key+'_rx.json'))))
  cat=cr(r['catalog']);profiles=cat['profiles']
  legal=next(int(i)for i,p in profiles.items()if not p['null']and p['kind']=='THREE')
  null=next(int(i)for i,p in profiles.items()if p['null'])
  other=next(int(i)for i,p in profiles.items()if not p['null']and p['kind']=='SINGLE')
  assert '4095'not in profiles
  headers=[]
  for name,pid in [('ordinary',legal),('paid_NULL',null),('other_legal_actual_profile',other),('illegal_profile',4095)]:
   wave=np.asarray(header.transmit(pid),np.float32);assert wave.shape==(68,2)and np.all((wave**2).sum(-1)==2)
   v=ledger.call('header_decode',name,lambda:header.receive(wave,30,profiles));save(out/(name+'_header_rx.json'),v)
   assert v['header_crc_ok']and v['header_ok']==(pid!=4095)
   if pid!=4095:assert v['profile_id']==pid
   headers.append(dict(case=name,result=v,record=pin(out/(name+'_header_rx.json'))))
  assert ledger.summary()['unresolved']==0
  save(out/'layouts.json',rows);save(out/'completion.json',dict(status='COMPLETE_NEW_LAYOUT_QUALIFICATION',request=pin(path),layouts=pin(out/'layouts.json'),headers=headers,ledger=ledger.summary(),seconds=time.monotonic()-start,noisy_body=0,noisy_header=0,image_frames=0))
  print(json.dumps(dict(qualified=sum(x.get('roundtrip_exact',False)for x in rows),excluded=sum(x['status']=='EXCLUDED_UNSUPPORTED'for x in rows),header_integration=4)))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
