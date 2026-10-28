"""Finite real LDPC/header probes; RX evidence is persisted before TX audits."""
from block_common import *
from phy_runtime import runtime,rng,counter,receive,save_npz,NAMESPACE
import traceback

def main(path,expected):
 assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']]
 q=cr(r['qualification']);assert q['status']=='COMPLETE_NEW_LAYOUT_QUALIFICATION'
 layouts={x['key']:x for x in cr(q['layouts'])if x.get('roundtrip_exact')}
 out=Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',r['caps']);start=time.monotonic();os.sched_setaffinity(0,set(r['cpu_affinity']))
 try:
  b,phy,header,packet=runtime();profiles=cr(r['catalog'])['profiles'];rows=[];actual_body=0;actual_headers=0
  for key in r['layout_keys']:
   old=layouts[key];g=old['geometry'];assert b.plan(g['k'],g['n'],2)==old['backend']
   for snr in [0,2,4]:
    statuses=[];undetected=0;correct=0;records=[]
    for offset in range(0,128,16):
     assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
     def batch():
      lengths=[[0,2,g['capacity']//2,g['capacity']][i%4]for i in range(offset,offset+16)]
      ids=[counter('P2_BODY',snr,-1,i,key,0)for i in range(offset,offset+16)]
      info=np.stack([packet.pack_group(rng('P2_PAYLOAD',snr,-1,i,key,0).integers(0,2,n,dtype=np.uint8),g['k'],explicit_empty=n==0)for i,n in zip(range(offset,offset+16),lengths)])
      mask=b.torch.as_tensor(phy.mask_batch(g['n'],ids,0,NAMESPACE+'_P2'))
      wave=phy.modulate_torch((b.encode(info,g['n'],2)+mask)%2,2).cpu().numpy()
      assert wave.shape==(16,g['N'],2)and np.all((wave**2).sum(-1)==2)
      noise=np.stack([rng('P2_AWGN',snr,-1,i,key,0).standard_normal((g['N'],2)).astype(np.float32)for i in range(offset,offset+16)])
      received=wave+noise*np.float32(10**(-snr/20))
      hard,dec=receive(b,phy,packet,received,g,snr,ids,0,[n==0 for n in lengths],NAMESPACE+'_P2')
      label=f'{key}_s{snr}_b{offset:03d}';p=out/(label+'.npz');save_npz(p,received=received,hard=hard)
      record=dict(layout=key,snr_db=snr,offset=offset,count=16,counters=ids,statuses=[d['status']for d in dec],accepted=[bool(d['accepted'])for d in dec],declared_lengths=[d.get('declared_length',-1)for d in dec],receiver_observation=pin(p),TX_truth_used=False)
      save(out/(label+'_rx.json'),record)
      exact=np.all(hard==info,axis=1);accept=np.asarray(record['accepted']);audit=dict(accepted=int(accept.sum()),correct=int((accept&exact).sum()),undetected=int((accept&~exact).sum()),received_record=pin(out/(label+'_rx.json')),truth_used_only_after_saved_RX=True)
      save(out/(label+'_audit.json'),audit);return record,audit
     rec,audit=ledger.call('body_batch',f'{key}_{snr}_{offset}',batch);actual_body+=16
     statuses+=rec['accepted'];correct+=audit['correct'];undetected+=audit['undetected'];records.append(audit['received_record'])
    rows.append(dict(layout_key=key,N=g['N'],rate=g['rate'],k=g['k'],n=g['n'],snr_db=snr,trials=128,accepted=sum(statuses),correct=correct,undetected=undetected,success_probability=sum(statuses)/128,records=records))
    print('PROBE_CELL',key,snr,sum(statuses),128,flush=True)
  headers=[]
  if r['include_header']:
   public_ids=sorted(map(int,profiles))
   for snr in [0,2,4]:
    for i in range(128):
     assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
     def htrial():
      pid=public_ids[int(rng('P2_HEADER_PROFILE',snr,-1,i,'HEADER').integers(len(public_ids)))];wave=np.asarray(header.transmit(pid),np.float32)
      assert np.all((wave**2).sum(-1)==2)
      y=wave+rng('P2_HEADER_AWGN',snr,-1,i,'HEADER').standard_normal((68,2)).astype(np.float32)*np.float32(10**(-snr/20))
      v=header.receive(y,snr,profiles);p=out/f'header_s{snr}_{i:03d}.npz';save_npz(p,received=y);rec=dict(snr_db=snr,replicate=i,**v,received=pin(p),TX_truth_used=False);save(p.with_suffix('.json'),rec)
      audit=dict(correct=bool(v['header_ok']and v['profile_id']==pid),undetected=bool(v['header_ok']and v['profile_id']!=pid));save(out/f'header_s{snr}_{i:03d}_audit.json',audit)
      return dict(**rec,**audit)
     headers.append(ledger.call('header_decode',f'{snr}_{i}',htrial));actual_headers+=1
  assert actual_body==len(r['layout_keys'])*384 and actual_headers==(384 if r['include_header']else 0)and ledger.summary()['unresolved']==0
  save(out/'body_cells.json',rows);save(out/'header_trials.json',headers)
  save(out/'completion.json',dict(status='COMPLETE_P2_PACKET_PROBES',request=pin(path),body_cells=pin(out/'body_cells.json'),header_trials=pin(out/'header_trials.json'),body_packets=actual_body,header_packets=actual_headers,physical_image_frames=0,ledger=ledger.summary(),seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
