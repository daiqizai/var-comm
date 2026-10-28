"""Actual complete N1024 image frames. RX accepts only waveform/public metadata."""
from block_common import *
from phy_runtime import runtime,rng,counter,receive,save_npz,NAMESPACE
import traceback,collections
SESSION=NAMESPACE+'_P3'

def receive_frames(y,snr,counters,profiles,backend,phy,header,packet,ledger,label,budget):
 """No TX profile, payload, candidate, source truth or expected decode is available."""
 assert y.ndim==3 and y.shape[1:]==(1024,2)
 rec=[];groups={};hard_arrays={}
 for i,a in enumerate(y):
  h=ledger.call('header_decode',f'{label}_{i}',lambda:header.receive(a[:68],snr,profiles))
  r=dict(header=h,profile=None,blocks=[],TX_truth_used=False);rec.append(r)
  if not h['header_ok']:continue
  p=profiles[str(h['profile_id'])];r['profile']=p
  if p['null']:continue
  offset=68
  for gi,g in enumerate(p['layouts']):
   empty=p['kind']=='THREE'and gi>0 and p['K']==0
   key=(gi,g['N'],g['rate'])
   groups.setdefault(key,[]).append((i,offset,g,empty))
   r['blocks'].append(None);offset+=g['N']
  assert offset==1024
 for (gi,N,rate),items in groups.items():
  for start in range(0,len(items),32):
   batch=items[start:start+32];g=batch[0][2];budget['decodes']+=len(batch)
   assert budget['decodes']<=budget['decode_limit']
   obs=np.stack([y[i,o:o+N]for i,o,_,_ in batch])
   hard,ds=ledger.call('body_decode_batch',f'{label}_{gi}_{N}_{rate}_{start}',lambda:receive(backend,phy,packet,obs,g,snr,[counters[i]for i,_,_,_ in batch],gi,[empty for _,_,_,empty in batch],SESSION))
   for (i,_,_,_),a,d in zip(batch,hard,ds):
    hard_arrays[f'hard_{i}_{gi}']=a
    rec[i]['blocks'][gi]=dict(group=gi,geometry=g,**{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in d.items()},hard_sha256=asha(a))
 return rec,hard_arrays

def main(path,expected):
 assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']]
 jobs=cr(r['inputs']);catalog=cr(r['catalog']);profiles=catalog['profiles']
 out=Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',r['caps']);start=time.monotonic();os.sched_setaffinity(0,set(r['cpu_affinity']))
 budget=dict(decodes=0,decode_limit=r['max_body_decodes']);rows=[];nbody=0;nframe=0
 try:
  b,phy,header,packet=runtime();waves={};batches=collections.defaultdict(list)
  for j in jobs:batches[(j['candidate_id'],j['snr_db'],j['replicate'])].append(j)
  assert len(jobs)==r['frames']and sum(len(j['layouts'])for j in jobs)==r['tx_body_blocks']
  for bi,((candidate,snr,replicate),batch)in enumerate(batches.items()):
   assert time.monotonic()-start<r['max_seconds']and not(L/'STOP').exists()
   label=f'b{bi:04d}';wave=np.empty((len(batch),1024,2),np.float32);truth={};txgroups=collections.defaultdict(list)
   ctrs=[j['counter']for j in batch]
   for i,j in enumerate(batch):
    pid=j['profile_id']
    if pid not in waves:waves[pid]=np.asarray(header.transmit(pid),np.float32)
    wave[i,:68]=waves[pid];offset=68
    for gi,(g,bits)in enumerate(zip(j['layouts'],j['payloads'])):
     info=packet.pack_group(np.asarray(bits,np.uint8),g['k'],explicit_empty=len(bits)==0)
     truth[i,gi]=info;txgroups[gi,g['N'],g['rate']].append((i,offset,g,info));offset+=g['N']
    assert offset==1024
   for (gi,N,rate),items in txgroups.items():
    for startidx in range(0,len(items),32):
     bb=items[startidx:startidx+32];g=bb[0][2];info=np.stack([z[3]for z in bb])
     coded=ledger.call('body_encode_batch',f'{label}_{gi}_{N}_{rate}_{startidx}',lambda:b.encode(info,g['n'],2))
     mask=b.torch.as_tensor(phy.mask_batch(g['n'],[ctrs[z[0]]for z in bb],gi,SESSION))
     ww=phy.modulate_torch((coded+mask)%2,2).cpu().numpy()
     for (i,o,_,_),w in zip(bb,ww):wave[i,o:o+N]=w
     nbody+=len(bb);assert nbody<=r['tx_body_blocks']
   assert np.all((wave**2).sum(-1)==2)
   noise=np.stack([rng('P3_AWGN',snr,j['source_index'],replicate,candidate,'FULL_FRAME').standard_normal((1024,2)).astype(np.float32)for j in batch])
   y=wave+noise*np.float32(10**(-snr/20))
   rx,hard=receive_frames(y,snr,ctrs,profiles,b,phy,header,packet,ledger,label,budget)
   yp=out/(label+'_observations.npz');save_npz(yp,received=y,**hard)
   rx_record=dict(batch=label,snr_db=snr,counters=ctrs,received=pin(yp),events=rx,TX_truth_used=False)
   save(out/(label+'_rx.json'),rx_record)
   # Only now compare saved RX outputs to TX, never modify decisions using truth.
   audits=[]
   for i,(j,v)in enumerate(zip(batch,rx)):
    header_wrong=bool(v['header']['header_ok']and v['header']['profile_id']!=j['profile_id']);wrong=[]
    for d in v['blocks']:
     if d['accepted']:
      gi=d['group'];a=hard[f'hard_{i}_{gi}'];expected_info=truth.get((i,gi))
      if header_wrong or expected_info is None or not np.array_equal(a,expected_info):wrong.append(gi)
    audit=dict(event_id=j['event_id'],header_accepted_wrong=header_wrong,body_accepted_wrong=wrong)
    audits.append(audit)
    rows.append(dict(event_id=j['event_id'],candidate_id=candidate,source_id=j['source_id'],source_index=j['source_index'],snr_db=snr,replicate=replicate,tx_profile_id=j['profile_id'],tx_null=j['null'],tx_fallback=j['fallback'],tx_source_bits=sum(map(len,j['payloads'])),tx_information_bits=sum(g['k']for g in j['layouts']),tx_body_blocks=len(j['layouts']),N=1024,Es=2,tx_energy=2048,rx_record=pin(out/(label+'_rx.json')),rx_index=i,**{k:v for k,v in audit.items()if k!='event_id'}))
   save(out/(label+'_audit.json'),dict(received_record=pin(out/(label+'_rx.json')),events=audits,truth_only_after_RX_saved=True))
   nframe+=len(batch)
   if bi%15==0:print('FULL_FRAME_BATCH',bi,'frames',nframe,'bodies',nbody,'decodes',budget['decodes'],flush=True)
  assert nframe==r['frames']and nbody==r['tx_body_blocks']and ledger.summary()['unresolved']==0
  save(out/'events.json',rows);save(out/'completion.json',dict(status='COMPLETE_P3_PHYSICAL_FRAMES',request=pin(path),events=pin(out/'events.json'),frames=nframe,tx_body_blocks=nbody,body_decode_attempts=budget['decodes'],header_decode_attempts=nframe,ledger=ledger.summary(),seconds=time.monotonic()-start,source_model_calls=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),budget=budget,frames=nframe,tx_body_blocks=nbody,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
