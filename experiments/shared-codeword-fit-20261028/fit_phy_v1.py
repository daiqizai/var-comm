"""Five new actual fixed frames; one shared body codeword, actual hard RX only."""
from fit_common import *
from fit_protocol import pack_compact,parse_compact
from phy_runtime import runtime,rng,counter,save_npz
import collections,traceback
SESSION='NESTCOM_SHARED_FIT_20261028_V1'
def receive_frames(y,snr,counters,profiles,b,phy,header,packet,ledger,label):
    rows=[];groups=collections.defaultdict(list);hard={}
    for i,a in enumerate(y):
        h=ledger.call('header_decode',f'{label}_{i}',lambda:header.receive(a[:68],snr,profiles));v=dict(header=h,profile=None,streams=[None]*3,status='HEADER_REJECT',local_status=[],TX_truth_used=False);rows.append(v)
        if not h['header_ok']:continue
        p=profiles[str(h['profile_id'])];v['profile']=p
        if p['null']:v['status']='PAID_NULL';continue
        groups[p['k']].append(i)
    for k,indices in groups.items():
        obs=np.stack([y[i,68:]for i in indices]);llr=phy.demap_torch(b.torch.as_tensor(obs,dtype=b.torch.float32),snr,2);mask=b.torch.as_tensor(phy.mask_batch(1912,[counters[i]for i in indices],0,SESSION),dtype=b.torch.float32)
        a=ledger.call('body_decode_batch',f'{label}_{k}',lambda:b.decode(llr*(1-2*mask),k,1912,2).cpu().numpy().astype(np.uint8))
        for i,information in zip(indices,a):
            p=rows[i]['profile'];hard['hard_'+str(i)]=information;v=rows[i];v['ldpc_convergence']='NOT_EXPOSED_BY_FROZEN_BACKEND';v['global_syndrome_gate']=False
            if p['mode']=='COMPACT':
                d=parse_compact(p['profile_id'],p['K'],information,k);v.update(d);v['streams']=[None if x is None else x.tolist()for x in d['streams']]
            else:
                ds=[packet.parse_group(information,k)]if p['mode'].startswith('S_')else[packet.parse_group(information[:398],398),packet.parse_group(information[398:517],119,explicit_empty=p['K']==0),packet.parse_group(information[517:],119,explicit_empty=p['K']==0)]
                v['streams']=[d['bits'].tolist()if d['accepted']else None for d in ds];v['local_status']=[d['status']for d in ds];v['status']='PARSED_LOCAL';v['groups']=[{k:(x.tolist()if isinstance(x,np.ndarray)else x)for k,x in d.items()}for d in ds]
    return rows,hard
def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']];jobs=cr(r['inputs']);profiles=cr(r['catalog']);out=Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',r['caps']);os.sched_setaffinity(0,set(r['cpu_affinity']));begin=time.monotonic();rows=[];ndec=0
    try:
        b,phy,header,packet=runtime();groups=collections.defaultdict(list)
        for j in jobs:groups[j['arm'],j['snr_db'],j['replicate']].append(j)
        for bi,((arm,snr,rep),batch)in enumerate(groups.items()):
            assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists();label=f'b{bi:03d}';wave=np.empty((len(batch),1024,2),np.float32);tx={};encgroups=collections.defaultdict(list);ctrs=[j['counter']for j in batch]
            for i,j in enumerate(batch):
                p=profiles[str(j['profile_id'])];wave[i,:68]=header.transmit(p['profile_id']);info=np.asarray(j['information'],np.uint8);assert info.shape==(p['k'],);tx[i]=info;encgroups[p['k']].append(i)
            for k,ii in encgroups.items():
                coded=ledger.call('body_encode_batch',f'{label}_{k}',lambda:b.encode(np.stack([tx[i]for i in ii]),1912,2));mask=b.torch.as_tensor(phy.mask_batch(1912,[ctrs[i]for i in ii],0,SESSION));symbols=phy.modulate_torch((coded+mask)%2,2).cpu().numpy()
                for i,w in zip(ii,symbols):wave[i,68:]=w
            assert np.all((wave**2).sum(-1)==2)
            noise=np.stack([rng('SHARED_FIT_AWGN_V1',snr,j['source_index'],rep,'FULL_FRAME').standard_normal((1024,2)).astype(np.float32)for j in batch]);y=wave+noise*np.float32(10**(-snr/20))
            received,hard=receive_frames(y,snr,ctrs,profiles,b,phy,header,packet,ledger,label);ndec+=len(hard)
            save_npz(out/(label+'_observations.npz'),received=y,**hard);rxpath=out/(label+'_rx.json');save(rxpath,dict(received=pin(out/(label+'_observations.npz')),events=received,public_counters=ctrs,snr_db=snr,TX_truth_used=False))
            for i,(j,v)in enumerate(zip(batch,received)):
                wrongh=bool(v['header']['header_ok']and v['header']['profile_id']!=j['profile_id']);wrong=[]
                for g,bits in enumerate(v['streams']):
                    if bits is not None and(wrongh or g>=len(j['payloads'])or bits!=j['payloads'][g]):wrong.append(g)
                rows.append(dict(event_id=j['event_id'],arm=arm,source_id=j['source_id'],source_index=j['source_index'],snr_db=snr,replicate=rep,tx_profile_id=j['profile_id'],source_bits=j['source_bits'],fields=j['fields'],padding=j['padding'],k=len(j['information']),n=1912,N=1024,Es=2,energy=2048,rx_record=pin(rxpath),rx_index=i,header_accepted_wrong=wrongh,payload_accepted_wrong=wrong,information_error_bits=int(np.count_nonzero(hard['hard_'+str(i)]!=tx[i]))if'hard_'+str(i)in hard and len(hard['hard_'+str(i)])==len(tx[i])else None))
            save(out/(label+'_event_audit.json'),rows[-len(batch):]);print('SHARED_FRAME',arm,snr,rep,len(rows),flush=True)
        assert len(rows)==r['frames']and len({x['event_id']for x in rows})==len(rows)
        save(out/'events.json',rows);save(out/'completion.json',dict(status='COMPLETE_ACTUAL_SHARED_FRAMES',events=pin(out/'events.json'),physical_frames=len(rows),header_decodes=len(rows),body_decodes=ndec,ledger=ledger.summary(),seconds=time.monotonic()-begin,source_inference=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),physical_frames=len(rows),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
