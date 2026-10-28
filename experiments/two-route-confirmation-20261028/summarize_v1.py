"""Closed exact received events and fixed16 records; no new inference/bootstrap."""
import pathlib,sys,json,csv,collections,shutil
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/shared-codeword-fit-20261028'))
from fit_common import *
from protocol import state_identity
L2=B/'two_route_confirmation_20261028_v1';O2=R/'results/two_route_confirmation_20261028_v1'
METRICS=('dino_cosine','psnr_db','lpips_alex','dreamsim','clip_image_cosine')

def main():
    dest=O2/'delivery';dest.mkdir(exist_ok=False);scope=read(O2/'scope.json');receipts=[]
    for role in ['a_phy','b_dependence']:
        wait=read(L2/(role+'_control')/'actual_wait.json');assert wait['returncode']==0 and gone(wait['owner'])and gone(wait['process'])
        done=read(L2/role/'completion.json');assert done['ledger']['unresolved']==0
        receipts.append(dict(role=role,wait=wait,completion=done))
    bundle=read(B/'nextscale_selective_blocks_20261028_v1/p3_replay_v2/receiver_cache.json')
    decode=dict(bundle['cache']);decode.update(read(L/'rx_clean/novel_decode_cache.json'));native=dict(bundle['native'])
    for n in read(L/'clean_native_all.json'):native[digest(n['state'])]=n
    content={}
    for n in native.values():
        ck=digest({k:v for k,v in n['state'].items()if k!='preprocessing'})
        if ck in content:assert content[ck]['pixel_sha256']==n['pixel_sha256']
        else:content[ck]=n
    scores=read(B/'nextscale_selective_blocks_20261028_v1/p1_summary/pair_scores.json');scores.update(read(L/'clean_pair_scores.json'))
    sources={s['source_id']:s for s in read(OLD/'source_manifest.json')}
    physical=read(L2/'a_phy/events.json')+read(L2/'a_prior_events.json');events={e['event_id']:e for e in physical}
    assert len(events)==len(physical)==scope['A']['new_physical_frames']+scope['A']['reused_distinct_physical_events']
    assert not any(e['header_accepted_wrong']or e['payload_accepted_wrong']for e in physical)
    seen={};rx={};used=set()
    for eid,e in events.items():
        p=e['rx_record']['path']
        if p not in seen:seen[p]=read(checked(e['rx_record']))
        v=seen[p]['events'][e['rx_index']];assert not v['TX_truth_used']
        image_hash=asha(np.full((3,256,256),.5,np.float32));status='GRAY_REJECT'
        if v['header']['header_ok']and not v['profile']['null']and v['streams'][0]is not None:
            p=v['profile'];assert p['mode']=='S_FILL'and p['k']==637 and p['n']==1912
            projection=dict(status='SOURCE_INPUT',codec='NEST',m=p['m'],K=p['K'],family='EC_VAR_PARTIAL'if p['K']else'EC_VAR_WHOLE',bits=v['streams'][0])
            key=digest(projection);assert key in decode and decode[key]['projection']==projection,'Actual received source cache miss'
            s=decode[key]['value'];used.add(key)
            st=state_identity(s['m'],s['coarse'],s['positions'],s['values'],'',bundle['route']);ck=digest({k:v for k,v in st.items()if k!='preprocessing'})
            assert ck in content,'Native received-state cache miss';image_hash=content[ck]['pixel_sha256'];status='FULL_ACCEPTED'
        key=digest(dict(source_id=e['source_id'],reference_sha256=sources[e['source_id']]['reference_sha256'],pixel_sha256=image_hash))
        assert key in scores,'Actual source/pixel score missing; do not substitute clean score'
        rx[eid]=dict(receive_status=status,pixel_sha256=image_hash,pair_key=key,**{k:scores[key][k]for k in METRICS})
    rows=[dict(v,**rx[v['physical_event_id']])for v in read(L2/'a_logical_events.json')];assert len(rows)==576
    csvsave(dest/'a_logical_frame_metrics.csv',rows);csvsave(dest/'a_physical_events.csv',physical)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['arm'],r['snr_db'],r['source_id']].append(r)
    def mean(rr):return {k:float(np.mean([float(r[k])for r in rr],dtype=np.float64))for k in METRICS}
    src=[]
    for(arm,snr,sid),rr in groups.items():
        assert len(rr)==3;src.append(dict(arm=arm,snr_db=snr,source_id=sid,source_index=rr[0]['source_index'],**mean(rr)))
    csvsave(dest/'a_source_means.csv',src);summary=[];counts=[]
    for arm in ['QUARTER','TOKEN']:
        for snr in [0,2,4]:
            rr=[r for r in src if(r['arm'],r['snr_db'])==(arm,snr)];assert len(rr)==32
            summary.append(dict(arm=arm,snr_db=snr,sources=32,logical_frames=96,**mean(rr)))
            statuses=collections.Counter(r['receive_status']for r in rows if(r['arm'],r['snr_db'])==(arm,snr))
            counts.extend(dict(arm=arm,snr_db=snr,status=k,count=v,denominator=96)for k,v in statuses.items())
    lookup={(r['arm'],r['snr_db'],r['source_id']):r for r in src};paired=[]
    for r in src:
        if r['arm']!='TOKEN':continue
        q=lookup['QUARTER',r['snr_db'],r['source_id']]
        paired.append(dict(comparison='TOKEN_MINUS_QUARTER',source_id=r['source_id'],source_index=r['source_index'],snr_db=r['snr_db'],**{k:r[k]-q[k]for k in METRICS}))
    csvsave(dest/'a_snr_summary.csv',summary);csvsave(dest/'a_receive_counts.csv',counts);csvsave(dest/'a_paired_source_differences.csv',paired)
    csvsave(dest/'a_paired_summary.csv',[dict(snr_db=snr,n_sources=32,**mean([r for r in paired if r['snr_db']==snr]))for snr in[0,2,4]])
    bsrc=L2/'b_dependence/source_pairs.csv';(dest/'b_source_pairs.csv').write_bytes(bsrc.read_bytes());br=list(csv.DictReader(bsrc.open()));assert len(br)==16
    bsummary=dict(sources=16,step=3000,j=256,scalar_match_fraction=sum(int(r['scalar_matches'])for r in br)/(16*64),bit_match_fraction=sum(int(r['bit_matches'])for r in br)/(16*256),exact_quantized_vectors=sum(r['exact_quantized_vector']=='True'for r in br),image_exact_pairs=sum(float(r['image_max_abs_difference'])==0 for r in br),max_pixel_absolute_difference=max(float(r['image_max_abs_difference'])for r in br),new_predicted_lpips_pairs=sum(r['predicted_lpips_reused']=='False'for r in br))
    for condition in ['received','predicted','received_minus_predicted']:
        for metric in ['psnr_db','lpips_alex','mse','D']:
            key=condition+'_'+metric;bsummary[key]=float(np.mean([float(r[key])for r in br],dtype=np.float64))
    save(dest/'b_summary.json',bsummary)
    for role in ['a_phy','b_dependence']:
        target=dest/'execution_receipts'/role;target.mkdir(parents=True)
        for name in ['owner','launch','actual_wait']:(target/(name+'.json')).write_bytes((L2/(role+'_control')/(name+'.json')).read_bytes())
        (target/'completion.json').write_bytes((L2/role/'completion.json').read_bytes())
    save(dest/'b_native_identity.json',read(L2/'b_dependence/native_identity.json'))
    waits={r['role']:r['wait']['seconds']for r in receipts};assert waits['b_dependence']<=1800 and sum(waits.values())<=3600
    closure=dict(status='COMPLETE_BOUNDED_CONFIRMATION_AND_DIAGNOSTIC',scope=pin(O2/'scope.json'),A=dict(new_physical_frames=scope['A']['new_physical_frames'],exact_prior_physical_reused=scope['A']['reused_distinct_physical_events'],distinct_physical=len(physical),logical_comparison_rows=576,independent_RX_cache_hits=len(used),new_source_inference=0,new_image_inference=0,new_metric_inference=0,accepted_wrong=0),B=bsummary,worker_seconds=waits,new_training_updates=0,bootstrap='NOT_RUN',scope_limit='Development32 and preselected development16; no common500 change, no final B1 selection or causal synthesis-insensitivity proof',old_training_stop_preserved=pin(B/'nested_residual_rd_20261028_v1/STOP')==scope['B']['training_stop'])
    save(dest/'closure.json',closure);save(L2/'summary_completion.json',dict(status='COMPLETE',closure=pin(dest/'closure.json')))
    print(json.dumps(dict(A=summary,B=bsummary,closure=closure)))
if __name__=='__main__':main()
