"""Replay received bitstrings first; join source identities and metrics only afterward."""
from fit_common import *
from fit_protocol import modes,OFFSETS
from protocol import state_identity
import collections,csv,traceback
METRICS=('dino_cosine','psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def main():
    out=L/'summary_v1';out.mkdir(exist_ok=False)
    try:
        old=B/'nextscale_selective_blocks_20261028_v1';bundle=read(old/'p3_replay_v2/receiver_cache.json');decode=dict(bundle['cache']);decode.update(read(L/'rx_clean/novel_decode_cache.json'));native=dict(bundle['native'])
        for n in read(L/'clean_native_all.json'):native[digest(n['state'])]=n
        content={}
        for key,n in native.items():
            ck=digest({k:v for k,v in n['state'].items()if k!='preprocessing'})
            if ck in content:assert native[content[ck]]['pixel_sha256']==n['pixel_sha256']
            else:content[ck]=key
        missing={};used=set()
        def take(p):
            key=digest(p);r=decode.get(key)
            if r is None:missing[key]=dict(type='SOURCE_STREAM',projection=p);return None
            assert r['projection']==p;used.add(key);return r['value']
        def state(m,C,pos,values,label):
            st=state_identity(m,C,pos,values,'',bundle['route']);ck=digest({k:v for k,v in st.items()if k!='preprocessing'});key=content.get(ck)
            if key is None:missing[ck]=dict(type='NATIVE_STATE',state=st);return dict(state=label,pending=True)
            return dict(state=label,state_key=key,pixel_sha256=native[key]['pixel_sha256'])
        gray=dict(state='GRAY',state_key=None,pixel_sha256=asha(np.full((3,256,256),.5,np.float32)))
        def replay(v):
            if not v['header']['header_ok']or v['profile']['null']:return {k:gray for k in ['ALL','SEQ','SEL']}
            p=v['profile'];streams=v['streams'];c=streams[0]
            if c is None:return {k:gray for k in ['ALL','SEQ','SEL']}
            if p['mode'].startswith('S_'):
                pp=dict(status='SOURCE_INPUT',codec='NEST',m=p['m'],K=p['K'],family='EC_VAR_PARTIAL'if p['K']else'EC_VAR_WHOLE',bits=c);s=take(pp)
                if s is None:return {}
                n=state(s['m'],s['coarse'],s['positions'],s['values'],'SINGLE');return {k:n for k in ['ALL','SEQ','SEL']}
            C=take(dict(type='C',m=p['m'],bits=c))
            if C is None:return {}
            tokens={};K=p['K'];positions=dict(A=list(range(K//2)),B=list(range(K//2,K)))
            for i,label in enumerate(['A','B'],1):
                if not positions[label]:tokens[label]=[];continue
                if streams[i]is None:continue
                vv=take(dict(type='BOUNDARY',m=p['m'],C_bits=c,positions=positions[label],bits=streams[i]))
                if vv is not None:tokens[label]=vv
            answer={}
            for mode,label in modes(True,'A'in tokens,'B'in tokens,K).items():
                if label=='GRAY':answer[mode]=gray;continue
                pos=[];vals=[]
                for role in ['A','B']:
                    if role in label:pos+=positions[role];vals+=tokens[role]
                answer[mode]=state(p['m'],C,pos,vals,label)
            return answer
        physical=[];replayed=[];seen={};receipts=[]
        for role in ['actual_frames_00','actual_frames_01']:
            w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and gone(w['owner'])and gone(w['process']);done=read(L/role/'completion.json');physical+=cr(done['events']);receipts.append(dict(completion=pin(L/role/'completion.json'),wait=pin(L/(role+'_control')/'actual_wait.json')))
        assert len(physical)==1440
        for e in physical:
            p=e['rx_record']['path']
            if p not in seen:seen[p]=cr(e['rx_record'])
            v=seen[p]['events'][e['rx_index']];assert not v['TX_truth_used'];replayed.append(dict(event_id=e['event_id'],modes=replay(v),TX_truth_used=False))
        save(out/'actual_RX_replay.json',replayed);save(out/'cache_misses.json',missing)
        if missing:
            save(out/'completion.json',dict(status='PENDING_ACTUAL_RX_MISSES',missing=len(missing),new_frames=1440,receipts=receipts));print(json.dumps(dict(missing=len(missing))));return
        scores=read(old/'p1_summary/pair_scores.json');scores.update(read(L/'clean_pair_scores.json'));sources={s['source_id']:s for s in read(OLD/'source_manifest.json')};rx={v['event_id']:v['modes']for v in replayed};rows=[]
        for e in physical:
            for mode in(['SINGLE']if e['arm'].startswith('S_')else['ALL','SEQ','SEL']):
                v=rx[e['event_id']]['SEL'if mode=='SINGLE'else mode];s=sources[e['source_id']];key=digest(dict(source_id=e['source_id'],reference_sha256=s['reference_sha256'],pixel_sha256=v['pixel_sha256']));assert key in scores,'Need actual source/pixel pair scoring; never substitute expected clean image'
                rows.append(dict(event_id=e['event_id'],arm=e['arm'],mode=mode,source_id=e['source_id'],source_index=e['source_index'],snr_db=e['snr_db'],replicate=e['replicate'],state=v['state'],pair_key=key,origin='NEW_ACTUAL_FRAME',**{k:scores[key][k]for k in METRICS}))
        oldnames={'S_m5_K27_r1-3':'OLD_STRONG_SINGLE','T_m4_K25_N598_r1-3_1-3':'OLD_BEST_THREE'};oldrows=list(csv.DictReader((OLD/'replay_metrics.csv').open()))
        reused=set()
        for v in oldrows:
            if v['candidate_id']not in oldnames:continue
            reused.add(v['event_id']);rows.append(dict(event_id=v['event_id'],arm=oldnames[v['candidate_id']],mode=v['mode'],source_id=v['source_id'],source_index=int(v['source_index']),snr_db=int(v['snr_db']),replicate=int(v['replicate']),state=v['state'],pair_key=v['pair_key'],origin='EXACT_OLD_FRAME_REUSE',**{k:float(v[k])for k in METRICS}))
        assert len(reused)==576;csvsave(O/'wireless_metrics.csv',rows);csvsave(O/'physical_events.csv',physical)
        groups=collections.defaultdict(list)
        for x in rows:groups[x['arm'],x['mode'],x['snr_db'],x['source_id']].append(x)
        def mean(rr):return {k:float(np.mean([r[k]for r in rr],dtype=np.float64))for k in METRICS}
        means=[]
        for(arm,mode,snr,sid),rr in groups.items():
            assert len(rr)==3;means.append(dict(arm=arm,mode=mode,snr_db=snr,source_id=sid,source_index=rr[0]['source_index'],**mean(rr)))
        csvsave(O/'source_means.csv',means);summary=[];overall=[];counts=[]
        for arm,mode in sorted({(x['arm'],x['mode'])for x in means}):
            for snr in[0,2,4]:
                rr=[x for x in means if(x['arm'],x['mode'],x['snr_db'])==(arm,mode,snr)];assert len(rr)==32;summary.append(dict(arm=arm,mode=mode,snr_db=snr,n_sources=32,logical_frames=96,**mean(rr)))
                cc=collections.Counter(x['state']for x in rows if(x['arm'],x['mode'],x['snr_db'])==(arm,mode,snr));assert sum(cc.values())==96
                counts.extend(dict(arm=arm,mode=mode,snr_db=snr,state=st,count=val,denominator=96)for st,val in cc.items())
            rr=[x for x in summary if(x['arm'],x['mode'])==(arm,mode)];overall.append(dict(arm=arm,mode=mode,n_sources=32,logical_frames=288,**mean(rr)))
        csvsave(O/'wireless_snr_summary.csv',summary);csvsave(O/'wireless_overall.csv',overall);csvsave(O/'receive_state_counts.csv',counts)
        control=sorted([r for r in overall if r['mode']=='SINGLE'],key=lambda r:(-r['dino_cosine'],r['arm']))[0];lookup={(x['arm'],x['mode'],x['snr_db'],x['source_id']):x for x in means};pairs=[]
        for x in means:
            ref=lookup[control['arm'],'SINGLE',x['snr_db'],x['source_id']];pairs.append(dict(arm=x['arm'],mode=x['mode'],reference=control['arm'],snr_db=x['snr_db'],source_id=x['source_id'],source_index=x['source_index'],**{k:x[k]-ref[k]for k in METRICS}))
        csvsave(O/'paired_source_differences.csv',pairs);paired=[]
        for arm,mode in sorted({(x['arm'],x['mode'])for x in pairs}):
            rr=[x for x in pairs if(x['arm'],x['mode'])==(arm,mode)];paired.append(dict(arm=arm,mode=mode,reference=control['arm'],n_sources=32,**mean(rr)))
        csvsave(O/'paired_differences.csv',paired)
        qualify=[r for r in overall if r['arm']in('SHARED_OLD','COMPACT_FIXED','COMPACT_TOKEN')and r['mode']=='SEL'and r['dino_cosine']>control['dino_cosine']]
        closure=dict(status='COMPLETE_A_32_SOURCE',best_single=control,qualifying_shared_candidates=qualify,recommendation='GATED_128_DEVELOPMENT_CONFIRMATION'if qualify else'NO_SHARED_ROUTE_EXPANSION',new_actual_frames=1440,exact_old_frames=576,total_physical_comparison_frames=2016,logical_replay_rows=len(rows),unique_RX_decode_cache_hits=len(used),new_RX_model_calls=0,new_metric_calls=0,header_accepted_wrong=sum(x['header_accepted_wrong']for x in physical),payload_accepted_wrong=sum(len(x['payload_accepted_wrong'])for x in physical),statistical_scope='32 calibration development sources; equal0/2/4dB, three noises averaged per source. New arms common AWGN; historical arms source-paired only. No bootstrap or held-out generalization claim.',receipts=receipts)
        save(O/'closure_32.json',closure);save(out/'completion.json',dict(status='COMPLETE_ACTUAL_FRAME_SUMMARY',closure=pin(O/'closure_32.json')));print(json.dumps(dict(overall=overall,closure=closure)))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
