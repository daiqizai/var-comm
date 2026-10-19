"""Read-only N1024 source/event/pixel/score audit. No scientific entry imports."""
from pathlib import Path
import csv, hashlib, json, os, sys, time, traceback, collections
import numpy as np

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
L = B.parent / 'n1024_consistency_20261019_v1'
O = R / 'results/n1024_consistency_20261019_v1'
T = B / 't1c_m10_test_20261016_v1'
EM = B / 'entropy_main_common500_20261013_v1'
SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction', 'dists', 'dreamsim', 'ms_ssim')
verified = {}

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def pin(p):
    p = Path(p)
    a = p.stat(); data = p.read_bytes(); z = p.stat()
    assert (a.st_size, a.st_mtime_ns) == (z.st_size, z.st_mtime_ns), str(p)
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def checked(d):
    p = Path(d['path'])
    assert p.is_absolute() and '..' not in p.parts and not p.is_symlink()
    if str(p) not in verified:
        actual = pin(p)
        assert actual['sha256'] == d['sha256'] and actual['bytes'] == d.get('bytes', actual['bytes']), str(p)
        verified[str(p)] = actual
    assert verified[str(p)]['sha256'] == d['sha256']
    return p

def read(p):
    return json.loads(Path(p).read_bytes())

def cr(d):
    return read(checked(d))

def csvread(p):
    p = Path(p); verified[str(p)] = pin(p)
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(p, v):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def csvsave(p, rows):
    assert rows
    with Path(p).open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())

def asha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def ident(pid):
    p = Path(f'/proc/{pid}/stat'); raw = p.read_text(); fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]))

def nt(e):
    m, k = e
    return sum(v*v for v in SIZES[:m]) + k

def choose(streams, family, snr):
    targets = {('WHOLE', 4):(7,0), ('WHOLE',10):(9,0), ('PARTIAL',4):(7,25), ('PARTIAL',10):(8,126)}
    target = targets.get((family,snr), (10,0))
    legal = [(m,k) for m,k in streams if nt((m,k)) <= nt(target) and (family == 'PARTIAL' or k == 0)]
    cap = {4:1245,10:2520,19:4751}[snr]
    for e in sorted(legal, key=nt, reverse=True):
        if len(streams[e]) <= cap:
            return e, cap, target
    raise AssertionError('No actual source fits; no synthetic null substituted')

def number(v):
    if str(v) in ('True', 'False'):
        return float(str(v) == 'True')
    return float(v)

def pixels(rec, cache):
    d = rec['image_archive']; p = checked(d); key = rec.get('image_key', 'image'); slot = rec.get('image_slot')
    signature = (str(p), key, slot)
    if signature not in cache:
        with np.load(p, allow_pickle=False) as z:
            a = z[key]; a = a if slot is None else a[int(slot)]
            assert a.shape == (3,256,256) and a.dtype == np.float32 and np.isfinite(a).all()
            cache[signature] = asha(a)
    assert cache[signature] == rec['image_sha256'], signature

def event_identity(pkt):
    tx = pkt['transmission']; ident = pkt.get('observation_identity', pkt.get('event', pkt))
    return dict(source_id=ident['source_id'], source_index=ident['source_index'], snr_db=ident['snr_db'], noise_seed=ident['noise_seed'],
                public_frame_counter=ident['public_frame_counter'], payload_sha256=ident['payload_sha256'],
                profile_id=tx['profile_id'], profile_key=tx['profile_key'], session=tx['session'],
                n=tx['n'], k=tx['k'], q=tx['q'], N=tx['N'], header_symbols=tx['header_symbols'],
                body_symbols=tx['body_symbols'], length_field_bits=tx['length_field_bits'], crc_bits=tx['crc_bits'])

def actual_rx_check(pkt, expected_bits):
    rx = pkt['actual_RX']; tx = pkt['transmission']
    assert tx['N'] == 1024 and tx['header_symbols'] == 68 and tx['body_symbols'] == 956
    assert tx['length_field_bits'] == 13 and tx['crc_bits'] == 16
    h = rx['header']; b = rx['body']
    if h['header_ok']:
        assert h['header_crc_ok'] and h['header_fields_legal']
        assert rx['rx_profile']['profile_id'] == h['profile_id']
    if b is not None and b['crc_accepted'] and b['parser_accepted']:
        # This is post-output audit only; no received value is corrected.
        a = np.asarray(b['payload'], dtype=np.uint8)
        assert len(a) == b['declared_length']
        return np.array_equal(a, expected_bits)
    return None

def main():
    start = time.monotonic(); out = L / 'audit_v2'; out.mkdir(); O.mkdir(exist_ok=True); assert not list(O.iterdir()), 'Prior output must never be overwritten'
    save(out/'launch.json', dict(process=ident(os.getpid()), parent_process=ident(os.getppid()), argv=sys.argv,
                               new_channel=0, new_models=0, new_scores=0, new_bootstrap=0))
    try:
        sd = read(T/'source/completion.json'); sources = cr(sd['sources'])
        assert len(sources) == 600 and sd['ledger']['unresolved'] == 0
        common = [s for s in sources if s['population'] == 'common500']; assert len(common) == 500
        catd = pin(R/'results/t1b_m10_calibration_20261016_v1/public_catalogue.json'); cat = cr(catd)
        profiles = {(p['m'],p['K'],p['q'],p['nominal_rate']):p for p in cat['profiles'] if p['family'] != 'EC_STATIC_WHOLE'}
        preg = read(R/'results/entropy_main_common500_20261013/registration_v4.json')
        assert preg['policies']['4']['candidate_id'] == 'm7_K25_q2_r2-3'
        assert preg['policies']['10']['candidate_id'] == 'm8_K126_q4_r2-3'
        wholecsv = R/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/per_frame.csv'
        oldw = {(int(r['source_index']),int(r['snr_db']),int(r['noise_seed'])):r for r in csvread(wholecsv) if r['family']=='EC_VAR_WHOLE' and int(r['snr_db']) in (4,10)}
        oldp = {(int(r['source_index']),int(r['snr_db']),int(r['noise_seed'])):r for r in csvread(R/'results/entropy_main_common500_20261013/frame_rows.csv') if int(r['snr_db']) in (4,10)}
        tc = {(r['source_index'],r['strategy_family'],r['noise_seed']):r for r in cr(read(T/'metrics/completion.json')['rows']) if r['population']=='common500'}
        tcf = {(int(r['source_index']),r['family'],int(r['noise_seed'])):r for r in csvread(R/'results/t1c_m10_test_20261016_v1/delivery/frame_scores_19dB.csv') if r['population']=='common500'}
        assert len(oldw)==len(oldp)==len(tc)==len(tcf)==3000
        wr = read(L/'whole_evidence_restored/restoration_manifest.json'); wmap = {v['relative']:v for v in wr['files']}; assert len(wmap)==3000
        bridge = B/'quality_metrics_supplement_20261014_v1/digital_input_bridge_v2'
        bm = read(bridge/'manifest.json'); bf = cr(bm['frame_index'])
        bmap = {(v['source_index'],int(v['snr_db']),v['noise_seed']):v for v in bf if v['method']=='EC_VAR_WHOLE' and int(v['snr_db']) in (4,10)}
        supd = read(B/'quality_metrics_supplement_20261014_v1/common500_metrics_v1/completion.json')
        supp = {(v['method'],int(v['source_index']),int(v['snr_db']),int(v['noise_seed'])):v for v in csvread(checked(supd['scores'])) if v['method'] in ('EC_VAR_WHOLE','EC_VAR_PARTIAL')}
        pd = read(EM/'visual/completion.json'); pframes = cr(pd['frames'])
        source_rows=[]; event_rows=[]; changed=[]; current_frames=[]; pixel_cache={}; counters=collections.Counter(); physicals={}
        for s in common:
            i=s['source_index']; sid=s['source_id']; checked(s['archive']); checked(s['old_metadata']); checked(s['old_streams']); checked(s['current_streams'])
            with np.load(s['archive']['path'], allow_pickle=False) as z:
                tokens=z['tokens']; a=z['pixels']
                assert tokens.dtype==np.int64 and tokens.shape==(680,)
                assert hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
                assert a.dtype==np.uint8 and a.shape==(3,256,256) and hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
            with np.load(s['current_streams']['path'],allow_pickle=False) as z:
                streams={(int(k.split('_')[0][1:]),int(k.split('_')[1][1:])):z[k] for k in z.files}
            with np.load(s['old_streams']['path'],allow_pickle=False) as z:
                for ep in s['old_endpoints']:
                    a=streams[ep['m'],ep['K']]; assert a.dtype==np.uint8 and a.ndim==1 and len(a)==ep['bits']
                    assert asha(a)==ep['payload_sha256'] and np.array_equal(a,z[f'm{ep["m"]}_K{ep["K"]}'])
            assert len(streams)==25 and len(streams[10,0])==s['m10']['arithmetic_bits'] and asha(streams[10,0])==s['m10']['payload_sha256']
            local_events={}
            for snr in (4,10):
                for seed in (6201,6202,6203):
                    idx=i*9+(0 if snr==4 else 3)+(seed-6201); vd=pframes[idx]; v=cr(vd); assert (v['source_index'],v['snr_db'],v['noise_seed'])==(i,snr,seed)
                    pkt=cr(v['physical_frame']); pr=oldp[i,snr,seed]; rec=dict(v['reconstruction'],image_key='image',image_slot=None)
                    pixels(rec,pixel_cache); assert rec['image_sha256']==pr['image_sha256']; sp=supp['EC_VAR_PARTIAL',i,snr,seed]
                    assert sp['reconstruction_rgb_sha256']==rec['image_sha256']
                    local_events['PARTIAL',snr,seed]=(v['physical_frame'],pkt,rec,{k:number(pr[k]) if k in pr else number(sp[k]) for k in METRICS},vd)
                    row=oldw[i,snr,seed]; rel=f'posthoc_render500_v1/physical_frames/{i:04d}/{snr}_{seed}_{int(row["received_profile_id"]):03d}.json'; wd=wmap[rel]; wp=cr(wd); bv=bmap[i,snr,seed]
                    rec=dict(image_archive=bv['archive'],image_key=bv['image_key'],image_slot=bv['image_slot'],image_sha256=bv['reconstruction_sha256'])
                    pixels(rec,pixel_cache); assert rec['image_sha256']==row['image_sha256']; sp=supp['EC_VAR_WHOLE',i,snr,seed]; assert sp['reconstruction_rgb_sha256']==rec['image_sha256']
                    local_events['WHOLE',snr,seed]=(wd,wp,rec,{k:number(row[k]) if k in row else number(sp[k]) for k in METRICS},pin(wholecsv))
            for family in ('WHOLE','PARTIAL'):
                for snr in (4,10,19):
                    ep, cap, target=choose(streams,family,snr); bits=streams[ep]; sha=asha(bits); q={4:2,10:4,19:6}[snr]; rate='5/6' if snr==19 else '2/3'; profile=profiles[ep[0],ep[1],q,rate]
                    assert profile['k']-29==cap
                    status_set=set(); oldbits_set=set(); oldend_set=set()
                    for seed in (6201,6202,6203):
                        if snr==19:
                            r=tc[i,family,seed]; d=r['physical_frame']; pkt=cr(d); rec=r['reconstruction']; pixels(rec,pixel_cache); met=r['metrics']; oldrow=tcf[i,family,seed]
                            assert all(number(oldrow[k])==float(met[k]) for k in METRICS) and oldrow['image_sha256']==rec['image_sha256']
                            original_row=pin(R/'results/t1c_m10_test_20261016_v1/delivery/frame_scores_19dB.csv')
                        else:
                            d,pkt,rec,met,original_row=local_events[family,snr,seed]
                        ei=event_identity(pkt); tx=pkt['transmission']; assert ei['source_id']==sid and ei['source_index']==i and ei['snr_db']==snr and ei['noise_seed']==seed
                        expected_fc=([1,4,7,10,13,19].index(snr)*500+i)*3+seed-6201; assert ei['public_frame_counter']==expected_fc
                        desired=dict(ei,payload_sha256=sha,profile_id=profile['profile_id'],profile_key=profile['profile_key'])
                        same=(ei==desired and tx['arithmetic_bits']==len(bits) and (tx['m'],tx['K'])==ep)
                        actual_rx_check(pkt,streams[tx['m'],tx['K']])
                        origin='EXACT_ORIGINAL_EVENT' if same else 'SOURCE_VERSION_OR_ENDPOINT_DIFFERENCE'
                        reuse=d if same else None; reuse_rec=rec if same else None; reuse_met=met if same else None
                        if not same and snr in (4,10):
                            alt=local_events['PARTIAL',snr,seed]
                            if event_identity(alt[1])==desired and alt[1]['transmission']['arithmetic_bits']==len(bits):
                                reuse,reuse_rec,reuse_met=alt[0],alt[2],alt[3];origin='EXACT_CROSS_FAMILY_EVENT'
                        if snr==19: assert same, ('T1C mismatch',i,family,seed)
                        oldep=(tx['m'],tx['K']); oldbits_set.add(tx['arithmetic_bits']);oldend_set.add(oldep);status_set.add(origin)
                        counters[family+'_'+str(snr)+'_'+origin]+=1
                        actualhash=asha(np.asarray(pkt['actual_RX']['body']['payload'],dtype=np.uint8)) if pkt['actual_RX']['body'] is not None and pkt['actual_RX']['body']['parser_accepted'] else None
                        if actualhash is not None: assert actualhash==ei['payload_sha256'], ('historical RX does not match TX',i,family,snr,seed)
                        er=dict(population='common500',source_index=i,source_id=sid,family=family,snr_db=snr,noise_seed=seed,
                                actual_m=ep[0],actual_K=ep[1],arithmetic_bits=len(bits),payload_sha256=sha,profile_id=profile['profile_id'],profile_key=profile['profile_key'],
                                old_m=oldep[0],old_K=oldep[1],old_arithmetic_bits=tx['arithmetic_bits'],old_payload_sha256=ei['payload_sha256'],old_profile_id=tx['profile_id'],old_profile_key=tx['profile_key'],
                                old_RX_status=pkt['actual_RX']['status'],old_image_sha256=rec['image_sha256'],old_score_pair_verified=True,source_side_byte_exact=sha==ei['payload_sha256'],
                                reuse_class=origin,quality_reusable=reuse is not None,physical_frame_path=d['path'],physical_frame_sha256=d['sha256'],
                                reused_physical_frame_path='' if reuse is None else reuse['path'],reused_image_sha256='' if reuse_rec is None else reuse_rec['image_sha256'],
                                new_channel_required=reuse is None,source_preprocessing_sha256=s['preprocessing_id'],tokens_sha256=s['tokens_sha256'])
                        event_rows.append(er)
                        if reuse is None:
                            key=digest(desired); changed.append(dict(event_key=key,**er,desired_event=desired,source_archive=s['archive'],stream_archive=s['current_streams'],stream_key=f'm{ep[0]}_K{ep[1]}',paid_profile=profile,old_physical_frame=d,old_reconstruction=rec,old_metrics=met))
                        else:
                            current_frames.append(dict(population='common500',source_index=i,source_id=sid,family=family,snr_db=snr,noise_seed=seed,m=ep[0],K=ep[1],arithmetic_bits=len(bits),source_capacity_bits=cap,utilization=len(bits)/cap,physical_frame=reuse,reconstruction=reuse_rec,metrics=reuse_met,reuse_class=origin))
                    assert len(oldbits_set)==len(oldend_set)==1
                    source_rows.append(dict(population='common500',source_index=i,source_id=sid,family=family,snr_db=snr,target_m=target[0],target_K=target[1],actual_m=ep[0],actual_K=ep[1],actual_tokens=nt(ep),q=q,nominal_rate=rate,n=956*q,k=cap+29,length_field_bits=13,CRC_bits=16,source_capacity_bits=cap,terminated_arithmetic_bits=len(bits),remaining_capacity_bits=cap-len(bits),utilization=len(bits)/cap,stream_version='T1C_FROZEN_H800_SOURCE600_OLD24_BYTE_EXACT',stream_archive=s['current_streams']['path'],stream_archive_sha256=s['current_streams']['sha256'],payload_sha256=sha,paid_profile_id=profile['profile_id'],paid_profile_key=profile['profile_key'],source_preprocessing_sha256=s['preprocessing_id'],tokens_sha256=s['tokens_sha256'],historical_terminated_bits=next(iter(oldbits_set)),historical_actual_m=next(iter(oldend_set))[0],historical_actual_K=next(iter(oldend_set))[1],quality_reuse_class=';'.join(sorted(status_set))))
            if (i+1)%50==0:print('AUDITED',i+1,'SOURCES','CHANGED_LOGICAL',len(changed),flush=True)
        assert len(source_rows)==3000 and len(event_rows)==9000 and len(current_frames)+len(changed)==9000
        summaries=[]; ends=[]
        for family in ('WHOLE','PARTIAL'):
            for snr in (4,10,19):
                rr=[r for r in source_rows if r['family']==family and r['snr_db']==snr];assert len(rr)==500
                a=np.asarray([r['utilization'] for r in rr]); bits=np.asarray([r['terminated_arithmetic_bits'] for r in rr]);margin=np.asarray([r['remaining_capacity_bits'] for r in rr])
                cnt=collections.Counter((r['actual_m'],r['actual_K']) for r in rr)
                summaries.append(dict(family=family,snr_db=snr,sources=500,source_records=500,logical_frames=1500,source_capacity_bits=rr[0]['source_capacity_bits'],mean_terminated_bits=float(bits.mean()),mean_utilization=float(a.mean()),median_utilization=float(np.median(a)),min_utilization=float(a.min()),p05_utilization=float(np.quantile(a,.05)),p95_utilization=float(np.quantile(a,.95)),max_utilization=float(a.max()),mean_remaining_bits=float(margin.mean()),endpoints=len(cnt),quality_reusable_logical=sum(r['quality_reusable'] for r in event_rows if r['family']==family and r['snr_db']==snr)))
                ends += [dict(family=family,snr_db=snr,m=e[0],K=e[1],sources=n,fraction=n/500) for e,n in sorted(cnt.items())]
        csvsave(O/'source_records_3000.csv',source_rows);csvsave(O/'family_snr_summary.csv',summaries);csvsave(O/'endpoint_distribution.csv',ends);csvsave(O/'event_reuse_audit_9000.csv',event_rows)
        save(L/'repair_events.json',changed);save(L/'reused_frames.json',current_frames)
        unique={x['event_key'] for x in changed}; budget=dict(changed_logical_frames=len(changed),unique_new_physical=len(unique),max_header_decodes=len(unique),max_body_decodes=len(unique),max_packet_decodes=2*len(unique),new_sources=0,new_snr=0,new_noise=0,new_TX_probability=0,new_tokenizer=0,new_policy_selection=0,execution_status='NOT_RUN_METADATA_ONLY')
        save(O/'repair_budget.json',budget)
        save(O/'input_manifest.json',dict(files=list(verified.values()),source600=sd['sources'],public_catalogue=catd,old_pixel_pairs_verified=len(pixel_cache),new100_included=False))
        save(O/'completion.json',dict(status='AUDIT_COMPLETE_REPAIR_PENDING' if changed else 'AUDIT_COMPLETE_ALL_REUSED',source_records=3000,event_rows=9000,counts=dict(counters),repair_budget=budget,repair_events=pin(L/'repair_events.json'),reused_frames=pin(L/'reused_frames.json'),new_scientific_calls=0,new_bootstrap=0,seconds=time.monotonic()-start))
        save(out/'completion.json',dict(delivery=pin(O/'completion.json'),new_scientific_calls=0));print(json.dumps(read(O/'completion.json'),indent=2))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),new_scientific_calls=0));raise

if __name__=='__main__':
    main()
