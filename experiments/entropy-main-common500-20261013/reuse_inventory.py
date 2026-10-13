"""Read-only E0 audit/export of already completed common500 reference results.

No models, channel draws, decoders, reconstruction, or bootstrap are imported.
Only this independent E0 output directory is written; historical files are hashed
before and after. Per-source averaging below is deterministic table aggregation.
"""
from pathlib import Path
import collections
import csv
import gzip
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parents[3]
TMP = BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011'
RT = ROOT.parents[1]/'var_comm_runtime_20261010'
OUT = ROOT/'results/entropy_main_common500_20261013/e0_reuse'
OLD = TMP/'postprocess_original500_v1'
T1 = ROOT/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3'
BPG = ROOT/'results/wcl_evidence_closure_20261009/T4_resources_cost/bpg_existing_v1'
FINAL = ROOT/'results/baseline_strength_20261011/paired_supplement_v1/common500'
CHAIN = ROOT/'results/baseline_strength_20261011/automatic_followthrough_v1'
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction')
SNRS = (4, 10, 19)
METHODS = ('EC_VAR_WHOLE', 'RAW_WHOLE', 'RAW_PARTIAL', 'P100000', 'Swin200000', 'BPG_ADAPTIVE')
PINS = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p):
    p = Path(p); PINS[str(p)] = sha(p)
    return dict(path=str(p), sha256=PINS[str(p)], bytes=p.stat().st_size)

def read(p):
    pin(p)
    return json.loads(Path(p).read_text())

def rows(p):
    pin(p)
    with Path(p).open(newline='') as stream:
        return list(csv.DictReader(stream))

def save(name, value):
    p = OUT/name
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text() == text, 'Existing audit differs; preserve and use separate version: '+str(p)
    else:
        p.write_text(text)

def csv_save(name, values):
    assert values
    p = OUT/name
    assert not p.exists(), 'Already exported; do not overwrite '+str(p)
    keys = list(dict.fromkeys(k for r in values for k in r))
    with p.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(values)

def number(value):
    return float({'True':1., 'False':0.}.get(value, value))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(TMP/'common500_original_sources/manifest.json')
    ids = manifest['source_ids']
    assert len(ids) == len(set(ids)) == manifest['source_count'] == 500
    assert ids == read(OLD/'source_ids.json')
    source_records = read(OUT/'source_records.json')
    assert source_records['source_ids'] == ids and source_records['source_count'] == 500
    for i, (record, original) in enumerate(zip(source_records['records'], manifest['records'])):
        assert record['source_id'] == ids[i] == original['source_id'] and record['source_index'] == i
        p = Path(record['archive']['path']); assert pin(p)['sha256'] == record['archive']['sha256'] == original['archive_sha256']
        c = Path(record['source_checkpoint']['path']); assert pin(c)['sha256'] == original['checkpoint_sha256']
        checkpoint = read(c)
        assert checkpoint['encoder_tokens_verified'] and checkpoint['asset_readback_exact']
        assert checkpoint['tokens_sha256'] == original['tokens_sha256'] == record['tokens_sha256']
        with np.load(p, allow_pickle=False) as z:
            tokens, pixels = z['tokens'], z['pixels']
            assert tokens.dtype == np.int64 and tokens.shape == (680,) and ((tokens >= 0) & (tokens < 4096)).all()
            assert pixels.dtype == np.uint8 and pixels.shape == (3,256,256)
            assert hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest() == record['tokens_sha256']
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == record['pixel_sha256']
    chain = read(CHAIN/'registration.json')
    assert [r['source_id'] for r in chain['populations']['common500']['records']] == ids
    for a, b in zip(chain['populations']['common500']['records'], source_records['records']):
        assert a['pixels']['sha256'] == b['archive']['sha256'] and a['preprocessing_id'] == b['preprocessing_id']

    source_means, summaries, original_summary_rows, inventory = [], [], [], []
    vectors = {}
    frame_audits = {}
    def add_means(method, values, origin):
        groups = collections.defaultdict(dict)
        for r in values:
            snr = int(r['snr_db']); metric = r['metric']; i = int(r['source_index'])
            if snr not in SNRS or metric not in METRICS: continue
            assert 0 <= i < 500 and r['source_id'] == ids[i]
            key = (method, snr, metric)
            assert i not in groups[key], ('duplicate source', key, i)
            groups[key][i] = number(r['mean'])
        assert len(groups) == 12
        for key, g in groups.items():
            assert sorted(g) == list(range(500))
            vector = np.array([g[i] for i in range(500)], dtype=np.float64)
            assert np.isfinite(vector).all()
            vectors[key] = vector
            source_means.extend(dict(method=method, snr_db=key[1], metric=key[2], source_index=i,
                source_id=ids[i], mean=g[i], source_count=500, noise_count=3, origin=origin) for i in range(500))

    def add_summary(method, values, origin):
        count = 0
        for r in values:
            s, m = int(r['snr_db']), r['metric']
            if s not in SNRS or m not in METRICS: continue
            assert int(r['source_count']) == 500 and int(r['noise_count']) == 3
            assert number(r['ci_low']) <= number(r['mean']) <= number(r['ci_high'])
            assert abs(number(r['mean'])-float(vectors[method,s,m].mean())) < 1e-12
            original_summary_rows.append(dict(method=method, origin=origin, original_row=r))
            summaries.append(dict(r, method=method, snr_db=s, metric=m,
                interval_provenance='ORIGINAL_PUBLISHED_INTERVAL_UNCHANGED', original_summary_file=origin))
            count += 1
        assert count == 12

    def aggregate_frames(method, values, seeds, origin):
        values = [r for r in values if int(r['snr_db']) in SNRS]
        keyed = {}
        for r in values:
            key = int(r['source_index']), int(r['snr_db']), int(r['noise_seed'])
            assert key not in keyed and 0 <= key[0] < 500 and r['source_id'] == ids[key[0]]
            assert key[2] in seeds
            keyed[key] = r
        assert set(keyed) == {(i,s,n) for i in range(500) for s in SNRS for n in seeds}
        result = []
        for s in SNRS:
            for metric in METRICS:
                for i in range(500):
                    mean = float(np.mean([number(keyed[i,s,n][metric]) for n in seeds], dtype=np.float64))
                    result.append(dict(source_index=i, source_id=ids[i], snr_db=s, metric=metric, mean=mean))
        frame_audits[method] = dict(rows=4500, source_count=500, noise_seeds=seeds, no_duplicates=True,
            full_grid=True, origin=origin, status_counts=dict(collections.Counter(
                r.get('source_status', r.get('failure_state', r.get('fallback','UNKNOWN'))) for r in values)))
        return result

    # Old raw arms: exact stored per-source averages; no recomputation or resampling.
    pin(OLD/'source_means.json.gz')
    with gzip.open(OLD/'source_means.json.gz', 'rt') as stream: raw_means = json.load(stream)
    raw_summary = read(OLD/'summary.json'); raw_points = read(OLD/'points.json')
    for method, prefix in [('RAW_WHOLE','RAW64_WHOLE_VAR_COMPLETION'),('RAW_PARTIAL','RAW64_PARTIAL_VAR_COMPLETION')]:
        selected=[]; selected_summary=[]
        for s in SNRS:
            point=prefix+'_SNR_'+str(s)
            assert raw_points[point]['noise_seeds'] == [6201,6202,6203]
            selected.extend(dict(r,snr_db=s) for r in raw_means if r['point_id']==point and r['metric'] in METRICS)
            selected_summary.extend(r for r in raw_summary if r['point_id']==point and r['metric'] in METRICS)
        add_means(method, selected, str(OLD/'source_means.json.gz'))
        add_summary(method, selected_summary, str(OLD/'summary.json'))

    # Prior whole-scale entropy result and actual lengths/fallbacks.
    t1_rows=[r for r in rows(T1/'per_frame.csv') if r['family']=='EC_VAR_WHOLE']
    t1_means=aggregate_frames('EC_VAR_WHOLE',t1_rows,[6201,6202,6203],str(T1/'per_frame.csv'))
    add_means('EC_VAR_WHOLE',t1_means,str(T1/'per_frame.csv'))
    add_summary('EC_VAR_WHOLE',[r for r in rows(T1/'summary.csv') if r['point_id'].startswith('EC_VAR_WHOLE_SNR_')],str(T1/'summary.csv'))
    length_rows=[r for r in rows(T1/'source_lengths.csv') if r['population']=='holdout' and r['family']=='EC_VAR_WHOLE']
    assert len(length_rows)==3000 and {(int(r['source_index']),int(r['complete_scale_m'])) for r in length_rows}=={(i,m) for i in range(500) for m in range(4,10)}
    for r in length_rows: assert r['source_id']==ids[int(r['source_index'])] and r['length_status']=='ACTUAL_ENCODED_LENGTH'
    whole_policy=read(T1/'frozen_policy.json')['policies']['EC_VAR_WHOLE']
    winners=read(ROOT/'experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_calibration_winners_v2.json')
    for s in SNRS:
        assert whole_policy[str(s)]['candidate_id']==winners['snrs'][str(s)]['original_final_whole_winner']
        assert all(r['candidate_id']==whole_policy[str(s)]['candidate_id'] for r in t1_rows if int(r['snr_db'])==s)
    chosen=[]
    for s in SNRS:
        for i in range(500):
            three=[r for r in t1_rows if int(r['snr_db'])==s and int(r['source_index'])==i]
            keys=('actual_m','K','transmitted_token_count','arithmetic_bits','actual_k','fallback_attempts','target_m')
            assert len(three)==3 and len({tuple(r[k] for k in keys) for r in three})==1
            r=three[0]
            chosen.append(dict(method='EC_VAR_WHOLE',source_index=i,source_id=ids[i],snr_db=s,target_m=int(r['target_m']),target_K=0,
                actual_m=int(r['actual_m']),actual_K=0,actual_T=int(r['transmitted_token_count']),arithmetic_bits=int(r['arithmetic_bits']),
                capacity_bits=int(r['actual_k'])-29,fallback_attempts_json=r['fallback_attempts'],original_file=str(T1/'per_frame.csv')))

    # New P/Swin already closed: use stored means, verify against complete frame grids.
    final_means=rows(FINAL/'source_means.csv'); final_summary=rows(FINAL/'summary.csv')
    for method in ('P100000','Swin200000'):
        frame_file=CHAIN/'common500'/(method+'.csv')
        aggregate=aggregate_frames(method,rows(frame_file),[2001,2002,2003],str(frame_file))
        source=[r for r in final_means if r['method']==method and int(r['snr_db']) in SNRS]
        table={(int(r['source_index']),int(r['snr_db']),r['metric']):number(r['mean']) for r in source}
        assert all(number(r['mean'])==table[int(r['source_index']),int(r['snr_db']),r['metric']] for r in aggregate)
        add_means(method,source,str(FINAL/'source_means.csv'))
        add_summary(method,[r for r in final_summary if r['method']==method],str(FINAL/'summary.csv'))

    bpg_rows=rows(BPG/'per_frame.csv')
    bpg_means=aggregate_frames('BPG_ADAPTIVE',bpg_rows,[2001,2002,2003],str(BPG/'per_frame.csv'))
    add_means('BPG_ADAPTIVE',bpg_means,str(BPG/'per_frame.csv'))
    add_summary('BPG_ADAPTIVE',rows(BPG/'quality_summary_existing.csv'),str(BPG/'quality_summary_existing.csv'))
    assert len(source_means)==36000 and len(summaries)==72 and len(vectors)==72

    # Inspect only the already registered H800 caches, not the whole PFS tree.
    cache_inputs=[]; cache_audit=[]
    p=RT/'qualification/h800_ep_full_source900_v1_attempt1/registered/run/worker_completion.json';d=read(p)
    cache_inputs.extend(d['existing100_streams']+d['new900_streams']);cache_audit.append(dict(path=str(p),population='calibration1000',count=1000))
    p=RT/'qualification/h800_ep_new100_source_v1_attempt1/registered/run/worker_completion.json';d=read(p)
    cache_inputs.extend(d['sources']);cache_audit.append(dict(path=str(p),population='new100',count=100))
    overlap=sorted(set(ids)&{r['source_id'] for r in cache_inputs})
    assert not overlap, 'Unexpected common500 overlap requires exact cache matching before launch'
    stats=OLD/'raw64_common500_source_statistics.py'
    assert pin(stats)['sha256']=='e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
    for method in METHODS:
        sample=next(r for r in summaries if r['method']==method)
        mean_sample=next(r for r in source_means if r['method']==method)
        inventory.append(dict(method=method,population='original_common500',source_count=500,source_order=str(OUT/'source_order.json'),snrs='4,10,19',
            noise_seeds='6201,6202,6203' if method.startswith(('RAW_','EC_')) else '2001,2002,2003',
            source_means=mean_sample['origin'],summary=sample['original_summary_file'],status='REUSE_VERIFIED_COMPLETE',
            reason='Complete source/order/three-noise metric grid and published mean reconciliation; original intervals unchanged'))
    inventory.append(dict(method='EC_VAR_PARTIAL',population='original_common500',source_count=500,source_order=str(OUT/'source_order.json'),snrs='4,10,19',noise_seeds='6201,6202,6203',
        source_means='',summary='',status='NEEDS_REGISTERED_4500_FRAMES',reason='No source-id overlap with the registered H800 calibration1000/new100 terminated-stream caches; generate fresh H800 streams from verified original common500 tokens'))
    save('original_summary_rows.json',original_summary_rows)
    save('frame_grid_audit.json',frame_audits)
    csv_save('source_means.csv',source_means);csv_save('summary.csv',summaries)
    csv_save('whole_source_streams.csv',chosen);csv_save('reuse_inventory.csv',inventory)
    audit=dict(status='E0_REFERENCE_DATA_AND_SOURCE_REUSE_VERIFIED',source_count=500,source_archives_and_tokens_rehashed=500,
        reference_methods=list(METHODS),reference_source_mean_rows=36000,reference_summary_rows=72,
        original_intervals_recomputed=0,new_models=0,new_channel_calls=0,new_decoders=0,new_bootstrap_calls=0,
        source_order_sha256=sha(OUT/'source_order.json'),source_records_sha256=sha(OUT/'source_records.json'),
        existing_H800_source_caches=cache_audit,common500_cache_overlap=overlap,
        exact_missing_new_method_frames=4500,existing_source_tokens_reused=500,fresh_source_TX_needed=500,
        T1_lengths=dict(path=str(T1/'source_lengths.csv'),selected_family='EC_VAR_WHOLE',population='holdout',all_endpoint_rows=3000,chosen_source_SNR_rows=1500),
        original_statistics=pin(stats),bootstrap_contract=dict(source_count=500,replicates=10000,seed=2026100701),
        noise_counter_rule='([1,4,7,10,13,19].index(snr)*500+source_index)*3+(noise_seed-6201)',
        noise_rule='t1_phy.standard_noise(source_id,seed); T1 original full1024 PCG64 standard variates',
        historical_host_note='Old raw/BPG/T1 quality is reused after source and frozen-method checks. New H800 recipe does not imply cross-host bitwise arithmetic or reconstruction equivalence.',
        direct_frame_tables=frame_audits,
        raw_frame_limitation='Raw per-source four-metric means and published intervals are exact reuses; raw failure diagnostics remain in T4 raw_existing_v1/failure_breakdown.csv. This export does not reconstruct unavailable historical per-frame metrics.',
        input_bindings=dict(PINS))
    for p,h in PINS.items(): assert sha(p)==h, 'Historical input changed: '+p
    save('audit.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','source_count','reference_source_mean_rows','reference_summary_rows','common500_cache_overlap','exact_missing_new_method_frames')}))

if __name__=='__main__':main()
