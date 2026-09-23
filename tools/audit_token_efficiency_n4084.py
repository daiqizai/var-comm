"""Audit existing N4084 comparison provenance without running or changing GPU work."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup'
COMPARISON = ROOT / 'results/review_20260923_phase2/comparison'
METHODS = ('pure_continuous', 'full_tx_control', 'light_tx', 'original_1024_Dc', 'raw_adaptive_m789_Dc', 'arithmetic_adaptive_m789_Dc', 'calibration_frozen_resource_lookup')
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')
SNRS = (1., 4., 7., 13., 19.)
SEEDS = (2001, 2002, 2003)
DC = 'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def rows(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))

def key(row):
    return row['image_id'], float(row['snr_db']), int(row['seed'])

def validate_table(data, sources, methods=METHODS):
    expected = set(itertools.product(methods, sources, SNRS, SEEDS))
    seen = set()
    for r in data:
        k = (r['method'], *key(r))
        require(k in expected and k not in seen, 'duplicate or unexpected source/SNR/noise/method')
        seen.add(k)
        require(int(r['source_index']) == sources[r['image_id']], 'source index mismatch')
        require(float(r['N']) == 4084 and float(r['E']) == 8168, 'resource ledger mismatch')
        require(r['decoder_sha256'] == DC, 'Decoder identity mismatch')
        require(all(math.isfinite(float(r[m])) for m in METRICS), 'nonfinite metric')
    require(seen == expected, 'missing source/SNR/noise/method')

def validate_timing(timings, consistency):
    methods = METHODS[:3]
    expected = set(itertools.product(methods, range(0,100,11), SNRS, (0,1)))
    def k(r):return r['method'],int(r['source_index']),float(r['snr_db']),int(r['repeat'])
    require(len(timings)==len(consistency)==len(expected)==300, 'timing row coverage')
    require({k(r) for r in timings}=={k(r) for r in consistency}==expected, 'timing key coverage')
    for r in timings:
        require(int(r['seed'])==2001, 'timing noise seed')
        require(all(math.isfinite(float(r[c])) and float(r[c])>=0 for c in ('tx_ms','rx_ms','total_ms')), 'invalid timing')
        require(abs(float(r['tx_ms'])+float(r['rx_ms'])-float(r['total_ms']))<1e-8, 'timing sum')
    for r in consistency:
        require(math.isfinite(float(r['max_abs_error'])) and 0<=float(r['max_abs_error'])<=2e-5, 'quality/timing mismatch')

def main():
    import numpy as np
    from token_efficiency.digital_grid import population
    from latent_enhancement_eval.runner import load_targets
    target = ROOT/'results/token_channel_efficiency_20260923/n4084_reference_audit_v1'
    require(not target.exists(), 'immutable audit already exists')
    bindings = {}
    def bind(p):
        p=Path(p);bindings[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = digest(p)
    def check_hash(p,sha):
        require(digest(p)==sha, f'hash mismatch: {p}');bind(p)
    records, image_bindings = population('development')
    old = load_targets()
    require(len(records)==len(old)==100, '100 original development sources required')
    sources={r['image_id']:i for i,r in enumerate(records)}
    require(list(sources)==read(COMPARISON/'source_identity.json')['development'], 'source identity list')
    for r,o in zip(records,old):
        require(r['image_id']==o['target']['image_id'] and np.array_equal(r['pixels'],o['pixels']), 'current/legacy source preprocessing mismatch')
    data=rows(COMPARISON/'per_frame.csv');validate_table(data,sources)
    for entry in read(COMPARISON/'lineage.json'):
        check_hash(ROOT/entry['source'],entry['sha256'])
    for p in [COMPARISON/'per_frame.csv',COMPARISON/'source_identity.json',COMPARISON/'lineage.json',Path(__file__)]:bind(p)
    evaluation=BASE/'research_20260923_evaluation';diagnostics=BASE/'research_20260923_diagnostics';policy=BASE/'research_20260923_system_policy_v2';digital=BASE/'research_20260923_digital_strict'
    edone=read(evaluation/'completion.json');ereg=read(evaluation/'registration.json');ddone=read(diagnostics/'completion.json');pdone=read(policy/'completion.json');frozen=read(policy/'frozen_policy.json');dreg=read(digital/'registration.json');digital_done=read(digital/'completion.json')
    for folder,done in [(evaluation,edone),(diagnostics,ddone)]:check_hash(folder/'per_frame.csv',done['per_frame_sha256'])
    check_hash(evaluation/'registration.json',edone['registration_sha256'])
    check_hash(digital/'registration.json',digital_done['registration_sha256'])
    for pop in ('calibration','development'):check_hash(digital/(pop+'.csv'),digital_done[pop+'_sha256'])
    check_hash(policy/'frozen_policy.json',pdone['policy_sha256'])
    check_hash(policy/'calibration.csv',frozen['calibration_sha256'])
    check_hash(digital/'calibration.csv',frozen['digital_calibration_sha256'])
    for p,sha in pdone['development_sources'].items():check_hash(p,sha)
    for receipt in [ereg,ddone,frozen,dreg]:
        for p,sha in receipt['source_snapshot'].items():check_hash(p,sha)
    for rec in [*ereg['selected'].values(),ddone['checkpoint'],*frozen['selected_arms'].values()]:
        p=Path(rec['checkpoint']);p=p if p.is_absolute() else ROOT/p;check_hash(p,rec['checkpoint_sha256'])
    require(dreg['decoder_state_sha256']==frozen['decoder_state_sha256']==DC,'frozen Dc scope')
    require(dreg['N']==4084 and dreg['E']==8168 and dreg['modes']==[7,8,9] and dreg['precision']=={'matmul_tf32':False,'cudnn_tf32':False},'digital protocol scope')
    lookup={}
    for folder,filename in [(evaluation,'per_frame.csv'),(diagnostics,'per_frame.csv'),(policy,'per_frame.csv'),(policy,'digital_adaptive_per_frame.csv')]:
        for r in rows(folder/filename):
            if folder==diagnostics:
                if r['method']!='actual_RX_AWGN':continue
                require(r['diagnostic_only']=='False','oracle diagnostic accidentally included')
                method='original_1024_Dc'
            else:method=r['method']
            lookup[(method,*key(r))]=r
    for r in data:
        origin=lookup[(r['method'],*key(r))]
        require(all(float(r[m])==float(origin[m]) for m in METRICS),'published metric differs from actual source row')
        if r['method'] in ereg['selected']:
            require(origin['checkpoint_sha256']==r['model_context_sha256']==ereg['selected'][r['method']]['checkpoint_sha256'],'selected identity')
    adaptive=read(policy/'digital_adaptive_policies.json')
    check_hash(digital/'calibration.csv',adaptive['calibration_sha256'])
    digital_rows={(r['family'],int(r['mode']),*key(r)):r for r in rows(digital/'development.csv')}
    for r in rows(policy/'digital_adaptive_per_frame.csv'):
        family=r['method'].split('_')[0];mode=adaptive['actions'][family]['4084']['Dc'][str(float(r['snr_db']))]['quality']
        require(int(r['selected_mode'])==mode,'digital development policy drift')
        origin=digital_rows[(family,mode,*key(r))]
        require(all(float(r[m])==float(origin['lpips' if m=='lpips_alex' else m]) for m in METRICS),'adaptive candidate mismatch')
    crows=rows(policy/'calibration.csv');scores={}
    for r in crows:scores.setdefault((float(r['snr_db']),r['method']),[]).append(float(r['utility']))
    for snr in SNRS:
        means={m:statistics.fmean(v) for (s,m),v in scores.items() if s==snr}
        require(frozen['actions'][str(snr)]==min(means,key=lambda m:(means[m],m)),'lookup calibration selection mismatch')
    for r in rows(policy/'per_frame.csv'):require(r['selected_endpoint']==frozen['actions'][str(float(r['snr_db']))],'lookup development policy drift')
    timings=rows(evaluation/'timing.csv');consistency=rows(evaluation/'quality_timing_consistency.csv');validate_timing(timings,consistency)
    for folder in [evaluation,diagnostics,policy,digital]:
        for p in folder.glob('*.json'):bind(p)
    for p in [evaluation/'timing.csv',evaluation/'quality_timing_consistency.csv',evaluation/'warmup.csv']:bind(p)
    target.mkdir(parents=True)
    result={'status':'REAL_N4084_PROVENANCE_AND_FROZEN_POLICY_AUDIT_PASS_COMPATIBILITY_PENDING','synthetic':False,'GPU_used':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_count':100,'frame_rows':len(data),'methods':list(METHODS),'N':4084,'E':8168,'source_preprocessing_equal':True,'source_snr_noise_coverage':'100 x 5 x 3 per method, failures retained','published_metrics_equal_original_rows':True,'selected_checkpoint_hashes_verified':True,'frozen_lookup_recomputed_from_calibration':True,'digital_selected_mode_and_candidate_verified':True,'oracle_rows_excluded':True,'original_timing_calls':300,'original_max_quality_timing_error':max(float(r['max_abs_error']) for r in consistency),'timing_hardware':ereg['hardware'],'timing_contract':ereg['quality_timing_contract'],'timing_scope':'historical separate; current uniform timing warmup/thermal/runtime acceptance pending','protocol_scope':'legacy QPSK m789/header68 raw or header92 arithmetic, distinct from new m6-m10 protocol; not interchangeable candidate cells','full_current_PHY_metric_compatibility_claimed':False,'pending':[{'item':'P4084 selected common quality/timing','execution':'already registered continuous_grid_v1 stage; no duplicate evaluation'},{'item':'legacy N4084 mixed/digital metric and full PHY identity','execution':'source/receipt audit complete; actual replay compatibility acceptance required before strict current-table reuse'},{'item':'external reference common metrics','execution':'existing reference_common_metrics_v1 serial worker; no duplicate launch'},{'item':'legacy uniform timing','execution':'keep original real timing separate until current endpoint and warmup scope replay'}],'source_preprocessing_ids':{r['image_id']:r['preprocessing_id'] for r in records},'bindings':bindings}
    (target/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','source_count','frame_rows','original_timing_calls','original_max_quality_timing_error']},indent=2))

if __name__=='__main__':main()
