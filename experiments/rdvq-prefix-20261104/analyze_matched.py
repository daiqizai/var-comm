"""Actual-bit P2 curves, source-paired effects, and separate training-seed spread."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json
import numpy as np
BUDGETS=[384,512,640,768,1024,1280,1536,2048,2560,3072,4096]
METRICS=['dists','dreamsim','psnr_db','lpips_alex']

def csv_write(p,rows):
    if not rows:return
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output'])
    identity=json.loads(Path(r['statistics_identity']).read_text());source_order=identity['sources'];assert len(source_order)==500
    bank=np.load(r['source_bank']);assert bank.shape==(10000,500) and hashlib.sha256(bank.tobytes()).hexdigest()==identity['array_sha256']
    closed=[];blocked=[];raws={};reference_hash={};counts=[]
    def load_scores(path,arm,seed):
        values={};refs={}
        for row in csv.DictReader(Path(path).open()):
            if row['split']!='test':continue
            sid=row['source'];t=int(row['slices']);key=(sid,t);assert key not in values
            values[key]=dict(bits=int(row['packet_bits']),slices=t,**{m:float(row[m]) for m in METRICS});refs[sid]=row['reference_sha256']
        assert len(values)==500*29 and set(refs)==set(source_order)
        if reference_hash:assert refs==reference_hash,'Preprocessed source identity mismatch'
        else:reference_hash.update(refs)
        assert all(np.isfinite([v[m] for m in METRICS]).all() for v in values.values())
        raws[(arm,seed)]=values
    load_scores(r['r0_scores'],'R0',0)
    for v in r['arms']:
        root=Path(v['output_root']);complete=root/'complete.json'
        if not complete.exists():blocked.append(dict(name=v['name'],root=str(root),reason='no successful arm closure'));continue
        done=json.loads(complete.read_text());assert done['passed'] and done['updates']==4000
        tests=[x for x in done['evaluations'] if x['step']==4000 and x['split']=='test'];assert len(tests)==1
        load_scores(tests[0]['scores'],done['arm'],done['seed']);closed.append(dict(name=v['name'],arm=done['arm'],seed=done['seed'],gpu_hours=done['gpu_hours']))
        counts.append(dict(name=v['name'],training_updates=4000,training_image_exposures=64000,rate_audit_source_encodings=8*128,quality_source_encodings=128*2+500,quality_nonnull_RX=28*(128*2+500),quality_NULL_outputs=128*2+500,four_metric_pairs=29*(128*2+500),training_GPU_hours=done['gpu_hours']))
    results=[];capacity=[];summary=[];paired=[];selection={}
    def stats(x):
        q=np.quantile(x[bank].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(q[0]),ci_high=float(q[1]),sources=500)
    for (arm,seed),data in raws.items():
        for b in BUDGETS+[-1]:
            rr=[]
            for sid in source_order:
                t=28 if b==-1 else max(t for t in range(29) if data[(sid,t)]['bits']<=b)
                v=data[(sid,t)];row=dict(arm=arm,seed=seed,source=sid,budget_bits=b,slices=t,packet_bits=v['bits'],**{m:v[m] for m in METRICS});rr.append(row);results.append(row)
            selection[(arm,seed,b)]={m:np.array([v[m] for v in rr]) for m in METRICS}
            capacity.append(dict(arm=arm,seed=seed,budget_bits=b,sources=500,null_count=sum(v['slices']==0 for v in rr),full_count=sum(v['slices']==28 for v in rr),mean_packet_bits=float(np.mean([v['packet_bits'] for v in rr])),mean_slices=float(np.mean([v['slices'] for v in rr])),hard_budget_violations=0 if b==-1 else sum(v['packet_bits']>b for v in rr)))
            for m in METRICS:summary.append(dict(arm=arm,seed=seed,budget_bits=b,metric=m,**stats(selection[(arm,seed,b)][m])))
    contrast_pairs=[]
    for seed in [2026110401,2026110402,2026110403]:
        for left,right in [('R3','R2'),('R3','R3_estimated'),('R2','R0'),('R3','R0')]:
            rseed=0 if right=='R0' else seed
            if (left,seed) in raws and (right,rseed) in raws:contrast_pairs.append((left,seed,right,rseed))
    for left,right in [('R1','R0'),('R2','R1'),('R3','R1')]:
        if (left,2026110401) in raws and (right,0 if right=='R0' else 2026110401) in raws:contrast_pairs.append((left,2026110401,right,0 if right=='R0' else 2026110401))
    seed_values=defaultdict(list)
    for left,seed,right,rseed in contrast_pairs:
        for b in BUDGETS+[-1]:
            for m in METRICS:
                d=selection[(left,seed,b)][m]-selection[(right,rseed,b)][m];st=stats(d);name=left+' minus '+right
                paired.append(dict(contrast=name,seed=seed,budget_bits=b,metric=m,**st));seed_values[(name,b,m)].append((seed,st['mean']))
    seed_spread=[]
    for (name,b,m),values in seed_values.items():
        xx=np.array([v for _,v in values]);seed_spread.append(dict(contrast=name,budget_bits=b,metric=m,training_seeds=len(values),mean_of_seed_effects=float(xx.mean()),sample_sd=float(xx.std(ddof=1)) if len(xx)>1 else '',minimum_seed_effect=float(xx.min()),maximum_seed_effect=float(xx.max()),seed_effects=json.dumps(values)))
    csv_write(O/'source_results.csv',results);csv_write(O/'summary.csv',summary);csv_write(O/'paired_deltas.csv',paired);csv_write(O/'capacity.csv',capacity);csv_write(O/'training_seed_variation.csv',seed_spread);csv_write(O/'actual_compute.csv',counts)
    (O/'statistics_identity.json').write_text(json.dumps(dict(source_bootstrap=identity,quality_policy='longest actually feasible native prefix; all28 endpoints plusNULL evaluated; no test tuning',CI_scope='Each interval resamples500 sources conditional on a particular trained checkpoint. Three-seed variability is reported separately, not as channel repeats or inferred from image CIs.',training_updates=4000,closed=closed,blocked=blocked),indent=2))
    assert all(v['hard_budget_violations']==0 for v in capacity)
    checks=[]
    for b in BUDGETS:
        d=[v for v in paired if v['contrast']=='R3 minus R2' and v['budget_bits']==b and v['metric']=='dists']
        ds=[v for v in paired if v['contrast']=='R3 minus R2' and v['budget_bits']==b and v['metric']=='dreamsim']
        checks.append(dict(budget_bits=b,matched_seeds=len(d),dists_negative_means=sum(v['mean']<0 for v in d),dists_negative_source_CI=sum(v['ci_high']<0 for v in d),dreamsim_negative_means=sum(v['mean']<0 for v in ds),dreamsim_negative_source_CI=sum(v['ci_high']<0 for v in ds)))
    (O/'hypothesis_observations.json').write_text(json.dumps(dict(observations=checks,automatic_P3_P4=False,interpretation='No success claim from R3 versus frozen R0 alone. Preserve mixed metrics, seed variability, estimated-rate control and full-representation effects.'),indent=2))
    (O/'complete.json').write_text(json.dumps(dict(passed=len(closed)==10,closed_arms=len(closed),blocked=blocked,mean_CI=len(summary),paired_CI=len(paired),new_model_calls=0,new_metric_calls=0,hard_budget_violations=0,hypothesis_status='requires interpretation of all favorable and unfavorable results'),indent=2))
if __name__=='__main__':main()
