"""Actual source-packet budget curves and labeled source-informed bound."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
BUDGETS=[384,512,640,768,1024,1280,1536,2048,2560,3072,4096]
METRICS=['dists','dreamsim','psnr_db','lpips_alex']
def csv_write(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output']);source=json.loads(Path(r['manifest']).read_text())['sources']
    raw=list(csv.DictReader(Path(r['scores']).open()));lookup={}
    for v in raw:
        key=(v['source'],int(v['slices']));assert key not in lookup
        lookup[key]={**{m:float(v[m]) for m in METRICS},'slices':key[1],'bits':int(v['packet_bits'])}
    groups={s:[v['source_id'] for v in source if v['split']==s] for s in ['development','calibration','test']};assert [len(groups[s]) for s in groups]==[128,1000,500]
    assert len(lookup)==1628*29
    policies={};cal=groups['calibration']
    def longest(s,b,target=28):
        return lookup[(s,max(t for t in range(target+1) if lookup[(s,t)]['bits']<=b))]
    for b in BUDGETS:
        candidates=[(np.mean([longest(s,b,t)['dists'] for s in cal]),t) for t in range(1,29)]
        # Lower DISTS then larger target wins an exact tie, fixed before test.
        score,t=min(candidates,key=lambda v:(v[0],-v[1]));policies[b]=dict(target=t,calibration_dists=score)
    (O/'calibrated_source_budget_targets.json').write_text(json.dumps(policies,indent=2))
    selected=[]
    for s in groups['test']:
        for b in BUDGETS:
            feasible=[lookup[(s,t)] for t in range(29) if lookup[(s,t)]['bits']<=b]
            best=min(feasible,key=lambda v:(v['dists'],-v['slices']))
            options={'longest_feasible':longest(s,b),'calibrated_target':longest(s,b,policies[b]['target']),'source_informed_DISTS_bound':best}
            for name,v in options.items():selected.append(dict(source=s,budget_bits=b,policy=name,slices=v['slices'],packet_bits=v['bits'],**{m:v[m] for m in METRICS}))
        full=lookup[(s,28)];selected.append(dict(source=s,budget_bits=-1,policy='complete_representation',slices=28,packet_bits=full['bits'],**{m:full[m] for m in METRICS}))
    csv_write(O/'test_source_results.csv',selected)
    rng=np.random.default_rng(2026110417);bank=rng.integers(0,500,(10000,500),dtype=np.int32);np.save(O/'source_bootstrap_bank.npy',bank,allow_pickle=False)
    bankid=dict(seed=2026110417,resamples=10000,sources=groups['test'],array_sha256=hashlib.sha256(bank.tobytes()).hexdigest());(O/'statistics_identity.json').write_text(json.dumps(bankid,indent=2))
    summaries=[];paired=[];index={(v['source'],v['budget_bits'],v['policy']):v for v in selected}
    def stats(x):
        ci=np.quantile(x[bank].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),sources=500)
    for b in BUDGETS+[-1]:
        names=['longest_feasible','calibrated_target','source_informed_DISTS_bound'] if b!=-1 else ['complete_representation']
        for name in names:
            rr=[index[(s,b,name)] for s in groups['test']]
            for m in METRICS:summaries.append(dict(budget_bits=b,policy=name,metric=m,**stats(np.array([v[m] for v in rr]))))
        if b!=-1:
            for compare,reference in [('calibrated_target','longest_feasible'),('source_informed_DISTS_bound','calibrated_target')]:
                for m in METRICS:
                    values=np.array([index[(s,b,compare)][m]-index[(s,b,reference)][m] for s in groups['test']]);paired.append(dict(budget_bits=b,contrast=compare+' minus '+reference,metric=m,**stats(values)))
    csv_write(O/'summary.csv',summaries);csv_write(O/'paired_deltas.csv',paired)
    capacity=[]
    for b in BUDGETS:
        rr=[index[(s,b,'longest_feasible')] for s in groups['test']]
        capacity.append(dict(budget_bits=b,sources=500,null_count=sum(v['slices']==0 for v in rr),full_count=sum(v['slices']==28 for v in rr),mean_packet_bits=float(np.mean([v['packet_bits'] for v in rr])),hard_budget_violations=sum(v['packet_bits']>b for v in rr)))
    csv_write(O/'capacity.csv',capacity)
    (O/'complete.json').write_text(json.dumps(dict(passed=True,sources=1628,test_sources=500,prefixes=1628*29,mean_ci=len(summaries),paired_ci=len(paired),all_budgets_feasible=all(v['hard_budget_violations']==0 for v in capacity),
        scope='source-codec P1; source-informed DISTS bound is not a measured wireless controller or a trained selector'),indent=2))
if __name__=='__main__':main()
