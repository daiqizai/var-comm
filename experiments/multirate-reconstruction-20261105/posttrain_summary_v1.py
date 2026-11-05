"""All fixed test sources, actual packet budgets and directly paired source CIs."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from runtime import sha,write
from posttrain_codec_v1 import BUDGETS,METRICS

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();r=json.loads(Path(a.protocol).read_text())
    root=Path(r['root']);out=root/'delivery';out.mkdir(exist_ok=False)
    sources=[s['source_id'] for s in json.loads(Path(r['manifest']).read_text())['sources']]
    bank=np.load(r['bootstrap_bank'],allow_pickle=False);identity=json.loads(Path(r['bootstrap_identity']).read_text())
    assert identity['sources']==sources and bank.shape==(10000,500) and hashlib.sha256(bank.tobytes()).hexdigest()==identity['array_sha256']
    assert sha(r['bootstrap_bank'])==r['bootstrap_bank_sha256']
    def stats(v):
        v=np.asarray(v,dtype=np.float64);assert v.shape==(500,) and np.isfinite(v).all()
        q=np.quantile(v[bank].mean(1),[.025,.975]);return dict(mean=float(v.mean()),ci_low=float(q[0]),ci_high=float(q[1]),sources=500)
    summaries=[];paired=[];capacity=[];allrows=[];status={};calls={}
    for size in [256,512]:
        error=root/f'resolution_{size}_failure.json'
        if error.exists():status[str(size)]={'status':'blocked','failure':json.loads(error.read_text())};continue
        rows=[]
        for shard in range(8):
            base=root/f'eval_{size}_{shard:02d}'
            for stage in ['tx','rx','score']:
                w=json.loads((root/'roles'/f'{size}_{shard:02d}_{stage}'/'actual_wait.json').read_text())
                done=json.loads((base/stage/'complete.json').read_text());assert w['returncode']==0 and w['child_gone'] and done['passed']
                calls[f'{size}_{shard:02d}_{stage}']=done['counters']
            rows.extend(csv.DictReader((base/'score/scores.csv').open()))
        assert len(rows)==8000
        lookup={(v['source'],int(v['budget_bits']),v['method']):v for v in rows};assert len(lookup)==len(rows)
        for budget in [b*(size//256)**2 for b in BUDGETS]:
            for method in ['original','new']:
                rr=[lookup[(sid,budget,method)] for sid in sources]
                assert all(int(v['packet_bits'])<=budget for v in rr)
                for metric in METRICS:summaries.append(dict(size=size,budget_bits=budget,method=method,metric=metric,**stats([float(v[metric]) for v in rr])))
                capacity.append(dict(size=size,budget_bits=budget,method=method,sources=500,null_count=sum(int(v['null']) for v in rr),mean_packet_bits=float(np.mean([int(v['packet_bits']) for v in rr])),mean_received_tokens=float(np.mean([int(v['received_tokens']) for v in rr])),full_count=sum(int(v['endpoint'])==(60 if method=='new' else 28) for v in rr),hard_budget_violations=0))
            for metric in METRICS:paired.append(dict(size=size,budget_bits=budget,contrast='new minus original',metric=metric,**stats([float(lookup[(sid,budget,'new')][metric])-float(lookup[(sid,budget,'original')][metric]) for sid in sources])))
        allrows.extend(rows);status[str(size)]={'status':'done','logical_positions':8000,'sources':500}
    for name,rows in [('frame_metrics.csv',allrows),('summary.csv',summaries),('paired_deltas.csv',paired),('capacity.csv',capacity)]:csv_write(out/name,rows)
    write(out/'actual_compute_and_reuse.json',calls);write(out/'statistics_identity.json',dict(identity,bank_file_sha256=r['bootstrap_bank_sha256']))
    report=['# Final-checkpoint source-codec comparison','', 'Final scheduled checkpoint; no test-based checkpoint or hyperparameter selection.','',
      'Both resolutions use the same 500 held-out source IDs and fixed center crops. All on-wire wrapper and entropy bytes count. New codec has a 136-bit wrapper; original RDVQ has a 72-bit wrapper with a fixed shared resolution. These are system-level actual source-packet comparisons, not equal-overhead ablations or radio-PHY results.',
      '', 'The original 256 codec uses identity-verified existing packets, arrays and metrics; one fixed cached pair per shard revalidates the metric route. At 512 the original codec uses its native four 256-patch representation and 28-slice stream. The new codec uses its trained whole-image representation. No favorable-result selection or parameter scan.', '', '| Resolution | Budget | Metric | New minus original [95% source-paired CI] |','|---|---:|---|---|']
    for v in paired:report.append(f"| {v['size']} | {v['budget_bits']} | {v['metric']} | {v['mean']:+.6f} [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}] |")
    report+=['','Positive PSNR favors new; negative LPIPS/DISTS/DreamSim favors new. All mixed and unfavorable values are retained. NULL remains gray and in the denominator. Unknown execution failures block that resolution and are not converted into successes or gray outputs.','',json.dumps(status,indent=2)]
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    ledger={}
    for f in (root/'roles').glob('*/actual_wait.json'):ledger[f.parent.name]=json.loads(f.read_text())
    write(out/'resource_ledger.json',ledger)
    write(out/'complete.json',dict(passed=all(v['status']=='done' for v in status.values()),resolution_status=status,mean_ci=len(summaries),paired_ci=len(paired),scientific_training_finished=True,new_PHY_calls=0,checkpoint=json.loads((root/'final_checkpoint.json').read_text())))

if __name__=='__main__':main()
