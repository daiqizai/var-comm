#!/usr/bin/env python3
"""Summarize a complete paired real-image table without recomputing image metrics."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
METRICS=('psnr_db','lpips_alex','dino_cosine')
def write_csv(path,rows):
    with Path(path).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(path,output):
    path=Path(path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(path.open()));groups={};keys=set();scopes={}
    for r in rows:
        key=(r['method'],int(r['source_index']),float(r['snr_db']),int(r.get('seed',r.get('noise_seed'))))
        if key in keys:raise RuntimeError('duplicate complete sample key')
        keys.add(key);groups.setdefault(key[:3],[]).append(r)
        scopes.setdefault(r['method'],set()).add(key[1:])
        if not all(np.isfinite(float(r[k])) for k in METRICS):raise RuntimeError('nonfinite metric')
    if any(v!=next(iter(scopes.values())) for v in scopes.values()):raise RuntimeError('method sample grids differ')
    per_source_snr=[dict(method=m,source_index=i,snr_db=s,seed_keys=','.join(sorted(r.get('seed',r.get('noise_seed')) for r in v)),**{k:float(np.mean([float(r[k]) for r in v])) for k in METRICS}) for (m,i,s),v in sorted(groups.items())]
    sources=sorted({r['source_index'] for r in per_source_snr});methods=sorted(scopes);arrays={}
    for m in methods:
        arrays[m]=np.array([[np.mean([r[k] for r in per_source_snr if r['method']==m and r['source_index']==i]) for k in METRICS] for i in sources])
    summary=[dict(method=m,sources=len(sources),frames=sum(r['method']==m for r in rows),**dict(zip(METRICS,arrays[m].mean(0).tolist()))) for m in methods]
    paired=[];rng=np.random.default_rng(2026092301);indices=rng.integers(0,len(sources),size=(10000,len(sources)))
    for a in methods:
        for b in methods:
            if a<=b:continue
            diff=arrays[a]-arrays[b];sample=diff[indices].mean(1)
            paired.append({'method':a,'control':b,'delta':{k:{'mean':float(diff[:,j].mean()),'ci95':np.quantile(sample[:,j],[.025,.975]).tolist()} for j,k in enumerate(METRICS)}})
    write_csv(output/'summary.csv',summary);write_csv(output/'per_source_snr.csv',per_source_snr)
    (output/'paired.json').write_text(json.dumps(paired,indent=2)+'\n')
    (output/'lineage.json').write_text(json.dumps({'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'source-level paired bootstrap, 10000 repeats; stored real metrics','rows':len(rows)},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();summarize(a.input,a.output)
