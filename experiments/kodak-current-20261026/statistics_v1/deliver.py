"""One-shot CPU statistics of closed Kodak scores using the original index bank."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(d):
    p=Path(d['path']);assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
    return p
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def gone(p):
    path=Path('/proc')/str(p['pid'])/'stat'
    if not path.exists():return True
    bits=path.read_text().rsplit(')',1)[1].split()
    return int(bits[19])!=int(p['start_ticks']) or bits[0]=='Z'

def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['sources']:checked(d)
    for d in r['closure_waits']:
        w=read(checked(d));assert type(w['returncode'])is int and w['returncode']==0 and w['actual_wait'] is True
        assert not any(w.get(k,False)for k in ('terminated','killed','timed_out'))
        assert w.get('metadata_error') is None
    for p in r['closed_processes']:assert gone(p),p
    completion=read(checked(r['metrics_completion']))
    assert completion['status']=='COMPLETE' and completion['rows']==1296 and completion['metrics']==4
    assert completion['scores']==r['frames']
    rows=list(csv.DictReader(checked(r['frames']).open()))
    bank=np.load(checked(r['bank']),allow_pickle=False)
    spec=importlib.util.spec_from_file_location('_kodak_stats',checked(r['analysis']))
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    out=Path(r['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    summary,paired,means,acceptance=a.summarize(rows,bank)
    for name,values in [('summary.csv',summary),('paired_differences.csv',paired),('source_means.csv',means),('acceptance_summary.csv',acceptance)]:a.csv_write(out/name,values)
    (out/'frame_metrics.csv').write_bytes(checked(r['frames']).read_bytes())
    endpoint=checked(r['endpoints']);(out/'endpoint_utilization.csv').write_bytes(endpoint.read_bytes())
    ep=list(csv.DictReader(endpoint.open()));assert len(ep)==144
    keys={(v['family'],int(v['source_index']),int(v['snr_db']))for v in ep}
    assert keys=={(f,i,s)for f in ('WHOLE','PARTIAL')for i in range(24)for s in (4,10,19)}
    ep_summary=[]
    for family in ('WHOLE','PARTIAL'):
        for snr in (4,10,19):
            group=[v for v in ep if v['family']==family and int(v['snr_db'])==snr]
            for v in group:assert float(v['utilization'])==int(v['terminated_arithmetic_bits'])/int(v['source_capacity_bits'])
            ep_summary.append(dict(family=family,snr_db=snr,source_count=24,mean_utilization=float(np.mean([float(v['utilization'])for v in group],dtype=np.float64)),mean_terminated_bits=float(np.mean([int(v['terminated_arithmetic_bits'])for v in group],dtype=np.float64)),source_capacity_bits=int(group[0]['source_capacity_bits']),definition='terminated arithmetic bits / source capacity; unchanged by reception outcome'))
    a.csv_write(out/'utilization_summary.csv',ep_summary)
    manifest={p.name:desc(p)for p in sorted(out.iterdir())if p.is_file()}
    save(out/'statistics_completion.json',dict(status='COMPLETE',registration=desc(path),logical_frames=1296,mean_conditions=72,paired_conditions=60,source_mean_rows=1728,original_bank=r['bank'],bootstrap_seed=2026100901,new_bootstrap_draws=0,source_count=24,noise_per_source=3,paired_unit='source image',new_models=0,new_channels=0,new_scores=0,elapsed_seconds=time.monotonic()-start,outputs=manifest))
    print(json.dumps(read(out/'statistics_completion.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--sha256',required=True);v=p.parse_args();main(v.registration,v.sha256)
