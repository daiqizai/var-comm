#!/usr/bin/env python3
"""Quantify actual historical impact without altering historical artifacts."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup';R=ROOT/'results/review_20260923_phase2'
M=('psnr_db','lpips_alex','dino_cosine')
def read(p):return list(csv.DictReader(p.open()))
def key(r):return (int(r['source_index']),float(r['snr_db']),int(r.get('seed',r.get('noise_seed'))))
def compare(old,new):
    expected={(i,s,n) for i in range(100) for s in (1.,4.,7.,13.,19.) for n in (2001,2002,2003)}
    a={key(r):r for r in old};b={key(r):r for r in new}
    if len(old)!=1500 or len(new)!=1500 or set(a)!=expected or set(b)!=expected:raise RuntimeError('impact pairing incomplete or duplicated')
    for k in a:
        if a[k]['image_id']!=b[k]['image_id']:raise RuntimeError('impact source identity mismatch')
        if not all(math.isfinite(float(r[m])) for r in (a[k],b[k]) for m in M):raise RuntimeError('nonfinite impact value')
    return {m:max(abs(float(a[k][m])-float(b[k][m])) for k in a) for m in M}
def main():
    paths=[B/'review_20260923_statistics/per_frame.csv',B/'research_20260923_system_policy_v2/digital_adaptive_per_frame.csv',
           B/'allocation_v1/per_frame.csv',B/'research_20260923_system_policy_v2/folded_development.csv']
    old,new,old_fold,new_fold=map(read,paths);impact={}
    for f in ('raw','arithmetic'):
        impact[f]=compare([r for r in old if r['N']=='4084' and r['renderer']=='Dc' and r['policy']=='quality' and r['family']==f],
                          [r for r in new if r['method']==f+'_adaptive_m789_Dc'])
    impact['folded512']=compare(old_fold,new_fold)
    result={'scope':'maximum absolute per-frame metric differences; complete source/SNR/seed and image identity checks',
        'compared_rows_per_method':1500,'metric_differences':impact,'original_artifacts_modified':False,
        'source_artifacts':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (R/'historical_impact.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(impact))
if __name__=='__main__':main()
