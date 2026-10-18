"""One formal source-bootstrap matrix, all seven metrics and direct paired deltas."""
import argparse,csv,json,math,os,pathlib,statistics as stdstats,time
import numpy as np
from data import METRICS,BASELINES,DIRECTIONS,means
def summarize(rows,source_ids,draws):
    assert len(source_ids)==len(set(source_ids)) and draws.ndim==2 and draws.shape[1]==len(source_ids)
    assert draws.min()>=0 and draws.max()<len(source_ids)
    mapping={};methods=BASELINES+('ARPC_ADAPTED',)
    for r in rows:
        key=(r['method'],int(r['snr_db']),r['metric'],r['source_id'])
        assert key not in mapping and math.isfinite(float(r['source_mean']));mapping[key]=float(r['source_mean'])
    assert set(mapping)=={(m,s,k,i) for m in methods for s in [4,10,19] for k in METRICS for i in source_ids}
    summaries=[];paired=[]
    def interval(x):
        distribution=x[draws].mean(axis=1);lo,hi=np.quantile(distribution,[.025,.975],method='linear')
        return {'mean':float(x.mean()),'ci_low':float(lo),'ci_high':float(hi)}
    for snr in [4,10,19]:
        for metric in METRICS:
            vectors={m:np.array([mapping[m,snr,metric,i] for i in source_ids],dtype=np.float64) for m in methods}
            for m,x in vectors.items():summaries.append({'method':m,'snr_db':snr,'metric':metric,'direction':DIRECTIONS[metric],'source_count':len(source_ids),**interval(x)})
            d=vectors['EC_VAR_PARTIAL']-vectors['ARPC_ADAPTED'];ci=interval(d)
            paired.append({'comparison':'NeST-Com_entropy_minus_ARPC','snr_db':snr,'metric':metric,'direction':DIRECTIONS[metric],'source_count':len(source_ids),**ci,
              'nest_favorable_sign':'negative' if DIRECTIONS[metric]=='lower' else 'positive',
              'interval_crosses_or_touches_zero':ci['ci_low']<=0<=ci['ci_high']})
    return summaries,paired
def main():
    import control as c
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args();assert c.desc(c.REG)['sha256']==a.expected_registration
    reg=json.loads(c.REG.read_text());c.success(c.RUN/'score.actual_wait.json')
    done=json.loads((c.OUT/'score_completion.json').read_text());assert done['unresolved']==0 and done['logical_frames']==900
    for d in done['outputs']:c.verify(d)
    with (c.OUT/'logical_scores.csv').open() as f:logical=list(csv.DictReader(f))
    pop=json.loads(pathlib.Path(reg['pair_plan']['path']).read_text())['sources'];ids=[s['source_id'] for s in sorted(pop,key=lambda s:s['rank'])]
    assert len(ids)==100;new=means(logical,set(ids))
    c.verify(reg['baseline_source_means'])
    with open(reg['baseline_source_means']['path']) as f:baseline=list(csv.DictReader(f))
    allrows=baseline+new
    bootstrap=c.RUN/'bootstrap';bootstrap.mkdir()
    c.put(bootstrap/'draw.reserved.json',{'replicates':10000,'seed':2026100701,'source_count':100,'source_ids':ids,'started_unix':time.time(),'draw_matrix_calls':1})
    # The single production RNG invocation. Shared indices preserve all pairings.
    draws=np.random.default_rng(2026100701).integers(0,100,size=(10000,100),dtype=np.int64)
    np.save(bootstrap/'source_indices.npy',draws,allow_pickle=False)
    c.put(bootstrap/'draw.complete.json',{'indices':c.desc(bootstrap/'source_indices.npy'),'draw_matrix_calls':1})
    summary,paired=summarize(allrows,ids,draws)
    for name,rows in [('arpc_source_noise_means.csv',new),('all_method_source_means.csv',allrows),('matched100_summary.csv',summary),('nest_minus_arpc_paired.csv',paired)]:
        with (c.OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    c.put(c.OUT/'bootstrap_identity.json',{'seed':2026100701,'replicates':10000,'draw_matrix_calls':1,'indices':c.desc(bootstrap/'source_indices.npy'),'source_ids':ids,
      'unit':'source; each source first averages three noise realizations','interval':'pointwise percentile 95%, NumPy quantile method=linear; no multiplicity adjustment',
      'paired_delta':'EC_VAR_PARTIAL source mean minus ARPC source mean before resampling; never subtract marginal intervals',
      'old_method_scores_recomputed':False,'NeST19':'T1-C m10 revised PARTIAL new_mean','noise':'ARPC and NeST6201/2/3; other historical baselines2001/2/3. Source pairing, not identical-noise coupling.'})
    c.put(c.OUT/'completion.json',{'status':'FIXED100_SEVEN_METRIC_MATCHED_COMPARISON_COMPLETE','registration_sha256':a.expected_registration,
      'source_count':100,'ARPC_logical_frames':900,'methods':5,'metrics':list(METRICS),'summary_cells':105,'paired_cells':21,'bootstrap_draw_matrix_calls':1,
      'unresolved':0,'policy_selection':False,'training':0,'new_baseline_scores':0,
      'outputs':[c.desc(c.OUT/n) for n in ['score_completion.json','arpc_source_noise_means.csv','all_method_source_means.csv','matched100_summary.csv','nest_minus_arpc_paired.csv','bootstrap_identity.json']]})
if __name__=='__main__':main()
