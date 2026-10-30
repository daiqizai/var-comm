"""Descriptive summaries from already-scored rows, without model or bootstrap calls."""
from opt_common import *
from route_statistics_v2 import csvout,METRICS
import csv,collections

def rows(p):
 with p.open()as f:return list(csv.DictReader(f))

def main():
 out=D/'delivery';assert out.is_dir();coverage=rows(out/'A_endpoint_coverage_and_conditional_means.csv')
 b=rows(D/'fit_B_delivery/frame_metrics.csv');bg=collections.defaultdict(list)
 for v in b:
  if v['variant']!='original':bg[(v['method'],v['N'],v['snr_db'],v['source_id'],v['noise_seed'],v['variant'].split('_s')[0].removeprefix('B_T'))].append(v)
 source=[];agg=collections.defaultdict(list)
 for key,g in sorted(bg.items()):
  assert len(g)==3 and len({v['variant']for v in g})==3
  for m in METRICS:
   vals=np.array([float(v[m])for v in g]);record=dict(method=key[0],N=key[1],snr_db=key[2],source_id=key[3],noise_seed=key[4],temperature=key[5],metric=m,generation_repetitions=3,mean=float(vals.mean()),sample_std=float(vals.std(ddof=1)),min=float(vals.min()),max=float(vals.max()),registered_generation_seeds=json.dumps([v['variant'].split('_s')[1]for v in g]),actual_generation_bypassed=all(not v['generation_seed']for v in g),not_best_seed_selection=True)
   source.append(record);agg[(key[0],key[1],key[2],key[5],m)].append(record)
 csvout(out/'B_generation_variability_per_source.csv',source)
 csvout(out/'B_generation_variability_summary.csv',[dict(method=k[0],N=k[1],snr_db=k[2],temperature=k[3],metric=k[4],sources=len(vs),mean_within_source_sample_std=float(np.mean([v['sample_std']for v in vs])),mean_within_source_range=float(np.mean([v['max']-v['min']for v in vs])),scope='Descriptive generation variation; 3 samples per actual received source state, not independent source uncertainty')for k,vs in sorted(agg.items())])
 save(out/'diagnostic_supplement_identity.json',dict(status='COMPLETE',input_A=pin(D/'fit_A_delivery/frame_metrics.csv'),input_B=pin(D/'fit_B_delivery/frame_metrics.csv'),new_scientific_calls=0,new_bootstrap_calls=0,A_coverage_rows=len(coverage),B_source_variability_rows=len(source)))
 print('DIAGNOSTIC_SUMMARIES',len(coverage),len(source))

if __name__=='__main__':main()
