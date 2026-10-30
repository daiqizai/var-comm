"""Descriptive summaries from already-scored rows, without model or bootstrap calls."""
from opt_common import *
from route_statistics_v2 import csvout,METRICS
import csv,collections

def rows(p):
 with p.open()as f:return list(csv.DictReader(f))

def main():
 out=D/'delivery';assert out.is_dir();a=rows(D/'fit_A_delivery/frame_metrics.csv');lookup={(v['method'],v['N'],v['snr_db'],v['source_id'],v['noise_seed'],v['variant']):v for v in a};groups=collections.defaultdict(list)
 for v in a:
  if v['variant']in ['A1','A2']:groups[(v['method'],v['N'],v['snr_db'],v['variant'])].append(v)
 coverage=[]
 for key,g in sorted(groups.items()):
  eligible=[v for v in g if v['receiver_status'].startswith('ACCEPTED')and tuple(json.loads(v['endpoint']))in [(7,0),(7,25)]]
  for label,vs in [('all_sources',g),('anchor_eligible_received_endpoint',eligible)]:
   if not vs:continue
   for m in METRICS:
    ds=[float(v[m])-float(lookup[(v['method'],v['N'],v['snr_db'],v['source_id'],v['noise_seed'],'original')][m])for v in vs]
    coverage.append(dict(method=key[0],N=key[1],snr_db=key[2],variant=key[3],subset=label,n_sources=len(vs),all_sources=len(g),anchor_eligible_sources=len(eligible),coverage=len(eligible)/len(g),metric=m,mean=float(np.mean([float(v[m])for v in vs])),mean_delta_from_same_source_original=float(np.mean(ds)),inference='Descriptive conditional subset; all-source paired inference remains original fit_A_delivery'))
 csvout(out/'A_endpoint_coverage_and_conditional_means.csv',coverage)
 b=rows(D/'fit_B_delivery/frame_metrics.csv');bg=collections.defaultdict(list)
 for v in b:
  if v['variant']!='original':bg[(v['method'],v['N'],v['snr_db'],v['source_id'],v['noise_seed'],v['temperature'])].append(v)
 source=[];agg=collections.defaultdict(list)
 for key,g in sorted(bg.items()):
  assert len(g)==3 and len({v['generation_seed']for v in g})==3
  for m in METRICS:
   vals=np.array([float(v[m])for v in g]);record=dict(method=key[0],N=key[1],snr_db=key[2],source_id=key[3],noise_seed=key[4],temperature=key[5],metric=m,generation_repetitions=3,mean=float(vals.mean()),sample_std=float(vals.std(ddof=1)),min=float(vals.min()),max=float(vals.max()),generation_seeds=json.dumps([v['generation_seed']for v in g]),not_best_seed_selection=True)
   source.append(record);agg[(key[0],key[1],key[2],key[5],m)].append(record)
 csvout(out/'B_generation_variability_per_source.csv',source)
 csvout(out/'B_generation_variability_summary.csv',[dict(method=k[0],N=k[1],snr_db=k[2],temperature=k[3],metric=k[4],sources=len(vs),mean_within_source_sample_std=float(np.mean([v['sample_std']for v in vs])),mean_within_source_range=float(np.mean([v['max']-v['min']for v in vs])),scope='Descriptive generation variation; 3 samples per actual received source state, not independent source uncertainty')for k,vs in sorted(agg.items())])
 save(out/'diagnostic_supplement_identity.json',dict(status='COMPLETE',input_A=pin(D/'fit_A_delivery/frame_metrics.csv'),input_B=pin(D/'fit_B_delivery/frame_metrics.csv'),new_scientific_calls=0,new_bootstrap_calls=0,A_coverage_rows=len(coverage),B_source_variability_rows=len(source)))
 print('DIAGNOSTIC_SUMMARIES',len(coverage),len(source))

if __name__=='__main__':main()
