"""Descriptive analysis of immutable completed results; no new trials or bootstrap."""
from opt_common import *
from route_statistics_v2 import csvout
import csv,collections,statistics

METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')

def rows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))

def main():
 out=D/'delivery/analysis_20261030';out.mkdir(exist_ok=False)
 inputs=[];result=[]
 def load(p):inputs.append(pin(p));return rows(p)
 def summarize(groups,study,total):
  for (snr,group),vs in sorted(groups.items()):
   for metric in METRICS:
    result.append(dict(study=study,snr_db=snr,group=group,observations=len(vs),full_denominator=total,metric=metric,conditional_mean_delta=statistics.mean(v[metric]for v in vs),contribution_to_full_mean=sum(v[metric]for v in vs)/total,interpretation='Post-hoc descriptive accounting only; selected success subsets are not an alternative performance estimate. No confidence interval or selection change.'))
 r=load(D/'fit_E_delivery/frame_metrics.csv');base={(v['source_id'],v['snr_db']):v for v in r if v['variant']=='original'};groups=collections.defaultdict(list)
 for v in r:
  if v['variant']!='E':continue
  a=base[(v['source_id'],v['snr_db'])];assert v['source_pixels_sha256']==a['source_pixels_sha256']and v['noise_seed']==a['noise_seed']
  changed=int(v['selected_rank'])>0
  group='unchanged'if not changed else('changed_both_accepted'if v['receiver_status']==a['receiver_status']=='ACCEPTED_SOURCE'else'changed_with_rejection')
  groups[(int(v['snr_db']),group)].append({m:float(v[m])-float(a[m])for m in METRICS})
 assert sum(len(g)for g in groups.values())==200;summarize(groups,'E_FIT',100)
 # Descriptive grouping does not remove failed events from the formal denominator.
 for snr in [7,10]:
  for m in METRICS:
   observed=sum(v[m]for (s,_),g in groups.items()if s==snr for v in g)/100
   official=next(x for x in rows(D/'fit_E_delivery/paired_deltas.csv')if x['method']=='NEST'and x['candidate']=='E'and x['snr_db']==str(snr)and x['metric']==m)
   assert abs(observed-float(official['mean_delta']))<1e-12
 r=load(D/'final_common500_delivery/frame_metrics.csv');base={(v['source_id'],v['channel_repeat']):v for v in r if v['method']=='original_NeST'and v['N']=='1024'and v['snr_db']=='7'};groups=collections.defaultdict(list)
 for v in r:
  if v['method']!='final_NeST'or v['N']!='1024'or v['snr_db']!='7':continue
  a=base[(v['source_id'],v['channel_repeat'])];assert v['source_pixels_sha256']==a['source_pixels_sha256']
  good_a=bool(a['image']);good_v=bool(v['image']);group={
   (False,False):'both_gray',(False,True):'original_gray_D_accepted',
   (True,False):'original_accepted_D_gray',(True,True):'both_accepted'}[(good_a,good_v)]
  groups[(7,group)].append({m:float(v[m])-float(a[m])for m in METRICS})
 assert sum(len(g)for g in groups.values())==1500;summarize(groups,'D_common500_N1024',1500)
 official=load(D/'final_common500_delivery/paired_deltas.csv')
 for m in METRICS:
  observed=sum(v[m]for g in groups.values()for v in g)/1500
  q=next(x for x in official if x['candidate']=='final_NeST'and x['reference']=='original_NeST'and x['N']=='1024'and x['snr_db']=='7'and x['metric']==m)
  assert abs(observed-float(q['mean_delta']))<1e-12
 csvout(out/'descriptive_failure_contributions.csv',result)
 timing=load(D/'delivery/timing_and_compute.csv')
 # All source metric and CI files remain immutable; include full lineage of the narrative.
 for p in [D/'delivery/route_summary.csv',D/'delivery/C_content_minus_control.csv',D/'delivery/C_missing_token_summary.csv',D/'delivery/A_endpoint_coverage_and_conditional_means.csv',D/'delivery/endpoint_and_failure_summary.csv',D/'resource_delivery/resource_ledger.json',D/'frozen_final.json']:inputs.append(pin(p))
 save(out/'identity.json',dict(status='COMPLETE_DESCRIPTIVE_ANALYSIS_ONLY',inputs=inputs,outputs=[pin(out/'descriptive_failure_contributions.csv')],new_PHY_calls=0,new_model_calls=0,new_metric_calls=0,new_bootstrap_draws=0,original_decisions_unchanged=True,all_full_denominator_means_match_official_paired_means=True,success_conditioning_is_posthoc_not_main_evidence=True,repeat_groups_not_independent_sources=True))
 print('CLOSED_RESULTS_ANALYSIS_COMPLETE',len(result))

if __name__=='__main__':main()
