"""Read-only count/identity/arithmetic validation of completed scientific CSVs."""
from opt_common import *
from route_statistics_v2 import METRICS
import csv,collections,subprocess

def rows(p):
 with Path(p).open()as f:return list(csv.DictReader(f))

def main():
 final=D/'final_common500_delivery';delivery=D/'delivery';out=delivery/'validation';out.mkdir(exist_ok=True)
 frame=rows(final/'frame_metrics.csv');source=rows(final/'per_source_metrics.csv');summary=rows(final/'summary.csv');paired=rows(final/'paired_deltas.csv')
 assert len(frame)==42000 and len(source)==14000 and len(summary)==112 and len(paired)==114
 manifest={s['source_id']:s for s in read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json')};assert len(manifest)==500
 fg=collections.defaultdict(list);sg=collections.defaultdict(dict);keys=set()
 for v in frame:
  assert v['source_id']in manifest and int(v['source_index'])==manifest[v['source_id']]['source_index']
  assert v['reference_sha256']==manifest[v['source_id']]['reference_sha256']
  k=(v['method'],int(v['N']),int(v['snr_db']),v['source_id']);event=(*k,v['noise_seed']);assert event not in keys;keys.add(event);fg[k].append(v)
  assert all(np.isfinite(float(v[m]))for m in METRICS)
 for v in source:
  k=(v['method'],int(v['N']),int(v['snr_db']),v['source_id']);g=fg[k]
  assert len(g)==3 and len({x['channel_repeat']for x in g})==3
  for m in METRICS:assert abs(float(v[m])-np.mean([float(x[m])for x in g]))<1e-12
  sg[k[:3]][k[3]]=v
 assert len(sg)==28 and all(set(g)==set(manifest)for g in sg.values())
 for v in summary:
  g=sg[(v['method'],int(v['N']),int(v['snr_db']))];m=v['metric'];mean=np.mean([float(x[m])for x in g.values()]);assert abs(mean-float(v['mean']))<1e-12
  assert float(v['ci95_low'])<=float(v['ci95_high'])
 fixed={(v['N'],v['snr_db']):v for v in read(D/'fit_baseline_denominators.json')['values']}
 for v in paired:
  if v['snr_db']=='core_mean':
   m=v['metric'].removeprefix('FIT_normalized_');ds=[]
   for sid in manifest:ds.append(np.mean([(float(sg[(v['candidate'],1024,c)][sid][m])-float(sg[(v['reference'],1024,c)][sid][m]))/fixed[(1024,c)][m]for c in [7,10]]))
  else:
   cell=(int(v['N']),int(v['snr_db']));m=v['metric'];a=sg[(v['candidate'],*cell)];b=sg[(v['reference'],*cell)];ds=[float(a[s][m])-float(b[s][m])for s in manifest]
  assert abs(float(v['mean_delta'])-np.mean(ds))<1e-12
  assert float(v['ci95_low'])<=float(v['ci95_high'])
 route=rows(delivery/'route_summary.csv');assert len(route)==13
 c=next(v for v in route if v['candidate']=='C_content_1000');assert json.loads(c['guard_violations'])[0]['cell']==[512,19]
 finalchoice=read(D/'frozen_final.json');assert finalchoice['configuration']=='original_receiver_plus_D_token'
 stages=read(delivery/'stage_completion.json')['stages'];assert stages['NEW200']['status']=='blocked' and stages['C_second_seed']['status']=='not-run-conditional'
 checks=dict(status='PASS',scope='Completed CSV arithmetic, counts, identifiers and gate reporting only; no metric or bootstrap rerun',frame_rows=len(frame),source_rows=len(source),cells=28,sources_per_cell=500,repeats_per_source=3,summary_CI_rows=112,direct_paired_CI_rows=114,all_source_ids_pixel_identities_and_indices_match=True,all_finite=True,source_mean_equals_three_frame_average=True,summary_equals_equal_source_mean=True,paired_mean_equals_direct_source_difference=True,fixed_FIT_denominators_used_for_core=True,C_guard_failure_visible=True,conditional_second_seed_not_misreported=True,NEW200_blocked_visible=True,inputs=[pin(final/x)for x in ['frame_metrics.csv','per_source_metrics.csv','summary.csv','paired_deltas.csv','statistics_identity.json']],scientific_calls_added=0)
 save(out/'data_checks.json',checks)
 report=(delivery/'REPORT_zh.txt').read_text();assert all(s in report for s in ['NEW200','1.088427','跨零','没有自动提交','H800','共享负载'])
 save(out/'claim_audit.json',dict(status='PASS_WITH_DECLARED_LIMITS',checks=pin(out/'data_checks.json'),claims=pin(delivery/'claim_evidence.json'),new_blind_test_claimed=False,all_metrics_SOTA_claimed=False,C_adopted=False,DreamSim_confirmation_significant_claimed=False,original_V6_untouched=True,figures_visual_review_required_separately=True))
 print('DELIVERY_DATA_CHECKS_PASS')

if __name__=='__main__':main()
