"""Recheck V8 claims directly against sealed CSVs, without model/statistical reruns."""
from audit_closed_claims import *
import math
def mean(vs,**k):return float(one(vs,**k)['mean'])
ext=[v for v in e1 if v['N']=='1024' and v['candidate']=='final_NeST' and v['reference'] in ['ARPC','SwinJSCC','Latent_JSCC','BPG_LDPC'] and v['metric'] in ['dists','dreamsim']]
assert len(ext)==32 and all(map(favorable,ext))
for m in ['SwinJSCC','BPG_LDPC']:
 for snr in [4,7,10,19]:assert mean(s1,method=m,N=1024,snr_db=snr,metric='psnr_db')>mean(s1,method='final_NeST',N=1024,snr_db=snr,metric='psnr_db')
ada_psnr=[v for v in e1 if v['candidate']=='final_NeST' and v['reference']=='Ada_frozen' and v['metric']=='psnr_db']
assert len(ada_psnr)==7 and all(map(favorable,ada_psnr))
relative={}
for metric,rounded in [('dists',8.7),('dreamsim',20.5),('lpips_alex',16.6)]:
 w=mean(s1,method='WHOLE_final_receiver',N=1024,snr_db=10,metric=metric);n=mean(s1,method='final_NeST',N=1024,snr_db=10,metric=metric)
 relative[metric]={'WHOLE':w,'NeST':n,'reduction_percent':100*(w-n)/w};assert round(relative[metric]['reduction_percent'],1)==rounded
cap=rows(P/'inputs/e1/capacity_summary.csv')
util=[100*float(one(cap,method=m,N=1024,snr_db=10)['utilization_mean']) for m in ['WHOLE_final_receiver','final_NeST']]
assert [round(x,2)for x in util]==[77.70,99.79]
capacity=rows(P/'inputs/e2/representation_and_capacity.csv');s2=rows(P/'inputs/e2/summary.csv');plateau={}
for m,expected in [('WHOLE',3.1),('Token_NeST',89.9)]:
 bits=[float(one(capacity,method=m,B_bits=b)['mean_actual_stream_bits']) for b in [1212,1302]]
 ds=[mean(s2,method=m,B_bits=b,metric='dreamsim') for b in [1212,1302]]
 plateau[m]={'budget_bits':[1212,1302],'stream_bits':bits,'additional_bits':bits[1]-bits[0],'dreamsim':ds}
 assert round(bits[1]-bits[0],1)==expected
for b in [620,1212,1302]:
 for m in ['dists','dreamsim']:assert favorable(one(e2,candidate='Token_NeST',reference='Ada',B_bits=b,metric=m))
for b in [2248,2577]:
 for m in ['dists','dreamsim']:assert float(one(e2,candidate='Token_NeST',reference='Ada',B_bits=b,metric=m)['ci95_low'])>0
def time(group,direction='total'):return float(one(e5,group=group,direction=direction)['equal_source_mean_seconds'])
n=time('Token_NeST_N1024_S10');w=time('WHOLE_N1024_S10');a=time('Ada_N1024_S10')
cost={'NeST_total_s':n,'WHOLE_total_s':w,'Ada_total_s':a,'Ada_over_NeST':a/n,'NeST_over_WHOLE_increase_percent':100*(n/w-1),'Token_over_Quarter_TX_increase_percent':100*(time('Token_NeST_N1024_S10','TX')/time('granularity_quarter','TX')-1)}
assert round(n,3)==.660 and round(a,3)==15.422 and round(a/n,1)==23.4
assert round(cost['NeST_over_WHOLE_increase_percent'],1)==13.1
assert round(cost['Token_over_Quarter_TX_increase_percent'],1)==16.0
ops=[v for v in rows(P/'inputs/e5/source_fitting_accounting.csv') if v['group']=='Token_NeST_N1024_S10' and v['phase']=='measured']
assert len(ops)==96
assert all(v['actual_candidate_length_tests']=='352' and v['fresh_source_probability_traversals']=='1' and v['cdf_construction_steps']=='9' for v in ops)
eff=one(rows(P/'inputs/e3/efficiency.csv'),method='token');assert float(eff['actual_length_comparisons'])==125.96
for metric in ['psnr_db','dists','dreamsim']:
 for c,r in [('B0','A0'),('C0','B0')]:assert favorable(one(e4,candidate=c,reference=r,snr_db=10,metric=metric))
fig=json.loads((P/'inputs/f1/selected_native_mapping.json').read_text())
for row in [1,2]:
 cell=[v for v in fig if v['row']==row and v['column']>0];best=next(v for v in cell if v['column']==3)
 for metric in ['dists','dreamsim']:assert best[metric]==min(v[metric]for v in cell)
old=P.parent/'nestcom_wcl_tokenwise_20261030_v7'
assert old.exists()
frozen=[p for p in (P/'inputs').rglob('*')if p.is_file() and p.suffix=='.csv']
assert all(p.read_bytes()==(old/p.relative_to(P)).read_bytes()for p in frozen)
assert all((P/f'figures/figure2.{x}').read_bytes()==(old/f'figures/figure2.{x}').read_bytes()for x in ['pdf','png','svg'])
assert all(p.read_bytes()==(old/p.relative_to(P)).read_bytes()for p in (P/'figures/individual').glob('*.png'))
checks.update(V8_editorial_only=True,NeST_WHOLE_favorable_intervals=35,external_two_distance_favorable_intervals=32,NeST_Ada_PSNR_favorable_intervals=7,
 relative_distance_reductions=relative,N1024_10_utilization_percent=util,E2_90_bit_diagnostic=plateau,processing_cost=cost,
 E3_cached_comparisons_mean=125.96,E5_actual_candidate_tests=352,E5_probability_traversals=1,E5_scale_CDF_steps=9,
 best_Fig3_values_bold_supported=True,frozen_CSV_files_unchanged=len(frozen),figure2_byte_identical=True,all_native_display_PNGs_byte_identical=True,
 new_scientific_model_PHY_metric_bootstrap_calls=0)
(P/'validation/claim_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
(P/'validation/data_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:checks[k]for k in ['abstract_word_count_whitespace','NeST_WHOLE_favorable_intervals','external_two_distance_favorable_intervals','E2_90_bit_diagnostic','processing_cost','frozen_CSV_files_unchanged']}))
