"""One source-paired summary using the existing diagnostic100 index bank."""
from p1_common import *
import collections
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def main():
 out=O/'p1_delivery';out.mkdir(exist_ok=False);r=read(O/'p1_registration.json');mr=read(O/'p1_metrics_registration.json')
 control=[];completions={};scores={}
 for role in ('three','clip'):
  w=read(L/('metrics_'+role+'_control')/'actual_wait.json');assert w['actual_wait'] and w['returncode']==0 and not w['timed_out']and gone(w['process'])and gone(w['parent_process']);control.append(pin(L/('metrics_'+role+'_control')/'actual_wait.json'))
  c=read(L/('metrics_'+role)/'completion.json');assert c['registration_sha256']==sha(O/'p1_metrics_registration.json')and all(st=='COMPLETE'for _,st,_ in c['counts']);completions[role]=c
  with checked(c['scores']).open()as f:
   for j in csv.DictReader(f):
    v=scores.setdefault(j['key'],{})
    for m in METRICS:
     if m in j:assert m not in v;v[m]=float(j[m])
 frames=cr(mr['logical']);assert len(frames)==1800
 allrows=[];group=collections.defaultdict(list)
 for f in frames:
  values=scores[f['pair_key']];assert set(values)==set(METRICS)
  row={k:v for k,v in f.items()if k!='physical_frame'}|dict(physical_path=f['physical_frame']['path'],physical_sha256=f['physical_frame']['sha256'],**values)
  allrows.append(row);group[f['family'],f['snr_db'],f['source_id']].append(row)
 bank=r['bootstrap'];indices=np.load(checked(bank['indices']),allow_pickle=False);assert indices.shape==(10000,100)and indices.dtype==np.int64
 source_ids=bank['source_ids'];sources=cr(r['sources']);assert [s['source_id']for s in sources]==source_ids
 means=[];vectors={};summary=[];paired=[];failures=[]
 def ci(a):
  assert a.shape==(100,)and np.isfinite(a).all();lo,hi=np.quantile(a[indices].mean(axis=1,dtype=np.float64),[.025,.975],method='linear');return dict(mean=float(a.mean(dtype=np.float64)),ci_low=float(lo),ci_high=float(hi))
 for fam in ('WHOLE','PARTIAL'):
  for snr in (8,10,12):
   data=[]
   for sid in source_ids:
    z=group[fam,snr,sid];assert len(z)==3 and {x['noise_seed']for x in z}=={6201,6202,6203};d=dict(source_id=sid,source_index=z[0]['source_index'],family=fam,TX_policy_SNR=10,snr_db=snr,noise_count=3,failed=sum(not x['accepted']for x in z),**{m:float(np.mean([x[m]for x in z],dtype=np.float64))for m in METRICS});means.append(d);data.append(d)
   failed=sum(d['failed']for d in data);failures.append(dict(family=fam,TX_policy_SNR=10,true_snr_db=snr,rejected=failed,total=300,rejection_rate=failed/300))
   for m in METRICS:
    a=np.asarray([d[m]for d in data],dtype=np.float64);vectors[fam,snr,m]=a;summary.append(dict(family=fam,TX_policy_SNR=10,true_snr_db=snr,metric=m,sources=100,logical_frames=300,**ci(a)))
 for snr in (8,10,12):
  for m in METRICS:paired.append(dict(comparison='PARTIAL_minus_WHOLE',TX_policy_SNR=10,true_snr_db=snr,metric=m,sources=100,**ci(vectors['PARTIAL',snr,m]-vectors['WHOLE',snr,m])))
 assert all(np.array_equal(vectors[f,10,m],vectors[f,12,m])for f in ('WHOLE','PARTIAL')for m in METRICS)
 assert all(np.array_equal(vectors['WHOLE',8,m],vectors['PARTIAL',8,m])for m in METRICS)
 csvsave(out/'frame_scores.csv',allrows);csvsave(out/'source_means.csv',means);csvsave(out/'summary.csv',summary);csvsave(out/'paired_differences.csv',paired);csvsave(out/'rejections.csv',failures)
 noise=np.load(checked(r['noise']),allow_pickle=False);csvsave(out/'noise_identity.csv',[dict(rank=i,source_id=s['source_id'],source_index=s['source_index'],noise_seed=seed,standard_sha256=ash(noise[i*3+seed-6201]))for i,s in enumerate(sources)for seed in (6201,6202,6203)])
 save(out/'source_list.json',sources);save(out/'statistics_identity.json',dict(bootstrap=bank,source_order=source_ids,new_random_draws=0,source_count=100,noise_average_first=True,paired_difference_before_resampling=True,interval='pointwise percentile95%, NumPy linear quantile; existing10000x100 bank',population='existing diagnostic subset; not a new blind test'))
 link=read(L/'link_closed_v1/completion.json');compute=dict(logical_receptions=1800,new_physical=1200,new_packet_decodes=2400,reused10dB_exact_events=600,new_source_probability=0,new_tokenizer=0,new_source_decoding=0,new_VAR_reconstructions=0,successful_logical_RX_reuses=1200,gray_logical_rejections=600,unique_source_pixel_pairs=261,new_PSNR_pairs=100,new_LPIPS_pairs=100,new_DreamSim_pairs=100,new_CLIP_pairs=161,new_metric_scalars=461,logical_metric_scalars=7200,unique_prior_metric_scalars_reused=583,new_bootstrap_indices=0,source_bootstrap_mean_CIs=24,paired_CIs=12,original_link_exit=1,postloop_assertion_metadata_closed=True,original_failure=link['original_failure'],metadata_closure=pin(L/'link_closed_v1/completion.json'),metric_completions=completions)
 save(out/'actual_compute_and_reuse.json',compute)
 save(out/'completion.json',dict(status='COMPLETE_1800_LOGICAL_RECEPTIONS_FOUR_METRICS',registration=pin(O/'p1_registration.json'),metric_registration=pin(O/'p1_metrics_registration.json'),actual_metric_waits=control,outputs=[pin(p)for p in sorted(out.iterdir())if p.is_file()],original_link_exit_preserved=1,metadata_closure_actual_exit=0,all_prescribed_conditions_reported=True,no_new_main_results=True))
 print(json.dumps(dict(rejections=failures,paired=paired,compute={k:v for k,v in compute.items()if k!='metric_completions'})))
if __name__=='__main__':main()
