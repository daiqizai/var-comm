"""One closed formal grid; frozen source bank, paired contrasts and denominators."""
from n512_common import *
from pipeline_prepare import closed
from select_calibration import csvsave
import csv,collections
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
METHODS=('NEST','WHOLE','ADA');SNRS=(4,10,19)

def ci(values,bank):
 v=np.asarray(values,dtype=np.float64);assert v.shape==(500,)and np.isfinite(v).all()
 boot=np.concatenate([v[bank[i:i+200]].mean(axis=1,dtype=np.float64)for i in range(0,len(bank),200)])
 lo,hi=np.quantile(boot,[.025,.975],method='linear')
 return dict(mean=float(v.mean(dtype=np.float64)),ci_low=float(lo),ci_high=float(hi))

def source_aggregate(rows,identity):
 groups=collections.defaultdict(dict)
 for f in rows:
  idx=int(f['source_index']);assert identity[idx]==f['source_id']
  k=(f['method'],int(f['snr_db']),idx);seed=int(f['noise_seed']);assert seed not in groups[k];groups[k][seed]=f
 assert set(groups)=={(m,s,i)for m in METHODS for s in SNRS for i in range(500)}
 out=[];vectors={}
 for (method,snr,idx),values in sorted(groups.items()):
  assert set(values)=={6201,6202,6203}
  for metric in METRICS:
   v=float(np.mean([values[s][metric]for s in (6201,6202,6203)],dtype=np.float64));assert np.isfinite(v)
   out.append(dict(method=method,snr_db=snr,source_index=idx,source_id=identity[idx],metric=metric,source_mean=v,noise_count=3))
   vectors.setdefault((method,snr,metric),np.full(500,np.nan))[idx]=v
 assert len(out)==18000;return out,vectors

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(d)for d in r['code']+r['inputs']]
 assert read(O/'formal_post_output_audit.json')['status']=='PASS'
 for role in ('formal_three_metrics','formal_clip_metrics'):closed(role)
 protocol=read(O/'protocol.json');policy=read(O/'frozen_policies.json');assert policy['protocol_sha256']==sha(O/'protocol.json')
 bank=np.load(checked(r['bootstrap']),allow_pickle=False);assert bank.shape==(10000,500)and np.issubdtype(bank.dtype,np.integer)and bank.min()==0 and bank.max()==499
 assert asha(bank)==r['bootstrap_array_sha256']
 sources=cr(protocol['sources']['common500']);identity={s['source_index']:s['source_id']for s in sources};assert set(identity)==set(range(500))and len(set(identity.values()))==500
 plan=read(L/'formal_plan/completion.json');logical=cr(plan['logical']);choices=cr(plan['choices']);assert len(logical)==13500
 pairplan=read(L/'formal_metric_plan/completion.json');physical=cr(pairplan['physical']);frames=cr(read(L/'formal_rx_plan/completion.json')['frames'])
 three=read(L/'formal_three_metrics/scores.json');clip=read(L/'formal_clip_metrics/scores.json');assert set(three)==set(clip)
 out=O/'delivery';out.mkdir(exist_ok=False);rows=[];physical_records={};directions=dict(psnr_db='higher',lpips_alex='lower',dreamsim='lower',clip_image_cosine='higher')
 for f in logical:
  key=f['physical_key']
  if key not in physical_records:physical_records[key]=cr(frames[key])
  wire=physical_records[key];p=physical[key];pk=p['pair_key'];c=choices[f['choice_id']];x=three[pk];y=clip[pk]
  assert x['reference_sha256']==y['reference_sha256']and x['pixel_sha256']==y['pixel_sha256']==p['pixel_sha256']
  assert wire['job']['source_id']==f['source_id']and wire['job']['snr_db']==f['snr_db']and wire['job']['noise_seed']==f['noise_seed']
  rx=wire['actual_RX'];h=rx['header'];b=rx.get('body',{});q=wire['projection']
  row={k:f[k]for k in ('source_index','source_id','method','snr_db','noise_seed','candidate_id')}
  row.update(N=512,header_symbols=68,body_symbols=444,nominal_Es=2,target_m=f['target'][0],target_K=f['target'][1],actual_m=c['m'],actual_K=c['K'],q=c['q'],rate=c['rate'],k=c['k'],profile_id=c['profile_id'],arithmetic_bits=c['arithmetic_bits'],class_bits=c['class_bits'],length_bits=13,crc_bits=16,padding_bits=c['padding'],arithmetic_capacity_bits=c['capacity'],utilization=c['arithmetic_bits']/c['capacity'],TX_NO_FIT=c['tx_no_fit'],header_accepted=bool(h['header_ok']),body_crc_accepted=bool(b.get('crc_accepted',False)),body_parser_accepted=bool(b.get('parser_accepted',False)),receiver_status=p['source_status'],accepted=p['source_status']=='ACCEPTED_SOURCE',fixed_gray=p['source_status']!='ACCEPTED_SOURCE',null_received=q.get('reason')=='EXPLICIT_NULL_PROFILE',physical_key=key,received_key=p['received_key'],pair_key=pk,reference_sha256=x['reference_sha256'],pixel_sha256=p['pixel_sha256'],three_score_origin=x['origin'],clip_score_origin=y['origin'])
  row.update({m:float(x[m]if m!='clip_image_cosine'else y[m])for m in METRICS});rows.append(row)
 source_rows,vectors=source_aggregate(rows,identity);summary=[];paired=[];interactions=[]
 for method in METHODS:
  for snr in SNRS:
   for metric in METRICS:summary.append(dict(method=method,snr_db=snr,metric=metric,direction=directions[metric],sources=500,noise_per_source=3,**ci(vectors[method,snr,metric],bank)))
 for other in ('ADA','WHOLE'):
  for snr in SNRS:
   for metric in METRICS:paired.append(dict(comparison='NEST_minus_'+other,snr_db=snr,metric=metric,direction=directions[metric],sources=500,noise_per_source=3,**ci(vectors['NEST',snr,metric]-vectors[other,snr,metric],bank)))
 prior={}
 with checked(r['N1024_paired_sources']).open()as f:
  for row in csv.DictReader(f):
   idx=int(row['source_index']);assert row['source_id']==identity[idx]and row['population']=='common500'
   key=(int(row['snr_db']),row['metric']);prior.setdefault(key,{})[idx]=float(row['nest_minus_ada'])
 for snr in SNRS:
  for metric in METRICS:
   assert set(prior[snr,metric])==set(range(500))
   diff=vectors['NEST',snr,metric]-vectors['ADA',snr,metric]-np.array([prior[snr,metric][i]for i in range(500)])
   interactions.append(dict(comparison='(NEST_minus_ADA)_N512_minus_N1024',snr_db=snr,metric=metric,direction=directions[metric],sources=500,interpretation='separately calibrated systems; not budget-only causal effect',**ci(diff,bank)))
 failures=[];capacities=[];endpoints=[]
 for method in METHODS:
  for snr in SNRS:
   fs=[f for f in rows if f['method']==method and f['snr_db']==snr];assert len(fs)==1500
   status=collections.Counter(f['receiver_status']for f in fs)
   failures.append(dict(method=method,snr_db=snr,logical_frames=1500,sources=500,TX_NO_FIT=sum(f['TX_NO_FIT']for f in fs),NULL_received=sum(f['null_received']for f in fs),header_reject=sum(not f['header_accepted']for f in fs),body_crc_reject=sum(f['header_accepted']and not f['body_crc_accepted']for f in fs),body_parser_reject=sum(f['body_crc_accepted']and not f['body_parser_accepted']for f in fs),source_decoder_reject=status.get('FIXED_GRAY_SOURCE_REJECT',0),accepted=sum(f['accepted']for f in fs),fixed_gray=sum(f['fixed_gray']for f in fs),status_counts=json.dumps(dict(status),sort_keys=True)))
   capacities.append(dict(method=method,snr_db=snr,logical_frames=1500,q=fs[0]['q'],rate=fs[0]['rate'],target_m=fs[0]['target_m'],target_K=fs[0]['target_K'],mean_arithmetic_bits=float(np.mean([f['arithmetic_bits']for f in fs])),mean_padding_bits=float(np.mean([f['padding_bits']for f in fs])),mean_utilization=float(np.mean([f['utilization']for f in fs])),full_representation_frames=sum((f['actual_m']==(256 if method=='ADA'else 10)and f['actual_K']==0 and not f['TX_NO_FIT'])for f in fs)))
   for (m,K),n in sorted(collections.Counter((f['actual_m'],f['actual_K'])for f in fs).items()):endpoints.append(dict(method=method,snr_db=snr,m=m,K=K,logical_frames=n))
 for name,values in [('frame_scores',rows),('source_scores',source_rows),('summary',summary),('paired_deltas',paired),('budget_interaction',interactions),('failure_summary',failures),('capacity_summary',capacities),('endpoint_distribution',endpoints)]:csvsave(out/(name+'.csv'),values)
 assert len(summary)==36 and len(paired)==24 and len(interactions)==12
 save(out/'statistics_identity.json',dict(registration=pin(request),protocol=pin(O/'protocol.json'),policies=pin(O/'frozen_policies.json'),bootstrap=r['bootstrap'],bootstrap_array_sha256=asha(bank),seed=2026100701,resamples=10000,new_random_draws=0,source_order=identity,unit='500 sources after averaging three outcomes',interval='percentile, numpy quantile linear, pointwise 95%',pairing='same source; N512 methods share channel noise; historical N1024 not claimed shared',paired_direction='NeST minus comparator',budget_interaction_definition='(NeST512-Ada512)-(NeST1024-Ada1024)',metrics=METRICS))
 save(out/'completion.json',dict(status='COMPLETE_N512_13500_LOGICAL_FRAMES',logical_frames=13500,per_method_SNR=1500,sources=500,mean_intervals=36,paired_intervals=24,budget_interaction_intervals=12,new_random_draws=0,outputs=[pin(p)for p in sorted(out.glob('*.csv'))]+[pin(out/'statistics_identity.json')]))
 print(json.dumps(dict(summary=summary,paired=paired,failures=failures)))

if __name__=='__main__':main(*sys.argv[1:])
