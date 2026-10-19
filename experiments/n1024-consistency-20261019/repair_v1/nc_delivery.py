"""Assemble the common500 grid and reuse the existing bootstrap index matrix."""
from nc_common import *
from nc_protocol import closed_wait,validate_full_grid,source_means,interval
import numpy as np,collections,traceback
def run(expected):
 r=registration('link',expected);out=T/'delivery';out.mkdir();light=O/'delivery';light.mkdir();start=time.monotonic()
 try:
  controls=[];done={}
  for role in ('phy','receiver','metrics'):
   c=T/(role+'_control');closed_wait(read(c/'actual_wait.json'),gone)
   assert not(c/'failure.json').exists() and not(T/role/'failure.json').exists()
   d=read(T/role/'completion.json');assert d['registration_sha256']==expected;done[role]=d
   controls.extend(pin(p) for p in [c/'launch.json',c/'actual_wait.json',c/'stdout.txt',c/'stderr.txt',T/role/'completion.json'])
  assert done['phy']['new_physical']==2421 and done['phy']['new_packets']['unresolved']==0
  assert done['receiver']['ledger']['unresolved']==done['metrics']['ledger']['unresolved']==0
  changed=checkread(done['metrics']['rows']);reuse=checkread(r['reused_frames']);rows=list(changed)
  for v in reuse:
   rows.append(dict(population=v['population'],source_global_index=v['source_index'],source_index=v['source_index'],source_id=v['source_id'],snr_db=v['snr_db'],noise_seed=v['noise_seed'],strategy_family=v['family'],candidate_id='FROZEN_EXISTING',actual_m=v['m'],actual_K=v['K'],arithmetic_bits=v['arithmetic_bits'],physical_reuse=True,physical_frame=v['physical_frame'],reconstruction=v['reconstruction'],metrics=v['metrics'],score_origin=v['reuse_class']))
  validate_full_grid(rows);rows.sort(key=lambda x:(x['source_index'],x['snr_db'],x['strategy_family'],x['noise_seed']))
  assert sum(x['physical_reuse'] for x in rows)==6579
  verified={};flat=[];status=collections.Counter();fallback=collections.Counter()
  for v in rows:
   assert set(v['metrics'])==set(METRICS)
   for d in (v['physical_frame'],v['reconstruction']['image_archive']):
    if d['path'] not in verified:checked(d);verified[d['path']]=d
   pkt=checkread(v['physical_frame']);rx=pkt['actual_RX'];rp=rx['rx_profile'];s=int(v['snr_db']);fam=v['strategy_family'];cap={4:1245,10:2520,19:4751}[s]
   assert pkt['transmission']['arithmetic_bits']==v['arithmetic_bits']
   assert (pkt['transmission']['m'],pkt['transmission']['K'])==(v['actual_m'],v['actual_K'])
   status[(fam,s,rx['status'])]+=1;fallback[(fam,s,v['actual_m'],v['actual_K'])]+=1
   flat.append(dict(population='common500',source_index=v['source_index'],source_id=v['source_id'],family=fam,snr_db=s,noise_seed=v['noise_seed'],sent_m=v['actual_m'],sent_K=v['actual_K'],arithmetic_bits=v['arithmetic_bits'],source_capacity_bits=cap,utilization=v['arithmetic_bits']/cap,N=1024,header_symbols=68,body_symbols=956,length_bits=13,CRC_bits=16,RX_status=rx['status'],received_m=None if rp is None else rp['m'],received_K=None if rp is None else rp['K'],physical_reused=v['physical_reuse'],score_origin=v['score_origin'],image_sha256=v['reconstruction']['image_sha256'],**v['metrics']))
  save(out/'metric_rows_9000.json',rows);csvsave(light/'frame_scores_9000.csv',flat)
  checked(r['bootstrap']['indices']);idx=np.load(r['bootstrap']['indices']['path'],allow_pickle=False)
  assert asha(idx)==r['bootstrap']['array_sha256'] and idx.shape==(10000,500) and idx.dtype==np.int64
  means=[];summary=[];paired=[]
  for snr in (4,10,19):
   for metric in METRICS:
    a={f:source_means(rows,snr,f,metric) for f in ('WHOLE','PARTIAL')}
    for fam in ('WHOLE','PARTIAL'):
     summary.append(dict(population='common500',family=fam,snr_db=snr,metric=metric,sources=500,noise_per_source=3,**interval(a[fam],idx)))
     means.extend(dict(population='common500',source_index=i,snr_db=snr,family=fam,metric=metric,source_mean=float(a[fam][i])) for i in range(500))
    paired.append(dict(population='common500',snr_db=snr,comparison='PARTIAL_minus_WHOLE',metric=metric,sources=500,noise_per_source=3,**interval(a['PARTIAL']-a['WHOLE'],idx)))
  csvsave(light/'summary_common500.csv',summary);csvsave(light/'paired_common500.csv',paired);csvsave(light/'source_means_common500.csv',means)
  csvsave(light/'RX_status_counts.csv',[dict(family=k[0],snr_db=k[1],status=k[2],logical_frames=v) for k,v in sorted(status.items())])
  csvsave(light/'actual_fallback_counts.csv',[dict(family=k[0],snr_db=k[1],m=k[2],K=k[3],logical_frames=v,sources=v//3) for k,v in sorted(fallback.items())])
  save(light/'closure_audit.json',dict(status='COMPLETE_N1024_VERSION_CONSISTENT_COMMON500',registration_sha256=expected,full_grid=r['full_grid'],controls=controls,stages=done,verified_files=list(verified.values()),source_utilization=r['source_utilization'],bootstrap=r['bootstrap'],new_bootstrap_draws=0,no_policy_selection=True,old_results_untouched=True,new100_pooled=False,metric_rows=pin(out/'metric_rows_9000.json')))
  save(light/'manifest.json',dict(registration_sha256=expected,files=[pin(p) for p in sorted(light.iterdir())]))
  completion=dict(status='COMPLETE_N1024_MINIMAL_VERSION_REPAIR',registration_sha256=expected,logical_frames=9000,new_physical=2421,reused_logical=6579,manifest=pin(light/'manifest.json'),seconds=time.monotonic()-start,new_bootstrap_draws=0,automatic_successor=False)
  save(light/'completion.json',completion);save(out/'completion.json',completion)
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
