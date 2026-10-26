"""Fixed A grids: complete source-level estimates using the existing 100-source bank."""
from common import *
from b_statistics import interval,METRICS
import argparse,itertools,numpy as np,traceback
ARMS=('WHOLE','HALF','QUARTER','EIGHTH')
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(exist_ok=False);dest=O/'a_delivery';assert not dest.exists();beg=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv))
 try:
  for d in r['waits']:
   w=cr(d);assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
  three=list(csv.DictReader(checked(r['three']).open()));clip=list(csv.DictReader(checked(r['clip']).open()));key=lambda v:(v['source_id'],int(v['snr_db']),int(v['noise_seed']),v['arm']);ci={key(v):v for v in clip};assert len(ci)==3600
  frames=[]
  for v in three:
   cv=ci[key(v)];assert cv['pixel_sha256']==v['pixel_sha256'];v['clip_image_cosine']=cv['clip_image_cosine'];frames.append(v)
  b=cr(r['bank']);bank=np.load(checked(b['indices']),allow_pickle=False);order=b['source_ids'];assert bank.shape==(10000,100)and len(set(order))==100
  exp={(s,snr,noise,arm)for s in order for snr in(4,10,19)for noise in(6201,6202,6203)for arm in ARMS};assert len(frames)==3600 and {key(v)for v in frames}==exp
  grouped={}
  for v in frames:grouped.setdefault((v['source_id'],int(v['snr_db']),v['arm']),[]).append(v)
  means=[];summary=[];paired=[]
  for snr in(4,10,19):
   index={}
   for arm in ARMS:
    for metric in METRICS:
     values=[]
     for sid in order:
      sub=grouped[sid,snr,arm];assert len(sub)==3;value=float(np.mean([float(v[metric])for v in sub]));values.append(value);means.append(dict(source_id=sid,source_index=int(sub[0]['source_index']),snr_db=snr,arm=arm,metric=metric,source_mean=value,noise_count=3))
     a=np.asarray(values,np.float64);assert np.isfinite(a).all();index[arm,metric]=a;summary.append(dict(snr_db=snr,arm=arm,metric=metric,logical_frames=300,**interval(a,bank)))
   for small,large in itertools.combinations(ARMS,2):
    for metric in METRICS:paired.append(dict(snr_db=snr,comparison=large+'_minus_'+small,metric=metric,**interval(index[large,metric]-index[small,metric],bank)))
  plan=cr(r['plan']);logical=cr(r['logical']);assert len(logical)==3600;grid={1:'WHOLE',2:'HALF',4:'QUARTER',8:'EIGHTH'};endpoint=[];efficiency=[]
  for snr in(4,10,19):
   for g,arm in grid.items():
    sub=[v for v in logical if v['grid']==g and v['snr_db']==snr];scores=[v for v in frames if v['arm']==arm and int(v['snr_db'])==snr];assert len(sub)==len(scores)==300
    counts={}
    for v in sub:counts[v['m'],v['K'],v['null']]=counts.get((v['m'],v['K'],v['null']),0)+1
    for (m,K,null),count in sorted(counts.items()):endpoint.append(dict(snr_db=snr,arm=arm,transmitted_m=m,transmitted_K=K,explicit_null=null,logical_frames=count,denominator=300))
    efficiency.append(dict(snr_db=snr,arm=arm,candidates=sub[0]['endpoint_count'],mean_arithmetic_utilization=float(np.mean([0 if v['null']else v['arithmetic_bits']/v['source_capacity_bits']for v in sub])),mean_transmitted_tokens=float(np.mean([0 if v['null']else OFFSETS[v['m']]+v['K']for v in sub])),mean_terminated_arithmetic_bits=float(np.mean([0 if v['null']else v['arithmetic_bits']for v in sub])),mean_descending_length_checks=float(np.mean([v['length_checks']for v in sub])),source_parse_accepted=sum(v['accepted']=='True'for v in scores),logical_frames=300,physical_events=len({v['event_key']for v in sub})))
  dest.mkdir();csvsave(dest/'frame_metrics.csv',frames);csvsave(dest/'source_means.csv',means);csvsave(dest/'summary.csv',summary);csvsave(dest/'paired_differences.csv',paired);csvsave(dest/'efficiency.csv',efficiency);csvsave(dest/'transmitted_endpoint_distribution.csv',endpoint)
  save(dest/'statistics_identity.json',dict(bank=r['bank'],source_order=order,new_random_draws=0,source_count=100,noise_count=3,source_aggregation='Mean three frozen noises per source, equal-weight all100 sources',paired='source differences before resampling; all six grid pairs',interval='pointwise percentile95, method=linear; no multiplicity adjustment',models_MCS_and_maximum_endpoint_fixed=True,not_historical_WHOLE_PARTIAL_joint_calibration=True,efficiency='terminated arithmetic bits divided by k-29 source capacity; null=0; transmitted endpoints distinct from acceptance',logical_frames=3600,physical_events=2214,packet_decodes=4428,unique_native_outputs=738,source_decode_new=732,source_decode_reused_qualification=6,qualification_source_decode_spent=49,metrics_new_pairs=plan['unique_new_pairs'],metrics_reused_B_logical=plan['reused_old_logical'],not_new_blind_test=True))
  save(dest/'completion.json',dict(status='A_SCIENCE_AND_STATISTICS_COMPLETE',registration=pin(path),means=len(summary),paired=len(paired),outputs=[pin(p)for p in dest.iterdir()],elapsed_seconds=time.monotonic()-beg,new_bootstrap_draws=0));print(json.dumps([v for v in paired if v['snr_db']==10],indent=2))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
