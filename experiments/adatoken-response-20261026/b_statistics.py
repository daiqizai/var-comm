"""Closed B scores, original fixed100 bootstrap bank; no new random draws."""
from common import *
import argparse,numpy as np,traceback
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def interval(v,bank):
 valid=np.isfinite(v);samples=v[bank];counts=np.isfinite(samples).sum(axis=1);assert np.all(counts>0)
 draws=np.nansum(samples,axis=1)/counts;lo,hi=np.quantile(draws,[.025,.975],method='linear');return dict(mean=float(np.mean(v[valid])),ci_low=float(lo),ci_high=float(hi),source_count=int(valid.sum()))
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(exist_ok=False);dest=O/'b_delivery';assert not dest.exists();beg=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv))
 try:
  for w in r['waits']:
   v=cr(w);assert type(v['returncode'])is int and v['returncode']==0 and gone(v['process'])and gone(v['parent_process'])
  triple=list(csv.DictReader(checked(r['three']).open()));clip=list(csv.DictReader(checked(r['clip']).open()));key=lambda v:(v['source_id'],int(v['snr_db']),int(v['noise_seed']),v['arm']);ci={key(v):v for v in clip};assert len(ci)==1800
  frames=[]
  for v in triple:
   c=ci[key(v)];assert c['pixel_sha256']==v['pixel_sha256'];v['clip_image_cosine']=c['clip_image_cosine'];frames.append(v)
  assert len(frames)==1800 and len({key(v)for v in frames})==1800
  b=cr(r['bank']);bank=np.load(checked(b['indices']),allow_pickle=False);order=b['source_ids'];assert bank.shape==(10000,100)and set(order)=={v['source_id']for v in frames};assert len(set(order))==100
  allkeys={key(v)for v in frames};assert allkeys=={(s,snr,n,a)for s in order for snr in(10,19)for n in(6201,6202,6203)for a in('A0','B0','C0')}
  summaries=[];paired=[];means=[]
  for snr in(10,19):
   for group in('all','positive_K','final_scale_positive_K'):
    index={}
    for arm in('A0','B0','C0'):
     for metric in METRICS:
      values=[];framecount=0
      for sid in order:
       sub=[v for v in frames if v['source_id']==sid and int(v['snr_db'])==snr and v['arm']==arm and(group=='all'or(v['accepted']=='True'and int(v['K']or 0)>0 and(group!='final_scale_positive_K'or int(v['m'])==9)))]
       value=float(np.mean([float(v[metric])for v in sub]))if sub else np.nan;values.append(value);framecount+=len(sub)
       if sub:means.append(dict(source_id=sid,source_index=int(sub[0]['source_index']),snr_db=snr,group=group,arm=arm,metric=metric,source_mean=value,noise_count=len(sub)))
      values=np.asarray(values,np.float64);index[arm,metric]=values
      if np.isfinite(values).any():summaries.append(dict(snr_db=snr,group=group,arm=arm,metric=metric,logical_frames=framecount,**interval(values,bank)))
    for first,second in(('B0','A0'),('C0','B0'),('C0','A0')):
     for metric in METRICS:
      d=index[first,metric]-index[second,metric]
      if np.isfinite(d).any():paired.append(dict(snr_db=snr,group=group,comparison=first+'_minus_'+second,metric=metric,**interval(d,bank)))
  final=[p for p in paired if p['group']=='final_scale_positive_K'and p['comparison']=='C0_minus_B0'];assert final and all(p['mean']==p['ci_low']==p['ci_high']==0 for p in final)
  dest.mkdir();csvsave(dest/'frame_metrics.csv',frames);csvsave(dest/'source_means.csv',means);csvsave(dest/'summary.csv',summaries);csvsave(dest/'paired_differences.csv',paired)
  save(dest/'statistics_identity.json',dict(bank=r['bank'],source_order=order,new_random_draws=0,group_rule='Within each predeclared group, average available original noises per source, then equal-weight eligible sources. Reuse original100-source draws with finite-group entries retained; conditional source bootstrap.',all_group_denominator=100,positive_group_is_postreception_diagnostic=True,interval='pointwise percentile95, method=linear; no multiplicity correction',paired='Form source differences before resampling; no marginal CI subtraction',not_theoretical_additive_PSNR=True))
  manifest=[pin(p)for p in dest.iterdir()];save(dest/'completion.json',dict(status='B_SCIENCE_AND_STATISTICS_COMPLETE',frames=1800,means=len(summaries),paired=len(paired),registration=pin(path),outputs=manifest,new_bootstrap_draws=0,elapsed_seconds=time.monotonic()-beg))
  print(json.dumps([p for p in paired if p['group']=='all'],indent=2))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
