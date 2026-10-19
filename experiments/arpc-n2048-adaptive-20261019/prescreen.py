"""Finite DINO-only nomination with explicit historical paired-success proxy."""
from shared import *
from adaptive_plan import layouts,policies,choose,RATES
import csv,collections,math
def bounds(gray,clean,prob,low,high):
 gain=clean-gray;ends=(gray+low*gain,gray+high*gain)
 return gray+prob*gain,min(ends),max(ends)
def nominate(rows):
 leaders=[]
 for g in layouts():
  group=[x for x in rows if x['layout_id']==g['layout_id']]
  assert group
  leaders.append(sorted(group,key=lambda x:(-x['clean_dino_mean'],-x['target']))[0])
 def order(x):return (-x['proxy_dino'],x['q'],RATES.index(x['nominal_rate']),-x['target'])
 best=sorted(leaders,key=order)[0]
 competitor=sorted([x for x in leaders if x is not best],key=lambda x:(-x['proxy_upper'],*order(x)))[0]
 best_lower=max(x['proxy_lower'] for x in leaders);possible=[x for x in leaders if x['proxy_upper']>=best_lower]
 selected=[best,competitor];uncovered=[x for x in possible if x not in selected]
 return dict(status='WITHIN_TWO_POLICY_CAP' if not uncovered else 'PENDING_MORE_THAN_TWO_PLAUSIBLE_POLICIES',selected_proposals=[x['policy_id']for x in selected],possible_best=[x['policy_id']for x in possible],uncovered=[x['policy_id']for x in uncovered],within_MCS_leaders=[x['policy_id']for x in leaders],additional_candidate_policies=len(uncovered),additional_cal1000_logical_upper=3000*len(uncovered),additional_subframes_upper=6000*len(uncovered),additional_decoder_upper=12000*len(uncovered))
def csvwrite(path,rows):
 with path.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 w=read(L/'controls/cal100_dino/actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 done=read(L/'cal100_scores_v1/worker/completion.json');assert done['ledger']['unresolved']==0
 for k in ('scores','metric_identity'):verify(done[k])
 new={(x['source_id'],x['image_float_sha256']):x for x in read(done['scores']['path'])}
 states=read(L/'cal100_scores_v1/all_states.json');quality={}
 for x in states:
  if x['dino'] is None:x['dino']=new[x['source_id'],x['image_float_sha256']]['dino']
  assert math.isfinite(x['dino']);quality[x['source_id'],x['codec'],x['endpoint']]=x['dino']
 assert len(quality)==3200
 graypath=OLD_O/'a2_metrics_v1/clean_quality.csv';cells_path=OLD_O/'a2_reliability_v1/reliability_cells.csv'
 with graypath.open()as f:gray={x['source_id']:float(x['dinov2_vitl14_cosine'])for x in csv.DictReader(f)if x['endpoint']=='0'}
 with cells_path.open()as f:cells={(int(x['snr_db']),int(x['q']),x['nominal_rate']):x for x in csv.DictReader(f)}
 src=read(OLD_O/'sources.json')['cal100'];ids=[x['source_id']for x in src];assert len(gray)==len(ids)==100 and set(gray)==set(ids)
 streams=collections.defaultdict(dict)
 for s in read(L/'b0/source_catalogue.json'):
  if s['cal100']:streams[s['source_id'],s['codec']][s['endpoint']]=s
 gm={x['layout_id']:x for x in layouts()};details=[];summaries=[]
 for codec in (0,1):
  for pol in policies(codec):
   g=gm[pol['layout_id']];values=[];actual=[]
   for sid in ids:
    end=choose(streams[sid,codec],codec,pol['target'],g);q=gray[sid] if end is None else quality[sid,codec,end];actual.append(end);values.append(q)
    s=streams[sid,codec].get(end)
    details.append(dict(codec=codec,source_id=sid,policy_id=pol['policy_id'],target=pol['target'],layout_id=g['layout_id'],actual_endpoint=end,object_bits=0 if s is None else s['bit_length'],capacity_bits=g['object_bits'],source_no_fit=end is None,clean_dino=q,gray_dino=gray[sid]))
   mean=sum(values)/100;graymean=sum(gray.values())/100
   for snr in (4,10):
    cell=cells[snr,g['q'],g['nominal_rate']];assert (int(cell['k']),int(cell['n']))==(g['k'],g['n'])
    probability=float(cell['success_rate']);low=float(cell['simultaneous95_bonferroni_low']);high=float(cell['simultaneous95_bonferroni_high']);point,lo,hi=bounds(graymean,mean,probability**2,low**2,high**2)
    summaries.append(dict(codec=codec,snr_db=snr,policy_id=pol['policy_id'],target=pol['target'],layout_id=g['layout_id'],q=g['q'],nominal_rate=g['nominal_rate'],actual_rate=g['actual_rate'],k=g['k'],n=g['n'],clean_dino_mean=mean,gray_dino_mean=graymean,proxy_dino=point,proxy_lower=lo,proxy_upper=hi,single_packet_trials=int(cell['trials']),single_packet_successes=int(cell['successes']),conditional_independence_pair_success_proxy=probability**2,source_no_fit_count=actual.count(None),endpoint_vector_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest()))
 out=O/'cal100_prescreen_v1';out.mkdir();decisions={str(codec)+'_'+str(snr):nominate([x for x in summaries if x['codec']==codec and x['snr_db']==snr])for codec in (0,1)for snr in (4,10)}
 csvwrite(out/'actual_fallback_quality.csv',details);csvwrite(out/'strategy_proxy.csv',summaries)
 passed=all(d['status']=='WITHIN_TWO_POLICY_CAP' for d in decisions.values())
 put(out/'nomination.json',dict(status='READY_FOR_B2_REGISTRATION' if passed else 'PENDING_UNCOVERED_CANDIDATE_UNCERTAINTY',decisions=decisions,B2_started=False,B3_started=False,clean_states=3200,all_strategies=384,nomination_metric='DINOv2-L only, every source actual legal length fallback',within_MCS_rule='Maximum mean actual clean DINO under common nonnegative single-MCS probability proxy. Higher target breaks exact ties; all candidates retained.',across_MCS_ties='Higher proxy DINO; lower q, lower nominal rate, higher target.',proxy_limitation='Squared historical single-packet joint header/body acceptance with Bonferroni bounds; conditional-independence/payload-transfer approximation, not measured new dual-fragment reliability. Source CRC/profile changed. B1 six noisy pairs qualify implementation only.',uncertainty_stop='If any third plausible MCS leader is uncovered, do not launch B2 or silently discard it; report additional costs. No new budget implied.',inputs=[pin(graypath),pin(cells_path),pin(L/'cal100_scores_v1/worker/completion.json'),pin(L/'cal100_verified_cache_v1/completion.json'),pin(O/'qualification/review.json')],source=pin(Path(__file__)),new_channel=0,new_models=0,new_bootstrap=0))
 print(json.dumps(dict(passed=passed,decisions=decisions),indent=2))
if __name__=='__main__':main()
