"""Original source-mean DINO selection, no testing or bootstrap."""
from tb_common import *
import tb_plan as plan,collections,traceback
def run(expected):
 r,out=stage_start('selection',expected)
 try:
  done=read(T/'metrics_recovery_v2/completion.json');assert done['registration_sha256']==expected
  rows=checkread(done['metric_rows']);inh=checkread(r['inheritance']);oldrows=checkread(inh['old_metric_rows']['full']);oldw=checkread(inh['old_full_winners']);groups=collections.defaultdict(list)
  for row in oldrows+rows:groups[row['snr_db'],row['candidate_id'],row['source_index']].append(row)
  sources=checkread(r['sources']);newcs={c['candidate_id']:c for c in plan.candidates()};summary=[];policies={};means={};source_means=[]
  for s in SNRS:
   legacy_ids=set(oldw['snrs'][str(s)]['means']);cids=sorted({cid for snr,cid,i in groups if snr==s});assert len(cids)==8
   for cid in cids:
    per=[]
    for i in range(1000):
     obs=groups[s,cid,i];assert len(obs)==3 and sorted(x['noise_seed']for x in obs)==list(SEEDS) and all(x['source_id']==sources[i]['source_id']for x in obs)
     v={k:math.fsum(x[k]for x in obs)/3 for k in METRICS};per.append(v);source_means.append(dict(snr_db=s,candidate_id=cid,source_index=i,source_id=sources[i]['source_id'],**v))
    v={k:math.fsum(x[k]for x in per)/1000 for k in METRICS};means[s,cid]=v
    if cid in legacy_ids:assert v==oldw['snrs'][str(s)]['means'][cid]
    summary.append(dict(snr_db=s,candidate_id=cid,origin='LEGACY_RETAINED'if cid in legacy_ids else'NEW_M10',source_count=1000,noise_count=3,**v))
   oldwhole=oldw['snrs'][str(s)]['original_final_whole_winner'];oldpartial=oldw['snrs'][str(s)]['winner'];policies[str(s)]={}
   for family in ('WHOLE','PARTIAL'):
    # Partial is a superset of whole endpoints, as in the original joint selection.
    eligible=[c for c in cids if (c in legacy_ids and (family=='PARTIAL'or c==oldwhole))or (c in newcs and(family=='PARTIAL'or newcs[c]['strategy_family']=='WHOLE'))]
    values={c:means[s,c]['dinov2_vitl14_cosine']for c in eligible};winner=plan.select_winner(values,legacy_ids);anchor=oldwhole if family=='WHOLE'else oldpartial
    policies[str(s)][family]=dict(winner=winner,eligible=eligible,DINO_means=values,old_winner=anchor,old_winner_retained=True,delta_vs_old=values[winner]-means[s,anchor]['dinov2_vitl14_cosine'],changed=winner!=anchor)
  result=dict(schema='T1B_M10_CAL1000_FROZEN_POLICIES_V1',status='T1B_CALIBRATION_COMPLETE_STOP_BEFORE_T1C',registration_sha256=expected,policies=policies,selection='three_noise_mean_within_source_then1000_source_mean_DINO_L',tie_break='legacy_before_new_on_exact_tie_then_candidate_id_lexical',all_legacy_candidates_retained=True,all_legacy_full_finalists_retained=True,source_ids=[s['source_id']for s in sources],T1C=False,LE3=False,training=False,automatic_followup=False)
  csvsave(out/'calibration_summary.csv',summary);csvsave(out/'source_means.csv',source_means);save(out/'frozen_policies.json',result)
  save(out/'completion.json',dict(status='COMPLETE_T1B_SELECTION',registration_sha256=expected,policies=pin(out/'frozen_policies.json'),summary=pin(out/'calibration_summary.csv'),source_means=pin(out/'source_means.csv'),new_scientific_calls=0))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
