"""One common DINO route; source-level selection, before common500 access."""
from study_common import *
from launch_extra_v2 import launch_extra_v2
import collections

def prepare(stage):
 out=L/('cal_'+stage+'_metric_plan');out.mkdir(exist_ok=False);p=read(L/('cal_'+stage+'_rx_plan/completion.json'));rx=cr(p['reuse'])
 for role in p['roles']:
  closed(role)
  for v in read(L/role/'outputs.json'):assert v['key']not in rx;rx[v['key']]=v
 sources=read(L/('cal_'+stage+'_plan/sources.json'));sm={s['source_id']:s for s in sources};jobs={j['key']:j for j in read(L/('cal_'+stage+'_plan/physical.json'))};pairs={};physical={}
 for key,f in cr(p['frames']).items():
  v=rx[f['received_key']];s=sm[jobs[key]['source_id']];pk=digest([s['reference_sha256'],v['pixel_sha256']])
  pairs.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],image=v['image'],pixel_sha256=v['pixel_sha256']))
  physical[key]=dict(pair_key=pk,received_key=f['received_key'],status=v['status'],image=v['image'],pixel_sha256=v['pixel_sha256'])
 save(out/'plan.json',dict(sources=sources,pairs=list(pairs.values())));save(out/'physical.json',physical);save(out/'completion.json',dict(status='COMPLETE_PLAN',sources=len(sources),physical=len(physical),unique_pairs=len(pairs),plan=pin(out/'plan.json')))
 return out

def start_dino(stage):
 assert stage in ['pilot','full'];out=prepare(stage);old=read(R/'results/adatoken_response_20261026_v1/c_predclass_cal1000_dino_pro_v1_registration.json')
 sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation
 code=relocation.relocate_value(old['code'])+[pin(R/'experiments/nestcom-n512-20261028/n512_common.py'),pin(R/'experiments/nestcom-n512-20261028/dino_worker.py'),pin(R/'tools/checkpoint_relocation_20261028.py'),pin(relocation.RECEIPT),pin(E/'calibration_scoring.py')]
 r=dict(plan=pin(out/'plan.json'),code=code,metric_config=old['metric_config'],cpu_affinity=[72,74],prior_score_dirs=[]if stage=='pilot'else[str(L/'cal_pilot_dino')])
 launch_extra_v2('cal_'+stage+'_dino',r,'dino_relocated.py',VPY,7200,gpu=0)

def select(stage):
 closed('cal_'+stage+'_dino');p=read(D/'protocol.json');logical=read(L/('cal_'+stage+'_plan/logical.json'));physical=read(L/('cal_'+stage+'_metric_plan/physical.json'));scores=read(L/('cal_'+stage+'_dino/scores.json'))
 sources=read(L/('cal_'+stage+'_plan/sources.json'));ids=[s['source_id']for s in sources];seeds=[4101]if stage=='pilot'else[4101,4102,4103]
 grouped=collections.defaultdict(lambda:collections.defaultdict(dict));failure=collections.Counter();nulls=collections.Counter();choices=read(L/('cal_'+stage+'_plan/choices.json'))
 for row in logical:
  key=(row['method'],row['snr_db'],row['candidate_id']);sid=row['source_id'];seed=row['noise_seed'];assert seed not in grouped[key][sid]
  f=physical[row['physical_key']];value=float(scores[f['pair_key']]['value']);assert np.isfinite(value);grouped[key][sid][seed]=value
  failure[key]+=int(f['status']!='ACCEPTED_SOURCE');nulls[key]+=int(choices[row['choice_id']]['NULL'])
 rows=[]
 for(method,snr,candidate),src in sorted(grouped.items()):
  assert set(src)==set(ids)and all(set(v)==set(seeds)for v in src.values());means=[np.mean(list(src[s].values()))for s in ids]
  rows.append(dict(method=method,snr_db=snr,candidate_id=candidate,source_count=len(ids),logical_count=len(ids)*len(seeds),mean_dino_cosine=float(np.mean(means)),failed_or_null=failure[method,snr,candidate],paid_NULL=nulls[method,snr,candidate]))
 root=D/(stage+'_calibration');root.mkdir(exist_ok=False);csvout(root/'candidate_scores.csv',rows);candidates=read(D/'public_directories.json')['candidates'];lookup={c['candidate_id']:c for c in candidates};selected={};detail=[]
 for snr in p['unified']['SNR']:
  selected[str(snr)]=[]if stage=='pilot'else{}
  for method in ['WHOLE','NeST','Ada']:
   pool=sorted((v for v in rows if v['method']==method and v['snr_db']==snr),key=lambda v:(-v['mean_dino_cosine'],v['candidate_id']));assert pool
   if stage=='pilot':
    top=[v['candidate_id']for v in pool[:3]];target=[256,0]if method=='Ada'else[10,0];anchor=next(c['candidate_id']for c in candidates if c['method']==method and c['q']==2 and c['rate']=='1/2'and c['target']==target)
    names=list(dict.fromkeys(top+[anchor]));selected[str(snr)].extend(lookup[n]for n in names);detail.append(dict(method=method,snr_db=snr,top3=top,anchor=anchor,advanced=names))
   else:selected[str(snr)][method]=lookup[pool[0]['candidate_id']];detail.append(pool[0])
 result=dict(status='COMPLETE_PILOT_FREEZE'if stage=='pilot'else'COMPLETE_POLICY_FREEZE',protocol=pin(D/'protocol.json'),summary=pin(root/'candidate_scores.csv'),selection=detail,source_count=len(ids),noise_seeds=seeds,metric='DINOv2-L cosine',source_equal_weighting=True,common500_used=False,actual_wait=pin(L/('cal_'+stage+'_dino_control/actual_wait.json')))
 result['candidates_by_snr'if stage=='pilot'else'policies']=selected;save(D/('pilot_finalists.json'if stage=='pilot'else'frozen_policies.json'),result)
 print('SELECTED',stage,json.dumps(detail),flush=True)
