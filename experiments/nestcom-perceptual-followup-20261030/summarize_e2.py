"""Error-free information-budget experiment, source-first and directly paired."""
from followup_common import *
from summarize_e1 import interval
from collections import defaultdict,Counter
from itertools import combinations
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  for role in ['E2_nest',*[f'E2_ada_{i:02d}'for i in range(4)],'E2_three','E2_clip','E2_dists']:closed(role)
  sets=cr(r['source_sets']);ids=sets['common500'];bank=np.load(checked(sets['bootstrap500']),allow_pickle=False);assert bank.shape==(10000,500)
  fresh={k:cr(r[k])for k in ['three','clip','dists']};frames=[];groups=defaultdict(list);events=set()
  for v in cr(r['logical']):
   v=dict(v);pk=v['pair_key']
   if v['metric_origin']=='NEW_FIVE_METRIC_GAP':
    for mode,ms in [('three',METRICS[:3]),('clip',METRICS[3:4]),('dists',METRICS[4:])]:
     q=fresh[mode][pk];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:q[m]for m in ms})
   assert all(np.isfinite(v[m])for m in METRICS)
   k=(v['method'],v['B_bits'],v['source_id']);ev=k+(v['generation_repeat'],);assert ev not in events;events.add(ev);groups[k].append(v);frames.append(v)
  assert len(frames)==15000 and len(groups)==9000
  source=[];cells=defaultdict(dict);extra=('source_bits','total_overhead','padding_bits','length_checks')
  for k,g in groups.items():
   n=3 if k[0]=='Ada'else 1;assert len(g)==n and{v['generation_repeat']for v in g}==set(range(n))
   v=dict(method=k[0],B_bits=k[1],source_id=k[2],source_index=g[0]['source_index'],generation_repeats=n,**{m:float(np.mean([x[m]for x in g]))for m in(*METRICS,*extra)});source.append(v);cells[k[:2]][k[2]]=v
  arrays={};summary=[];paired=[];efficiency=[]
  for (method,budget),g in sorted(cells.items()):
   assert set(g)==set(ids)
   for m in METRICS:
    a=np.array([g[s][m]for s in ids]);arrays[method,budget,m]=a;mean,lo,hi=interval(a,bank);summary.append(dict(method=method,B_bits=budget,metric=m,n_sources=500,logical=1500 if method=='Ada'else 500,mean=mean,ci95_low=lo,ci95_high=hi))
   efficiency.append(dict(method=method,B_bits=budget,n_sources=500,**{m:float(np.mean([g[s][m]for s in ids]))for m in extra}))
  for budget in [620,1212,1302,1940,2248,2577]:
   for base,candidate in [('WHOLE','Token_NeST'),('Ada','Token_NeST'),('Ada','WHOLE')]:
    for m in METRICS:
     mean,lo,hi=interval(arrays[candidate,budget,m]-arrays[base,budget,m],bank);paired.append(dict(candidate=candidate,reference=base,B_bits=budget,metric=m,n_sources=500,mean_delta=mean,ci95_low=lo,ci95_high=hi))
  dst=D/'e2';dst.mkdir(exist_ok=True)
  for name,data in [('frame_metrics',frames),('source_means',source),('summary',summary),('paired_deltas',paired),('efficiency',efficiency)]:csvout(dst/(name+'.csv'),data)
  save(dst/'statistics_identity.json',dict(source_order=ids,bootstrap=sets['bootstrap500'],source_set_registration=r['source_sets'],source_first=True,direct_source_paired=True,replicates=10000,pointwise95_percentile=True,generation_repeats_are_averaged_metrics_not_images=True,no_quality_selection=True,channel='error-free, no new wireless frames',claim_scope='Frozen complete systems, not isolated representation-order causality'))
  save(dst/'actual_compute_and_reuse.json',dict(receivers={x:read(L/x/'completion.json')for x in ['E2_nest',*[f'E2_ada_{i:02d}'for i in range(4)]]},metrics={k:read(L/('E2_'+k)/'completion.json')for k in ['three','clip','dists']},logical=15000,source_method_budget_units=9000,new_PHY=0,logical_reuse_counts=dict(Counter(v['metric_origin']for v in frames))))
  lines=['# E2：相同信息位预算下的冻结系统比较','','每个预算使用同一 common500。完整粗尺度/逐 token NeST 最大端点均为 (10,0)，Ada 最多 256 token。公共开销 57 bit，Ada 另付 10 bit 类别。独立终止码流按实际码长装入，真实传递并独立解码；没有新无线帧。Ada 三个预登记生成重复先按源平均指标。','','|预算 bit|对照|指标|均值差|95% 来源配对区间|','|---:|---|---|---:|---|']
  for v in paired:lines.append(f"|{v['B_bits']}|{v['candidate']}−{v['reference']}|{v['metric']}|{v['mean_delta']:.8f}|[{v['ci95_low']:.8f}, {v['ci95_high']:.8f}]|")
  lines+=['','PSNR/CLIP-I 正差有利，三个距离负差有利。所有预算及不利结果保留；本实验不单独排除 tokenizer、prior、decoder 和类别条件的差异。']
  (dst/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8');save(out/'completion.json',dict(status='COMPLETE',logical=15000,mean_CIs=90,paired_CIs=90,new_models=0,new_PHY=0,seconds=time.monotonic()-beg));print('E2_COMPLETE',len(frames),len(paired))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
