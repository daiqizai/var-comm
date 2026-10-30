"""Controlled common-ceiling endpoint granularity, all metrics and contrasts."""
from followup_common import *
from summarize_e1 import interval
from collections import defaultdict,Counter
from itertools import combinations
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  for role in ['E3_rx','E3_three','E3_clip','E3_dists']:closed(role)
  sets=cr(r['source_sets']);ids=sets['control100'];bank=np.load(checked(sets['bootstrap100']),allow_pickle=False);assert bank.shape==(10000,100)
  fresh={k:cr(r[k])for k in ['three','clip','dists']};frames=[];groups=defaultdict(list);events=set()
  for v in cr(r['logical']):
   v=dict(v);pk=v['pair_key']
   if v['metric_origin']=='NEW_FIVE_METRIC_GAP':
    for mode,ms in [('three',METRICS[:3]),('clip',METRICS[3:4]),('dists',METRICS[4:])]:
     q=fresh[mode][pk];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:q[m]for m in ms})
   assert all(np.isfinite(v[m])for m in METRICS)
   k=(v['method'],v['source_id']);ev=k+(v['channel_repeat'],);assert ev not in events;events.add(ev);groups[k].append(v);frames.append(v)
  assert len(frames)==1200 and len(groups)==400
  source=[];cells=defaultdict(dict);extra=('source_bits','capacity_bits','utilization','actual_length_comparisons','candidate_count')
  for k,g in groups.items():
   assert len(g)==3 and{v['channel_repeat']for v in g}=={0,1,2}
   v=dict(method=k[0],N=1024,snr_db=10,source_id=k[1],source_index=g[0]['source_index'],channel_events=3,**{m:float(np.mean([x[m]for x in g]))for m in(*METRICS,*extra)});source.append(v);cells[k[0]][k[1]]=v
  arrays={};summary=[];paired=[];efficiency=[]
  for method,g in sorted(cells.items()):
   assert set(g)==set(ids)
   for m in METRICS:
    a=np.array([g[s][m]for s in ids]);arrays[method,m]=a;mean,lo,hi=interval(a,bank);summary.append(dict(method=method,N=1024,snr_db=10,metric=m,n_sources=100,logical=300,mean=mean,ci95_low=lo,ci95_high=hi))
   efficiency.append(dict(method=method,N=1024,snr_db=10,n_sources=100,target='(8,126)',q=4,rate='2/3',**{m:float(np.mean([g[s][m]for s in ids]))for m in extra}))
  for base,candidate in combinations(['whole','quarter','eighth','token'],2):
   for m in METRICS:
    mean,lo,hi=interval(arrays[candidate,m]-arrays[base,m],bank);paired.append(dict(candidate=candidate,reference=base,N=1024,snr_db=10,metric=m,n_sources=100,mean_delta=mean,ci95_low=lo,ci95_high=hi))
  dst=D/'e3';dst.mkdir(exist_ok=False)
  for name,data in [('frame_metrics',frames),('source_means',source),('summary',summary),('paired_deltas',paired),('efficiency',efficiency)]:csvout(dst/(name+'.csv'),data)
  save(dst/'statistics_identity.json',dict(source_order=ids,bootstrap=sets['bootstrap100'],source_set_registration=r['source_sets'],source_first=True,direct_source_paired=True,replicates=10000,pointwise95_percentile=True,common_model_MCS_ceiling_fitting_noise=True,new_quality_selector=False,counts_are_length_comparisons_not_model_forwards_or_latency=True))
  save(dst/'actual_compute_and_reuse.json',dict(physical=read(L/'E3_phy/completion.json'),receiver=read(L/'E3_rx/completion.json'),metrics={k:read(L/('E3_'+k)/'completion.json')for k in ['three','clip','dists']},logical=1200,logical_reuse_counts=dict(Counter(v['metric_origin']for v in frames))))
  lines=['# E3：固定端点上限的粒度控制','','100 个固定源、N=1024、10 dB、16-QAM/rate 2/3、共同最大端点 (8,126)、共同逐图噪声及装入规则。保留所有 1,200 个逻辑结果。这里只改变候选粒度；不复用条件不同的旧 Table I 数字。','', '|候选−基线|指标|均值差|95% 来源配对区间|','|---|---|---:|---|']
  for v in paired:lines.append(f"|{v['candidate']}−{v['reference']}|{v['metric']}|{v['mean_delta']:.8f}|[{v['ci95_low']:.8f}, {v['ci95_high']:.8f}]|")
  lines+=['','PSNR/CLIP-I 正差有利，三个距离负差有利。码长检查次数比较已独立终止的码长，不等于额外先验前向或实测处理时间；计时另见 E5。']
  (dst/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8');save(out/'completion.json',dict(status='COMPLETE',logical=1200,mean_CIs=20,paired_CIs=30,new_models=0,new_PHY=0,seconds=time.monotonic()-beg));print('E3_COMPLETE',len(frames),len(paired))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
