"""Five-metric ordered receiver interventions; direct source-paired intervals."""
from followup_common import *
from summarize_e1 import interval
from collections import defaultdict,Counter
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 os.sched_setaffinity(0,set(r['cpu_affinity']));out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  for role in ['E4_interventions','E4_three','E4_clip','E4_dists']:closed(role)
  sets=cr(r['source_sets']);ids=sets['control100'];bank=np.load(checked(sets['bootstrap100']),allow_pickle=False)
  assert len(ids)==100 and bank.shape==(10000,100)and ids==sets['common500'][:100]
  old={}
  for v in cr(r['e1_frames']):
   if v['method']=='final_NeST':old.setdefault(v['dists_pair_key'],v)
  fresh={k:cr(r[k])for k in ['three','clip','dists']};frames=[];groups=defaultdict(list);events=set()
  for v in cr(r['logical']):
   v=dict(v);pk=v['pair_key'];s=v['source_id'];assert s in ids
   if v['reuse_reason']=='NEW_INTERVENTION':
    for mode,ms in [('three',METRICS[:3]),('clip',METRICS[3:4]),('dists',METRICS[4:])]:
     q=fresh[mode][pk];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:q[m]for m in ms})
   else:
    q=old[pk];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:q[m]for m in METRICS})
   assert all(np.isfinite(v[m])for m in METRICS)
   k=(v['arm'],v['snr_db'],s);ev=k+(v['channel_repeat'],);assert ev not in events;events.add(ev);groups[k].append(v);frames.append(v)
  assert len(frames)==1800 and len(groups)==600
  source=[];cells=defaultdict(dict)
  for k,g in groups.items():
   assert len(g)==3 and{v['channel_repeat']for v in g}=={0,1,2}
   v=dict(arm=k[0],N=1024,snr_db=k[1],source_id=k[2],source_index=g[0]['source_index'],channel_events=3,**{m:float(np.mean([x[m]for x in g]))for m in METRICS});source.append(v);cells[k[:2]][k[2]]=v
  arrays={};summary=[];paired=[]
  for k,g in sorted(cells.items()):
   assert set(g)==set(ids)
   for m in METRICS:
    a=np.array([g[s][m]for s in ids]);arrays[*k,m]=a;mean,lo,hi=interval(a,bank)
    summary.append(dict(arm=k[0],N=1024,snr_db=k[1],metric=m,n_sources=100,logical=300,mean=mean,ci95_low=lo,ci95_high=hi))
  for snr in [7,10]:
   for c,b in [('B0','A0'),('C0','B0'),('C0','A0')]:
    for m in METRICS:
     mean,lo,hi=interval(arrays[c,snr,m]-arrays[b,snr,m],bank);paired.append(dict(candidate=c,reference=b,N=1024,snr_db=snr,metric=m,n_sources=100,mean_delta=mean,ci95_low=lo,ci95_high=hi))
  assert len(summary)==30 and len(paired)==30
  dst=D/'e4';dst.mkdir(exist_ok=False)
  for name,data in [('frame_metrics',frames),('source_means',source),('summary',summary),('paired_deltas',paired)]:csvout(dst/(name+'.csv'),data)
  save(dst/'statistics_identity.json',dict(source_order=ids,bootstrap=sets['bootstrap100'],source_set_registration=r['source_sets'],replicates=10000,pointwise95_percentile=True,direct_source_paired=True,source_first=True,all_logical_denominators_retained=True,ordered_interventions_not_independent_contribution_fractions=True,final_scale_audit=cr(r['final_scale_audit'])))
  save(dst/'actual_compute_and_reuse.json',dict(interventions=read(L/'E4_interventions/completion.json'),metrics={k:read(L/('E4_'+k)/'completion.json')for k in ['three','clip','dists']},logical=1800,new_PHY=0,logical_reuse_counts=dict(Counter(v['reuse_reason']for v in frames))))
  lines=['# E4：相同实收前缀的有序干预','', '固定 common500 manifest 前 100 源、N=1024、7/10 dB、原三个信道重复。A0 忽略边界；B0 仅加残差；C0 同时更新后续条件。五指标全部保留。区间为先源内平均后的直接来源配对 95% percentile bootstrap；不是三项可独立相加的贡献比例。','', '|SNR|对照|指标|均值差|95% CI|','|---|---|---|---:|---|']
  for v in paired:lines.append(f"|{v['snr_db']}|{v['candidate']}−{v['reference']}|{v['metric']}|{v['mean_delta']:.8f}|[{v['ci95_low']:.8f}, {v['ci95_high']:.8f}]|")
  lines+=['','PSNR/CLIP-I 正差有利；LPIPS/DreamSim/DISTS 负差有利。跨零区间不宣称方向确定。没有新增无线帧、重新选端点或训练。C0 的资格回放逐像素一致；19 dB 最终尺度的结构检查独立记录于 statistics_identity.json。']
  (dst/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
  save(out/'completion.json',dict(status='COMPLETE',logical=1800,mean_CIs=30,paired_CIs=30,new_models=0,new_PHY=0,seconds=time.monotonic()-beg));print('E4_COMPLETE',len(frames),len(paired))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
