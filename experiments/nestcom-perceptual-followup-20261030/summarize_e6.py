"""Actual fixed-policy rejection curves, with gray outputs in overall quality."""
from followup_common import *
from summarize_e1 import interval
from collections import defaultdict,Counter
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']));out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  for role in ['E6_phy','E6_dreamsim','E6_dists']:closed(role)
  sets=cr(r['source_sets']);ids=sets['control100'];bank=np.load(checked(sets['bootstrap100']),allow_pickle=False);assert bank.shape==(10000,100)
  scores={k:cr(r[k])for k in ['dreamsim','dists']};frames=[];g=defaultdict(list);events=set()
  for v in cr(r['logical']):
   v=dict(v)
   if v['metric_origin']=='NEW_TWO_METRIC_GAP':
    for m in ['dreamsim','dists']:
     q=scores[m][v['pair_key']];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v[m]=q[m]
   assert np.isfinite(v['dreamsim'])and np.isfinite(v['dists']);k=(v['method'],v['snr_db'],v['source_id']);ev=k+(v['channel_repeat'],);assert ev not in events;events.add(ev);g[k].append(v);frames.append(v)
  assert len(frames)==30000 and len(g)==1500
  src=[];cells=defaultdict(dict)
  for(method,snr,sid),v in g.items():
   assert len(v)==20 and{a['channel_repeat']for a in v}==set(range(20));counts=Counter(a['outcome']for a in v);good=[a for a in v if a['outcome']=='SUCCESS'];row=dict(method=method,N=1024,design_snr_db=10,snr_db=snr,source_id=sid,source_index=v[0]['source_index'],events=20,successes=len(good),frame_rejection_rate=1-len(good)/20,header_reject=counts['HEADER_REJECT'],body_or_profile_parse_reject=counts['BODY_OR_PROFILE_PARSE_REJECT'],source_decode_reject=0,NULL=counts['NULL'])
   for m in ['dists','dreamsim']:row[m]=float(np.mean([a[m]for a in v]));row[m+'_successful_sum']=float(sum(a[m]for a in good))
   src.append(row);cells[method,snr][sid]=row
  summary=[]
  for(method,snr),v in sorted(cells.items()):
   assert set(v)==set(ids);a=[v[s]for s in ids];success=sum(x['successes']for x in a);row=dict(method=method,N=1024,design_snr_db=10,snr_db=snr,n_sources=100,logical_frames=2000,total_success=success,header_reject=sum(x['header_reject']for x in a),body_or_profile_parse_reject=sum(x['body_or_profile_parse_reject']for x in a),source_decode_reject=0,NULL=sum(x['NULL']for x in a),success_conditional_denominator=success)
   for m in ['frame_rejection_rate','dists','dreamsim']:
    mean,lo,hi=interval(np.array([x[m]for x in a]),bank);row.update({m:mean,m+'_ci95_low':lo,m+'_ci95_high':hi})
   for m in ['dists','dreamsim']:row[m+'_successful_only']=sum(x[m+'_successful_sum']for x in a)/success if success else None
   summary.append(row)
  dst=D/'e6';dst.mkdir(exist_ok=False)
  for name,value in [('frame_metrics',frames),('source_means',src),('summary',summary)]:csvout(dst/(name+'.csv'),value)
  save(dst/'statistics_identity.json',dict(source_order=ids,bootstrap=sets['bootstrap100'],source_first=True,replicates=10000,overall_quality_includes_all_gray=True,successful_only='Sum of successful-event metric / number of successful events; blank when zero successes; a conditional descriptive statistic, not a fixed population comparison',transmitter_design_snr=10,true_noise_variance_at_RX=True,original_generation_repeat0_fixed=True,threshold_localization='Only the five tested SNR values; do not infer an exact discontinuity'))
  save(dst/'actual_compute_and_reuse.json',dict(physical=read(L/'E6_phy/completion.json'),metrics={k:read(L/('E6_'+k)/'completion.json')for k in ['dreamsim','dists']},actual_frames=30000,extra_render_calls=0,successful_outputs='exact received projection plus seed reuse only',logical_metric_origins=dict(Counter(v['metric_origin']for v in frames))))
  lines=['# E6：固定 10 dB 策略的接收阈值诊断','','100 个源、三方法、五个实际 SNR、每源 20 次新信道噪声，共 30,000 个实际帧。TX 保持原 profile、payload、public counter/session 与扰码；只有 AWGN 改变。RX 使用真实噪声方差。全部成功输出按实际收到的前缀和原生成种子复用；全部拒收为灰图。','','|方法|实际 SNR|成功 / 2000|拒收率|整体 DISTS|整体 DreamSim|成功条件 DISTS|成功条件 DreamSim|','|---|---:|---:|---:|---:|---:|---:|---:|']
  for v in summary:
   cond=[('—'if v[k]is None else f'{v[k]:.6f}')for k in ['dists_successful_only','dreamsim_successful_only']]
   lines.append(f"|{v['method']}|{v['snr_db']}|{v['total_success']}|{v['frame_rejection_rate']:.4f}|{v['dists']:.6f}|{v['dreamsim']:.6f}|{cond[0]}|{cond[1]}|")
  lines+=['','成功条件质量只描述成功事件；分母随 SNR 改变。整体质量保留拒收，才能反映固定策略的实际风险。本实验没有重选策略、重传或训练，不支持未经测试的 SNR 鲁棒性主张。']
  (dst/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8');save(out/'completion.json',dict(status='COMPLETE',actual_frames=30000,cells=15,source_rows=1500,new_models=0,new_PHY=0,seconds=time.monotonic()-beg));print('E6_COMPLETE',len(frames))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
