"""Score the native timed outputs and seed control, without retiming."""
from study_common import *
from low_finish import wait
import five_metrics,traceback,csv

def prepare():
 closed('compiled_quality_restore_v2');root=L/'compiled32_metric_plan';root.mkdir(exist_ok=False)
 ss=read(L/'compiled_quality_plan/sources.json');sm={s['source_id']:s for s in ss};rows=read(L/'compiled_quality_restore_v2/outputs.json');pairs={}
 for row in rows:
  s=sm[row['source_id']];pk=digest([s['reference_sha256'],row['pixel_sha256']]);row['pair_key']=pk
  pairs.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=row['pixel_sha256'],image=row['image']))
 save(root/'sources.json',ss);save(root/'pairs.json',list(pairs.values()));save(root/'plan.json',dict(sources=ss,pairs=list(pairs.values())));save(root/'rows.json',rows)
 return root

def finish():
 score=five_metrics.gather('compiled32');rows=read(L/'compiled32_metric_plan/rows.json');old=read(L/'compiled_quality_plan/old_rows.json')
 for v in rows:v.update({m:float(score[v['pair_key']][m])for m in METRICS})
 rows=old+rows;ss=read(L/'compiled_quality_plan/sources.json');ids=[s['source_id']for s in ss]
 oldstat=R/'results/nestcom_timing_fairness_20261102_v1/paired_prefill_delivery/statistics_identity.json';stat=read(oldstat)
 assert ids==stat['source_order'];bp=Path(stat['bank_PFS_path']);assert sha(bp)==stat['bank_sha256'];bank=np.load(bp,allow_pickle=False);assert bank.shape==(10000,32)and bank.min()>=0 and bank.max()<32
 arms=['original_timed','compiled_timed','compiled_fixed_old_seed'];vectors={}
 for arm in arms:
  rr={v['source_id']:v for v in rows if v['arm']==arm};assert set(rr)==set(ids)
  for m in METRICS:vectors[arm,m]=np.array([float(rr[s][m])for s in ids]);assert np.isfinite(vectors[arm,m]).all()
 def ci(v):
  z=v[bank].mean(1);return dict(mean=float(v.mean()),ci_low=float(np.quantile(z,.025)),ci_high=float(np.quantile(z,.975)),sources=32,distinct_channel_events_per_source=1)
 summary=[dict(arm=a,metric=m,**ci(v))for(a,m),v in vectors.items()]
 paired=[dict(comparison=a+'-'+b,metric=m,**ci(vectors[a,m]-vectors[b,m]))for a,b in [('compiled_timed','original_timed'),('compiled_fixed_old_seed','original_timed'),('compiled_timed','compiled_fixed_old_seed')]for m in METRICS]
 out=D/'compiled32';out.mkdir(exist_ok=False);csvout(out/'frame_metrics.csv',rows);csvout(out/'source_means.csv',rows);csvout(out/'summary.csv',summary);csvout(out/'paired_deltas.csv',paired)
 save(out/'statistics_identity.json',dict(bank=pin(bp),source_order=ids,old_identity=pin(oldstat),replicates=10000,confidence=.95,direct_source_paired=True,timing_repetitions_are_not_independent_quality_events=True))
 counts={k:closed(k)for k in ['compiled_quality_restore_v2','compiled32_three','compiled32_clip','compiled32_dists']}
 save(out/'actual_compute_and_reuse.json',dict(stages=counts,original_metric_pairs_exact_reuse=32,logical_arm_outputs=96,new_wireless_events=0,new_TX=0,new_timing=0,earlier_import_failure_preserved=pin(D/'compiled_restore_import_revision.json')))
 tr=list(csv.DictReader((R/'results/nestcom_timing_fairness_20261102_v1/paired_prefill_delivery/summary.csv').open()))
 joint=[dict(arm='compiled_timed',**{m:next(x['mean']for x in summary if x['arm']=='compiled_timed'and x['metric']==m)for m in METRICS},**{x['direction']+'_ms':float(x['mean_ms'])for x in tr},source_count=32,timing_reused=True,scope='Same timed implementation/source32; supplemental diagnostic, not common500')]
 csvout(out/'quality_latency.csv',joint)
 lines=['# 快速 Ada 的质量与对应计时','', '同一32源、N1024/10 dB。实际计时输出全部从保存的接收信号恢复并逐哈希匹配；新旧已收到的 token 完全相同。五次计时重复是同一事件，不按五个独立质量观测统计。','', '编译实现的码流变化会改变由码流派生的尾部种子，因此同时报告实际计时输出和固定旧种子的控制。后者仅改变接收数值实现；不是额外无线事件。原顺序实现继续用于本轮主校准，32源结果不替换common500。','', '| 对比 | 指标 | 源均值差 | 95%来源配对区间 |','|---|---|---:|---|']
 lines += [f"| {v['comparison']} | {v['metric']} | {v['mean']:.7f} | [{v['ci_low']:.7f}, {v['ci_high']:.7f}] |"for v in paired]
 lines += ['', '对应计时仅与 compiled_timed 质量配对：'+', '.join(f"{v['direction']}={float(v['mean_ms']):.2f} ms"for v in tr)+'。这些为既有合格计时结果，本轮没有重跑计时。','', '差值方向：PSNR/CLIP-I越大越好；LPIPS/DreamSim/DISTS越小越好。完整有利、不利和跨零区间均保留。']
 (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save(L/'compiled32_finish/completion.json',dict(status='COMPLETE',sources=32,arms=3,mean_CIs=15,paired_CIs=15,report=pin(out/'REPORT_zh.md')))

def main():
 out=L/'compiled32_finish';out.mkdir(exist_ok=False);save(out/'owner.json',dict(owner=ident(os.getpid()),created=time.time()))
 try:
  wait('compiled_quality_restore_v2');root=prepare()
  for mode in ['three','clip','dists']:five_metrics.start('compiled32',mode,root);wait('compiled32_'+mode)
  finish()
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
