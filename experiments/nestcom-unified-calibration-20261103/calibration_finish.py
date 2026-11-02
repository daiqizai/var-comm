"""Formal source-paired five-metric statistics, using the fixed common500 bank."""
from study_common import *
import calibration_scoring,five_metrics,collections

def prepare():
 root=calibration_scoring.prepare('formal');p=read(root/'plan.json');save(root/'sources.json',p['sources']);save(root/'pairs.json',p['pairs']);return root

def finish():
 ss=read(L/'cal_formal_plan/sources.json');ids=[s['source_id']for s in ss];sets=read(F/'source_sets.json');assert ids==sets['common500']
 bank=np.load(checked(sets['bootstrap500']),allow_pickle=False);assert bank.shape==(10000,500)and bank.min()>=0 and bank.max()<500
 scores=five_metrics.gather('cal_formal');logical=read(L/'cal_formal_plan/logical.json');physical=read(L/'cal_formal_metric_plan/physical.json');choices=read(L/'cal_formal_plan/choices.json');frames=[];grouped=collections.defaultdict(list)
 for v in logical:
  ch=choices[v['choice_id']];f=physical[v['physical_key']];q=scores[f['pair_key']]
  r=dict(v,**{k:ch[k]for k in ['m','K','q','rate','source_bits','capacity_bits','utilization','NULL','class_bits','actual_length_comparisons']},receiver_status=f['status'],received_key=f['received_key'],image=f['image'],pixel_sha256=f['pixel_sha256'],pair_key=f['pair_key'],**{m:float(q[m])for m in METRICS})
  assert all(np.isfinite(r[m])for m in METRICS);frames.append(r);grouped[r['method'],r['snr_db'],r['source_id']].append(r)
 assert len(frames)==18000 and len({(r['method'],r['snr_db'],r['source_id'],r['channel_repeat'])for r in frames})==18000
 source=[];vectors={};capacity=[]
 for method in ['WHOLE','NeST','Ada']:
  for snr in [4,7,10,19]:
   rr=[]
   for sid in ids:
    a=grouped[method,snr,sid];assert sorted(r['noise_seed']for r in a)==[6201,6202,6203]
    v=dict(source_id=sid,source_index=a[0]['source_index'],method=method,N=1024,snr_db=snr,repeats=3,failed_or_null=sum(r['receiver_status']!='ACCEPTED_SOURCE'for r in a),paid_NULL=sum(r['NULL']for r in a),**{m:float(np.mean([r[m]for r in a]))for m in (*METRICS,'utilization','source_bits','class_bits','actual_length_comparisons')});source.append(v);rr.append(v)
   for m in METRICS:vectors[method,snr,m]=np.array([v[m]for v in rr])
   capacity.append(dict(method=method,N=1024,snr_db=snr,sources=500,logical=1500,failed_or_null=sum(r['failed_or_null']for r in rr),paid_NULL=sum(r['paid_NULL']for r in rr),**{m:float(np.mean([v[m]for v in rr]))for m in ['utilization','source_bits','class_bits','actual_length_comparisons']}))
 def ci(v):
  a=v[bank].mean(1);return dict(mean=float(v.mean()),ci_low=float(np.quantile(a,.025)),ci_high=float(np.quantile(a,.975)),sources=500,repeats=3)
 summary=[dict(method=method,N=1024,snr_db=snr,metric=m,**ci(v))for(method,snr,m),v in vectors.items()]
 paired=[dict(comparison=a+'-'+b,N=1024,snr_db=snr,metric=m,**ci(vectors[a,snr,m]-vectors[b,snr,m]))for a,b in [('NeST','WHOLE'),('NeST','Ada'),('WHOLE','Ada')]for snr in [4,7,10,19]for m in METRICS]
 out=D/'formal';out.mkdir(exist_ok=False)
 for name,rows in [('frame_metrics',frames),('source_means',source),('summary',summary),('paired_deltas',paired),('capacity_summary',capacity)]:csvout(out/(name+'.csv'),rows)
 save(out/'statistics_identity.json',dict(bank=sets['bootstrap500'],source_order=ids,replicates=10000,confidence=.95,unit='source after averaging three actual channel outcomes',direct_paired=True,noise_seeds=[6201,6202,6203],failures_kept=True))
 stages={}
 for stage in ['pilot','full','formal']:
  roles=read(L/('cal_'+stage+'_phy_dispatch.json'))['roles']+read(L/('cal_'+stage+'_rx_plan/completion.json'))['roles']+(['cal_'+stage+'_dino']if stage!='formal'else['cal_formal_'+m for m in ['three','clip','dists']])
  stages[stage]={role:closed(role)for role in roles}
 save(out/'actual_compute_and_reuse.json',dict(stages=stages,source800=closed('cal_token_source800'),directory_qualification=closed('directory_qualification'),frozen_policy=pin(D/'frozen_policies.json')))
 lines=['# N1024统一校准与正式评测','', 'WHOLE、NeST(Token)、Ada使用相同12个MCS、固定pilot100一噪声和cal1000三噪声、DINOv2目标和失败灰图。NeST目标上限在25个Quarter位置预登记，目标以下允许全部逐token端点；这不是651个目标上限的穷举优化。WHOLE有7个完整尺度上限，Ada上限为256。','', '每方法/候选的12-bit目录在初筛前公开冻结，固定策略上下文不随源图变化，实际端点仍由收到的受保护profile恢复。正式common500在策略冻结后评测，每方法每SNR1500逻辑事件，失败/NULL保留分母。','', '| 对比 | SNR | 指标 | 均值差 | 95%来源配对区间 |','|---|---:|---|---:|---|']
 lines += [f"| {v['comparison']} | {v['snr_db']} | {v['metric']} | {v['mean']:.7f} | [{v['ci_low']:.7f}, {v['ci_high']:.7f}] |"for v in paired]
 (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save(L/'calibration_chain/completion.json',dict(status='COMPLETE_SCIENCE',formal_logical=18000,quality_mean_CIs=60,quality_paired_CIs=60,report=pin(out/'REPORT_zh.md'),manuscript_and_publication='PENDING'))
