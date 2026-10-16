"""One full-denominator paired statistical delivery, with no model/channel calls."""
from tc_common import *
from tc_protocol import closed_wait,source_means,interval,POPS
import numpy as np,collections,traceback,shutil
FOUR=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
def run(expected):
 r=registration('link',expected);out=T/'delivery';out.mkdir();light=O/'delivery';light.mkdir();ledger=Ledger(out/'calls.sqlite',{'bootstrap_draw_bank':2});start=time.monotonic()
 try:
  controls=[];done={}
  for stage in ('source','phy','receiver','metrics'):
   c=T/(stage+'_control');w=read(c/'actual_wait.json');closed_wait(w,gone);assert not(c/'failure.json').exists() and not(T/stage/'failure.json').exists()
   d=read(T/stage/'completion.json');assert d['registration_sha256']==(sha(O/'source_registration.json') if stage=='source' else expected)
   controls.extend([pin(c/'launch.json'),pin(c/'actual_wait.json'),pin(c/'stdout.txt'),pin(c/'stderr.txt'),pin(T/stage/'completion.json')]);done[stage]=d
  rows=checkread(done['metrics']['rows']);assert len(rows)==3600
  assert done['phy']['new_physical']==1125 and done['phy']['new_packets']['unresolved']==0
  for d in (done['source'],done['receiver'],done['metrics']):assert d['ledger']['unresolved']==0
  # Closed per-file audit. No extra computation of pixels, models, PHY or scores.
  verified={};statuses=collections.Counter();flat=[];prefix=collections.Counter()
  for v in rows:
   for d in (v['physical_frame'],v['reconstruction']['image_archive']):
    if d['path'] not in verified:checked(d);verified[d['path']]=d
   packet=checkread(v['physical_frame']);rx=packet['actual_RX'];rp=rx['rx_profile'];status=rx['status'];statuses[(v['population'],v['strategy_family'],status)]+=1
   prefix[(v['population'],v['strategy_family'],v['actual_m'],v['actual_K'])]+=1
   flat.append(dict(population=v['population'],source_index=v['source_index'],source_id=v['source_id'],snr_db=19,noise_seed=v['noise_seed'],family=v['strategy_family'],candidate_id=v['candidate_id'],sent_m=v['actual_m'],sent_K=v['actual_K'],arithmetic_bits=v['arithmetic_bits'],source_capacity_bits=4751,length_field_bits=13,CRC_bits=16,utilization=v['arithmetic_bits']/4751,rx_m=None if rp is None else rp['m'],rx_K=None if rp is None else rp['K'],RX_status=status,physical_reused=v['physical_reuse'],score_origin=v['score_origin'],image_sha256=v['reconstruction']['image_sha256'],**{k:v['metrics'].get(k,'') for k in FOUR+('dists','dreamsim','ms_ssim')},**{'old_'+k:v['legacy_metrics'].get(k,'') for k in FOUR+('dists','dreamsim','ms_ssim')}))
  csvsave(light/'frame_scores_19dB.csv',flat)
  shutil.copyfile(checked(done['source']['lengths']),light/'m10_source_lengths_test600.csv')
  summaries=[];paired=[];means=[];banks=[]
  for pop,(n,seeds) in POPS.items():
   idx=ledger.call('bootstrap_draw_bank',pop,lambda:np.random.default_rng(2026100701).integers(0,n,size=(10000,n),dtype=np.int64))
   bp=out/(pop+'_bootstrap_indices.npy')
   with bp.open('xb') as f:np.save(f,idx)
   banks.append(dict(population=pop,seed=2026100701,replicates=10000,indices=pin(bp),array_sha256=asha(idx)))
   for metric in FOUR+(('dists','dreamsim','ms_ssim') if pop=='common500' else ()):
    a={fam:source_means(rows,pop,fam,metric) for fam in ('WHOLE','PARTIAL')};old={fam:source_means(rows,pop,fam,metric,'legacy_metrics') for fam in ('WHOLE','PARTIAL')}
    for fam in ('WHOLE','PARTIAL'):
     summaries.append(dict(population=pop,sources=n,noise_per_source=3,snr_db=19,family=fam,metric=metric,direction='lower' if metric in ('lpips_alex','dists','dreamsim') else 'higher',**interval(a[fam],idx),old_mean=float(old[fam].mean())))
     means.extend(dict(population=pop,source_index=i,family=fam,metric=metric,new_mean=float(a[fam][i]),old_mean=float(old[fam][i]),delta=float(a[fam][i]-old[fam][i])) for i in range(n))
    for name,diff in [('PARTIAL_minus_WHOLE',a['PARTIAL']-a['WHOLE']),('revised_WHOLE_minus_old_WHOLE',a['WHOLE']-old['WHOLE']),('revised_PARTIAL_minus_old_PARTIAL',a['PARTIAL']-old['PARTIAL'])]:paired.append(dict(population=pop,sources=n,noise_per_source=3,snr_db=19,comparison=name,metric=metric,**interval(diff,idx)))
  csvsave(light/'summary_19dB.csv',summaries);csvsave(light/'paired_19dB.csv',paired);csvsave(light/'source_means_19dB.csv',means)
  csvsave(light/'actual_fallback_counts.csv',[dict(population=k[0],family=k[1],m=k[2],K=k[3],logical_frames=v,sources=v//3) for k,v in sorted(prefix.items())]);csvsave(light/'RX_status_counts.csv',[dict(population=k[0],family=k[1],status=k[2],logical_frames=v) for k,v in sorted(statuses.items())])
  save(light/'closure_audit.json',dict(status='ACTUAL_COMPLETE_T1C_ONLY',registration_sha256=expected,source_registration=pin(O/'source_registration.json'),controls=controls,scientific_stages=done,exact_grid=r['exact_grid'],verified_files=list(verified.values()),bootstrap=banks,statistical_ledger=ledger.summary(),legacy_common500_whole_streams_changed_on_current_frozen_runtime=True,old_scientific_results_unchanged=True,automatic_successor=False,backup_deferred_by_user=True))
  lines=['# T1-C：m10 修正版 19 dB 测试集结果','', '仅原 common500 与独立 new100、两个冻结熵码家族、每源原三次噪声。两个集合分别统计。固定 N=1024（68 头＋956 正文）、Es=2、q=6、k/n=4780/5736；13 bit 长度＋16 bit CRC 后实码容量4751 bit。失败输出保留分母。', '', '600次 m10 TX 已全部完成；原24终点全部逐字节一致。common500仅6/500、new100仅1/100实际装下完整 m10。其余严格按校准后冻结的整尺度/部分尺度顺序回退，未再选择策略或检查点。', '', '3600逻辑帧中2454帧复用（其中378帧来自另一家族的完全相同既有事件）；1146个改变的逻辑帧合并为1125个不同物理发送，上限2250包。相同实际发送与实际RX输入仅执行一次。各阶段实际调用数见closure_audit.json。', '', '**必须披露的版本差异：**旧common500整尺度来自较早4090环境，其码流与当前冻结H800实现不同。因此500源的整尺度旧版本不能直接继承；其中126源可从另一家族的完全相同既有事件继承，其余374源需要更新发送；与旧整尺度的差值混合了实现环境/码流版本和m10开放的影响，不能全部归因于m10。旧版本及旧分数仍保留。', '', '## 全分母均值与95%来源级区间','', '|集合|家族|指标|修正版均值|95%区间|旧均值|','|---|---|---|---:|---|---:|']
  for s in summaries:lines.append(f'|{s["population"]}|{s["family"]}|{s["metric"]}|{s["mean"]:.9g}|[{s["ci_low"]:.9g}, {s["ci_high"]:.9g}]|{s["old_mean"]:.9g}|')
  lines+=['','## 直接配对差值（PARTIAL减WHOLE）','','|集合|指标|均值差|95%直接配对区间|','|---|---|---:|---|']
  for s in paired:
   if s['comparison']=='PARTIAL_minus_WHOLE':lines.append(f'|{s["population"]}|{s["metric"]}|{s["mean"]:.9g}|[{s["ci_low"]:.9g}, {s["ci_high"]:.9g}]|')
  lines+=['','先对每源三次噪声平均，再对来源直接配对重采样10000次，seed2026100701。没有相减边际区间；没有混合500和100两集合。common500延续原四指标及三项补充指标；new100延续原四指标，不新增其三指标任务。全部指标不论方向均报告。','','4/10dB、P100k、Swin200k、raw、BPG、HiFi、LE3和已完成六方法计时均未重跑。旧时间对应旧实现版本，本次不作新端点速度或跨设备一致性声明。未训练、未再选策；备份按用户要求延后。交付后停止。','']
  (light/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf8')
  (light/'RESULTS_en.md').write_text('The corrected whole-scale and partial-scale entropy-coded strategies were evaluated at 19 dB on the original common500 and independent new100 sets, separately, with three fixed noise realizations per source. Full m10 fit the 4751-bit arithmetic capacity for 6/500 and 1/100 sources, respectively; all others followed the frozen fallback chains. The 3600 logical frames required 1125 distinct new physical transmissions, with 2454 logical frames reused (including 378 exact cross-family events) after exact evidence checks. The older common500 whole-scale bitstreams differ from the current frozen H800 implementation; its old-to-new changes therefore cannot be attributed solely to adding m10. All failures remain in the denominators. Tables report source-mean metrics and directly paired 95% bootstrap intervals (10000 draws; seed 2026100701), with no policy reselection or additional training. Existing 4/10 dB results and other methods are unchanged.\n',encoding='utf8')
  files=[pin(p) for p in sorted(light.iterdir())];save(light/'manifest.json',dict(files=files,registration_sha256=expected))
  completion=dict(status='COMPLETE_T1C_19DB_DELIVER_AND_STOP',registration_sha256=expected,logical_frames=3600,manifest=pin(light/'manifest.json'),statistical_ledger=ledger.summary(),seconds=time.monotonic()-start,new_models=0,new_channel=0,new_selection=0,backup_deferred=True)
  save(light/'completion.json',completion);save(out/'completion.json',completion)
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,ledger=ledger.summary()));raise
if __name__=='__main__':run(sys.argv[1])
