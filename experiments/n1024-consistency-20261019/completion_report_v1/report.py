"""Closed-delivery reporting only. No model, PHY, scoring or resampling entry."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,os,time,traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
L=B.parent/'n1024_consistency_20261019_v1'
O=R/'results/n1024_consistency_20261019_v1'
REG='4789b2595190262a5970f985d25f420663d8225321830a770c4400fc82487670'
INPUT_MANIFEST_SHA='544bcfb90e9c85bfef6f24e72fc358219c126656f4bddae5ffa5005dfeb8d93b'
M=[('psnr_db','PSNR (dB)',1,'higher'),('lpips_alex','LPIPS',1,'lower'),('dreamsim','DreamSim',1,'lower'),('convnext_top1_source_prediction','Agreement (%)',100,'higher'),('dinov2_vitl14_cosine','DINO similarity',1,'higher'),('dists','DISTS',1,'lower'),('ms_ssim','MS-SSIM',1,'higher')]
FAMS=('WHOLE','PARTIAL');SNRS=(4,10,19);NOISE=(6201,6202,6203)
STYLE=dict(families={'WHOLE':dict(label='Complete scales + VAR',color='#767676',marker='s',linestyle='--',markerfacecolor='white',linewidth=1.5),'PARTIAL':dict(label='NeST-Com (entropy-coded)',color='#3775BA',marker='o',linestyle='-',markerfacecolor='#3775BA',linewidth=1.7)},width_mm=180,height_mm=55,label_pt=8.5,tick_pt=7.5,legend_pt=8,marker_pt=4.5,png_dpi=600,gray_preview_dpi=300)
INPUTS={}
def pin(p):
 p=Path(p);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 return dict(path=str(p),bytes=a.st_size,sha256=h.hexdigest())
def checked(d):
 p=Path(d['path']);assert p.is_absolute() and '..' not in p.parts
 for q in (p,*p.parents):assert not q.is_symlink()
 v=pin(p);assert v['sha256']==d['sha256'] and v['bytes']==d.get('bytes',v['bytes']),str(p)
 INPUTS[str(p)]=v;return p
def read(p):return json.loads(Path(p).read_bytes())
def cr(d):return read(checked(d))
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvread(p):
 checked(pin(p))
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvsave(p,rows):
 assert rows
 with Path(p).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(v):
 if v in ('True','False'):return float(v=='True')
 x=float(v);assert math.isfinite(x);return x
def gone(v):
 p=Path('/proc')/str(v['pid'])/'stat'
 if not p.exists():return True
 a=p.read_text().rsplit(')',1)[1].split();return int(a[19])!=v['start_ticks'] or a[0]=='Z'
def wait_ok(v,outer=False):
 assert type(v['returncode']) is int and v['returncode']==0
 assert all(v[k] is False for k in ('terminated','killed','timed_out')) and v['metadata_error'] is None
 assert gone(v['process'])
 if not outer:assert v['actual_wait'] is True and gone(v['parent_process'])
def closure():
 reg=O/'repair_v1/registration.json';assert pin(reg)['sha256']==REG;INPUTS[str(reg)]=pin(reg)
 r=read(reg);assert r['caps']['physical']==2421 and r['full_grid']['logical']==9000
 for d in r['code']:checked(d)
 for role in ('phy','receiver','metrics','delivery'):
  c=L/'repair_v1'/(role+'_control');w=read(c/'actual_wait.json');wait_ok(w)
  assert not(c/'failure.json').exists() and not(L/'repair_v1'/role/'failure.json').exists()
  for name in ('prepared.json','launch.json','actual_wait.json','stdout.txt','stderr.txt'):checked(pin(c/name))
  d=read(L/'repair_v1'/role/'completion.json');assert d['registration_sha256']==REG
  checked(pin(L/'repair_v1'/role/'completion.json'))
  outer=L/'repair_root_v1'/role;w=read(outer/'actual_wait.json');wait_ok(w,outer=True)
  assert not(outer/'failure.json').exists() and read(outer/'completion.json')['status']=='COMPLETE'
  assert gone(read(outer/'supervisor.json')['process']) and gone(read(outer/'launch.json')['parent_process'])
  for name in ('actual_wait.json','completion.json','supervisor.json','launch.json'):checked(pin(outer/name))
 D=O/'repair_v1/delivery';done=read(D/'completion.json');assert done['registration_sha256']==REG and done['logical_frames']==9000
 manifest=cr(done['manifest']);assert manifest['registration_sha256']==REG
 for d in manifest['files']:checked(d)
 assert checked(done['manifest'])==D/'manifest.json';checked(pin(D/'completion.json'))
 audit=read(D/'closure_audit.json');assert audit['status']=='COMPLETE_N1024_VERSION_CONSISTENT_COMMON500' and audit['new_bootstrap_draws']==0 and audit['new100_pooled'] is False
 assert audit['bootstrap']['indices']['sha256']=='5ca12fe75af6a09e1d3c18e205d7394bf4cdca53c25f7ca21005e5de30117634'
 assert audit['stages']['phy']['new_physical']==2421 and audit['stages']['phy']['new_packets']['unresolved']==0
 assert audit['stages']['receiver']['ledger']['unresolved']==audit['stages']['metrics']['ledger']['unresolved']==0
 assert audit['stages']['receiver']['truth_reads']==audit['stages']['receiver']['TX_CDF_reads']==0
 checked(r['source_utilization']);assert r['source_utilization']['sha256']=='c752d4b5ffac234db39d8e9c4021bd36d5501d7e6262fbc0762ca78e5a98af20'
 return r,D,audit
def validate_tables(D):
 summary=csvread(D/'summary_common500.csv');paired=csvread(D/'paired_common500.csv');means=csvread(D/'source_means_common500.csv');frames=csvread(D/'frame_scores_9000.csv')
 assert len(summary)==42 and len(paired)==21 and len(means)==21000 and len(frames)==9000
 sm={(r['family'],int(r['snr_db']),r['metric']):r for r in summary};pm={(int(r['snr_db']),r['metric']):r for r in paired}
 assert len(sm)==42 and len(pm)==21
 assert set(sm)=={(f,s,m[0]) for f in FAMS for s in SNRS for m in M}
 assert set(pm)=={(s,m[0]) for s in SNRS for m in M}
 for v in summary+paired:
  assert v['population']=='common500' and int(v['sources'])==500 and int(v['noise_per_source'])==3
  for k in ('mean','ci_low','ci_high'):number(v[k])
  assert number(v['ci_low'])<=number(v['ci_high'])
 for v in paired:assert v['comparison']=='PARTIAL_minus_WHOLE'
 keys={(int(v['source_index']),v['family'],int(v['snr_db']),int(v['noise_seed'])) for v in frames}
 assert keys=={(i,f,s,n) for i in range(500) for f in FAMS for s in SNRS for n in NOISE}
 assert sum(v['physical_reused']=='True' for v in frames)==6579
 assert all(v['population']=='common500' and int(v['N'])==1024 and int(v['header_symbols'])==68 and int(v['body_symbols'])==956 and int(v['length_bits'])==13 and int(v['CRC_bits'])==16 for v in frames)
 source_rows=csvread(O/'source_records_3000.csv');sr={(int(v['source_index']),v['family'],int(v['snr_db'])):v for v in source_rows}
 assert len(source_rows)==len(sr)==3000
 for v in frames:
  q=sr[int(v['source_index']),v['family'],int(v['snr_db'])]
  assert v['source_id']==q['source_id']
  for a,b in [('sent_m','actual_m'),('sent_K','actual_K'),('arithmetic_bits','terminated_arithmetic_bits'),('source_capacity_bits','source_capacity_bits')]:assert int(v[a])==int(q[b])
  assert math.isclose(number(v['utilization']),number(q['utilization']),rel_tol=0,abs_tol=1e-15)
 # Check saved source means against saved frame scores; this is reporting arithmetic, not rescoring or resampling.
 groups=collections.defaultdict(list)
 for v in frames:
  for metric,_,_,_ in M:groups[(int(v['source_index']),v['family'],int(v['snr_db']),metric)].append(number(v[metric]))
 mk=set()
 for v in means:
  k=(int(v['source_index']),v['family'],int(v['snr_db']),v['metric']);assert k not in mk;mk.add(k)
  a=groups[k];assert len(a)==3 and math.isclose(number(v['source_mean']),math.fsum(a)/3,rel_tol=0,abs_tol=1e-12)
 for f,s,metric in sm:
  a=[number(v['source_mean']) for v in means if v['family']==f and int(v['snr_db'])==s and v['metric']==metric]
  assert len(a)==500 and math.isclose(math.fsum(a)/500,number(sm[f,s,metric]['mean']),rel_tol=0,abs_tol=1e-12)
 for s,metric in pm:
  assert math.isclose(number(pm[s,metric]['mean']),number(sm['PARTIAL',s,metric]['mean'])-number(sm['WHOLE',s,metric]['mean']),rel_tol=0,abs_tol=1e-12)
 return sm,pm,frames
def legacy_means(sm):
 inp=O/'input_manifest.json';d=pin(inp);assert d['sha256']==INPUT_MANIFEST_SHA;manifest=cr(d);by={x['path']:x for x in manifest['files']}
 whole=R/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/per_frame.csv'
 sup=B/'quality_metrics_supplement_20261014_v1/common500_metrics_v1/frame_scores.csv'
 checked(by[str(whole)]);checked(by[str(sup)])
 a=csvread(whole);b=csvread(sup)
 wa={(int(x['source_index']),int(x['snr_db']),int(x['noise_seed'])):x for x in a if x['family']=='EC_VAR_WHOLE' and int(x['snr_db']) in (4,10)}
 wb={(int(x['source_index']),int(x['snr_db']),int(x['noise_seed'])):x for x in b if x['method']=='EC_VAR_WHOLE' and int(x['snr_db']) in (4,10)}
 expected={(i,s,n) for i in range(500) for s in (4,10) for n in NOISE};assert set(wa)==set(wb)==expected
 out=[]
 for s in (4,10):
  for metric,label,factor,direction in M:
   values=[number(wa[i,s,n][metric] if metric in wa[i,s,n] else wb[i,s,n][metric]) for i in range(500) for n in NOISE]
   old=math.fsum(values)/1500;new=number(sm['WHOLE',s,metric]['mean'])
   out.append(dict(family='WHOLE',snr_db=s,metric=metric,old_mean=old,version_consistent_mean=new,mean_difference=new-old,CI='NOT_COMPUTED_NO_NEW_BOOTSTRAP',interpretation='Source implementation version correction under the unchanged 4/10 dB policy; not an m10 or training effect'))
 return out
def pretty(v,scale=1):return f'{number(v)*scale:.4f}'
def tables(out,sm,pm):
 rows=[]
 for s in SNRS:
  for metric,label,factor,direction in M:
   w=sm['WHOLE',s,metric];p=sm['PARTIAL',s,metric];d=pm[s,metric]
   rows.append(dict(snr_db=s,metric=metric,label=label,display_scale=factor,whole_mean=number(w['mean'])*factor,whole_low=number(w['ci_low'])*factor,whole_high=number(w['ci_high'])*factor,partial_mean=number(p['mean'])*factor,partial_low=number(p['ci_low'])*factor,partial_high=number(p['ci_high'])*factor,paired_partial_minus_whole=number(d['mean'])*factor,paired_low=number(d['ci_low'])*factor,paired_high=number(d['ci_high'])*factor))
 csvsave(out/'manuscript_quality_table.csv',rows)
 tex=['% common500; direct saved paired intervals; no new resampling','\\begin{tabular}{rlrrr}','\\toprule','SNR & Metric & Complete + VAR & NeST-Com & $\\Delta_{\\rm partial-whole}$ \\\\','\\midrule']
 for x in rows:
  lab=x['label'].replace('%',r'\%').replace('_',r'\_')
  show=lambda a,b,c:f'{a:.4f} [{b:.4f}, {c:.4f}]'
  tex.append(f"{x['snr_db']} & {lab} & {show(x['whole_mean'],x['whole_low'],x['whole_high'])} & {show(x['partial_mean'],x['partial_low'],x['partial_high'])} & {show(x['paired_partial_minus_whole'],x['paired_low'],x['paired_high'])} \\\\")
 tex+=['\\bottomrule','\\end{tabular}','% Brackets: 95% source-paired bootstrap intervals from the existing saved index bank.','% Agreement values are percentages; its delta is in percentage points.','']
 (out/'quality_and_paired_table.tex').write_text('\n'.join(tex),encoding='utf8')
 return rows
def plotting(out,sm,pm):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.8,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':True})
 ranges={}
 def series(ax,metric,label,factor,direction,paired=False,gray=False):
  limits=[];handles=[];xx=np.asarray(SNRS,dtype=float)
  for fam in (('PARTIAL',) if paired else FAMS):
   sty=STYLE['families'][fam];rr=[pm[s,metric] if paired else sm[fam,s,metric] for s in SNRS]
   mean=np.asarray([number(r['mean']) for r in rr])*factor;lo=np.asarray([number(r['ci_low']) for r in rr])*factor;hi=np.asarray([number(r['ci_high']) for r in rr])*factor
   if paired and direction=='lower':mean,lo,hi=-mean,-hi,-lo
   limits.extend(lo);limits.extend(hi);limits.extend(mean);color=('#272727' if fam=='PARTIAL' else '#767676') if gray else sty['color']
   # Draw the saved interval itself; even an estimate outside a percentile CI is not clipped or altered.
   ax.vlines(xx,lo,hi,color=color,linewidth=1,zorder=2)
   ax.plot(xx,lo,ls='none',marker='_',ms=4.6,mew=1,color=color,zorder=2)
   ax.plot(xx,hi,ls='none',marker='_',ms=4.6,mew=1,color=color,zorder=2)
   line,=ax.plot(xx,mean,label=sty['label'],color=color,lw=sty['linewidth'],ls=sty['linestyle'],marker=sty['marker'],ms=4.5,mfc=color if fam=='PARTIAL' else 'white',mec=color,mew=1,zorder=3);handles.append(line)
  if paired:limits.append(0.)
  span=max(limits)-min(limits);pad=max(span*.17,0.01 if factor==100 else .001)
  low=min(limits)-pad;high=max(limits)+pad;ax.set_ylim(low,high);ax.set_xlim(2.8,20.2);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)',labelpad=3);ax.tick_params(length=3,width=.7,pad=2)
  if paired:ax.axhline(0,color='#767676',lw=.7,ls=':',zorder=1)
  ranges[('paired_' if paired else 'quality_')+metric]=dict(ymin=low,ymax=high,xlim=[2.8,20.2],xticks=list(SNRS))
  return handles
 def export(fig,stem,gray=False):
  if not gray:
   fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.svg'))
  fig.savefig(out/(stem+('_grayscale' if gray else '')+'.png'),dpi=300 if gray else 600)
 def label_delta(metric,label,direction):
  return {'psnr_db':'ΔPSNR (dB)','lpips_alex':'LPIPS reduction','dreamsim':'DreamSim reduction','convnext_top1_source_prediction':'ΔAgreement (pp)'}[metric]
 for stem,metrics,paired in [('quality_main',M[:4],False),('quality_supplement',M[4:],False),('same_prior_paired',M[:4],True)]:
  for gray in (False,True):
   fig,axes=plt.subplots(1,len(metrics),figsize=(180/25.4,55/25.4));fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.74,wspace=.42 if len(metrics)==4 else .3);handles=[]
   for k,(ax,m) in enumerate(zip(axes,metrics)):
    metric,label,factor,direction=m;h=series(ax,*m,paired=paired,gray=gray);handles=h if not handles else handles
    title=label_delta(metric,label,direction) if paired else label+(' ↑' if direction=='higher' else ' ↓')
    ax.set_title(f'({chr(97+k)}) {title}',pad=5)
   if not paired:fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=2,frameon=False,handlelength=2.8,columnspacing=2)
   else:fig.text(.54,.935,'Partial scales versus complete scales (same frozen prior)',ha='center',va='center',fontsize=8)
   export(fig,stem,gray);plt.close(fig)
  for k,m in enumerate(metrics):
   fig,ax=plt.subplots(figsize=(60/25.4,55/25.4));fig.subplots_adjust(left=.23,right=.97,bottom=.23,top=.84)
   series(ax,*m,paired=paired);metric,label,_,direction=m;title=label_delta(metric,label,direction) if paired else label+(' ↑' if direction=='higher' else ' ↓');ax.set_title(title,pad=6)
   export(fig,stem+'_'+metric);plt.close(fig)
 save(out/'plot_axis_ranges.json',ranges);save(out/'style.json',STYLE)
 captions={
 'quality_main':'Same-prior comparison on common500 at N=1024 (68 header and 956 body symbols; nominal Es=2). Points are means after averaging three fixed noise realizations per source; error bars are saved 95% source bootstrap intervals. Both families use the same frozen source implementation. Complete-scale results at 4 and 10 dB incorporate the documented minimal source-version repair; the two entropy-coded families at 19 dB use the corrected T1-C m10 results. Failed transmissions remain in the denominator. Agreement is the fraction of images whose ConvNeXt top-1 prediction matches that of the source image. No new bootstrap samples were drawn for this figure.',
 'quality_supplement':'Additional metrics for the same common500 paired comparison and fixed outputs. All three metrics are reported using the frozen metric implementations and preprocessing. Points and error bars are read from the closed delivery, with no rescoring or new resampling.',
 'same_prior_paired':'Directly paired source-level differences between partial-scale and complete-scale entropy coding under the same frozen prior. PSNR and agreement are partial minus complete; LPIPS and DreamSim reductions are complete minus partial. Agreement differences are percentage points. Error bars use the saved direct paired intervals; lower-is-better metrics negate the saved difference and exchange its interval endpoints. Zero, negative and crossing-zero results are retained.'}
 save(out/'captions_en.json',captions)
def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);args=p.parse_args();assert args.expected_registration==REG
 r,D,audit=closure();sm,pm,frames=validate_tables(D)
 target=O/'completion_report_v1';target.mkdir();start=time.monotonic()
 try:
  changes=legacy_means(sm);csvsave(target/'legacy_whole_version_delta.csv',changes);table=tables(target,sm,pm)
  calls=audit['stages'];origins=collections.Counter(x['score_origin'] for x in frames)
  summary=dict(registration_sha256=REG,logical_frames=9000,sources=500,source_records=3000,reused_physical_logical_frames=6579,new_physical_events=2421,packet_ledger=calls['phy']['new_packets'],unique_new_RX_inputs=calls['phy']['unique_new_RX_inputs'],receiver_ledger=calls['receiver']['ledger'],metric_ledger=calls['metrics']['ledger'],new_four_metric_pairs=calls['metrics']['actual_new_four_pairs'],new_supplemental_pairs=calls['metrics']['actual_new_supplemental_pairs'],score_origins=dict(origins),bootstrap=audit['bootstrap'],new_bootstrap_draws=0,old_results_preserved=True,new100_pooled=False)
  save(target/'actual_compute_and_reuse.json',summary)
  lines=['# N1024 主证据一致性核验：最终同版本结果','', '本交付仅覆盖原 common500、两种冻结熵编码家族、4/10/19 dB 与每源原三次噪声，共 9,000 个逻辑方法帧。每源先平均三次噪声后统计；不混入 new100、cal1000、Kodak 或 fixed16。','', '原 4/10 dB 回退策略保持不变；19 dB 两家族沿用 T1-C m10 修正版。旧完整尺度实现的源码字节差异经最小修复后，源码利用率和质量现在对应同一冻结实现。旧结果和全部失败保留。','','## 已完成的实际计算与复用','',f"- 6,579 个逻辑帧复用，2,421 个差异事件实际重做；正式 PHY 的包账本见 actual_compute_and_reuse.json。",f"- 新物理接收事件形成 {calls['phy']['unique_new_RX_inputs']} 个待独立恢复的唯一 RX 输入，其余已核验的实际 RX→像素缓存直接复用。",f"- 实际新增四指标像素对 {calls['metrics']['actual_new_four_pairs']} 个，实际新增 DISTS/DreamSim/MS-SSIM 像素对 {calls['metrics']['actual_new_supplemental_pairs']} 个；失败输出仍计入全部分母。",'- 无新 TX 概率遍历、tokenizer、训练、策略选择、checkpoint 选择或新增测试源。','- 原 10,000×500 bootstrap 索引 SHA 5ca12fe75af6a09e1d3c18e205d7394bf4cdca53c25f7ca21005e5de30117634 直接复用；本收尾入口只读取已保存区间，不重评分、不重建、不再重采样。','','## 全指标结果及直接配对区间','','表中差值均为部分尺度减完整尺度；LPIPS、DreamSim、DISTS 的负差值有利于部分尺度。Agreement 使用百分数，差值为百分点。','', '|SNR|指标|完整尺度均值 [95% CI]|部分尺度均值 [95% CI]|直接配对差 [95% CI]|','|---:|---|---|---|---|']
  for x in table:
   q=lambda a,b,c:f'{a:.4f} [{b:.4f}, {c:.4f}]'
   lines.append(f"|{x['snr_db']}|{x['label']}|{q(x['whole_mean'],x['whole_low'],x['whole_high'])}|{q(x['partial_mean'],x['partial_low'],x['partial_high'])}|{q(x['paired_partial_minus_whole'],x['paired_low'],x['paired_high'])}|")
  lines+=['','## 旧新版本差异','','legacy_whole_version_delta.csv 列出旧完整尺度 4/10 dB 与本次同版本结果的均值差。这里的变化包含源码实现版本修正，不能解释成新增 m10、训练或策略搜索的收益。这份差异表只给均值，不把两个边际区间相减，也不额外生成新的旧新差值区间。','','## 图表及可追溯性','','- quality_main / quality_supplement：两家族七指标的完整均值和区间；各面板单独导出。','- same_prior_paired：直接配对差值图；所有负值和跨零区间保留。','- quality_and_paired_table.tex：LaTeX 三线表；manuscript_quality_table.csv 为无舍入底表。','- 原 figures/source_utilization_ecdf 保持不变，按已有 3,000 条实码长记录引用。利用率曲线不代替无线质量。','- input_manifest.json 绑定本报告直接读取的所有输入，含原科学实际退出、修复交付、历史对照分数及源码。','- 本脚本保存成功仅表示导出，actual_visual_review.json 必须由实际打开彩色、灰度及论文尺寸结果的审查补齐后才可宣称视觉交付完成。','']
  (target/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf8')
  plotting(target,sm,pm)
  checked(pin(Path(__file__)));save(target/'input_manifest.json',dict(files=list(INPUTS.values()),skills=['C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md','references/design-theory.md','references/common-patterns.md','references/demos.md'],utilization_figure_reused_without_render=pin(O/'figures/manifest.json')))
  save(target/'manifest.json',dict(files=[pin(p) for p in sorted(target.iterdir()) if p.is_file()],registration_sha256=REG))
  save(target/'completion.json',dict(status='REPORT_EXPORTED_VISUAL_REVIEW_PENDING',registration_sha256=REG,manifest=pin(target/'manifest.json'),seconds=time.monotonic()-start,new_scientific_calls=0,new_bootstrap_draws=0))
 except BaseException:save(target/'failure.json',dict(traceback=traceback.format_exc(),new_scientific_calls=0));raise
if __name__=='__main__':main()
