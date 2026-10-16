from pathlib import Path
import json,csv,hashlib,shutil
H=Path(__file__).resolve().parent;S=H/'snapshot';O=S/'results/submission_closure_20261016_v1';F=H/'deliverable'
def read(p):return json.loads(Path(p).read_bytes())
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf8')))
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(n,x):(F/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def text(n,s):(F/n).write_text(s,encoding='utf8',newline='\n')
def csvsave(n,rs):
 with(F/n).open('w',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def mdtable(headers,rs):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rs)+'\n'
summ=rows(O/'fixed16/summary.csv');methods=['RAW_PARTIAL','P100000','Swin200000','HiFiSwin200000'];labels=['NeST-Com (raw)','Latent JSCC (100k)','SwinJSCC (adapted,200k)','HiFi (Swin200k)'];metrics=['psnr_db','dists','dreamsim','ms_ssim'];table=[]
for m,l in zip(methods,labels):
 table.append([l]+[f"{float(next(r['mean']for r in summ if r['method']==m and r['metric']==k)):.6f}"for k in metrics])
text('fixed16_table.md','# Fixed16：N1024，13 dB，原生浮点输出\n\n'+mdtable(['方法','PSNR ↑','DISTS ↓','DreamSim ↓','MS-SSIM ↑'],table)+'''\n全部原16来源进入统计；上表为均值，完整中位数、逐源值和直接同源差值另附。raw 的 DISTS / DreamSim 较低，但 PSNR / MS-SSIM 低于其余三方法；不据此宣布总体优于 HiFi。

数字 raw 使用原6201信道种子，P/Swin使用2001，HiFi后验23。只有 Swin 与 HiFi 共享实际接收观测。source pairing 不等于共享信道噪声。这里没有13 dB熵码实验，也不能并入common500。

新增48个raw三指标分数和64个PSNR；原144个分数以相同像素SHA核验后逐字文本复用。RGB256原生float32；PSNR采用float64 MSE。DISTS原256RGB，DreamSim官方ensemble及原224 bicubic antialias浮点预处理，MS-SSIM原256RGB、11窗口、sigma1.5、data_range1。精确版本/权重/参数见metric_registration.json和fixed16/runtime_identity.json。没有新bootstrap，表内不提供总体显著性。

可用英文图注：Descriptive comparison on all 16 predeclared development sources at N=1024 and 13 dB, using archived native float32 reconstructions. Raw uses channel seed6201, continuous methods seed2001, and HiFi posterior seed23. Only SwinJSCC and HiFi share the actual received observation. We report all-source means, medians and direct source-paired differences without a new bootstrap. No inference is made about common500 or population-level superiority.
''')
stats=read(O/'m10/length_statistics.json');fit=rows(O/'m10/m10_fit_by_capacity.csv')
text('m10_candidate_decision.md',f'''# m10：可以装入，须修正候选范围；第一批在此停止

原 calibration1000，完整第10尺度(10,0)=680token。新完成1000次TX、前8源独立RX，所有8次token精确往返；同一次TX输出的24,000个旧终点码流与当前H800已闭合旧流逐字节一致。没有额外TX、图像重建、信道或评分。

实际含finish码长：min **{stats['min']}**，median **{stats['median']}**，P95 **{stats['p95']}**，max **{stats['max']}** bit。P95为线性分位数。0源超过8191，但这不保证别的数据集不会溢出。13bit长度与16bit CRC另占29bit。

'''+mdtable(['q','标称码率','实际k/n','算术容量','可装 / 1000'],[[r['q'],r['nominal_rate'],r['k']+'/'+r['n'],r['arithmetic_capacity_bits'],r['fits']]for r in fit])+'''
因此最大4751bit容量可装18源，4273bit可装4源；其余十档未观察到可装。不得以比例低为由忽略，也不以raw8160bit排除熵码m10。只证明这1000源的源码可装性，不证明可靠性、质量提升或任意测试源可装性。

## 缓存与真实调用

恢复了历史m10的1000源码资产，但当前H800与该历史缓存的全部1000个m9流不一致；来源/680token一致不足以证明整数CDF和实际码流一致。本轮因此未复用旧m10科学结果，按授权做了一次当前环境TX遍历。历史缓存和失败/诊断记录保留，未用历史16/1000替代本轮18/1000。

## 仅报告的候选修正

所有旧候选保留。新增最大目标m10必须同时进入整尺度与允许部分尺度两家族：整尺度依次m10,m9…m4；部分尺度依次m10，再按累计token数递减检查全部原24终点。不能因K=0误走整尺度回退。RX仅知道付费的实际终点/profile，不能免费得知TX目标家族。详见m10/candidate_delta_manifest.json及逐源变化CSV。

在这批源码上，只有q6×{3/4,5/6}新增目标改变实际前缀，合计22个source–MCS组合，去重18个来源。两家族×三SNR×三噪声的m10改变子集为396逻辑方法帧；若相同付费profile真正共享，则是198物理帧、396包译码。**这只是改变子集，不是完整补测预算。**旧最大目标在其余来源的回退不变，但旧full1000仅测过筛后的finalists，不能把所有旧MCS当成已完成。

对这两个MCS、两家族、4/10/19dB、原1000×3噪声完整候选网格，保守上界为 **36,000逻辑方法帧 / 72,000包译码**（零复用、每帧头+正文）。在下一批登记前，必须完成原事件的逐帧静态匹配，明确实际profile/波形/接收目录/噪声/输出一致才扣减；当前不虚报精确净新增物理帧。相同m10两个家族应共享实际付费profile，不能为声称复用而借用旧y。新公共ID尚未分配；旧非法ID的接收语义影响仍须核查。

源码TX与tokenizer在cal1000上无新缺口。最多18个独特m10完整端点的干净质量/输出尚未核验；本轮未生成这些图。独立源码资格已完成前8源，但不把它们误当18个可装来源的质量证据。后续接收/恢复数量必须按实际事件登记。选择仍限原cal1000 DINO和既有有限筛选，保留旧赢家为锚；不得用common500/new100质量挑选。

条件任务T1-B、T1-C均NOT_RUN。规则冻结后common500/new100源码核查最多另600TX；若两家族三工作点全变，测试逻辑上界9000+1800分别统计，不与本轮混合。最终实现若改变，只讨论受影响端点计时，不自动重测六方法。

## English result statement

On the original 1,000 calibration sources, fully terminated entropy coding of all 680 tokens produced lengths of 3,823–6,855 bits (median 6,071; 95th percentile 6,568.05). Four sources fit the 4,273-bit capacity and 18 fit the maximum 4,751-bit capacity. All eight independently decoded source streams recovered the exact tokens, and all 24,000 legacy endpoint streams matched the current frozen implementation bit for bit. Full-scale m10 therefore cannot be excluded solely by the earlier interface bound. Symmetric candidate corrections for both whole- and partial-scale strategy families are required; no new wireless quality, reliability, or performance gain is claimed here.
''')
# Exact physical-profile table, including different raw wire budgets.
raw=read(F/'raw_frozen_profile_evidence.json')['rows'];phy=[]
for row in raw:
 p=row['profile'];g=p['groups'][0];l=g['layout'];r=row['policy']
 phy.append(dict(method='RAW_'+r['family'],snr_db=r['snr_db'],target_m=r['m'],target_K=r['K'],q={'QPSK':2,'16QAM':4,'64QAM':6}[r['groups'][0]['modulation']],nominal_rate=r['groups'][0]['nominal_rate'],k=r['groups'][0]['actual_k'],n=r['groups'][0]['actual_n'],bg=l['bg'],z=l['z'],body_symbols=r['used_body_symbols'],idle_symbols=r['idle_symbols'],length_bits=0,crc_bits=16,decoder=l['decoder_config']['cn_update'],num_iter=l['decoder_config']['num_iter'],early_stop=l['decoder_config']['early_stop'],rate_matching='RV0 filler shortening + first2Z/tail puncturing; original profile',candidate_id=r['candidate_id']))
layout=rows(O/'facts/entropy_12_actual_ldpc_layouts.csv');p=read(S/'results/entropy_main_common500_20261013/e0_reuse/frozen_configs.json')['policies'];w=read(O/'facts/whole_policy_metadata.json')['data']['policies']['EC_VAR_WHOLE']
for family,policies in [('EC_VAR_PARTIAL',p),('EC_VAR_WHOLE',w)]:
 for snr,r in policies.items():
  l=next(l for l in layout if int(l['q'])==r['q']and int(l['k'])==r['k']and int(l['n'])==r['n'])
  phy.append(dict(method=family,snr_db=int(snr),target_m=r['target_m'],target_K=r.get('target_K',0),q=r['q'],nominal_rate=r['nominal_rate'],k=r['k'],n=r['n'],bg=l['bg'],z=int(l['z']),body_symbols=956,idle_symbols=0,length_bits=13,crc_bits=16,decoder=l['cn_update'],num_iter=int(l['num_iter']),early_stop=l['early_stop'],rate_matching='RV0 filler shortening + first2Z/tail puncturing; zero repetition',candidate_id=r['candidate_id']))
csvsave('physical_workpoints.csv',phy)
text('implementation_fact_sheet.md','''# Implementation facts — frozen evidence, no new model search

## Architecture and weights

The actual VAR builder uses depth16, width1024, 16 attention heads, MLP ratio4, vocabulary4096, latent channels32, VAE base channels160, quantization residual sharing4. The ten patch sizes are 1,2,3,4,5,6,8,10,13,16 (680 tokens). Null class1000 and the frozen integer-CDF / completion rules are retained. Evidence: src/var_comm/next_scale_prior.py; the registered author-root models/__init__.py and models/var.py; source_registration.json and model_routes.json lineage.

Actual full-loaded parameter counts from the completed timing runtime: VAE/tokenizer+original decoder **108,948,355**; VAR **310,283,520**; adapted D_c **64,663,171**; P communication module **495,208**; SwinJSCC **33,073,488**. All are frozen in evaluation. The P communication count is not the end-to-end P system total.

CPU weight-tensor inventory confirms VAE encoder44,107,872; quant_conv9,248; quantizer parameters168,064 (209,024 stored elements minus40,960 EMA-usage buffer); original decoder64,653,923; post_quant_conv9,248. Tokenizer-side subtotal is **44,285,184**. VAR's VAE quantizer proxy is an unregistered tuple referencing the same quantizer: do not count it twice. D_c deep-copies decoder+post_quant_conv and has separate adapted weights. The unique NeST-Com tokenizer+VAR+D_c parameter subtotal is **419,231,875**; a runtime retaining unused original VAE decoder/post-conv stores **483,895,046** parameters. These are weight counts, not FLOPs, transmitted bits or an online latency claim. No system model constructor or inference was called for this metadata inventory.

VAE file SHA256 `7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c`; VAR-d16 `4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb`; D_c checkpoint update0038000 `e3d757a5a16206f5f9be345d0e6330b0f1fd6c78bf167670e70378b429d931c4`. Actual state identities are retained in the registration/model routes; file SHA and state SHA are different domains. P100k and Swin200k are the completed selected final checkpoints, not older diagnostic P/Swin80k.

Raw TX uses the frozen tokenizer and fixed-width token serialization without loading a VAR probability model for source coding. Entropy TX performs conditional VAR CDF evaluation. Both receivers use their actual received tokens, frozen completion and D_c; TX CDF, source pixels and untransmitted tokens are not receiver inputs. Entropy CRC/arithmetic failure rejects the stream; raw keeps its historically registered hard-token/KEEP rule. This is not the layered DROP-only proposal.

## D_c adaptation

Selected checkpoint is the registered update38,000 decoder, not a new training run. Original Stage-A optimizer AdamW, learning rate1e−5, weight decay0, gradient clip1, logical batch16/microbatch4. Each batch has exactly8 continuous F,4 quantized Fq,4 interpolated inputs; independent alpha Uniform[0,1]. Loss is image MSE +0.1 LPIPS. Original20k source images; no new spatial augmentation. Encoder/quantizer/VAR frozen; only copied decoder modules adapted. Full originalcal1000 selects the minimum50/25/25 mixture image loss; DINO is absent from this optimizer/checkpoint selection. Minimum4k and safetycap40k are protocol bounds, not claims of convergence. See experiments/var-latent-enhancement-20260917/configs/experiment.json, docs/stage_A_execution.md and source training.py.

## Physical configuration

`physical_workpoints.csv` lists both raw/entropy whole/partial configurations at4,10,19dB, including real k,n,BG,Z. `facts/entropy_12_actual_ldpc_layouts.csv` lists all12 original entropy MCS. These values are copied from actual constructor qualification, not recomputed by an approximate capacity formula. Entropy main q2/r2/3 uses BG2,Z128,k1274,n1912; q4/r2/3 uses BG2,Z256,k2549,n3824; q6/r5/6 uses BG1,Z224,k4780,n5736. Actual rate is k/n and includes the13+16bit source framing overhead. Sionna2.2.0 5G LDPC, RV0 shortening/filler removal, first2Z and tail puncturing, no repetition; flooding boxplus-phi, VN sum,20 iterations, no early stop, llr_max20, hard information output. Original raw19 whole uses only616 body symbols and340 idle; do not claim every raw waveform occupies956.

Header is68 complex uses and total budget1024. Entropy body956, nominal constellation Es2, originalAWGN/numerical protocol. Original q/r restrictions remain; no LE2 r1/3 added to entropy-main candidates. Frozen targets and actual fallback differ; entropy partial checks actual terminated lengths along its paid prefix chain. Main19 partial targetm9+192 is616tokens, not fullm10. Catalogue/profile additions are pending a separate correction, not deployed by this audit.

## Sources, preprocessing and statistical roles

`cohort_source_ids.csv`, `cohort_roles.csv`, `cohort_intersections.csv` contain the exact original lists and direct intersections. Train20k comes from the registered ImageNet training parquet; cal1000/development100/common500/new100 are ImageNet validation images used in different internal roles. Official validation is not synonymous with a fresh blind internal test. The HiFi16 are a subset of development100; all other pairwise source-ID intersections among these seven roles are zero. Available recorded canonical/flip pixel hashes also show zero cross-role matches except that expected16. This does not exclude arbitrary crops, all historical exposure, perceptual duplicates or pretrained-model training overlap.

Original ImageNet preprocessing is frozen RGB center256 under the original next_scale_prior/cache lineage; train-cache hashes record original and flip variants where applicable. No source pixels are regenerated in this batch. Kodak uses all24 numeric-order original lossless PNGs, center crop256 with integer floor offsets, no resizing; Kodak evidence here remains raw transport, not entropy transport or full-resolution transmission. The precise source/pixel identities and preprocessing are in the cohort evidence and registration, not inferred solely from filenames.

Common500 has already been evaluated historically. Development100 and cal1000 have been used for development/selection. New100 was content-excluded against the recorded-study registry before its original confirmation, and is now also already evaluated. None is rebranded as a newly unseen test.

## Metrics, timing and availability

All common500 original and supplementary metrics/95% source intervals are reused unchanged; each source first averages its three noises. No marginal intervals are subtracted to fabricate a paired interval. Raw whole lacks the three supplementary metrics and remains explicitly unavailable; no extra4500 pairs were scored. Fixed16 uses nativefloat, frozen official DreamSim ensemble, original DISTS/MS-SSIM and newPSNR; precise settings/version/weight hashes are in metric_registration.json and fixed16/runtime_identity.json.

The completed six-method shared-H800 timing table is included unchanged as reference implementation cost. Fixed method order, sharedGPU/resource admission revisions,576 reused calls and1152 continuation calls are disclosed in its original report;1728 totalcalls/864measurements, not864 independent sources. Do not infer interference-free exact speed ratios, combine differentGPUs or claim new timing for m10. No six-method retiming was run.

Repository visibility and access were not changed. Code path and reproducibility metadata are provided within the existing repository; no unapproved public download availability is promised. Author names, affiliations, corresponding author, licensing declarations and venue-specific availability wording: **AUTHOR_TO_COMPLETE**.
''')
# Original completed evidence is copied byte-exact, never re-estimated.
reuse=[('results/entropy_main_common500_20261013/analysis_v1/mechanism_four_methods.csv','same_prior_four_methods.csv'),('results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/paper_timing_table.csv','shared_H800_timing.csv'),('results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/REPORT.md','shared_H800_timing_original_report.md'),('results/entropy_main_common500_20261013/analysis_v1/length_summary.csv','source_capacity_summary.csv')]
for src,dst in reuse:shutil.copyfile(S/src,F/dst)
assets=[]
for role,path,status,gap in [('m10_new_cal1000',O/'m10/m10_lengths.csv','CLOSED_1000_TX_8_RX','quality and wireless conditional correction not run'),('fixed16_native_four',O/'fixed16/frame_scores.csv','CLOSED_112_NEW_144_REUSED','no source pixels missing'),('common500_existing',S/'results/entropy_main_common500_20261013/analysis_v1/summary_all_methods.csv','EXISTING_REUSED','m10 comparator scope pending'),('supplement_common500',S/'results/quality_metrics_supplement_20261014_v1/common500/summary.csv','EXISTING_REUSED_81000_SCALARS','RAW_WHOLE supplementary metrics intentionally absent'),('shared_H800_timing',S/'results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/paper_timing_table.csv','EXISTING_CLOSED','not m10 timing'),('historical_m10',H/'m10_restoration_verified.json','RESTORED_NOT_CURRENT_REUSE_ELIGIBLE','all1000 current oldm9 streams differ'),('raw_native_restoration',H/'raw_restoration_verified.json','EXACT_RESTORED_NO_NEW_RENDER','none'),('seven_cohorts',F/'cohort_source_ids.csv','METADATA_AUDITED','not universal perceptual/pretrained dedup')]:
 assets.append(dict(role=role,actual_path=str(path),sha256=pin(path)['sha256'],exists=True,status=status,reuse_condition='exact source/pixel/protocol identity; historical stage closure',gap=gap))
csvsave('asset_inventory.csv',assets)
claims=[('m10 cannot fit','REJECTED','m10/m10_fit_by_capacity.csv','18/1000 fit maximum;4/1000 next; neither wireless gain nor universal bound'),('raw perceptual advantage on fixed16','DESCRIPTIVE_ONLY','fixed16/summary.csv','DISTS/DreamSim lower;PSNR/MS-SSIM lower too;different seeds;no population superiority'),('entropy main compared fairly with full scale','CORRECTION_REQUIRED','m10/candidate_delta_manifest.json','oldm9-limited results retained; symmetricm10 candidates pending'),('source capacity utilization','SUPPORTED_EXISTING_SCOPE','source_capacity_summary.csv','actual arithmetic bits/(k−29),not channel-symbol savings'),('six-method speed ratios','NOT_CAUSAL','shared_H800_timing_original_report.md','sharedGPU/fixedorder andadmissionrevision; referencecostonly'),('independent cohorts','LIMITED_ID_HASH_AUDIT','cohort_intersections.csv','HiFi16subsetdevelopment;no universal duplicate/pretrained exclusion'),('raw Kodak generalizes entropy','UNSUPPORTED','implementation_fact_sheet.md','keep raw version tag'),('LE2 group improvement','NOT_OBSERVED','../layered_entropy_le2_20261014_v1/delivery_v2/REPORT_zh.md','existingnegativeproxy preserved;noLE3'),('new branch experiments','NOT_RUN','m10_candidate_decision.md','T1-B/T1-C/LE3/LE4/retiming/training/channel/diffusion excluded')]
csvsave('claim_evidence_matrix.csv',[dict(claim=a,status=b,evidence=c,limitation=d)for a,b,c,d in claims])
# Verify every export, then record the actual visual review already performed by root.
manifest=read(H/'figures_v2/manifest.json');pngs=[]
for fig in manifest['figures']:
 for r in fig['files']:
  q=H/'figures_v2'/Path(r['path']).name;assert q.stat().st_size==r['bytes']and pin(q)['sha256']==r['sha256']
  if q.suffix=='.png':pngs.append(r)
assert len(pngs)==9
save('actual_visual_review.json',dict(status='PASS',reviewer='/root',all_nine_pngs_actually_opened=True,manifest=pin(H/'figures_v2/manifest.json'),PNG=pngs,checks=['all16 fixed order retained','first2 indices0,25','labels and metric directions legible','no cropping or method-specific enhancement','full PDF/editable SVG/600dpiPNG hashes verified','main chart explicitly provisional until m10 correction'],dedicated_failure_example='not added; complete16 gallery retains all structural/identity discrepancies without estimating hallucination rates'))
text('REPORT_zh.md','''# 投稿前第一批交付：m10 边界与固定16图补齐

开始与本轮科学闭合时HEAD均为`bdb3e3c12d2ea2c18f4371e11cd912d5e8529100`，与执行包参考一致；没有把旧README中的等待状态当成新任务。范围为T0、T1-A、T2及T3资料准备，交付后停止。旧raw R1 NO_GO、LE0/LE1、LE2和已完成六方法计时均保留。

## 结论

1. **m10可以装入，必须修正候选范围。** 当前cal1000实际终止码长3823/6071/6568.05/6855 bit（min/median/P95/max）。q6,r3/4容量4273可装4源；q6,r5/6容量4751可装18源。其余十档零。8次独立RX全部通过，24,000旧终点码流完全一致。完整尺度与部分尺度两家族都应纳入m10，不能只增强本文方法。
2. **原生raw已补齐固定16表。** 新增48个raw感知/结构评分和64个PSNR，复用144个原分数。raw在DISTS、DreamSim均值上更低，在PSNR、MS-SSIM上更低。全部16来源和三种方向取舍都报告，不宣称总体优于HiFi，不混入common500或13dB熵码结果。
3. **资料准备已齐。** 七种源角色清单/交集、真实LDPC表、分组件参数和训练事实、全部旧指标、同先验四方法表、容量利用率、共享H800原计时、九套正式导出图（含全16图库）均可追溯。新的common500主图标为m10对照待修订，未覆盖旧图或改旧成绩。

## 后续工作量与停止边界

只在两档MCS出现m10新增实际前缀，22个source–MCS组合、去重18源。两家族×三SNR×三噪声，改变子集396逻辑帧；若真正同profile则198物理帧。完整候选校准上界36,000逻辑帧、72,000包，必须先逐帧证明可复用部分再给净新增数；不能误把改变子集当完整补测。最多18个m10干净端点质量缺口，cal1000源码TX/tokenizer无新增缺口。详情及源ID见候选差异清单。

**T1-B、T1-C均未执行。** 无新信道包、无线图像、图像恢复、扩散、训练、bootstrap、计时、LE3/LE4或功率扩展。未恢复任何旧活动调度；暂不新增科学备份包。第一批停止，后续校准/测试须另有范围授权。

## 文件导航

- [m10判断与下一批缺口](m10_candidate_decision.md)；[逐源码长](m10/m10_lengths.csv)；[12档装入表](m10/m10_fit_by_capacity.csv)；[候选差异登记](m10/candidate_delta_manifest.json)。
- [固定16小表及图注](fixed16_table.md)；[64图像对四指标](fixed16/frame_scores.csv)；[均值中位数](fixed16/summary.csv)；[直接逐源差值](fixed16_source_paired_differences.csv)。
- [实现事实表](implementation_fact_sheet.md)；[物理工作点](physical_workpoints.csv)；[源角色](cohort_roles.csv)；[交集](cohort_intersections.csv)；[资产](asset_inventory.csv)；[论断边界](claim_evidence_matrix.csv)。
- [待修订主图](figures_v2/common500_main_provisional.png)；[固定16量化](figures_v2/fixed16_descriptive.png)；[预定前两张](figures_v2/fixed16_first_two.png)；[全部图库1](figures_v2/fixed16_gallery_1.png)、[2](figures_v2/fixed16_gallery_2.png)、[3](figures_v2/fixed16_gallery_3.png)、[4](figures_v2/fixed16_gallery_4.png)；[单码字系统图](figures_v2/single_codeword_system.png)。同名PDF/SVG及600dpiPNG齐。
- [全部指标原值和区间](figures_v2/common500_all_metrics.csv)；[同先验四方法](same_prior_four_methods.csv)；[原共享计时](shared_H800_timing.csv)。RAW_WHOLE没有三项补充指标，明确留空而非新评分。

## 工程记录

历史m10恢复和当前缓存比较均保留；当前1000个旧m9流不匹配，所以按授权独立编码当前m10。科学两任务均真实退出0且父子进程gone，ledger未解决项0。首次绘图因BPG旧/新命名不一致在布局阶段exit1，未计算任何科学分数；第二版仅统一显示别名并保留原方法标签，实际退出0，全部九张PNG已打开审查。中途一次SSH在CPU差异表执行前断开，未产生输出或科学调用；核验后仅生成CPU派生表。原证据不删改。
''')
text('README.md','# Submission closure batch1\n\nSee [Chinese delivery report](REPORT_zh.md), [m10 decision](m10_candidate_decision.md), and [implementation facts](implementation_fact_sheet.md). This is a closed first batch, not approval to launch m10 calibration, test updates, layered LE3/LE4, training, wireless trials or retiming.\n')
print('REPORT_FILES',[(p.name,p.stat().st_size)for p in F.iterdir()])
