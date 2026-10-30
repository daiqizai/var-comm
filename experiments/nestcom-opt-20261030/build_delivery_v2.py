"""Assemble completed evidence. No scoring, selection, training or scientific retry."""
from opt_common import *
from route_statistics_v2 import csvout
import csv,collections,shutil,subprocess

def rows(path):
 with Path(path).open()as f:return list(csv.DictReader(f))

def text(path,value):
 with Path(path).open('x',encoding='utf8')as f:f.write(value)

def ci(v):return f"{float(v['mean_delta']):+.6f} [{float(v['ci95_low']):+.6f}, {float(v['ci95_high']):+.6f}]"

def main():
 final=D/'final_common500_delivery';dev=D/'development_summary';timing=D/'timing_delivery';resource=D/'resource_delivery'
 identity=read(final/'statistics_identity.json');assert identity['logical']==42000 and identity['source_rows']==14000
 budget=read(resource/'resource_ledger.json');assert budget['all_owner_and_child_identities_gone'] and budget['all_scientific_calls_complete']
 assert read(timing/'identity.json')['status']=='COMPLETE'
 assert read(L/'final_successors_control/actual_wait.json')['returncode']==0
 freeze=read(D/'frozen_final.json');assert freeze['configuration']=='original_receiver_plus_D_token'
 route=rows(dev/'route_summary.csv');cgate=read(D/'fit_C_complete_delivery/gate.json')['gates']
 for v in route:
  if v['candidate']in cgate:
   g=cgate[v['candidate']];v['guard_violations']=json.dumps(g['guard_violations']);v['all7_fit_gate']=g['fit_pass'];v['source_report']=str((D/'fit_C_complete_delivery').relative_to(R))
  elif v['route']=='D'and v['candidate']=='D_token':v['all7_fit_gate']=read(D/'fit_D_complete_delivery/gate.json')['gates']['D_token']['fit_pass']
  elif v['route']in ['A','B']:v['all7_fit_gate']=v['fit_core_gate']
  elif v['route']in ['C','D']:v['all7_fit_gate']='NOT_RUN_NOT_SELECTED_CHECKPOINT_OR_GRID'
  else:v['all7_fit_gate']='NOT_REQUIRED_AFTER_CORE_NO_GO'
 timing_rows=rows(timing/'timing_and_compute.csv');timing_lookup={(v['variant'],v['component']):v for v in timing_rows if v['snr_db']=='pooled_core'}
 aliases={'A1':'A','B_T0.3':'B03','B_T0.7':'B07'}
 for v in route:
  name=aliases.get(v['candidate'],v['candidate']);v['timing_variant']=name if (name,'TX_total')in timing_lookup else 'NOT_SEPARATELY_TIMED_VERIFICATION_OR_OTHER_CHECKPOINT'
  for component in ['TX_total','RX_total']:
   v[component+'_mean_seconds']=timing_lookup.get((name,component),{}).get('mean_seconds','')
 out=D/'delivery';out.mkdir(exist_ok=False)
 csvout(out/'route_summary.csv',route)
 shutil.copyfile(D/'fit_C_delivery/content_minus_selected_control.csv',out/'C_content_minus_control.csv')
 ct=[]
 for root in ['fit_C_token_diagnostics','fit_C_guard_token_diagnostics']:
  ct.extend(rows(D/root/'missing_token_diagnostics.csv'))
 groups=collections.defaultdict(list)
 for v in ct:groups[(v['method'],v['N'],v['snr_db'],v['variant'],v['logits'])].append(v)
 csummary=[]
 for k,vs in sorted(groups.items()):
  csummary.append(dict(method=k[0],N=k[1],snr_db=k[2],variant=k[3],logits=k[4],eligible_source_rows=len(vs),unique_sources=len({v['source_id']for v in vs}),mean_missing_CE=float(np.mean([float(v['missing_ce'])for v in vs])),mean_missing_accuracy=float(np.mean([float(v['missing_accuracy'])for v in vs])),mean_missing_positions=float(np.mean([float(v['missing_positions'])for v in vs])),population='Supported partial endpoints only; not all-source image population; targets joined after RX exit'))
 csvout(out/'C_missing_token_summary.csv',csummary)
 es=read(L/'fit_E_score/selection.json');frequency=[]
 for snr in [7,10]:
  vs=[v for v in es if v['snr_db']==snr];assert len(vs)==100
  for rank in [0,1,2]:frequency.append(dict(N=1024,snr_db=snr,selected_rank=rank,sources=sum(v['selected_rank']==rank for v in vs),total_sources=100,selection_before_actual_channel=True))
 csvout(out/'E_selection_frequency.csv',frequency)
 for name in ['own_baseline_normalized.csv']:shutil.copyfile(dev/name,out/name)
 paired=rows(dev/'paired_deltas.csv')+[dict(population='common500_REGRESSION',route='FINAL',**v)for v in rows(final/'paired_deltas.csv')]
 source=rows(dev/'per_source_metrics.csv')+[dict(population='common500_REGRESSION',route='FINAL',**v)for v in rows(final/'per_source_metrics.csv')]
 csvout(out/'paired_deltas.csv',paired);csvout(out/'per_source_metrics.csv',source)
 for src,name in [(final/'summary.csv','final_summary.csv'),(final/'endpoint_and_failure_summary.csv','endpoint_and_failure_summary.csv'),(timing/'timing_and_compute.csv','timing_and_compute.csv'),(timing/'timing_raw.csv','timing_raw.csv'),(resource/'actual_workers.csv','actual_workers.csv'),(resource/'actual_calls.csv','actual_calls.csv'),(resource/'device_budget.csv','device_budget.csv')]:shutil.copyfile(src,out/name)
 workers=rows(resource/'actual_workers.csv');completions=[]
 for w in workers:
  p=L/w['role']/'completion.json'
  if p.exists():completions.append(dict(role=w['role'],route=w['route'],timing_only=w['timing_only'],actual_wait=read(L/(w['role']+'_control/actual_wait.json')),completion=read(p),completion_identity=pin(p)))
 save(out/'actual_compute_and_reuse.json',dict(status='ACTUAL_CLOSED_ONLY',workers=completions,all_waits_and_calls=pin(resource/'resource_ledger.json'),logical_results_not_model_call_counts=True,repeated_timing_calls_separate_from_quality=True,historical_final_rows_reused=30000,historical_native_arrays_verified=9296,historical_missing_CLIP_only_new_pairs=2188,new_final_logical=12000,new_final_physical_events=11223,exact_reused_final_physical_events=777,new_final_RX_expected_unique_decode=3383,actual_values_in_per_worker_ledgers=True))
 stages={
  'P0':dict(status='done',evidence='Original/zero-corrector real replay and protocol/data identities passed.'),
  'A':dict(status='done',decision='NO_GO',reason='A1 DreamSim worsens; A2 verification retained, not a replacement candidate.'),
  'B':dict(status='done',decision='NO_GO',reason='Both fixed low temperatures fail FIT; all three generation repeats retained.'),
  'C_primary':dict(status='done',decision='NO_GO',reason='Content and same-capacity control each 3000 updates; selected 1000-step content fails N512/19 DreamSim guard.'),
  'C_second_seed':dict(status='not-run-conditional',reason='Primary downstream guard failed; triggering condition not met, not resource failure.'),
  'D':dict(status='done',decision='CONFIRM_PASS',reason='Token-wise grid selected at FIT; frozen before CONFIRM; all seven guards passed.'),
  'E':dict(status='done',decision='NO_GO',reason='Only 8/200 shorter selections; all-source metrics fail, including one real rejection.'),
  'combinations':dict(status='done',reason='Only D passed. R_det is unchanged; R_det+D is exact standalone D result. No redundant run.'),
  'leave_one_out':dict(status='done',reason='Only retained component is D. Removing D exactly gives original baseline; reuse exact CONFIRM and common500 records.'),
  'common500':dict(status='done',logical_results=42000,source_rows=14000,role='Previously examined regression, not new blind validation.'),
  'NEW200':dict(status='blocked',reason='No auditable unused raw validation pool: registered root absent on both PRO and main.',evidence=pin(D/'new200_availability.json')),
  'NEW200_contact_sheet':dict(status='blocked',reason='Depends on unavailable NEW200; no quality-based replacement.'),
  'known_Pastries_Dog_final_contact':dict(status='blocked',reason='Exact legacy IDs absent from common500. No join by unrelated integer source_index, no extra receiver restoration registered.',evidence=pin(D/'display_source_rule.json')),
  'deployment_timing':dict(status='done',reason='Same GPU, two CPU affinity, 16 fixed sources, 2 warmups/3 measured; actual E scoring included.'),
  'git_publication':dict(status='not-run-user-directed',reason='Latest work order explicitly prohibits automatic commit/push. Review package only.'),
 }
 save(out/'stage_completion.json',dict(status='COMPLETED_FEASIBLE_SCOPE_WITH_EXPLICIT_BLOCKED_NEW200',stages=stages,not_full_independent_validation=True,missing_evidence_not_filled_by_historical_test=True))
 save(D/'stage_status.json',dict(delivery=pin(out/'stage_completion.json'),all_scientific_workers_exited=True,Git_commit_push=False))
 save(out/'configuration_index.json',dict(defaults=pin(D/'protocol_defaults.json'),resolved=pin(D/'protocol_resolved_p0.json'),source_manifest=pin(D/'source_manifest.json'),frozen_final=pin(D/'frozen_final.json'),D_confirmation=pin(D/'confirm_D_delivery/gate.json'),C_selection_before_guards=pin(D/'C_selected_before_guards.json'),C_checkpoint_recovery=pin(D/'model_storage_and_recovery.json'),statistical_identity=pin(final/'statistics_identity.json'),resource_ledger=pin(resource/'resource_ledger.json'),timing=pin(timing/'identity.json'),historical_no_rescore=pin(L/'historical_regression_inputs/completion.json')))
 # Read actual frozen results; no conclusions are prefilled from predicted outcomes.
 pp=rows(final/'paired_deltas.csv');ss=rows(final/'summary.csv');cap=rows(final/'endpoint_and_failure_summary.csv')
 pmap={(v['candidate'],v['reference'],int(v['N']),v['snr_db'],v['metric']):v for v in pp}
 lines=['NeST-Com 五路线优化：实际执行与交付报告','基准 HEAD：9d55aacfcd740034ce85014ae65f20a33893b59c','',
 '本轮 A–E 首轮均已完成。只有 D（相同最大端点/MCS 下逐 token 装入）通过预定 FIT 与 CONFIRM，最终配置为原接收器 + D。A/B/C/E 的失败和混合结果完整保留。原暂停研究未恢复；唯一新增训练为本工作单授权的 C 主种子匹配两臂。',
 '本报告覆盖可执行范围。NEW200 因缺少可审计的未使用原始源池而阻塞；common500 是历史已查看集上的回归，不替代独立留出验证。没有自动提交或推送 Git。','',
 '一、身份、协议和统计',
 '原生 256×256 RGB；N1024 的 4/7/10/19 dB，N512 的 4/10/19 dB；68-symbol header、13-bit length + CRC16、Es=2。D 只细化候选，不扩大原最大端点、不改 MCS；独立终止实际码长判断，不裁切长码流。12-bit 目录合法性已核验，旧 profile 含义保留。',
 'FIT100/CONFIRM100 从排除指定历史开发集合后的 612 个合格 cal1000 来源按固定 hash 切分；C 为固定 10000 训练源、m7/m8 共 20000 条件。训练、开发、common500 的完整源 ID 与像素 hash 分别核验。',
 'B 先生成重复、再信道重复、再等权源平均。其他方法先平均信道重复。拒收/NULL 灰图不移除。直接 source-paired bootstrap；CONFIRM 归一化分母固定为 FIT 均值。common500 复用原 10000×500 索引并核对完整源顺序。逐点区间不是多重比较同时覆盖保证。',
 '原 N1024 历史 H800 原生数组与原数值路由保留，新计算在 PRO6000 FP32/TF32off。跨方法保留各自历史噪声身份，不声称所有方法共享同一噪声。CLIP 新特征只走正式 torch2.10 路由；合法已有标量不重新评分。','',
 '二、全部路线筛选（FIT 核心 J；小于 1 有利，不等同统计显著）']
 for v in route:lines.append(f"{v['candidate']}: J={float(v['core_J']):.9f}; LPIPS归一化={float(v['normalized_lpips']):.9f}; DreamSim归一化={float(v['normalized_dreamsim']):.9f}; 平均PSNR差={float(v['mean_core_psnr_delta']):+.6f} dB; {v['status']}; 保护线={v['guard_violations']}")
 lines += ['',
 'A：原锚定 decoder 的复用提高部分 PSNR，却在核心 DreamSim 明显退化。第二训练种子仅作预定验证，未用它替换主候选追胜。',
 'B：T=0.3 和 0.7 均未通过；低温采样在本冻结系统中没有满足共同感知目标。三次生成结果均参与均值，没有挑最好 seed。',
 'C：content/no_content 各 3000 updates，独立在 1000/2000/3000 中选择；两者均选择 1000。content 核心 J=0.994034822，control=0.994778766，内容增量很小。冻结后的 N512/19 DreamSim 恶化 1.088427%，超过预定 1% 保护线（control 为 1.153741%）。因此不启动第二种子，不调整 checkpoint 回避保护线。token CE/accuracy 诊断及所有 14 个含 step0 的训练 checkpoint 身份与恢复信息保留；不以训练损失替代图像证据。',
 'E：200 个核心源×工作点中 8 个选了更短端点。实际重发改变的帧后出现一次拒收；灰图按合同保留。全样本 LPIPS/DreamSim 均未达到推进条件。不得据成功模拟的局部目标下降删除该失败，或事后根据已发生信道换回 e0。',
 'D：FIT Token 优于 Eighth；两网格均保留原上限。逐 token 配置冻结后，CONFIRM 的固定 FIT 分母 LPIPS 差为 -0.033444165 [-0.057228058, -0.011514610]，DreamSim 差为 -0.031989851 [-0.066229798, +0.001220903]。LPIPS 排除零且另一项上界小于 +0.01，七点保护线通过。DreamSim 的确认区间跨零，不能写成两项均显著改善。',
 '只有 D 进入组合，故原 receiver+D 与独立 D 完全相同；去掉 D 就是原系统，直接复用同身份记录，无额外组合/消融科学调用。','',
 '三、最终 common500：最终 NeST − 原 NeST，全部四指标']
 for N,snr in [(1024,4),(1024,7),(1024,10),(1024,19),(512,4),(512,10),(512,19)]:
  lines.append(f'N={N}, SNR={snr} dB: '+ '; '.join(m+' '+ci(pmap[('final_NeST','original_NeST',N,str(snr),m)])for m in ['psnr_db','lpips_alex','dreamsim','clip_image_cosine']))
 lines += ['核心固定 FIT 分母汇总：']
 for m in ['lpips_alex','dreamsim']:lines.append(m+' '+ci(pmap[('final_NeST','original_NeST',1024,'core_mean','FIT_normalized_'+m)]))
 lines += ['','四、完整系统取舍：最终 NeST − Ada（固定帧、预测类别适配）']
 for N,snr in [(1024,4),(1024,7),(1024,10),(1024,19),(512,4),(512,10),(512,19)]:
  lines.append(f'N={N}, SNR={snr}: '+ '; '.join(m+' '+ci(pmap[('final_NeST','Ada_frozen',N,str(snr),m)])for m in ['psnr_db','lpips_alex','dreamsim','clip_image_cosine']))
 lines += ['PSNR/CLIP-I 正差有利于 NeST，LPIPS/DreamSim 负差有利。WHOLE 同接收规则对照及 WHOLE−Ada 直接配对值在 CSV。不同 tokenizer/prior/decoder/类别机制意味着完整系统比较，不将所有差距归给端点。没有重新选择 Ada 策略，也没有宣称通信全指标 SOTA。','','五、容量、失败与原生数组']
 for v in cap:lines.append(f"{v['method']} N{v['N']} {v['snr_db']}dB: {v['logical']}逻辑位置，灰图{v['fixed_gray']}，源码利用率{v['mean_source_utilization']}，状态{v['receive_status_counts']}")
 lines += ['利用率为算术源码 bit /（body 信息容量−29bit），不是总空口效率。端点/回退分布以逐帧 CSV 和 endpoint_and_failure_summary 为准。所有显示图从 hash 核验的原生数组导出，没有美化或质量选图。',
 '最终 common500 共 42000 逻辑位置、14000 源级行；历史 30000 行合法复用。新 12000 逻辑位置为 D 的 10500 加此前缺失的 7dB WHOLE 1500。物理事件 11223 新执行 + 777 精确复用。历史缺失 CLIP 的 2188 对补单指标；其余历史指标未重跑。实际新接收/指标调用与缓存计数见 actual_compute_and_reuse.json，每个 worker 的 SQLite 账本为准。',
 '', '六、实测部署成本']
 for name in ['original','A','B03','B07','C_content_1000','C_no_content_1000','D_eighth','D_token','E']:
  t=timing_lookup[(name,'TX_total')];r=timing_lookup[(name,'RX_total')]
  lines.append(f"{name}: TX mean/median/p95={float(t['mean_seconds']):.6f}/{float(t['median_seconds']):.6f}/{float(t['p95_seconds']):.6f}s; RX={float(r['mean_seconds']):.6f}/{float(r['median_seconds']):.6f}/{float(r['p95_seconds']):.6f}s; 独立终止均值{float(t['mean_independent_terminations']):.3f}; 额外序列化模型{t['extra_serialized_checkpoint_bytes']} bytes")
 lines += ['同一实际 GPU/精度、2 个固定 CPU 核、batch1、固定16源、每源工作点2暖身+3计时；共享负载，非独占硬件。TX/RX 总时间为匹配样本各组件实测时间之和，不是单次端到端秒表。TX 包含新 tokenizer/概率遍历/独立终止，E 包含候选独立解码、生成、Dc 与 LPIPS；没有离线图像缓存隐藏部署成本。A/B/C 复用完全相同原 TX 的计时组件。B 计时为一张输出，质量均值用三个生成重复，分别列示。',
 'A 同时保存原 decoder 与锚定 decoder；C 额外 2899456 参数/11597824 FP32 state bytes；E 需 AlexNet + LPIPS 标定权重。序列化训练 checkpoint 大小含优化器等，不等于部署参数大小；峰值 allocated/reserved 和 state bytes 完整列于 CSV。加载时间不计稳定部署延迟，但计入资源预算。',
 '', '七、资源、失败与未完成项',
 f"累计 GPU 设备墙时 {budget['total_GPU_device_hours']:.6f} h；最多同时 {budget['peak_owned_GPU_workers']} 个本轮 GPU worker。各路线均在原额度内，未重置预算或互借；新 PFS 目录 {budget['new_PFS_tree_bytes']} bytes，小于 80 GiB。CPU 进程等待墙时单列，不冒充 CPU 实际计算时间。",
 '原 C_guard_metric_queue 资源竞态拒绝，退出1且零科学调用，后续缺失评分由独立登记入口首次完成。原 benchmark_tx 在编码前把 raw像素hash 与 dtype+shapehash 混比，退出1、只加载模型、零编码计时；32输入逐项核验后新入口只修正断言，失败和20.906秒加载费用保留。历史只读汇总 v1–v4 的 schema 差异失败保留；修订没有重跑已有接收/评分。',
 'NEW200 无原始合格源池，独立新留出与其联系表明确 blocked。Pastries95/Dog21 的完整旧 source ID 不在 common500；不能误用另一个集合的相同整数索引。未擅自补开展示重建。C 机制图使用预先固定的 FIT hash顺序前2例，属于开发诊断，不称代表性或独立测试。',
 '所有运行已实际退出；没有以运行中快照冒充完成。checkpoint/缓存/原生大数组和日志仍在 PFS。恢复清单保留 C 权重、优化器、RNG、源和配置身份，但不会自动恢复未通过保护线的训练。',
 '', '八、交付与解释边界',
 '结果支持在本冻结上限/MCS和模型下进一步细化装入；不能推广为逐图单调改善、最优粒度或已证实独立留出优势。A/B/C/E 的 NO_GO 是本配置和预算下的证据，不是方向永久无效。',
 '本轮不改旧 V6 论文/旧主图，不恢复此前停止支线，不自动 commit/push。代码、配置、全部候选/逐帧/逐源/配对数据、图形、资源账本、工程失败与 blocked 状态提供为可审查轻量包。仓库检查及实际图像审阅另附 validation。']
 text(out/'REPORT_zh.txt','\n'.join(lines)+'\n')
 text(out/'README_zh.txt','本轮五路线实际结果入口\n\n先读 REPORT_zh.txt 与 stage_completion.json。A/B/C/E未通过，D通过FIT/CONFIRM并完成common500回归；NEW200受阻，不能宣称独立新留出完成。\nroute_summary.csv保留全部候选；per_source_metrics.csv/paired_deltas.csv显式标注FIT、CONFIRM、common500，禁止跨样本混算。原逐帧数据在各fit_*_delivery、confirm_D_delivery、final_common500_delivery/frame_metrics.csv。\nconfiguration_index.json连接冻结配置/身份；actual_compute_and_reuse.json与资源CSV记录真实执行。figures为可编辑SVG、PDF、PNG；原生数组位于PFS，图源manifest含完整hash。\n本轮没有自动commit/push；原V6未改写。验证结果见validation，完整待办/阻塞见stage_completion.json。\n')
 save(out/'claim_evidence.json',dict(final_configuration='original_receiver_plus_D_token',development_confirmation=pin(D/'confirm_D_delivery/gate.json'),all_negative_candidates=pin(out/'route_summary.csv'),regression=pin(final/'statistics_identity.json'),new_holdout_completed=False,DreamSim_confirm_CI_excludes_zero=False,not_SOTA_claim=True,not_per_image_monotonic=True,not_optimal_granularity=True,not_exclusive_hardware_timing=True,C_guard_failure_not_hidden=True))
 print('DELIVERY_ASSEMBLED',len(route),len(source),len(paired))

if __name__=='__main__':main()
