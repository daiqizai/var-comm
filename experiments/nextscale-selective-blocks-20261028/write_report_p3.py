"""Assemble the closed P0–P3 delivery from exact saved results and receipts."""
from block_common import *
import csv,collections,difflib
def rows(name):return list(csv.DictReader((O/name).open()))
def main():
 assert read(O/'p3_final_audit.json')['status']=='PASS'
 c=read(O/'p3_closure.json');nom=read(O/'nomination.json');overall=rows('p3_candidate_overall.csv');paired=rows('paired_mean_differences.csv');mech=rows('selective_mechanism_contribution.csv');physical=rows('p3_physical_events.csv');cat=read(O/'candidate_catalog.json')
 budgets=[]
 for folder in sorted(L.glob('*_control')):
  p=folder/'actual_wait.json'
  ps=[p]if p.exists()else sorted(folder.glob('*.actual_wait.json'))
  for p in ps:
   w=read(p);r=cr(w['request'])if 'request'in w else {};gpu=bool(r.get('environment',{}).get('CUDA_VISIBLE_DEVICES',''))
   budgets.append(dict(role=folder.name,receipt=pin(p),seconds=w['seconds'],GPU=gpu,returncode=w['returncode']))
 for p in [O/'p0_engineering/actual_wait.json',*sorted((L/'p3_launch_once').glob('*.actual_wait.json'))]:
  w=read(p);budgets.append(dict(role=p.parent.name,receipt=pin(p),seconds=w.get('seconds',w.get('elapsed_seconds',0)),GPU=False,returncode=w['returncode']))
 gh=sum(r['seconds']for r in budgets if r['GPU'])/3600;ch=sum(r['seconds']for r in budgets if not r['GPU'])/3600;assert gh<32 and ch<32
 rx=read(L/'p3_replay_v2/rx_replay.json');decode_reuses=sum(len(r['result']['reused_decode_keys'])for r in rx)
 resource=dict(status='WITHIN_FROZEN_LIMITS',gpu_device_hours=gh,cpu_worker_hours=ch,limit_each_hours=32,measurement='Owner/process elapsed wall time; GPU stages classified as device-hours, CPU-only stages as worker-hours; scientific/control/audit receipts through final image audit and figure export. Repository publication checks are separate engineering work.',rows=budgets,P3_actual=c['actual_counts'],P3_decode_lookup_reuses=decode_reuses,P3_unique_decoder_input_keys=384,P3_native_images_reused=286,P3_source_image_pairs_reused=318,P3_metric_inference_calls=0,P3_render_calls=0,P3_prior_calls=0)
 save(O/'resource_accounting.json',resource)
 costs={(r['source_id'],int(r['m']),int(r['K'])):r for r in rows('source_lengths.csv')};frame_cost=[]
 for r in physical:
  p=cat['profiles'][r['tx_profile_id']];extra=0
  if p['kind']=='THREE'and not p['null']and p['K']>0:extra=int(costs[r['source_id'],p['m'],p['K']]['AC_difference'])
  frame_cost.append(dict(event_id=r['event_id'],candidate_id=r['candidate_id'],source_index=r['source_index'],profile_id=p['profile_id'],N=1024,Es=2,energy=2048,source_bits=int(r['tx_source_bits']),information_bits=int(r['tx_information_bits']),syntax_bits=29*len(p['layouts']),extra_syntax_vs_single=58 if p['kind']=='THREE'else 0,independent_AC_increment_vs_equivalent_prefix=extra if not p['null']else'',actual_m=p['m'],actual_K=p['K'],paid_NULL=p['null'],source_capacity_utilization=float(r['source_capacity_utilization'])))
 csvsave(O/'p3_frame_accounting.csv',frame_cost)
 layouts=read(L/'p2_layout_qualification/layouts.json');csvsave(O/'phy_layouts.csv',[dict(layout=r['key'],**r['geometry'],roundtrip_exact=r.get('roundtrip_exact',False),backend=json.dumps(r['backend'],sort_keys=True))for r in layouts])
 save(O/'codec_proof.json',dict(status='PASS',source_independence=pin(O/'codec_source_proof.json'),full_CAB_pixels=pin(O/'render_alignment_proof.json'),P1_gate=pin(O/'p1_gate.json'),actual_frame_replay=pin(L/'p3_replay_v2/completion.json'),saved_output_audit=pin(O/'p3_final_audit.json'),unknown_or_wrong_accepted_content=0,new_RX_source_oracle=False))
 correction='''# P3 RX cache identity correction

The first CPU-only cache preparation stopped before saving a cache or processing a P3
reception. It asserted that every `preprocessing` field was identical. This field
actually records SHA256 of each source's preprocessed uint8 pixels, as defined by
`experiments/nestcom-n512-20261028/prepare_assets.py` and `nest_source_worker.py`.
It is a reference-image identity, not a numerical image-processing pipeline version.

The original p3_rx_replay.py, failure.json and exit-1 receipt remain unchanged.
Independent p3_rx_replay_v2.py indexes already verified received-token states by
all renderer inputs: coarse tokens, positions, values, model hashes, numeric route,
completion settings and endpoint. The original full state and its source pixel
hash remain stored and hash-checked. Collisions must have identical native pixels.
The renderer never reads the `preprocessing` label or source pixels; it consumes
the actual tokens and native model state. This is verified against the preserved
renderer source, not assumed from equal token counts.

No source ID or expected TX tokens enter the replay function. Unknown accepted
wire bits are preserved as pending native RX work, never replaced with expected
tokens or gray. There were zero such misses in the observed P3 events. Native
images are looked up only after exact independent-RX stream memoization. Actual
source/image metric pairs are joined after saved RX; original reference hashes
and all 286 used native arrays were independently verified. No PHY, successful
source decode, image generation or metric inference was repeated by this repair.
'''
 (O/'CACHE_IDENTITY_CORRECTION.md').open('x',encoding='utf8').write(correction)
 diff=''.join(difflib.unified_diff((E/'p3_rx_replay.py').read_text().splitlines(True),(E/'p3_rx_replay_v2.py').read_text().splitlines(True),fromfile='p3_rx_replay.py',tofile='p3_rx_replay_v2.py'))
 (O/'p3_cache_revision.diff').open('x').write(diff)
 receipts=O/'p3_closed_receipts';receipts.mkdir(exist_ok=False)
 role_names=['p2_nomination','p3_phy_00','p3_phy_01','p3_phy_02','p3_phy_03','p3_replay','p3_replay_v2','p3_summary','p3_final_audit','p3_figures']
 for role in role_names:
  for folder in [L/role,L/(role+'_control')]:
   if not folder.exists():continue
   for p in folder.iterdir():
    if p.is_file()and(p.name in ('failure.json','completion.json','registration.json','actual_wait.json')or p.name.endswith('.actual_wait.json')):
     (receipts/(folder.name+'__'+p.name)).open('xb').write(p.read_bytes())
 for p in (L/'p3_launch_once').glob('*.actual_wait.json'):(receipts/('p3_launch_once__'+p.name)).open('xb').write(p.read_bytes())
 best=c['best_measured_SEL'];single=c['best_measured_single'];cb=sum(int(r['C_success_A_fail_B_success'])for r in mech if r['metric']=='dino_cosine');best_delta=next(float(r['dino_cosine'])for r in paired if r['candidate_id']==best['candidate_id']and r['contrast']=='SEL_minus_SEQ'and r['snr_db']=='overall')
 lines=['# 同尺度分块保护与选择性恢复：P0–P3 完整交付','',f"结论：**{c['recommendation']}**。源码独立恢复和 C+B 干净状态价值成立；在本次预定候选、固定 TX 策略与 0/2/4 dB 等权分布下，尚未补偿分包与短码块成本。P4/P5 不启动；原 WCL 与旧负结果保持不变。",'',
 '## 先回答六个问题','',
 '1. **删除 A 后，B 能否独立恢复？能。** 128/128 删除 A 的实收源码恢复逐 token 精确一致；128/128 完整往返通过；删除 C 时不调用先验恢复 B。128 个全 CAB 原生图像与等价连续前缀逐像素相同。',
 '2. **C+B 是否有质量价值？有平均价值，但不保证逐图改善。** 四个预定配置的平均 DINO(C+B)−DINO(C) 为 +0.053907、+0.030066、+0.035832、+0.061830；所有逐源负值与其他四指标保留。',
 f'3. **真实信道中是否出现 C+B？出现 {cb}/3168 个三块物理帧（{100*cb/3168:.4f}%）。** 这是 11 个候选各 288 帧的描述性合计，不能当作一个部署策略的发生率。逐候选/逐 SNR 见 state_counts.csv。',
 f'4. **SEL−SEQ 的贡献多大？依候选而定，存在正值、零值和负值。** 本轮最佳 SEL 候选的总体 ΔJ_SEL−SEQ = {best_delta:+.12f}；下面报告全部候选。逐源、逐噪声用同一波形回放，频率×实际 C+B−C 的分解恒等式逐项通过。',
 f"5. **是否胜过重新校准单块？没有。** 最佳已测 SEL J={best['dino_cosine']:.15f}，最佳已测单块 J={single['dino_cosine']:.15f}，差 {c['delta_J_best_SEL_minus_best_single']:+.15f}。单块最优只指四个实际提名候选，不声称全部 100 候选均完成无线最优化。",
 '6. **是否进入 P4？不建议自动推进。** 没有候选同时满足 J_SEL>最佳已测单块 且 J_SEL>自身 SEQ。按原门槛停止，不扩大尺度/位置/码率/信道/功率，不训练。','',
 '## 固定范围与统计口径','',
 '- 内容无关选取 cal1000 内 32 源，保留 source_id/source_index；不是 common500 正式测试，也不是独立盲测。',
 '- TX 名义 2 dB，同一候选跨实际 0/2/4 dB；每源每点三次噪声。先平均每源三次噪声，再等权源图，最后等权三个 SNR。J 始终为 DINOv2-L cosine；PSNR↑、LPIPS-Alex↓、DreamSim↓、CLIP-I↑同步完整报告。',
 '- 本轮没有新增 bootstrap/CI/显著性检验。全部差值是来源配对描述量；候选间使用各自登记噪声，ALL/SEQ/SEL 则严格共用同一完整接收记录。',
 '- N=1024=68+956，Es=2，总能量2048。头部为实际 ±(1+j) 两点映射；正文 QPSK。每块 L13+padding+CRC16，独立算术 finish 和 LDPC。',
 '- 120 三块+100 单块全部参加代理筛选。按固定分层规则选出 11 三块+4 单块，tie 采用 candidate_id 升序。代理独立包假设只用于提名，不作为完整无线结果。','',
 '## 所有实际提名候选','',
 '|候选 ID|J proxy|J 实测 SEL/单块|ΔJ SEL−SEQ|C+B 帧数 / 288|','|---|---:|---:|---:|---:|']
 for n in nom['selected']:
  rr=next(r for r in overall if r['candidate_id']==n['candidate_id']and r['mode']==('SEL'if n['kind']=='THREE'else'SINGLE'))
  if n['kind']=='THREE':d=next(r['dino_cosine']for r in paired if r['candidate_id']==n['candidate_id']and r['contrast']=='SEL_minus_SEQ'and r['snr_db']=='overall');ct=sum(int(r['C_success_A_fail_B_success'])for r in mech if r['candidate_id']==n['candidate_id']and r['metric']=='dino_cosine');delta=f'{float(d):+.9f}'
  else:ct='—';delta='—'
  lines.append(f"|{n['candidate_id']}|{n['J_proxy']:.9f}|{float(rr['dino_cosine']):.9f}|{delta}|{ct}|")
 lines+=['','目标端点不同的候选可能按实际码长退到同一配置，但固定事件身份和噪声不同，仍各报告其 288 帧。不能将这些小差异解读为目标端点本身的收益。','',
 '## 最佳已测单块与最佳 SEL 的五指标','',
 '|指标|最佳单块|最佳 SEL|SEL−单块|','|---|---:|---:|---:|']
 for k in ['dino_cosine','psnr_db','lpips_alex','dreamsim','clip_image_cosine']:lines.append(f"|{k}|{single[k]:.9f}|{best[k]:.9f}|{best[k]-single[k]:+.9f}|")
 lines+=['',f"单块：`{single['candidate_id']}`；三块：`{best['candidate_id']}`。两者均按原 J 选择，未按报告指标换目标。",'',
 '![所有候选和最佳已测策略](figures/p3_candidate_comparison.png)','![全部三块候选的机制贡献](figures/p3_selective_mechanism.png)','',
 '## 失败、成本与实际计算','',
 '- P2：28 个新几何各一次无噪声往返通过，另4个头部工程例；10,752有噪正文包+384头部，全部原记录保留。',
 '- P3：4,320完整物理帧 / 10,656实际发送正文块 / 4,320实际头部译码 / 9,029实际正文译码尝试。其余正文仍发送并计费，因接收头部失败或合法 NULL 而不译码。',
 '- 每候选/SNR完整分母96，每候选总288。三块 ALL/SEQ/SEL 产生9,504逻辑回放，加单块1,152，共10,656指标记录；这些不是新增物理帧。',
 '- 3次头部拒收、540个合法付费NULL、673次非NULL粗块/单块CRC拒收；无通过校验却内容错误的头部或正文。全部计入质量分母。A/B 的CRC拒收、条件不可用、付费空块逐帧分开保存。',
 '- 原完整四配置源开销：固定字段比单块多58bit；实测独立AC额外0–4bit；合计58–62bit，平均60.21875bit。真实回退后的逐帧码长、字段与利用率另见 p3_frame_accounting.csv，NULL从未静默省能量。',
 '- P1 已完成32概率遍历、0 tokenizer；正常状态928，包含247旧原生接收复用、553新连续状态、128新C+B；另128次全CAB对齐验证。新decoder总809次，正常状态与技术验证分开记账。',
 f'- P3 独立接收缓存按实际收到的码流与粗尺度依赖精确复用 {decode_reuses} 次，覆盖384种实际解码输入身份；复用286张原生图和318个源图—重建评分对（含32灰图对）。P3新增VAR/decoder/指标推理均0，未把TX真值补给RX。',
 f'- 实测科学/控制/审计资源累计：GPU {gh:.6f}设备小时，CPU {ch:.6f}worker小时，均低于各32小时固定上限。详细逐阶段等待记录见 resource_accounting.json；仓库工程检查单列，不是科学试验。','',
 '## 实现修订与验收','',
 '首次P3 CPU缓存准备错误地假定 source pixel hash 是统一预处理流程ID，执行在任何P3接收回放前退出1。原脚本/失败/哈希保留；独立v2修订依据实际接收内容查缓存，原full-key不改，碰撞要求像素相同。详见 CACHE_IDENTITY_CORRECTION.md 和最小差异。未重跑任何成功科学调用。',
 '验收通过：540个完整观测批次/4320帧和9029正文hard数组哈希一致；53,280个指标数值与既有精确评分对一致；286原生重建、32原图哈希实际核验；37个候选×模式总体均值与逐帧重算一致。P1全CAB像素一致与删除A/B源码证明保持有效。','',
 '## 文件与复现','',
 '- `nomination.json`、`p2_proxy_all_candidates.csv`：220候选和全部固定提名原因；`p2_actual_length_fallback.csv`保留逐源回退。',
 '- `p3_physical_events.csv`：真实帧、接收状态、回退与错误审计；观测大数组留在登记PFS路径。',
 '- `replay_metrics.csv`、`p3_source_means.csv`：全部回放与来源聚合；`p3_candidate_snr_summary.csv`和`p3_candidate_overall.csv`为五指标完整结果。',
 '- `paired_source_differences.csv`、`paired_mean_differences.csv`：SEL−SEQ及SEL−最佳已测单块的直接来源配对差。',
 '- `state_counts.csv`、`selective_mechanism_contribution.csv`：完整分母、机制频率×实际质量贡献。',
 '- `codec_proof.json`、`phy_layouts.csv`、`p3_final_audit.json`、`p3_closed_receipts/`：资格与真实闭合证据。',
 '- `figures/`：两个结果图的PDF/SVG/PNG、输入哈希和候选标签映射；`experiments/nextscale-selective-blocks-20261028/plot_p3.py`可复现。',
 '- 历史P1报告、原scope/协议/登记保持原字节。权重、原图、token/数组、环境与大日志不进入Git。','',
 '本轮所有 P0–P3 科学步骤已完成，无待运行科学项；P4/P5按授权边界和实际NO_GO停止。最终仓库检查、视觉检查和推送回执由交付时的独立验收记录确认。','']
 (O/'REPORT_zh.md').open('x',encoding='utf8').write('\n'.join(lines))
 (O/'README.md').write_text('# Same-scale selective blocks: completed P0–P3\n\n[完整结果与负结果结论](REPORT_zh.md) · [协议](../../experiments/nextscale-selective-blocks-20261028/PROTOCOL.md)\n\n4,320 full frames completed. NO_GO_STOP; P4/P5 not run. See p3_closure.json, p3_final_audit.json and resource_accounting.json. Historical REPORT_P1.md remains an immutable milestone.\n',encoding='utf8')
 print(json.dumps(dict(report=str(O/'REPORT_zh.md'),gpu_hours=gh,cpu_hours=ch,decode_cache_reuses=decode_reuses,CB_frames=cb,best_SEL_minus_SEQ=best_delta)))
if __name__=='__main__':main()
