"""Make a reader report from closed CSVs; no model, metric or resampling calls."""
from pathlib import Path
import argparse,csv,hashlib,json,math,time

METRICS=('dists','dreamsim','psnr_db','lpips_alex')
BUDGETS=(384,512,640,768,1024,1280,1536,2048,2560,3072,4096)

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def ci(r,digits=6):
    return f"{float(r['mean']):+.{digits}f} [{float(r['ci_low']):+.{digits}f}, {float(r['ci_high']):+.{digits}f}]"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--delivery',required=True);a=ap.parse_args()
    O=Path(a.delivery).resolve()
    ss=rows(O/'source/summary.csv');sp=rows(O/'source/paired_deltas.csv');cap=rows(O/'source/capacity.csv')
    cs=rows(O/'control/expected_summary.csv');cp=rows(O/'control/expected_paired_deltas.csv')
    policies=rows(O/'control/frozen_reference_policies.csv');rel=rows(O/'control/reliability.csv')
    assert (len(ss),len(sp),len(cs),len(cp))==(136,88,48,24)
    for data in (ss,sp,cs,cp):
        assert all(int(r['sources'])==500 for r in data)
        for r in data:
            assert all(math.isfinite(float(r[k])) for k in ('mean','ci_low','ci_high'))
            assert float(r['ci_low'])<=float(r['ci_high'])
    assert [int(r['budget_bits']) for r in cap]==list(BUDGETS)
    assert all(int(r['hard_budget_violations'])==0 and int(r['sources'])==500 for r in cap)
    keys=[(r['budget_bits'],r['policy'],r['metric']) for r in ss]
    assert len(keys)==len(set(keys))
    by={(int(r['budget_bits']),r['policy'],r['metric']):r for r in ss}
    paired={(int(r['budget_bits']),r['contrast'],r['metric']):r for r in sp}
    ctrl={(int(r['N']),int(r['snr_db']),r['metric']):r for r in cp}
    for b in BUDGETS:
        for p in ('longest_feasible','calibrated_target','source_informed_DISTS_bound'):
            assert all((b,p,m) in by for m in METRICS)
    raw=json.loads((O/'control/complete.json').read_text())
    figure=json.loads((O/'figures/figure_identity.json').read_text())
    assert figure['plotted_mean_and_CI_records']==66
    visual=json.loads((O/'validation/visual_review.json').read_text())
    assert visual['actual_opened'] and visual['passed']
    for name,h in visual['artifact_sha256'].items():assert sha(O/name)==h
    main=[
        '# RDVQ P0/P1 实际码流与控制参考：闭合报告',
        '',
        '状态：P0、P1 已真实完成；P2 匹配训练正在运行。本报告不是 P2 有效性的结论，也不是整个首批任务的完成声明。旧 NeST-Com WCL、旧实验及暂停支线保持原状态。',
        '',
        '## 1. 本轮回答了什么',
        '',
        '公开 RDVQ 权重在独立终止的原生 slice 前缀上能够完成实际码流发送、独立接收及缺失索引补全。真实 source-packet 长度随预算形成有效质量曲线；500 个固定测试源在 3072/4096 bit 已全部达到完整表示，余量属于表示饱和。',
        '',
        '以实测可靠性建立的逐图参考控制器只带来很小的 DISTS 改善，PSNR 均值在六个工作点均下降，DreamSim/LPIPS 也有不利结果。这不足以支持增加复杂学习控制器；后续仍按预登记的 P2 比较真实比特预算训练与普通 token 前缀训练。',
        '',
        '## 2. 模型、数据与口径',
        '',
        '- RDVQ 上游固定为 `a2744eedd803d9e20b55b4b0c935ec211a07d184`；共同权重 `lam_4.8_tau_0.1.pt`，SHA-256 `827aaab038626bb306fc30fac37634eece418dd02bd4617f88e65d8d19e92294`。',
        '- 原生 4×4、8×8、16×16 三尺度、336 个位置、28 个 causal slice。top-k/escape tensor-rANS，K=1024、CDF precision=16；FP32、TF32 关闭、math SDPA、确定性 argmax 补全。接收端只读取实际包与公共模型，不读取未传真值 suffix。',
        '- 72-bit 源包封装包含版本/端点、两个码流长度及 CRC32。rANS 状态已包含在实际流长中。NULL 源包仍收费 72 bit 并输出固定灰图。图中横轴是允许的源包预算，实际平均码长单列，二者不混淆。',
        '- 固定 20,000 训练 ID、1,000 开发源、1,000 校准源、500 测试源，JPEG CRC/字节 SHA 与原生像素身份均核验。跨四划分未见精确原生像素重复；训练内保留一对重复（308608/373199），因此 20,000 ID 对应 19,999 唯一原生像素。未作近重复或上游预训练逐图排除认证。',
        '- P1 使用前 128 个固定开发源、全部 1,000 校准源、500 测试源。新源编码 1,620 次，精确复用 P0 的 8 次。新增独立非 NULL 重建 45,360 次、灰图 1,620 个；复用 P0 的 224+8 个输出。四指标共 47,212 对，46,980 对新增评分、232 对精确复用。',
        '- source-level 区间使用固定 500 源顺序和 10,000 次来源 bootstrap；配对差异直接从相同源计算，不相减边际区间。136 个均值区间、88 个源编码配对区间，以及 48 个控制期望均值/24 个直接配对区间已闭合。',
        '',
        '## 3. R0 真实源包预算曲线',
        '',
        '下表为 longest-feasible 规则；DISTS、DreamSim、LPIPS 越低越好，PSNR 越高越好。全部 500 源保留。完整均值区间见 [source/summary.csv](source/summary.csv)，配对结果见 [source/paired_deltas.csv](source/paired_deltas.csv)。',
        '',
        '| 预算 bit | 实际均值 bit | 完整表示 /500 | DISTS | DreamSim | PSNR dB | LPIPS |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in cap:
        b=int(r['budget_bits']);v=[float(by[(b,'longest_feasible',m)]['mean']) for m in METRICS]
        main.append(f"| {b} | {float(r['mean_packet_bits']):.3f} | {r['full_count']} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.4f} | {v[3]:.6f} |")
    main += [
        '',
        '全部 11 个预算的实际长度检查均为零超预算、零 NULL。完整测试包长度为 576–2856 bit，均值 2066.416 bit。3072/4096 bit 的质量相同不能解读为拟合失败。640/1280/2560 bit 是 P2 未见训练预算诊断点；此处尚未训练。',
        '',
        '校准集固定目标在 384/640/768 bit 分别为 15/22/25 slices，其余为 28；测试上并不持续优于最长可行规则。逐图 DISTS 参考在 TX 使用源图及所有可行候选重建，是昂贵的源参考上限，不是已训练的低成本预测器，也不观察未来噪声。它在 384 bit 的 DISTS 差值较明显，但 PSNR 下降，其他预算上的收益很小且存在感知指标取舍：',
        '',
        '| 预算 bit | DISTS 参考−校准目标 | DreamSim 参考−校准目标 | PSNR 参考−校准目标 |',
        '|---:|---|---|---|',
    ]
    for b in (384,1024,2048,4096):
        contrast='source_informed_DISTS_bound minus calibrated_target'
        main.append('| '+str(b)+' | '+' | '.join(ci(paired[(b,contrast,m)]) for m in METRICS[:3])+' |')
    main += [
        '',
        '图： [PDF](figures/p1_actual_packet_curves.pdf) · [SVG](figures/p1_actual_packet_curves.svg) · [600 dpi PNG](figures/p1_actual_packet_curves.png)。两个面板、三个规则、11 个预算，共 66 个均值及 66 个区间；181.864×71.12 mm。色彩、灰度与实际 PDF 渲染均已打开检查，见 validation/visual_review.json。',
        '',
        '## 4. 可靠性测量与控制参考',
        '',
        '新动态 PHY 使用 64-symbol protected header（38 个信息位：版本、MCS、源包字节长度、CRC16；LDPC 后 128 bit/QPSK），body 为 N−64 个符号。源包的 72-bit 包装另计；不沿用旧 NeST 的 68-symbol header 或 29-bit body 字段。',
        '',
        '校准网格为 N=512/1024、SNR=4/10/19 dB、12 个 MCS、14/28-slice 目标、100 个固定校准源，共 14,400 个实际无线帧。8,381 个接受包均精确匹配独立接收缓存，6,019 个拒收事件保留灰图；未知接收包与错误接受包均为零。没有新增测试 PHY 或从 TX 真值补 RX。',
        '',
        'C0 在独立校准集选择 SNR-only 策略；C1 利用源图及实测成功概率选择参考动作。下列测试量是“条件于实测可靠性表的期望质量”，不是测试无线帧实测均值。来源 bootstrap 不包含可靠性表估计误差，后者由 reliability.csv 中 Wilson 二项区间单列。所有选定 C0 动作均在 100 次校准测量中成功；其 FER 95% Wilson 上限仍为 0.0369935，不写成真实 FER 为零。',
        '',
        '| N | SNR dB | C0 MCS | 目标 slice |',
        '|---:|---:|---|---:|',
    ]
    mcs={1:'QPSK 2/3',5:'16-QAM 2/3',6:'16-QAM 3/4',11:'64-QAM 5/6'}
    for r in policies:main.append(f"| {r['N']} | {r['snr_db']} | {mcs[int(r['mcs'])]} | {r['target']} |")
    main += ['', '| N / SNR | C1−C0 DISTS [95% CI] | C1−C0 DreamSim [95% CI] | C1−C0 PSNR dB [95% CI] |', '|---|---|---|---|']
    for r in policies:
        n,g=int(r['N']),int(r['snr_db'])
        main.append(f'| {n} / {g} | '+' | '.join(ci(ctrl[(n,g,m)]) for m in METRICS[:3])+' |')
    main += [
        '',
        'LPIPS 六个均值差也都为正（不利），完整四指标及区间保留于 control/expected_paired_deltas.csv。C1 大量动作变化只改变并列的 MCS，并未改变源前缀或成功概率；动作改变率不是质量收益。N1024/19 dB 中 474/500 个动作改变，仅 29 个前缀不同、零个成功概率改变，见 decisions/。当前结果不支持为该初始化额外训练复杂控制器。此判断不排除 P2 表示学习可能改变控制器可获收益。',
        '',
        '## 5. 率代理、开销与实际预算',
        '',
        '初始固定 128 开发源的 teacher-forced 硬索引率代理，每个端点的实际−估计中位数约为 −8.55 至 −4.03 bit。1024-bit 端点匹配由未修正 113/128 提升到同批审计修正后的 122/128；这是同批校准诊断，不是独立测试泛化结论。1536-bit 修正后仍有 1/128 个代理选择超预算，最终真实字节检查及回退保持零硬违反。',
        '',
        '45,584 个已有相邻非空端点长度关系中没有下降，12,343 个相等。这是本次缓存的经验结果，不是 tensor-rANS 单调性证明；实际适配器仍逐候选检验，不借用旧 NeST 的算术编码长度公式。',
        '',
        '384-bit 最长可行点的 72-bit 包装平均占实际包 24.41%，加上 rANS 原生状态约 45.54%；1024-bit 时约为 7.47%/14.10%。因此低预算下终止与包装确有明显开销，当前全部方法一致收费，未按测试质量切换封装。精确分布及各预算数值见 rates/。',
        '',
        '## 6. 资源、失败与后续边界',
        '',
        '- P0 总收费 0.08347030765 GPU-device-hour，包含失败零科学尝试和 CUDA 可见性探针；P1 源链收费 7.00941781839 GPU-device-hours。GPU 角色和 CPU PHY 独立列账；父进程 CPU 包含子进程，不与叶角色重复相加。',
        '- 14,400 个 PHY 事件合计约 769.618 CPU-process-seconds；控制分析 4.600 CPU-process-seconds。源码分析/报告/绘图额外 CPU 见 resource_receipts.csv 及独立报告回执，不算 GPU 节省。',
        '- 旧网络下载失败、Python 头文件缺失、NVML/CUDA UUID 路径不匹配、原确定性 backward 资格失败和指标导入路径失败均保留。独立修订经过精确缺口核验，未重跑成功科学调用。完整 22,500 源校验已通过；原传输失败不能被抹去，也不等于最终数据仍缺失。',
        '- P2 已按共同初始化、数据、损失、4000 步、global batch16 注册 10 个训练臂：R1 一个种子，R2/R3/R3-estimated 各三个。固定 2000 步开发检查与 4000 步开发/测试评估；全状态滚动保存每 50 步。训练数值恢复按 1e−6 容差验证，不承诺原生 CUDA bicubic backward 逐位确定。',
        '- P2 190 GPU-hour 预留低于阶段 256-hour 上限，不是已消耗值。实际结果必须通过 R3−R2 与 R3−R3-estimated 匹配对比；不能仅胜冻结 R0 就宣布机制有效。当前没有 P2 质量结论、p95 正式时延、跨数据集迁移或 P3/P4 完成声明。',
        '',
        '复现入口：[配置/运行解释](../../../experiments/rdvq-prefix-20261104/reporting/REPRODUCING.md)、[源包/独立 RX 实现](../../../experiments/rdvq-prefix-20261104/README.md)、[数据身份](source_identities.csv)、[原始 PFS 证据索引](raw_evidence_index.json)、[失败记录](retained_failures.json)。权重、数组、源码包缓存与大日志仅在 PFS，未提交 Git。',
        '',
    ]
    report=O/'REPORT_zh.md';assert not report.exists();report.write_text('\n'.join(main),encoding='utf8')
    audits=dict(created=time.time(),passed=True,source_summary_rows=len(ss),source_paired_rows=len(sp),control_summary_rows=len(cs),control_paired_rows=len(cp),
        sources_per_statistic=500,budget_bits=BUDGETS,hard_budget_violations=0,figure_means=66,figure_CIs=66,
        no_new_model_metric_or_statistical_calls=True,control_scope='conditional expected quality, not test PHY',P2='ACTIVE; no efficacy conclusion',
        input_sha256={str(p.relative_to(O)):sha(p) for p in [O/'source/summary.csv',O/'source/paired_deltas.csv',O/'source/capacity.csv',O/'control/expected_summary.csv',O/'control/expected_paired_deltas.csv']},
        report_sha256=sha(report),visual_review_sha256=sha(O/'validation/visual_review.json'))
    with (O/'validation/data_review.json').open('x') as f:json.dump(audits,f,indent=2)
    print(json.dumps(dict(report=str(report),passed=True,source_rows=len(ss),control_rows=len(cs))))

if __name__=='__main__':main()
