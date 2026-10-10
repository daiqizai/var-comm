"""Read-only N1024 delivery index; no scientific imports or mutable old outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit("Output must be new; sealed output is never overwritten")
    base = root / "results/wcl_evidence_closure_20261009"
    scriptdir = root / "experiments/wcl-evidence-closure-20261009/scripts"
    original = root / ".research/main_raw64_20261007/take_over_v1/final_publication_r6/actual_staging_r6/results/main_raw64_20261007/final_common500_r6"
    pins = {
        "summary.csv": "5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859",
        "paired.csv": "dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235",
    }
    for n, h in pins.items():
        assert sha(original / n) == h, n
    groups = [
        ("T1", "T1_entropy_whole/v3", "completion.json", "outputs"),
        ("T2", "T2_calibration_audit/v1", "completion.json", "outputs"),
        ("T3", "T3_swin_sideinfo", "completion.json", "outputs"),
        ("T4_current", "T4_resources_cost/summary_completed_v2", "completion.json", "outputs"),
        ("T4_external", "T4_resources_cost/external_timing_reuse_v1", "completion.json", "outputs"),
        ("T5_original", "T5_figures", "DELIVERY_MANIFEST.json", "files"),
        ("T5_external", "T5_figures/external_completed_v1", "DELIVERY_MANIFEST.json", "files"),
        ("T5_mechanism", "T5_figures/mechanism_entropy_completed_v2", "DELIVERY_MANIFEST.json", "files"),
        ("T5_entropy", "T5_figures/entropy_quality_completed_v1", "DELIVERY_MANIFEST.json", "files"),
        ("T5_cost", "T5_figures/quality_tx_cost_completed_v1", "DELIVERY_MANIFEST.json", "files"),
    ]
    verified, provenance, inventory = [], {}, []
    for label, rel, sealname, field in groups:
        folder = base / rel
        seal = folder / sealname
        d = json.loads(seal.read_text(encoding="utf-8"))
        assert isinstance(d[field], dict) and d[field]
        paths = []
        for name, binding in d[field].items():
            normalized = name.replace("\\", "/")
            if normalized.startswith("/home/") or ":/" in normalized:
                p = folder / normalized.rsplit("/", 1)[-1]
            else:
                p = folder / normalized
            h = binding["sha256"] if isinstance(binding, dict) else binding
            assert p.is_file() and sha(p) == h, str(p)
            if isinstance(binding, dict) and "bytes" in binding:
                assert p.stat().st_size == binding["bytes"], str(p)
            paths.append(p)
            verified.append({"group": label, "sealed_by": str(seal), "local_path": str(p), "sha256": h, "bytes": p.stat().st_size})
        provenance[label] = {"directory": str(folder), "seal": str(seal), "seal_sha256": sha(seal), "status": d.get("status"), "verified_output_count": len(paths)}
        for p in sorted(set(paths + [seal])):
            inventory.append({"group": label, "repo_relative_path": p.relative_to(root).as_posix(), "local_absolute_path": str(p), "sha256": sha(p), "bytes": p.stat().st_size, "type": p.suffix.lstrip("."), "remote_repo_path": "/home/liulu/projects/VAR_COMM/" + p.relative_to(root).as_posix()})
    # Original statistics and selected report extracts keep the CSV's numeric strings.
    raw = [r for r in read_csv(original / "paired.csv") if r["method"].startswith("RAW64_PARTIAL_VAR_COMPLETION_SNR_") and r["reference"] == r["method"].replace("PARTIAL", "WHOLE") and r["metric"] in ["psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "convnext_top1_source_prediction"]]
    ec = [r for r in read_csv(base / "T1_entropy_whole/v3/paired.csv") if r["method"].startswith(("EC_STATIC_WHOLE_SNR_", "EC_VAR_WHOLE_SNR_")) and r["reference"] == "RAW64_PARTIAL_VAR_COMPLETION_SNR_" + r["snr_db"]]
    assert len(raw) == 24 and len(ec) == 24
    for r in raw + ec:
        assert r["source_count"] == "500" and r["noise_count"] == "3"
        assert float(r["ci_low"]) <= float(r["mean"]) <= float(r["ci_high"])
    m9 = [r for r in read_csv(base / "T1_entropy_whole/v3/m9_sendable_summary.csv") if r["population"] == "holdout"]
    cost = [r for r in read_csv(base / "T1_entropy_whole/v3/quality_vs_tx_cost.csv") if r["metric"] == "dinov2_vitl14_cosine"]
    assert len(m9) == 6 and len(cost) == 12
    # Nothing above modifies a result. Create the independent textual delivery only now.
    out.mkdir(parents=True)
    write_csv(out / "verified_files.csv", verified)
    write_csv(out / "delivery_files.csv", inventory)
    write_csv(out / "report_paired_evidence_original_precision.csv", raw + ec)
    write_csv(out / "report_m9_evidence_original_precision.csv", m9)
    write_csv(out / "report_timing_evidence_original_precision.csv", cost)
    def link(path, label=None):
        p = Path(path)
        assert p.exists(), p
        return f"[{label or p.name}](<{p.as_posix()}>)"
    def item(rel, name=None):
        return link(base / rel, name)
    def rowtable(rows, direction):
        lines = ["| SNR (dB) | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2-L | Δprediction agreement (pp) |", "|---:|---:|---:|---:|---:|"]
        for snr in sorted({int(r["snr_db"]) for r in rows}):
            parts = [str(snr)]
            for metric in ["psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "convnext_top1_source_prediction"]:
                matches = [r for r in rows if int(r["snr_db"]) == snr and r["metric"] == metric]
                assert len(matches) == 1
                r = matches[0]
                scale = 100 if metric == "convnext_top1_source_prediction" else 1
                parts.append("{:+.6f} [{:+.6f}, {:+.6f}]".format(*(float(r[k]) * scale for k in ["mean", "ci_low", "ci_high"])))
            lines.append("| " + " | ".join(parts) + " |")
        return direction + "。表内为均值及原有逐点 95% CI；显示舍入，原精度见配套 CSV。\n\n" + "\n".join(lines)
    public = {"EC_STATIC_WHOLE": "固定统计熵编码整尺度 + VAR", "EC_VAR_WHOLE": "VAR 条件熵编码整尺度 + VAR", "RAW64_WHOLE_VAR_COMPLETION": "原始 token 完整尺度 + VAR", "RAW64_PARTIAL_VAR_COMPLETION": "原始 token 部分尺度 + VAR"}
    m9table = ["| 家族 | SNR | 完整 m9 可发送源图 / 500 | 可发送比例 | 实际选中 m9 的源图数 | 净源码容量 (bit) |", "|---|---:|---:|---:|---:|---:|"]
    for r in m9:
        m9table.append(f"| {public[r['family']]} | {r['snr_db']} | {r['m9_fits_frozen_MCS_count']} | {100*float(r['m9_fits_fraction_all_sources']):.1f}% | {r['actual_selected_m9_count']} | {r['source_capacity']} |")
    costtable = ["| 方案 | SNR | TX mean (ms) | TX median (ms) | RX mean (ms) |", "|---|---:|---:|---:|---:|"]
    for r in cost:
        family = r["point_id"].rsplit("_SNR_", 1)[0]
        costtable.append(f"| {public[family]} | {r['snr_db']} | {float(r['tx_mean_ms']):.3f} | {float(r['tx_median_ms']):.3f} | {float(r['rx_mean_ms']):.3f} |")
    now = datetime.now(timezone.utc).isoformat()
    report = f"""# FINAL_REPORT 模板：已验证 N1024 结论，T6 待实际完成

生成时间（UTC）：{now}。状态：**N1024 证据已完成；整个 T0–T6 任务尚未完成。T6 N2048 正在运行，当前文件不得作为最终全部完成报告。** 本独立版本不覆盖任何已封印结果；负责执行的主任务在实际 T6 closure、评分、统计及图片完成后另建最终报告。

科学快照为 `14b09ecd72984fb39c683d1a62bcd3221c69142f`；原 common500 结果来自已发布提交 `252176e041758ecb2d3e81b6fde5b587e7e17bb7`。新增熵码比较为已查看的原 500 源图上的**事后新增同源比较**，策略只在原 calibration1000 按 DINOv2-L 选择。保留原 raw KEEP、模型、Dc、预算及失败定义；新策略不回填旧主方法。

## 1. 加强整尺度对照后，原部分尺度还有多少增量？

T2 对原 433 个独立线协议动作中的全部 106 个整尺度动作进行 100 源图 pilot；随后 10/19 dB 各在 1,000 原校准源图 × 3 噪声复核 10 个整尺度动作及一个固定部分尺度参考，覆盖全部 7 个整尺度满预算保护动作。两个原整尺度赢家均保留，分别为 profile 274 和 270，因此无需另跑其原 500 图测试。此结论仅覆盖本次有限候选审查，不证明全部 106 动作的 full1000 最优或全协议空间最优。{item('T2_calibration_audit/v1/REPORT_T2.md')}

{rowtable(raw, '原始 token 部分尺度 − 原始 token 完整尺度')}

7 dB 四指标增量精确为零；13 dB 的质量增量小，一致率均值为零。4、13、19 dB 的一致率差值区间含零，不能称所有指标均有确定优势。T2 的加强审查仅覆盖 10/19 dB；其他四档仍是原冻结比较。

面对更强的同先验 VAR 熵编码整尺度对照，4/10 dB 保留原部分尺度的质量优势；19 dB 则由熵编码获得更高 PSNR、较低 LPIPS 与更高 DINO，原部分尺度不再具有质量优势。一致率差值精确为零，区间并非零宽。

{rowtable([r for r in ec if r['method'].startswith('EC_VAR_')], 'VAR 条件熵编码完整尺度 − 原始 token 部分尺度')}

{rowtable([r for r in ec if r['method'].startswith('EC_STATIC_')], '固定统计熵编码完整尺度 − 原始 token 部分尺度')}

固定统计方案在三个已测点的 PSNR、LPIPS、DINO 均弱于原部分尺度；不能据此把较强的 VAR 熵编码结果省略。完整 132 条配对比较见 {item('T1_entropy_whole/v3/paired.csv')}。

## 2. 熵编码是否消除了容量瓶颈？错误传播与代价如何？

在本次 19 dB 冻结 MCS 下，VAR 条件熵编码让 500/500 源图的完整 m9 码流满足计费后的源码容量，确实消除了这些图的 raw m9 容量限制。固定统计编码仅 2/500 满足。4/10 dB 不能据此声称 m9 普遍可发；回退由实际码长决定，禁止逐图按质量选择。m9 的 raw 源长是 5,088 bit，19 dB 净算术码流容量是 4,751 bit；辅助长度和 CRC 已计入。

{chr(10).join(m9table)}

上述 m9 长度全部实际编码，无未知长度被算作成功或失败。收发独立源码解码及 96/96 同 token 恢复图像相等检查见 {item('T1_entropy_whole/v3/roundtrip_validation.json')}。冻结配置表中的 `NOT_QUALIFIED_BY_SOURCE_ONLY_CHECK` 是源长检查阶段遗留描述，不是后续正式 PHY 运行的状态；不能把源长检查本身当作 LDPC 资格。正式结果依据后续实际链路、接收及 closure，详见 v3 的绑定记录。

两个熵码家族各 4,500 帧，共 9,000 帧；六个方法×SNR 条件的头部拒收、正文 CRC 拒收、正文解析失败、算术源码拒收、灰图及已解码 token 错误计数均为零。这说明这些已选工作点未观察到相应失败，**不能据此估计出现比特错误后的熵码传播危害，更不能宣称任意信道下无错误传播**。原 raw KEEP 保留；在同一实际接收记录上的 raw CRC-DROP 诊断，10 dB 部分尺度将 66/1,500 个正文 CRC 拒收帧变为灰图，而原 KEEP 的灰图数为零。其 DINO 相对 KEEP 下降，说明失败规则确实影响部署比较；该诊断不取代原主方法。{item('T1_entropy_whole/v3/failure_breakdown.csv')}；{item('T1_entropy_whole/v3/raw_crc_drop_breakdown.csv')}。

T4 新统一计时已真实完成：四方案 × 16 固定 development 源图 × 3 SNR ×（3 预热 + 3 实测）= 1,152 次执行，其中 576 次计入统计，每方法×SNR 48 个实测调用。CPU uint8 图像到完整 I/Q 为 TX，接收 I/Q 到单张最终 RGB 为 RX；含视觉编码、概率、实际熵码及回退、FEC 和恢复，不使用最终 token/图像缓存免除在线成本。RTX 4090 D、FP32、冻结数值设置、6 CPU 线程/2 interop；不含加载、I/O、指标、排队与空口传播。

{chr(10).join(costtable)}

19 dB VAR 熵码的 TX 均值 155.844 ms，原部分尺度 52.731 ms；其质量改善伴随更大 TX 成本。固定统计编码更快但质量较弱。这里质量来自 500 源图 × 3 噪声，计时来自 16 development 源图 × 3 次重复，不能解释为同一批逐图质量与耗时关联，也不能用小样本 p95 宣称普适尾延迟。原始数字和 RX/E2E 更多统计见 {item('T1_entropy_whole/v3/quality_vs_tx_cost.csv')}。

实际能量另见 {item('T4_resources_cost/summary_completed_v2/energy_summary.csv')}：保留实测 E 与 rho=E/(2N)、总体标准差及分位数，不以理想平均能量替代逐帧能量；不平移原质量曲线制造严格等能量结果。SOURCE_UNFIT 没有发射波形时为 NA，并单列计数。

## 3. Swin 控制开销是否有明显可直接修正的问题？

N1024 适配版使用 C6、768 个正文复符号、256 个头部复符号。73 bit 辅助载荷实际由每图 32-bit float32 归一化功率与 41-bit 内容相关六通道子集编号组成；加 CRC16 与 6 个终止 bit，再卷积编码和重复保护成为 512 个发射 bit。N、C 与工作 SNR 已预共享，不能再次把它们说成可删除的头部开销。

固定六个通道不等于固定通道位置。审计中 20 个缓存条件出现 3 个 mask 与 20 个功率值；20/20 同接收观测原生路径核验通过，44/44 既有诊断文件 SHA 一致。不能免费提供每图 mask/功率，也不能仅缩短保护头部就声称多传了已训练 latent。当前检查点只训练 C6/C13，C13 正文已需 1,664 符号；C7 代码可接受但没有训练支持。

因此本轮保留原合法付费适配，未引入新紧凑头部策略，也没有所谓“修正后新性能”可报告。合法的有损元数据量化或不同保护仍是未评价新协议，审计不证明 256 头部符号全局最优。结论仅针对 **SwinJSCC-80k (adapted)**；step 80000 为用户指定检查点，不是充分收敛或官方最佳。训练与校准仅覆盖 1–13 dB，19 dB 保留范围外标记。{item('T3_swin_sideinfo/side_information_audit.md')}；{item('T3_swin_sideinfo/training_scope.md')}。

旧外部统一耗时在 {item('T4_resources_cost/external_timing_reuse_v1/external_timing_single_output.csv')}：latent JSCC、Swin 与 **native256 BPG**，N1024、7/13/19 dB、development16。它们不与新 4/10/19 计时混合；native BPG 的链路耗时只对可发送样本定义，不把 SOURCE_UNFIT 写成零耗时，不冒充自适应 BPG 的耗时。历史 BPG 单独 exit receipt 本地缺失，保留已发布 completion/输出哈希与审计证据边界。HiFi 不新增统一计时。

## 4. 第二预算是否支持相同机制？

**待实际 T6 结果，当前没有可写的结论。** 预定 N2048、100 个预登记确认源图、4/10/19 dB、每图 3 噪声、三数字方案共 2,700 方法帧。当前执行负责人报告资格与校准已开始；本次索引未连接服务器，不能把脚本、synthetic 测试、校准中间结果或本地准备称为正式测试完成。

最终报告补入时必须绑定：实际 100 源图清单与内容去重范围；冻结校准策略和事先选定的单一熵码家族；真实 render/score/stat completion 与父进程 wait/exit；2,700 帧、36 summary 和 36 paired 行；失败及零增量；各工作点 m10/K0、部分尺度选择完整尺度比例；跨预算机制图。不可从原 500 source reference 缓存冒充新 100 源图，不称超出可审计哈希范围的全球从未见过盲测。

原 N1024 500 源图与新 N2048 100 源图属于不同总体；跨预算图只展示各预算内部 partial−whole，不能把绝对质量差归为预算效应。两个预算不足以推断等质量带宽节省百分比。完整十尺度可行后回退完整尺度或增量为零，应保留为机制边界。

## 5. 论文目前应主张什么？

N1024 已验证的是固定长度 raw token 接口下、有限已测工作点的源信息与纠错保护折中：允许部分尺度能在若干 SNR 改善质量，且两处扩大整尺度候选审查后原比较保持。它不是每个 SNR 都有效，不能将 7 dB 零增量或 13 dB 小增量省略；也不是对所有同先验源码接口普遍最优。

较强的 VAR 条件熵编码在 19 dB 消除了这批图的完整 m9 容量限制并提升三项质量，同时发送端计算更重。因此目前应同时报告**固定长度接口下的有限增量**与**质量—发送端计算取舍**，明确外部适配模型范围。是否可进一步提出跨预算机制结论，等待 T6 实测；不为保住宽泛优势而重选样本、扩大预算或开启新模型。

所有区间均为源图先平均三次噪声后原有配对统计，10,000 次 bootstrap、seed 2026100701、逐点 95% CI，未做多重比较校正。本文整理不再 bootstrap；LPIPS 不反转符号，ConvNeXt 是原图预测一致率而非真标签分类准确率。区间含零不证明等价。图表和原精度证据入口见同目录 INDEX.md。
"""
    (out / "FINAL_REPORT_TEMPLATE.md").write_text(report, encoding="utf-8")
    index = f"""# N1024 已完成交付索引（独立版本）

更新时间（UTC）：{now}。N1024 T1–T5 已完成的本地交付经过本轮逐文件 SHA 复核；**T6 N2048 运行中，不计为完成**。本目录仅导航、原精度摘录及报告模板，0 模型、0 PHY、0 bootstrap、0 新计时；不覆盖旧封印或发布过滤器。所有数据图 PDF/SVG 为矢量线条与文字；重建样例的照片保留真实栅格像素，标题为矢量。PNG 为 600 dpi。

## 数据及报告

| 内容 | 当前入口 | 口径与边界 |
|---|---|---|
| 总报告草稿 | {link(out/'FINAL_REPORT_TEMPLATE.md')} | 已回答五个科学问题；T6 留待实测，不是全任务完成报告 |
| T0 盘点 | {item('T0_inventory/inventory.md')}，{item('T0_inventory/provenance.json')} | 历史盘点；后续完成状况以本索引和各新版本为准 |
| T1 整尺度熵码 | {item('T1_entropy_whole/v3/REPORT_T1.md')}，{item('T1_entropy_whole/v3/completion.json')} | v3 含真实在线 TX/RX 成本；两家族各 500×3SNR×3噪声，事后同源比较 |
| T1 原精度质量及诊断 | {item('T1_entropy_whole/v3/summary.csv')}，{item('T1_entropy_whole/v3/paired.csv')}，{item('T1_entropy_whole/v3/per_frame.csv')}，{item('T1_entropy_whole/v3/raw_crc_drop_per_frame.csv')} | 原 KEEP 不变，CRC-DROP 仅同 RX 诊断；禁止把两种 CI 相减 |
| T1 长度/冻结 | {item('T1_entropy_whole/v3/source_lengths.csv')}，{item('T1_entropy_whole/v3/m9_sendable_summary.csv')}，{item('T1_entropy_whole/v3/frozen_policy.json')}，{item('T1_entropy_whole/v3/roundtrip_validation.json')} | 实际纯熵码、m4 下界按长度回退、source-only 字段不代替 PHY 资格 |
| T2 候选审查 | {item('T2_calibration_audit/v1/REPORT_T2.md')}，{item('T2_calibration_audit/v1/candidate_coverage.csv')}，{item('T2_calibration_audit/v1/full_calibration_rankings.csv')} | 10/19 dB 两原整尺度赢家不变；未重选原部分尺度，未新跑 holdout |
| T3 Swin 审计 | {item('T3_swin_sideinfo/side_information_audit.md')}，{item('T3_swin_sideinfo/training_scope.md')}，{item('T3_swin_sideinfo/adapter_validation.json')}，{item('T3_swin_sideinfo/header_budget.csv')} | 原合法付费适配保留；未训练 C7、不删内容相关 mask/power，19 dB 范围外 |
| T4 当前汇总 | {item('T4_resources_cost/summary_completed_v2/REPORT_T4.md')}，{item('T4_resources_cost/summary_completed_v2/resource_summary.csv')}，{item('T4_resources_cost/summary_completed_v2/energy_summary.csv')}，{item('T4_resources_cost/summary_completed_v2/failure_breakdown.csv')} | 实际资源和能量；SOURCE_UNFIT 能量 NA；未改旧图质量 |
| T4 新四臂计时 | {item('T4_resources_cost/summary_completed_v2/timing_per_call.csv')}，{item('T4_resources_cost/summary_completed_v2/timing_summary.csv')} | 4/10/19 dB；development16×3 实测重复；576 实测/576 预热，真实 exit0 |
| T4 旧外部计时 | {item('T4_resources_cost/external_timing_reuse_v1/README.md')}，{item('T4_resources_cost/external_timing_reuse_v1/external_timing_single_output.csv')} | 7/13/19 dB，P/Swin/native256 BPG；642 原输出 SHA 验证；与新计时分列、不补缺点 |
| T6 | {item('T6_budget2048/consumer_review_v1/review.json')} | 仅独立消费者工程审查，不是实际科学完成；执行与最终结果由根任务补入 |

## 实际图片入口

每个下列 stem 对应 `.pdf`、`.svg`、`.png`，完整逐文件绝对路径与 SHA 见 {link(out/'delivery_files.csv')}。独立子图与所有全样例页都在各组目录。

| 图组 | 入口 | 范围 |
|---|---|---|
| 原完整尺度与部分尺度四指标曲线 | {item('T5_figures/fig_t5_same_prior_N1024.png', 'PNG')} / {item('T5_figures/fig_t5_same_prior_N1024.pdf','PDF')} / {item('T5_figures/fig_t5_same_prior_N1024.svg','SVG')} | 原500、六SNR；2×2与四独立子图 |
| 原 partial−complete 配对增量 | {item('T5_figures/fig_t5_partial_minus_complete.png','PNG')} / {item('T5_figures/fig_t5_partial_minus_complete.pdf','PDF')} / {item('T5_figures/fig_t5_partial_minus_complete.svg','SVG')} | 原500、六SNR；7 dB零及13 dB小增量保留 |
| 完整外部五列固定样例 | {item('T5_figures/external_completed_v1/fig_t5_external_completed_N1024_10dB_page01.png','10 dB PNG')} / {item('T5_figures/external_completed_v1/fig_t5_external_completed_N1024_10dB_all16.pdf','10 dB全16 PDF')} | 1/10/19 dB、全部固定16、每SNR四页；共12页，BPG/Swin实际缺项已补完 |
| 第五熵码列机制样例 v2 | {item('T5_figures/mechanism_entropy_completed_v2/fig_t5_mechanism_entropy_N1024_19dB_page01.png','19 dB PNG')} / {item('T5_figures/mechanism_entropy_completed_v2/fig_t5_mechanism_entropy_N1024_19dB_all16.pdf','19 dB全16 PDF')} | 4/10/19 dB、固定16、共12页，EC_VAR由校准预选；不得使用排版拒收v1 |
| 两熵码与原部分尺度曲线 | {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.png','PNG')} / {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.pdf','PDF')} / {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_quality_N1024.svg','SVG')} | 4/10/19 dB，500×3；2×2与四独立子图 |
| EC−raw partial 配对增量 | {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.png','PNG')} / {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.pdf','PDF')} / {item('T5_figures/entropy_quality_completed_v1/fig_t5_entropy_minus_partial_N1024.svg','SVG')} | 方向未翻转，19 dB熵码优势及零一致率保留 |
| 四方案质量与 TX 代价 | {item('T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.png','19 dB mean PNG')} / {item('T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.pdf','PDF')} / {item('T5_figures/quality_tx_cost_completed_v1/fig_t1_quality_vs_tx_mean_N1024_19dB.svg','SVG')} | 4/10/19 dB，各mean与median；共六组2×2，TX轴为log(ms)；质量500与计时16不同总体 |

固定源顺序为 `[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]`。全部为 development 样例，不伪称 500 图随机展示。page01 用清单前四个源；所有16源全部交付，不按各方法挑噪声。Direct 和 partial+VAR 共享原实际接收 token 与 Dc，Direct 未传残差置零、不是未知 token 填索引0。不同方法仍保留各自预定 noise namespace；同源不意味着同 RX。

原根目录 `T5_figures` 的 external13dB 五列、1/10/19 四列机制图继续有效，原 README 中缺图说明是生成时状态；后续补齐以本索引的新子目录为准，不修改旧 README/seal。`mechanism_entropy_completed_v1` 因标题裁切被拒收，仅 v2 交付。synthetic N2048 排版测试不得纳入论文数据图。

已存在的完整外部曲线：{link(root/'paper/figures/adaptive_bpg_holdout500/fig_adaptive_bpg_main_N1024.png')}。HiFi 仅在真实共同 N1024/13 dB 既有子集单独展示：{link(root/'paper/figures/hifi_development_N1024_13dB_group02')}；没有把其他 SNR 图片当成缺失工作点，也没有新增 HiFi 采样。

所有当前 T5 图组已有实际 PNG 打开检查记录，见各目录 `visual_QA.json`。本次只复核这些已完成文件的 SHA，未冒称重新逐张打开。重建页的照片不是矢量重建，但标题可编辑；数据曲线 PDF/SVG 为真正矢量。

## 可重现脚本与版本差异

本地脚本根目录：`{scriptdir.as_posix()}`。远端仓库：`/home/liulu/projects/VAR_COMM`。根任务已报告远端交付 SHA 验证完成，本次整理未使用 SSH。远端归并凭据：`outputs/WCL-EVIDENCE-CLOSURE-20261009/N1024_final_small_deliveries_v1_remote_verification.json`；它是根任务提供的路径，本索引不冒称独立读取远端。

- 原六点图：`plot_existing_t5.py`。
- 外部 fixed16 已封印生产脚本 SHA `9cfd10f7b84ec2fa0f6ded2c14f6a1db7042337c74f05ac0190466651fefb3b4`，保留在**远端** `experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples.py`。
- 第五列机制 v2：SHA `279a229c7c50b5d17c7f3215aad4f743fe943f3c16241a07da2016e52f74a5f4`。本地为 `plot_t5_completed_examples.py`，**远端实际安装名称为 `plot_t5_completed_examples_delivery_279a229c.py`**。原 README 的脚本名不修改；重现 v2 时请用对应内容哈希，不能误用远端旧同名脚本。新版也支持外部模式，但不是旧外部 production seal 的原脚本身份。
- 熵码曲线：`plot_t5_entropy_quality.py`，SHA `ed901d0b6049bc57afb66c0c45b0e71fb46133a4bbc7721c57a0dcd6b9981491`。
- 质量/代价图：`plot_t1_quality_tx_cost.py`，SHA `a6798d9a4172a5cff92a49a5f5741bd150754adec18df674dc2852b9c4b1e47e`。

重现所有图时使用新空输出目录，输入真实完成资产和冻结文件。机制 v2 远端入口示例（路径参数由已封印 manifest 绑定）：

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples_delivery_279a229c.py --mode mechanism --root . --entropy-manifest outputs/WCL-EVIDENCE-CLOSURE-20261009/T5_examples/entropy_fixed16_v1/display_manifest.json --entropy-assets outputs/WCL-EVIDENCE-CLOSURE-20261009/T5_examples/entropy_fixed16_v1 --entropy-freeze outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/selected_entropy_family.json --out results/wcl_evidence_closure_20261009/T5_figures/mechanism_entropy_REPLOT
```

旧和新版输入命令更多选项见各组 README；原结果不覆盖。`delivery_files.csv` 列每个本地文件的准确绝对路径、仓库相对路径与对应远端仓库路径，不声称每个二进制都已 git 跟踪。发布继续遵循已有过滤规则，不为了 PDF/PNG 改过滤器。

## 仍未关闭的任务

T6：预定第二预算 N2048 的实际测试、评分、源配对统计和跨预算机制图等待执行闭环。这里不发布任何 N2048 科学均值。完成后根任务应依据真实结果填入总报告的第4问及对第5问的影响，保留零/负结果与不同源总体边界。
"""
    (out / "INDEX.md").write_text(index, encoding="utf-8")
    provenance.update({"original_common500": {"commit": "252176e041758ecb2d3e81b6fde5b587e7e17bb7", "directory": str(original), "sha256": pins}, "T6_status": "RUNNING_NOT_COMPLETE", "T6_status_source": "parent executing task update; no remote access by index builder", "remote_merge_evidence_source": "parent report, remote path only; not read locally", "script_sha256": sha(__file__), "generated_utc": now})
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Recheck all old result seals after writing the new report.
    for r in verified:
        assert sha(r["local_path"]) == r["sha256"], r["local_path"]
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
    completion = {"status": "N1024_DELIVERY_INDEX_COMPLETE_T6_PENDING", "verified_existing_output_count": len(verified), "indexed_file_count": len(inventory), "old_results_unchanged": True, "T6_complete": False, "FINAL_REPORT_is_template": True, "new_model_calls": 0, "new_packet_calls": 0, "new_bootstrap_calls": 0, "new_timing_calls": 0, "outputs": files}
    (out / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "status": completion["status"], "verified": len(verified), "indexed": len(inventory), "completion_sha256": sha(out / "completion.json")}, indent=2))


if __name__ == "__main__":
    main()
