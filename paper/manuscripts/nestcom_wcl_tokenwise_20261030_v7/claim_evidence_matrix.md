# V7 主张与证据

基准提交：`e38822dce0832dfb4e85ffe5466ae66a7cec300f`。本稿主配置为 `original_receiver_plus_D_token`，保留原 tokenizer、VAR、适配后 D_c 和 argmax 接收机，仅采用当前逐 token 端点。历史 Quarter、Ada-argmax、coarse-only、Kodak 与 N2048 结果不改名为新主配置结果。

以下路径以本稿目录为起点。完整实验目录为 `results/nestcom_perceptual_followup_20261030_v2/`；已闭合 E1/E3/E4 的发布副本位于其中 `closed_evidence_v1/`。数值自动核对记录为 `validation/closed_claim_audit.json`。E2/E5/E6 的独立 actual-wait 完成身份见 inputs 中各自 execution_closure.json。

| 主张 | 人群与条件 | 证据与直接比较 | 允许范围与限制 | 状态 |
|---|---|---|---|---|
| 完整粗尺度加逐尺度残差形成粗到细表示 | 原冻结 VAR tokenizer/VAR-d16 | 原方法代码、十尺度定义；`sections/method.tex`、Fig.1 | 模型选型动机，不证明表示顺序的独立因果优势，不称残差严格等于高频 | done |
| 651 个合法整数 token 端点可由付费 profile 目录标识 | m=4…9 的整数 K，加 (10,0)，30…680 tokens | `inputs/e1/frozen_policy_metadata.json` 中目录身份和实际 12-bit 约束 | 主系统端点总数；不是每个固定 ceiling 下均有 651 候选 | done |
| 固定 MCS 下实际独立终止长度决定最长可装入前缀 | 原冻结策略，N1024 与 N512 | 方法预算式；`inputs/e1/frozen_policy_metadata.json`、`capacity_identity.json` | 非质量选择，不假定码长单调；N1024 无可行端点报错、N512 付费 NULL 各自保留 | done |
| 新 NeST 优于 WHOLE 的同模型重建 | common500，每源三原信道重复，七个预算/SNR 点 | `inputs/e1/paired_deltas.csv`，candidate=final_NeST，reference=WHOLE_final_receiver | 五指标、七点的逐点来源配对区间均方向有利；不是逐图保证，也不是多重比较校正结论 | done |
| N1024/10 利用率 77.70%→99.79% | 同上，固定家族策略 | `inputs/e1/capacity_summary.csv` | 分母为源码信息容量，不能称总空口效率 | done |
| 原 Quarter→Token 的 DISTS/DreamSim 改善为六点而非七点 | common500、原策略 | `inputs/e1/paired_deltas.csv`，reference=original_NeST | N1024/7 两项区间跨零；与同 ceiling E3 是不同实验 | done |
| 细化网格有质量收益与额外长度检查 | control100、N1024/10、q4/r2/3、共同最大 (8,126)、共享实际噪声 | `inputs/e3/summary.csv`、`paired_deltas.csv`、`efficiency.csv` | Whole/Quarter/Eighth/Token 分别 5/20/39/352 候选；检查次数不等于模型前向或耗时 | done；实际耗时见 E5/S7，早停比较与计时全扫描分列 |
| 边界 token 同时参与残差与后续条件 | control100、7/10 dB、同一实际接收前缀 | `inputs/e4/paired_deltas.csv`：B0−A0、C0−B0；`closed_evidence_v1/e4/final_scale_identity_check.json` | 有序干预，不是额外 Token−Quarter 增益的分解；A0 仍有先验补全；19 dB 最后尺度无更细条件更新 | done |
| 六系统比较存在感知取舍 | N1024、common500、4/7/10/19 dB | Fig.2、`inputs/e1/summary.csv`、`paired_deltas.csv` | 完整系统比较，tokenizer/prior/decoder/类别条件不同；10 dB Ada 的 DISTS/DreamSim 更好，19 dB Ada 的 DISTS 更好 | done |
| 低预算支持更好的源重建，但不能称感知全面领先 | N512、common500、4/10/19 dB | Table III、E1 five-metric direct pairs | NeST 相对 Ada 的 PSNR/LPIPS 三点有利；19 dB DISTS/DreamSim 有利于 Ada；CLIP-I 均值 Ada 更高，10/19 dB 区间跨零 | done |
| 图3展示真实可见优势 | N1024/7，source84 Television、143 Patterned leather，各法固定原 repeat0 | `inputs/f1/fig3_metrics.csv`、`selected_native_mapping.json`、完整图池索引 | 明确按优势选择；不是随机或代表样本，不改变主统计；不同方法历史噪声身份不强称相同 | done |
| 相同信息位预算下三系统重建关系 | common500、六 B 值，WHOLE/Token 各一次，Ada 三固定生成重复 | `inputs/e2/summary.csv`、`paired_deltas.csv`、`execution_closure.json`，15000结果 | NeST−WHOLE全30区间有利；NeST−Ada高两预算DISTS/DreamSim不利，1940 bits DreamSim跨零；不隔离模型混杂 | done |
| 端到端处理成本与粒度代价 | timing32，16组，同PRO6000、2CPU核，2热身+3测量 | `inputs/e5/timing_summary.csv`、`execution_closure.json`；5120记录、48汇总 | 全部实际编解码；BPG CPU-only，ARPC IPC计入；是软件处理成本，不是空口时延；完整候选扫描与早停位置分开 | done |
| 固定 10 dB 策略接近门限的拒收风险 | control100、8/8.5/9/9.5/10 dB、20噪声，30k帧 | `inputs/e6/summary.csv`、`execution_closure.json` | 全部失败保留；8 dB全拒收、10 dB零观察失败；不声称低误帧率保证或SNR鲁棒 | done |

置信区间均直接从同来源差值构造。500 源与 100 源有各自完整顺序和固定 bootstrap bank；不相减边际区间，不切片 500 源 bank 冒充 100 源 bank。
