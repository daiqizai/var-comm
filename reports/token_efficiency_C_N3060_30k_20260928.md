# N3060 / m6 首个 seed 的 30k 校准边界

北京时间 2026-09-28（原计算 UTC 2026-09-27）。原 scoped token_efficiency.C_train 的 H6-V/H6-P 均完成 30000 次更新。原校准规则正式 extend=false，两臂保留 27500 selected；不称理论收敛。此报告交付缓存复用校准资产，正式 selected development、实际逐帧能量与完整在线 TX/RX 计时仍待原队列执行。

## 结果与原决策

| 臂 | selected 更新 | selected utility | 30k utility | 最后两段相对改善 |
|---|---:|---:|---:|---:|
| H6-V | 27500 | 0.026196066380416355 | 0.026339326869758466 | +0.572104% / -0.546878% |
| H6-P | 27500 | 0.026468030497416233 | 0.026483488106820732 | +0.319067% / -0.058401% |

原规则要求一臂连续两个 2500 步区间均改善至少 0.2%，才成对再延长 10000；本边界两臂均不满足。30k 退步保留，没有使用 development 选模。

selected 共同 checkpoint SHA：66b6cd526d07017a3018b65cc5a45be0a07319d6b733ea586f861104f7cc6226。
30k terminal SHA：b0f775574bac330b9a36294a7eecc2941ca9bce7446bad379d12b8b522d1abab。
20k parent SHA：3b1300660c69dffd59a62301d7b125f089832f932625cd801980ab8677d8b649。
registration SHA：729b7dc6e9279a7505007494477f01ef1a9dfb2dc5fb234f4cedf087293f61e5。原训练 parent=null、parent_updates=0，20k parent 是本次延长的边界引用。

十三轮 0–30000 的完整校准共 390000 条（1000 原源 × 五 SNR × 三 noise × 两臂 × 十三轮）。本次只新增四轮 120000 条，初始 270000 条按已发布 index/CSV SHA 引用。全部逐源键、预处理/噪声身份、指标公式、配对失败、均值和全历史选模已重新核验。这是复用数字 cache 的评分，不能计作 390000 次独立数字 PHY 传输。每臂末轮保留 1 次 header 失败及 2829 次 body CRC 失败，后者 1 dB 为 2821、4 dB 为 8；两类可重叠。

## 原调度回执缺口

原外层 thermal stop 传播为内层 requested_stop，末轮全校准、terminal checkpoint、selected 和 completion 随后真实落盘；外层 stopping 分支未归档原 stage。原重启依据 completion 作出停止决策并接续后续 seed。独立工具 tools.publish_c_n3060_extension --until 30000 --allow-thermal-receipt-gap 核验原 launch、PID/start_ticks、退出身份、热标志、时间顺序、完整无失败日志、原重启及正式决策后捕获终端资产。原 outputs 和活动代码未改，未补造调度 receipt 或 exit0。

明确状态 REAL_N3060_EXTENSION_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING；complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。原 completion mtime 为 1790526380.3438215。实际内外 launch/stop/resume 与原 console 在本发布的 thermal/ 中逐 SHA 保留。原 20k 回执缺口也保留于历史发布。

## 资源及验证范围

N3060 = 68 付费头 + 1200 数字 body + 1792 连续残差。source payload=1092 bit；头 class10+mode2+CRC16+tail6，经母码68 bit/136 coded slots；body 母码2228 bit、2400 coded slots。E6120 是原登记约束，校准没有新增逐帧 E 测量；训练/校准 seconds 不是完整在线 TX/RX 时间。

已核验 57 项合并源码/资产绑定、10 个 calibration tensor SHA、两 role cache completion/registration、210 份 shard 元数据，以及真实 20k/30k/selected CPU checkpoint 的模型状态、填充优化器和恢复状态。训练大 tensor 未全部重哈希，保留原实际 loader 的验证范围。

本发布 index 包含 30 个文件、30612598 字节，单文件小于 10 MB；全部 SHA/大小和两张 SVG XML 通过。两图由同一 calibration_curve.csv 生成，另以同数据 PNG 作视觉审阅。

根目录 228 项 CPU 测试、repository/release/fsck 及排除原 console 保留空格的 diff 检查已通过。独立远端检出验证在提交后执行；最终结果以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_N3060_30k_verification/receipt.json 为准。此处不预报独立验证成功，也不把 CPU 发布一致性视作新增 GPU 质量。

发布 index SHA：9eaa57b2a71cebbba465f45c4ef0ba6692361858e28921aba06219ce490b28ac。
未修改发布器 SHA：6ec34cf8b6b7e6879ac9fd3ea8ac946c1b13871d8718436275800fc5aeb10920。

## 数据与待办

结果目录：results/token_channel_efficiency_20260923/C_extensions/N3060_m6_seed2026092304_until30000。内含新增原 CSV/meta、逐源均值、完整历史分项/SNR 曲线、两 SVG、selected/parent lineage、正式 decision、资源账本、原回执缺口证据及 SHA 索引；旧记录通过 history_references.json 引用，目录不可覆盖。

仍须完成另外两个训练 seed、selected 正式 development/诊断/计时、四个历史 GPU worker、全方法配对统计和最终合并报告。新 holdout 与内容选择器继续暂缓；源图 bootstrap 与训练 seed 变异分开。自动跟进继续。
