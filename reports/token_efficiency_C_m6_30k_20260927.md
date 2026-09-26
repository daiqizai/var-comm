# C m6：30k 配对延长校准结果（2026-09-27）

H6-V/H6-P 各完成 30,000 次更新，原调度器在 30k 正式登记 `extend=false`。两臂最终 selected 均为 30k；这表示登记规则停止，不证明理论收敛。原队列随后自动进入 m7 的 20k→30k 配对延长。

| 臂 | selected | 完整校准 utility | 25k→27.5k 改善 | 27.5k→30k 改善 |
|---|---:|---:|---:|---:|
| H6-V | 30000 | 0.019709816085345423 | +0.190896% | +0.389324% |
| H6-P | 30000 | 0.019656061023722093 | −0.556747% | +0.631520% |

原规则要求至少一臂连续两段均改善 >=0.2%，两臂共同延长；本次两臂均不满足。P 在 27.5k 的退步完整保留。每臂 30k 的 15,000 条记录有 0 个 header 失败、2,825 个 body CRC 失败（1dB 2,816、4dB 9）；失败并未剔除。

## 数据、身份与核验范围

结果目录：`results/token_channel_efficiency_20260923/C_extensions/m6_seed2026092304_until30000`。本次新增 22.5k/25k/27.5k/30k 四轮 120,000 条完整校准记录、逐源均值、SNR 分项、全历史曲线和两张 SVG；最初九轮 270,000 条通过不可变发布索引和 CSV SHA 引用。独立 CPU 发布器重审十三轮共 390,000 条完整 source/SNR/noise/method 键、公式、均值、失败、全历史最优选模以及正式决策。两臂与十三个 checkpoint 复用同一数字 cache；这些重评分不代表新增独立数字 PHY 传输。

- registration SHA：`25fcd28c9ec902a0e725bbca572ff17561ae014cdb4e6a201f2338ae9a257d11`。
- 两臂 selected/terminal checkpoint SHA：`a98929f4b89e8aaff73f65dfcff016ba75fe3fe8af0a141a64b36e29a728e9a0`。
- 原 20k parent SHA：`14c8fa4ce75fb25fc09cbbfcc28a92f3309bd2af16d25ba2c3ddfa34c274238e`。
- 63 项绑定、10 份校准 tensor SHA、20k/30k 真实 CPU checkpoint 的填充 optimizer、模型及恢复状态通过；训练大 tensor 仅保留原 loader 核验范围，本发布器未重哈希全部训练 tensor。
- N4084=68 付费控制+1200 数字+2816 连续，source 1092 bit、body 母码2228、发送2400 coded slots。E8168 是登记约束，校准记录没有新增逐帧实测 E。训练/校准秒数不是完整在线 TX/RX 时间。

## 热暂停造成的原调度回执缺口

原 m6 最后 attempt 已记录 thermal stop；外层 C_followups 同时记录 thermal stop。训练在处理该请求期间完成 30k 全校准、写出 terminal checkpoint、selected、`MILESTONE_COMPLETE` 与 completion。源码证据表明，外层 runner 的 `self.stopping` 分支在归档阶段 snapshot 前退出；重启后 lifecycle 依据已写 completion 和完整校准登记停止决策并转入 m7，因此缺少原 `C_N4084_m6_seed2026092304_until30000` stage receipt。

默认 publisher 在创建结果目录前拒绝了这项缺失，原拒绝日志保存在 outputs/monitoring。此次新增未绑定 CPU 审计器及显式 `--recover-finalized-boundary` 分支：仅接受正式停止的终端边界，同时核验原 launch/进程退出、内外层实际热标志和时间顺序、无失败终端日志、重启登记、状态与 checkpoint/selected/decision 全部一致。随后以独立身份捕获只读原文件，不写回或伪造原 scheduler stage/snapshot，不重启训练。

本次发布状态明确为 `REAL_C_EXTENSION_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING`，`complete_stage=false`、`terminal_artifacts_verified=true`；原进程退出码未持久化，保持 `process_returncode=null`。这份独立终端资产验算证明登记校准结果与真实权重一致，不能改称原调度回执齐全或已证明退出码为0。原记录、失败日志和暂停证据均保留。活动调度、模型、协议、缓存与热保护源码完全未改。

## 后续范围

本结果是校准里程碑，不是 development 质量验收、实际 E/完整在线计时或整份合并交付。m7/m8/pure 延长、N3060、另外两个训练 seed、selected 实际评测和计时、诊断、四个历史 GPU worker 以及最终全方法配对统计仍由原队列继续。源图 bootstrap 和训练 seed 变异分开；新 holdout 与内容选择器仍暂缓。
