# N3060 / m6 首轮 20k 校准阶段（2026-09-27）

原登记 H6-V / H6-P 已各完成 20,000 更新。九轮 0/2500/5000/7500/10000/12500/15000/17500/20000 全量校准共 270,000 行，覆盖同一 1000 源 × 五 SNR × 三噪声 seed × 两臂。两臂 selected 均为 20,000。原调度器仅根据校准正式登记成对延长至 30,000，已由原队列自动接续；没有另启训练，也没有重复训练 P3060。

| 臂 | selected | 校准 utility | 最后两段相对改善 | 原决策 |
|---|---:|---:|---:|---|
| H6-V | 20000 | 0.026456227790750565 | +0.517951% / +0.884973% | extend=true |
| H6-P | 20000 | 0.026788107907709975 | +1.073060% / +0.211064% | extend=true |

以上是训练停止规则的校准证据，不是理论收敛或 development 质量结论。每臂末轮 15,000 行保留 1 次 header 失败、2829 次 body CRC 失败（1 dB 为 2821，4 dB 为 8，两类可重叠）。九个 checkpoint 与两臂复用数字缓存，因此 270,000 行重评分不是 270,000 次独立数字 PHY 传输。

## 身份与资源

- 引擎：原 `token_efficiency.C_train`，N3060 / m6 / seed2026092304；`parent=null`、`parent_updates=0`，不是原 N4084 m6 的接续目录。
- registration SHA：`729b7dc6e9279a7505007494477f01ef1a9dfb2dc5fb234f4cedf087293f61e5`。
- selected / terminal checkpoint SHA：`3b1300660c69dffd59a62301d7b125f089832f932625cd801980ab8677d8b649`。
- N3060 = 68 付费头 + 1200 数字主体 + 1792 连续增强。source payload 1092 bit，header class/mode 10/2 bit，header CRC/tail 16/6 bit、母码 68、发送 coded slots 136；body CRC/tail 16/6 bit、母码 2228、发送 coded slots 2400。
- E6120 为原登记约束；本次校准 CSV 没有新的逐帧能量测量。训练及校准秒数不作为完整在线 TX/RX 时间。
- m6 由原 N4084 primary 1/4/7 dB 全量校准选择，候选和 P3060 复用回执随本包保留。既有 P3060 selected37500 不重训；原 8 项 N3060 pretrain cached/online 资格不等于 selected 完整 development 验收。

## 原 stage 回执缺口

原 inner PID3538300/start116161516 与 enclosing C_followups PID3538036/start116156340 已实际退出。外层在 1790515554.566372 记录 thermal（83 C / 软件热标志），内层在 1790515554.9685059 接收传播后的 requested_stop（80 C / 软件热标志）。原训练完成 20k 全校准、terminal checkpoint、selected 与 completion；completion mtime 为 1790515848.6486952。外层停止分支没有归档 `C_N3060_m6_seed2026092304_until20000` stage。

原 C_followups 于 1790515903.1828787 重新登记 PID3551272/start116405907，读取真实 completion 与完整校准后登记 until30000。此为已核实的热暂停归档缺口，未发现训练日志 Traceback/RuntimeError/FAILED。没有伪造 stage、退出码或通过重跑填补历史。

新增独立 CPU `tools/publish_c_n3060_initial.py` 使用明确的 `--allow-thermal-receipt-gap` 分支捕获该初始边界资产：**complete_stage=false、terminal_artifacts_verified=true、process_returncode=null**。它检查原 scoped 引擎/命令、已退出进程、内外热证据和时间顺序、完整终端日志、原重启、真实 20k completion/填充 optimizer/恢复 RNG 状态及正式校准决策。它只发布资产证据，不重构原调度回执、不宣称 finalization；现有只允许已停止边界的 N4084 `--recover-finalized-boundary` 工具完全未改。本次 extend=true 仅由原 scheduler 决策，发布工具不启动或登记训练。

## 交付数据与验证

结果：`results/token_channel_efficiency_20260923/C_initial_milestones/N3060_m6_seed2026092304_20k`。

独立 CPU 发布已核验 57 项合并绑定、10 个 calibration tensor SHA、两 role 共 210 份 shard 元数据/registration/completion、原预处理与噪声身份、九轮完整键和源顺序、指标公式、配对失败标志、均值及全历史最优 selected。训练大 tensor 只保留真实 loader 的验证范围和封存清单；本发布器没有重新哈希全部训练 tensor。

发布包含全部原始校准 CSV/receipt、逐源逐 SNR 均值、分项和 SNR 曲线、selected、资源账本、decision、资格、缓存 lineage、原 thermal 证据和逐文件索引。269 项索引文件共 65,117,825 字节，每项小于 10 MB，SHA/大小通过；两 SVG 语法及同源数据 PNG 视觉审查通过。原 console 按 SHA 保留，可能含历史尾随空格。已发布目录不可覆盖，不重跑 main。

新增五项 CPU 回归覆盖 N3060 资源/父级/seed 身份、完整成对校准边界、缺 stage 默认拒绝、已有 stage 不得由 gap 选项替代、原成对延长规则；完整 221 项 CPU 检查通过。这些是工程与发布一致性验证，不是新增 GPU 质量实验。独立远端发布复验回执位于 `outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_N3060_20k_verification/receipt.json`，实际成功回执决定远端验收，不以启动或旧 RUNNING registration 代替。

N3060 后续校准延长、另外两个训练 seed、正式 selected development/实际 E/统一在线计时/诊断、四个历史 GPU worker、全方法配对统计和最终合并交付仍待完成。新 holdout 与内容选择器保持暂缓；活动实验源码、原模型、loss、FEC 和硬件设置均未改。

