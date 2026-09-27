# N4084 m6 第二训练 seed 的初始 20k 校准资产

2026-09-28，北京时间。seed 2026092404 的 H6-V / H6-P 各完成 20000 更新；原 short_prefix.train 从零训练，parent=null、parent_updates=0。九轮完整校准（0 至 20000，每 2500）共 270000 条记录和 90000 条逐源噪声均值已独立 CPU 核验并封存。本次没有运行新的 GPU development 或在线计时，也不代表整个合并实验交付完成。

两臂均 selected 20000，utility 分别为 0.020225655451323838 / 0.019880569598187382。共同 checkpoint SHA256：
6b2a628f83c2bf58b6be76c7923ec7ce6be67ec81a1cdea486cfb737709d9fa5

registration SHA256：
39ef84cc5d32d36897c1e5ed58c9ac709db1e8fbe14409ea994bd4576211bc66

V 的最后两段相对改善为 0.515359% / 0.299748%；P 为 0.141163% / 0.949610%。V 满足原规则的连续两段至少 0.2%，原 at_20000 正式决定 extend=true、until=30000、paired_arms_extend_together=true。P 第一段不达阈值也保留。原队列已接续 30k；本发布器不启动训练，不把 20k 称为收敛。

每臂末轮 15000 条中 header 失败 0、body CRC 失败 2825（1dB 2816，4dB 9），所有失败保留。校准复用同一数字 cache，两臂九轮评分不是 270000 次独立数字 PHY 传输。N4084=68+1200+2816；source 1092 bit、header 12 bit 加 CRC16/tail6、body 母码 2228 bit、发送 2400 coded slots。E8168 是原登记约束，不是本校准逐帧新增实测。训练/校准秒数不等于完整在线 TX/RX 时间。正式 selected development、实际能量和统一在线计时仍待原队列执行。训练 seed 变异与源图 bootstrap 分开。

## 原热暂停回执缺口

outer 3736963/start119887723 在 1790551032.2175727 因 thermal 请求停止（80C、软件热标志）；inner 3737256/start119892901 在 1790551033.870581 收到传播的 requested_stop（78C、软件热标志）。原训练写完 20k 完整校准、checkpoint、selected、completion，completion mtime=1790551342.3164527；外层停止分支未归档原 C_N4084_m6_seed2026092404_until20000 stage。两原进程已实际退出。原 resumed C_followups 3740470/start119955070 在 1790551394.804959 读取终端资产，登记延长并接续训练。

独立发布显式使用 allow-thermal-receipt-gap，核验准确原命令、进程身份与退出、双方热证据、严格时间顺序、无未知失败的终端日志、原重启及正式决策；原 launch/stop/console 和终端资产按 SHA 原样封存。状态为 REAL_C_REPEAT_INITIAL20K_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING；complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。没有伪造原 stage、exit0 或 finalization，没有重训补回执，也没有改写原 outputs。

## 复核与发布范围

54 项合并绑定、冻结 candidate m6、第一 seed 已发布 index、Decoder、真实梯度/能量/优化器隔离及 bitwise resume qualification、两 role 缓存 completion、210 shard 元数据和 10 calibration tensor SHA 通过。大训练 tensor 未由本 CPU 发布器全部重新哈希，仍明确限定原 loader 验算范围。全部 270000 条 source/preprocessing/SNR/noise/arm 键、公式、完整均值、配对失败、全历史选模最小值，以及真实 terminal/selected 填充 optimizer 与 RNG 恢复 state 已核验；CUDA 被屏蔽。

首次正式调用在创建目录前拒绝原 hybrid completion 的字段 schema：原 short_prefix.train 不写 synthetic 字段，scoped C_train 则写 false。仅修正两个未绑定 repeat CPU 发布器，允许已核验 hybrid 臂的原缺字段 schema；pure 仍必须显式 false，任意臂显式 true/null/0/string 均拒绝。两项新增回归覆盖初始与扩展路径；原失败日志保留于 outputs/monitoring，不是 GPU 实验失败，所有活动实验/worker 源码未改。

结果目录：results/token_channel_efficiency_20260923/C_initial_milestones/N4084_m6_seed2026092404_20k。
47 项索引文件共 63777146 字节，每个文件小于 10 MB；全部 SHA/大小、两 SVG XML 及同数据 PNG 视觉检查通过。包含全部原校准 CSV、源均值、SNR/分项曲线、资源账本、selected/lineage、原决策及缺口证据。目录不可覆盖，勿重跑 main。未上传权重、原图或缓存 tensor。

index SHA256：b8c74b6e171c9ffcbe60d680bba5152f7237643f688bc0216c2b2a0e144b8573
initial publisher SHA256：4718909ec3d39da9f6542a348cf6b8fe42d629dd7ea9f70a05a74926dcc0ad6f
extension publisher SHA256：11a4689411e78a0cb7104bce21a7170a84c795f5e5b5d57079a729c047040f99

提交前运行完整 CPU/repository/release/fsck 检查。独立远端检出验收的实际状态以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m6_seed2026092404_20k_verification/receipt.json 为准；CPU 发布一致性不是新增 GPU 质量验收。后续所有额外 seed、正式 selected 评测/诊断/时间、四历史 GPU worker 及最终全方法配对交付仍待完成。
