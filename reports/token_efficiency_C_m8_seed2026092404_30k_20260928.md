# N4084 m8 第二训练 seed 的 30k 延长校准资产

2026-09-28，北京时间。seed 2026092404 的唯一 H8-V 完成 30000 更新，无 H8-P；原 short_prefix.train 从零训练，parent=null、parent_updates=0。十三轮完整校准（0 至 30000，每 2500）共 195000 条记录已独立 CPU 重审。本次新增四轮 60000 条及 20000 条逐源噪声均值，初始九轮 135000 条由已发布 index 和逐文件 SHA 引用。本次是校准资产交付，正式 selected development、完整在线计时及整个合并实验尚未完成。

selected 为 30000 步，utility=0.02710457253693603，checkpoint SHA256：
1df01beab8fa4f2e4b09a45c0581f8043d7f64930b011fa5ee90c02c86bd76d6

30k utility=0.02710457253693603；terminal checkpoint SHA256：
1df01beab8fa4f2e4b09a45c0581f8043d7f64930b011fa5ee90c02c86bd76d6

registration SHA256：
419905d41f4517f960a4f00d8ab8f2fd0db5d2bb752d646ca0cd42895c6d205b

真实 20k extension parent checkpoint SHA256：
8221fc07089824f6967c259aa38f39ca5bff0c565d45acc67eb8f539c1947b3e

末两段相对改善 0.195699% / 0.150894%。原 at_30000 正式 extend=false，until=30000；遵守连续两段至少 0.2% 的原规则，不把规则停止或延长称为理论收敛。本次两段均低于 0.2%，原队列正式停止 m8 延长；25k utility 的短暂退步与末段 MSE 增长均保留。单臂登记不解释为双臂。CPU 发布器不启动训练。

末轮 15000 条中 0 header 失败、3055 body CRC 失败（1dB 3000、4dB 55），全部保留。十三轮复用原数字 cache，195000 条评分不是独立数字 PHY 传输。N4084=68+2992+1024；source3060bit、header12bit 加 CRC16/tail6，body 母码6164bit，puncture 发送5984coded slots。E8168 仅为原登记约束，不是本校准新增逐帧实测。训练/校准秒数不能代替完整在线 TX/RX 时间，源图 bootstrap 与训练 seed 变异分开。

## 阶段证据

原热暂停归档缺口保留：outer 3953704/start123864078 于 1790591597.9680061 记录 thermal；inner 3953943/start123869261 于 1790591599.6608632 接收 requested_stop，双方原热标志和严格时间序列已核验。30k 完整终端资产于 mtime=1790591757.9151678 写完，两原进程已实际退出，外层 stopping 分支未归档原 stage。原重启 3961306/start123996630 于 1790591810.4085417 读取资产后登记正式决策。
独立 CPU 发布器显式使用 allow-thermal-receipt-gap，核验原 launch/准确命令/退出身份/双方热证据/完整无未知失败日志/重启与正式决策，封存真实终端资产。complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。不伪造 stage、exit0 或 finalization，不重训补回执，不写回原调度 outputs。

发布状态 REAL_C_REPEAT_EXTENSION_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING。原活动实验/worker 源码、视觉模型、loss、FEC 母码、硬件和队列均未修改；未新建 holdout 或内容选择器。

## 数据复核与发布

54 合并绑定、原 candidate、第一 seed 发布 index、Decoder、真实 qualification、两 role cache completion/registration、210 shard 元数据及10 calibration tensor SHA 通过。大训练 tensor 未由本发布器全部重哈希，仍限定原 loader 验算范围。十三轮完整 source/preprocessing/SNR/noise/arm 键、公式、均值、失败、全历史 selected 最小值、20k 发布 index/全部文件 SHA 和正式决策通过；真实 parent/terminal/selected 的 CPU checkpoint 参数、已填充 optimizer 与 RNG 恢复状态通过，CUDA 屏蔽。

结果目录：results/token_channel_efficiency_20260923/C_extensions/N4084_m8_seed2026092404_until30000。
30 项索引文件，共 17191384 字节；每项小于10MB，全部 SHA/大小、两 SVG XML 及同源 PNG 视觉检查通过。包含新增原 CSV、源均值、完整历史曲线、SNR 分项、资源账本、selected/lineage、原决策与阶段证据。目录不可覆盖，勿重跑 main。旧校准逐 SHA 引用，未上传原图、权重或大缓存。

index SHA256：15da7516b760a1cefafb32984bc9d57f084a8362b67b1ba1fa06fb5ab330494c
publisher SHA256：11a4689411e78a0cb7104bce21a7170a84c795f5e5b5d57079a729c047040f99

提交前运行 root CPU/repository/release/fsck 检查。独立远端检出复验以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m8_seed2026092404_30k_verification/receipt.json 实际成功回执为准；CPU 发布一致性不是新增 GPU 质量。后续第二 seed fresh pure、第三 seed 及原校准决定的延长、正式 selected 评测/诊断/计时、四历史 GPU worker、最终全方法配对报告与独立远端验收仍待完成。
