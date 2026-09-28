# N4084 m6 第二训练 seed 的 40k 延长校准资产

2026-09-28，北京时间。seed 2026092404 的 H6-V / H6-P 各完成 40000 更新，原 short_prefix.train 从零训练（parent=null、parent_updates=0）。十七轮完整校准（0 至 40000，每 2500）共 510000 条记录已独立 CPU 重审；本次新增四轮 120000 条记录及 40000 条逐源噪声均值，此前十三轮 390000 条通过 20k/30k已发布 index 和逐文件 SHA 引用。本次校准资产交付不代表正式 selected development、在线计时或整个合并实验完成。

两臂 selected 均为 40000，utility V=0.01955087842933523、P=0.019440962665419406。共同 terminal/selected checkpoint SHA256：
d8d107591c69782a240fb00c62f77a775ea80f5274b3abff4999a8233e7d21ad

registration SHA256：
39ef84cc5d32d36897c1e5ed58c9ac709db1e8fbe14409ea994bd4576211bc66

真实 30k extension parent checkpoint SHA256：
69caefad3bb900cdd5ad4620348b4f580230e2b654f64ef4196847c82d7ac488

V 最后两段相对改善为 0.267139% / 0.170973%；P 为 0.455292% / 0.154185%。两臂最后一段均低于 0.2%，原 at_40000 正式决定 extend=false、paired_arms_extend_together=true、until=40000。m6 按原规则停止，原队列已实际接续同 seed 的 m8；不把规则停止称为理论收敛，CPU 发布器不启动训练。

每臂末轮 15000 条中 header 失败 0、body CRC 失败 2825（1dB 2816、4dB 9），全部保留。十七轮复用原数字 cache，510000 条评分不是独立数字 PHY 传输。N4084=68+1200+2816；source1092bit、header12bit 加 CRC16/tail6、body 母码2228bit、发送2400coded slots。E8168 仅为原登记约束，不是本校准新增逐帧实测。训练/校准秒数不能代替完整在线 TX/RX 时间。正式 selected development、实际能量、统一在线计时仍待原队列执行；训练 seed 变异与源图 bootstrap 分开。

## 原热暂停归档缺口

outer 3861541/start122261300 在 1790574888.3082209 记录 thermal（84C、软件热标志）；inner 3861800/start122266479 在 1790574889.9519155 收到传播的 requested_stop（80C、软件热标志）。40k 完整校准、terminal checkpoint、selected 和 completion 在 mtime=1790575042.8600986 完成。外层 stopping 分支未归档 C_N4084_m6_seed2026092404_until40000 stage，原两进程已实际退出。原重启 3864955/start122325060 于 1790575094.7140124 读取终端资产并登记停止及接续 m8。

未改的独立 CPU tools.publish_c_repeat_extension 显式使用 allow-thermal-receipt-gap，核验原准确命令、原进程身份/退出、双方热证据、严格时间顺序、无未知失败的完整日志、原重启与正式决策、终端资产后封存。状态 REAL_C_REPEAT_EXTENSION_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING；complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。未伪造原 stage、exit0 或 finalization，未重训补回执，未改写原 outputs。本次按原正式停止决定封存边界，缺失调度回执仍原样披露。

## 数据复核与发布

54 合并绑定、原 candidate、第一 seed 发布 index、Decoder、真实 qualification、两 role cache completion/registration、210 shard 元数据、10 calibration tensor SHA 均通过。大训练 tensor 未由本发布器全部重哈希，仍明确限定原 loader 验算范围。全部十七轮 source/preprocessing/SNR/noise/arm 键、公式、完整均值、配对失败、全历史 selected 最小值、原 20k/30k 发布 index/所有文件 SHA 和正式延长、停止决策通过；真实 30k/40k CPU checkpoint 的参数、填充 optimizer 与 RNG 恢复状态通过。CUDA 被屏蔽。

结果目录：results/token_channel_efficiency_20260923/C_extensions/N4084_m6_seed2026092404_until40000。
30 项索引文件共 30662054 字节，每个小于10MB；全部 SHA/大小、两 SVG XML 及同源 PNG 视觉检查通过。包含新增原校准 CSV、源均值、完整历史曲线与 SNR 分项、资源账本、selected/lineage、原决策和热缺口证据。旧记录按已发布 SHA 引用，不重复上传。目录不可覆盖，勿重跑 main。未上传原图、权重或缓存 tensor，活动实验/worker 源码和队列均未修改。

index SHA256：f70263ca22426faec9b011fc9ee677c00e36c8867bc9bdb1014065bd6fd990b1
extension publisher SHA256：11a4689411e78a0cb7104bce21a7170a84c795f5e5b5d57079a729c047040f99

提交前运行 root 完整 CPU/repository/release/fsck 检查。独立远端检出验收的实际状态以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m6_seed2026092404_40k_verification/receipt.json 为准；CPU 发布一致性不是新增 GPU 质量。后续第二 seed m8/fresh pure、第三 seed、正式 selected 评测/诊断/计时、四历史 GPU worker 和最终全方法配对交付仍待完成。
