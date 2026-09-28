# C 第二训练 seed H8-V 初始 20k 校准发布

日期：2026-09-28。N4084 / seed2026092404，原 short_prefix.train 从零训练，parent=null、parent_updates=0。唯一登记臂 H8-V 已完成 20000 更新；没有 H8-P。原调度 at_20000 决策 extend=true，接续至 30000，不能将本阶段称为收敛或完整实验交付。

## 实际校准与选模

原 1000 校准源、五个 SNR、三个噪声 seed，在 0、2500、5000、7500、10000、12500、15000、17500、20000 九轮形成 135000 条完整记录及 45000 条逐源噪声均值。独立 CPU 发布器核验全部 source / preprocessing / SNR / noise / run 键、指标公式、均值、失败记录与全历史选模。

H8-V selected=20000，utility=0.02727551174464946；末两区间改善为 +0.592121% / +0.254459%，均满足原连续两段至少 0.2% 的条件，因此原规则延长 10000 更新。决策的六项原始证据 SHA 均已核验，没有使用 development。

20k 的 15000 条记录保留 0 次 header 失败、3055 次 body CRC 失败，其中 1dB 为 3000、4dB 为 55。所有失败均进入原统计。各轮复用原数字 cache；135000 条重评分不是 135000 次独立数字 PHY 传输。

## 执行、资源与可追溯性

本次原 stage 完整，准确命令 short_prefix.train --group m8 --seed 2026092404 --until 20000、returncode=0 与不可变 completion snapshot 一致；complete_stage=true、terminal_artifacts_verified=true、process_returncode=0。此前 m6 的热暂停回执缺口仍按其历史发布保留，不能套用于本次。

- registration SHA：419905d41f4517f960a4f00d8ab8f2fd0db5d2bb752d646ca0cd42895c6d205b
- selected / terminal checkpoint SHA：8221fc07089824f6967c259aa38f39ca5bff0c565d45acc67eb8f539c1947b3e
- 原 immutable completion snapshot SHA：655ea157d344227a796622f897fb000d8c9e5a8c84926034f7b52d9a66510274
- 发布器 SHA：4718909ec3d39da9f6542a348cf6b8fe42d629dd7ea9f70a05a74926dcc0ad6f
- 发布 index SHA：5db61347d1aed1fcd4542c840f32380c37db441c4465ca7eb0eec947c286b089

真实 terminal CPU 参数、已填充 optimizer 和恢复 RNG/state、原实际 GPU 梯度/能量/optimizer 隔离/bitwise resume qualification 及 probe 丢弃身份已核验。54 项合并绑定、两 role 缓存 completion/registration、210 份 shard 元数据及 10 份 calibration tensor SHA 通过。训练大 tensor 未在本发布器中全部重哈希，仍明确限定原 loader 已验证的范围。

N4084 = NH68 + ND2992 + NA1024。source payload 为 3060bit；头 class10 + mode2 + CRC16 + tail6，经母码形成 68bit、发送136 coded slots；body 母码6164bit，原 puncture 发送5984 coded slots。source bit 不等于信道 N。E8168 是原登记约束，校准 CSV 没有新的逐帧实测 E；训练/校准耗时不作为完整在线 TX/RX 时间。

## 数据与审阅

结果目录：[N4084_m8_seed2026092404_20k](../results/token_channel_efficiency_20260923/C_initial_milestones/N4084_m8_seed2026092404_20k/index.json)。42 项索引文件共 33769603 字节，逐文件 SHA/大小与每项小于10MB检查通过。包含全部九轮原 CSV、逐源均值、按 SNR 曲线、selected、正式 decision、lineage、资源账本及原完整 stage。两 SVG 已完成 XML 检查，并以同源 CSV 渲染五指标 PNG 完成视觉审阅。发布目录不可覆盖。

本报告与轻量结果在根工程正常提交、推送，并由独立 checkout 执行 CPU/repository/release/fsck、全部发布 SHA、135000 条 CSV/45000 源均值、正式决策与真实 terminal CPU 状态复验；实际完成状态以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m8_seed2026092404_20k_verification/receipt.json 为准，未生成成功回执前不宣称远端验收通过。

本次没有修改活动模型、视觉权重、loss、FEC、协议、硬件或队列。正式 selected development 质量、逐帧实际能量与完整在线计时仍待原队列执行；图像 bootstrap 与训练 seed 变异须分开。新 holdout 和内容选择器继续暂缓。后续另一个 seed、纯连续直接控制、诊断、四个历史 GPU worker 和最终全方法配对报告仍待完成。
