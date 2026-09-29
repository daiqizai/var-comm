# C 第三训练 seed H8-V 初始 20k 校准发布

日期：2026-09-29。N4084 / seed2026092504，原 short_prefix.train 从零训练，parent=null、parent_updates=0。唯一 H8-V 已完成 20000 更新，无 H8-P。原 at_20000 决策 extend=true，已自动进入 30000 更新阶段；本阶段不代表收敛或完整交付。

## 校准、选模与失败记录

原 1000 校准源 × 五 SNR × 三噪声 seed，九轮 0 至 20000 更新共 135000 条完整校准记录、45000 条逐源噪声均值。全部 source/preprocessing/SNR/noise/run 键、指标公式、均值、失败和全历史选模已由独立 CPU 发布器核验。

selected=20000，utility=0.0272466456703376。最后两区间改善 +1.234375% / +0.302822%，满足原连续两段至少 0.2% 的条件，因此原规则延长 10000 更新。六项原决策证据 SHA 均通过，未使用 development 选模。

末轮 15000 条记录保留 0 次 header 失败、3055 次 body CRC 失败，其中 1dB 3000、4dB 55。各轮复用原数字 cache；135000 次评分并非同数量独立数字 PHY 传输。

## 原热暂停回执缺口

原 inner 170587/start131162757 于 1790664474.4192328、outer 170316/start131157570 于 1790664482.8337572 分别记录 thermal，均 81C 且软件热标志为 true。完整 20k 校准、terminal checkpoint、selected 和 completion 于 mtime 1790664635.5645204 写完；两原进程实际退出，外层 stopping 分支未归档原 stage。原重启 177731/start131284315 于 1790664687.2570026 读取终端资产并正式登记延长。

未改 CPU publisher 的显式 allow-thermal-receipt-gap 分支核验原内外 launch、已退出身份、双方热证据、严格时间序列、完整无未知失败日志、原重启和正式决策。发布状态为 REAL_C_REPEAT_INITIAL20K_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING：complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。没有伪造原 stage、exit0、finalization，也没有重训补回执或写回原调度记录。

- registration SHA：ce5931402b1f34967b0d7da26e3ac3535fe38cd0b1c87cd5d8e7c3fa7323f2bd
- selected / terminal checkpoint SHA：e253cdfc35ab11c5e66863e074c78e54f7527643d85f5b054aeb402996b34d26
- 捕获的 completion SHA：6e4c5ccbd4dd43dca77207a25356f67a870c7fbc2c4ec12f83140b8a1efcc7ea
- CPU publisher SHA：184d4a66c77eb4718202cebb0cb357bd41a30eadf14ed20043d6a99865d9cedf
- 发布 index SHA：ebe6dbc3708aed5d57437b66d5cc2018059afff39a1e23d8cdc1fddc8f252f3d

54 项合并绑定、10 份 calibration tensor SHA、两 role 缓存 completion/registration、210 份 shard 元数据，以及真实 terminal CPU 参数、已填充 optimizer、RNG 和恢复 state 均通过。原真实 GPU qualification 与 probe 丢弃身份保留；本 CPU 发布没有新增 GPU 验收。训练大 tensor 未全部重哈希，范围仍限定原 loader 验证。

## 资源与数据

N4084 = NH68 + ND2992 + NA1024。source payload 3060bit；头 class10 + mode2 + CRC16 + tail6，经母码形成 68bit、发送 136 coded slots；body 母码 6164bit，原 puncture 发送 5984 coded slots。source bit 不等于信道 N。E8168 是登记约束，校准 CSV 不含新的逐帧测量 E；训练/校准秒数不是完整在线 TX/RX 时间。

结果：[N4084_m8_seed2026092504_20k](../results/token_channel_efficiency_20260923/C_initial_milestones/N4084_m8_seed2026092504_20k/index.json)。47 项索引文件共 33773331 字节，逐文件 SHA/大小及每项小于 10MB 核验通过。包含全部九轮 CSV、逐源均值、SNR 曲线、selected、decision、lineage、资源账本与原热暂停证据。两 SVG 已通过 XML 检查，同源五指标 PNG 已完成视觉审阅。发布目录不可覆盖。

根工程 264 项 CPU 及 repository/release/fsck 已全部通过；独立远端完整复验按原流程执行，成功前不宣称远端验收通过。独立回执入口为 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m8_seed2026092504_20k_verification/receipt.json。

## 后续范围

此前 14 组已完成的 selected development 评测属于独立已发布阶段，其中不包含本第三 seed H8-V。当前仍需本臂 30k 校准决策、第三 seed fresh P4084、最终完整 selected 评测/实际 E/在线计时/诊断、四个历史 GPU worker 和最终全方法配对报告。图像 bootstrap 与训练 seed 变异分开，新 holdout 和内容选择器继续暂缓。活动模型、视觉权重、loss、FEC、协议、硬件和队列均未修改。
