# N4084 pure 第二训练 seed 的 30k 延长校准资产

2026-09-28，北京时间。seed 2026092404 的唯一 P4084 完成 30000 更新，原 token_efficiency.C_train scoped_N4084 从零训练，parent=null、parent_updates=0。十三轮完整校准（0 至 30000，每 2500）共 195000 条记录已独立 CPU 重审。本次新增四轮 60000 条及 20000 条逐源噪声均值，初始九轮 135000 条由已发布 index 和逐文件 SHA 引用。本次是校准资产交付，正式 selected development、完整在线计时及整个合并实验尚未完成。

selected 保留 30000 步，utility=0.016116970843588933，checkpoint SHA256：
d5cb513fcb9741db1cd882772adbca382e0053a83f398ea7a26c95bb0c62f94e

30k utility=0.016116970843588933；terminal checkpoint SHA256：
d5cb513fcb9741db1cd882772adbca382e0053a83f398ea7a26c95bb0c62f94e

registration SHA256：
81ef88ebbadb9be16c17f5ff52ca2ad10dc2073a7c28a6a2df34093dea5c3dd0

真实 20k extension parent checkpoint SHA256：
afe13cd12777d54973a0db9b1e98d14727a898c5007cc36fbe0562ddb199a358

末两段相对改善 0.881347% / 0.001967%。原 at_30000 正式 extend=false，until=30000；遵守连续两段至少 0.2% 的原规则，不把规则停止或延长称为理论收敛。单臂登记不解释为双臂。末段图像 MSE、LPIPS 与 U_image 均小幅退步，而 normalized_latent 改善使原选模 utility 略降，完整保留，不改选模规则。CPU 发布器不启动训练。

纯连续无数字 CRC；body_crc_ok=not_applicable，header_ok=1仅无头序列化约定，不代表数字 PHY 成功。十三轮复用原 F cache，195000 条评分不是新增独立 PHY 传输。N4084 全部连续，source bits 不适用，数字头/控制/FEC/coded slots 均为0。E8168 仅为原登记约束，不是本校准新增逐帧实测。训练/校准秒数不能代替完整在线 TX/RX 时间，源图 bootstrap 与训练 seed 变异分开。

## 阶段证据

原 stage 完整：准确命令 token_efficiency.C_train --N 4084 --group pure --seed 2026092404 --until 30000、returncode=0；不可变 completion snapshot SHA256 92f895573f4841e9a80e618de253153f9be864a02044bc9ad1bd0aaccff2bbbb。默认发布分支通过，未使用热缺口分支。

发布状态 REAL_C_REPEAT_EXTENSION_CALIBRATION_VERIFIED。原活动实验/worker 源码、视觉模型、loss、FEC 母码、硬件和队列均未修改；未新建 holdout 或内容选择器。

## 数据复核与发布

55 合并绑定、原 candidate、第一 seed 发布 index、Decoder、真实 qualification、两 role cache completion/registration（F 历史 source_snapshot 按第一 seed 发布身份核验，当前 runtime 绑定独立核验）、210 shard 元数据及10 calibration tensor SHA 通过。大训练 tensor 未由本发布器全部重哈希，仍限定原 loader 验算范围。十三轮完整 source/preprocessing/SNR/noise/arm 键、公式、均值、失败、全历史 selected 最小值、20k 发布 index/全部文件 SHA 和正式决策通过；真实 parent/terminal/selected 的 CPU checkpoint 参数、已填充 optimizer 与 RNG 恢复状态通过，CUDA 屏蔽。

结果目录：results/token_channel_efficiency_20260923/C_extensions/N4084_pure_seed2026092404_until30000。
25 项索引文件，共 18122421 字节；每项小于10MB，全部 SHA/大小、两 SVG XML 及同源 PNG 视觉检查通过。包含新增原 CSV、源均值、完整历史曲线、SNR 分项、资源账本、selected/lineage、原决策与阶段证据。目录不可覆盖，勿重跑 main。旧校准逐 SHA 引用，未上传原图、权重或大缓存。

index SHA256：05c3bfbaa9e3c0fe74b84a30cd8c9153c4570d68a2ddbb037da8d0d847abcf47
publisher SHA256：11a4689411e78a0cb7104bce21a7170a84c795f5e5b5d57079a729c047040f99

提交前运行 root CPU/repository/release/fsck 检查。独立远端检出复验以 outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_pure_seed2026092404_30k_verification/receipt.json 实际成功回执为准；CPU 发布一致性不是新增 GPU 质量。后续第三 seed 及原校准决定的延长、正式 selected 评测/诊断/计时、四历史 GPU worker、最终全方法配对报告与独立远端验收仍待完成。
