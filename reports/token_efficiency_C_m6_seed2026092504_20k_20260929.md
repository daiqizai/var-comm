# C 第三训练 seed m6 初始 20k 校准发布

日期：2026-09-29。N4084 / seed2026092504，原 short_prefix.train 从零训练，parent=null、parent_updates=0；H6-V/H6-P各完成20000更新。原at_20000正式决定成对延长至30000，两臂均满足原连续两段至少0.2%的条件；不是收敛或完整交付。

## 完整校准与选择

原1000校准源、五SNR、三个原噪声seed，0至20000每2500更新全校准，九轮270000条完整记录、90000逐源噪声均值。完整source/preprocessing/SNR/noise/run键、指标公式、均值、配对失败及全历史选模通过独立核验。六项原决策evidence SHA通过，未使用development。

| 臂 | selected更新 | utility | 15000→17500改善 | 17500→20000改善 |
|---|---:|---:|---:|---:|
| H6-V | 20000 | 0.020083768816214674 | +1.792933% | +1.152207% |
| H6-P | 20000 | 0.020064594345757115 | +0.579590% | +0.277211% |

V在12500至15000的utility从.02062686349839593退步至.02068880940183687；P在17500至20000的MSE从.006202315134927631退步至.006212035036263599，其他分项改善使utility下降。全部曲线与退步保留，不修改原选择规则。20k每臂0header失败、2825bodyCRC失败（1dB2816、4dB9）。复用同一数字cache的两臂九轮评分，不是270000次独立数字PHY传输；校准结果不能当正式development质量。

## 原阶段回执缺口

原内外层分别检测到thermal，双方均有软件热标志。inner PID4173540/start127635148于1790628636.8183398记录thermal79C；outer PID4173297/start127629965于1790628645.1328974记录thermal83C。本次不是外层requested_stop传播顺序。完整20k终端资产mtime=1790628755.8763912；两原进程实际退出，外层stopping分支未归档stage。原重启4176832/start127696340于1790628807.5053458读取资产登记延长。

未改CPU发布器显式allow-thermal-receipt-gap核验原launch/退出身份、双方热标志、严格时间顺序、完整无未知失败console、终端资产及原重启决策。原始证据封存在thermal目录。明确状态：REAL_C_REPEAT_INITIAL20K_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING；complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。不伪造原stage/exit0/finalization，不重训补回执，不写回原调度outputs。

- registration SHA：d9ab9cd035f3dbca5ca23ddc12e3a084582cfcf24a4df9e2b34cb1f4b1c6a392
- 两臂selected与terminal checkpoint SHA：a45ab60decc802cd15fcf9b4399b92476dfa2381e2d70a4df16f1c8bb694db2f
- 原发布器 SHA：184d4a66c77eb4718202cebb0cb357bd41a30eadf14ed20043d6a99865d9cedf
- 发布index SHA：e7a15f80b33feca1a7a5140cb7b39a0cada1be2d1386e8d3ff05814a9643433f

真实20k参数、已填充optimizer和RNG恢复state经CUDA屏蔽CPU验证通过；原真实GPU梯度/能量/optimizer隔离/bitwise resume qualification与probe丢弃身份核验通过。54合并绑定、210cache shard元数据、两role completion/registration与10calibration tensor SHA核验通过。训练大tensor没有由发布器全量重哈希，保留原loader实际验证范围。

## 资源与发布审阅

N4084=NH68+ND1200+NA2816。source payload1092bit；header class10bit+mode2bit+CRC16+tail6，母码68bit、发送136coded slots；body CRC16+tail6，母码2228bit、发送2400coded slots。source bit不能代替信道N。E8168仅原登记约束，校准CSV不含新的逐帧实测E。训练16017.511226654053秒、校准3190.592846393585秒不是完整在线TX/RX时间；正式selected development、实际E和在线计时仍待原队列。

结果：[完整索引](../results/token_channel_efficiency_20260923/C_initial_milestones/N4084_m6_seed2026092504_20k/index.json)。47索引文件共63774199字节，全部SHA/大小通过且每项<10MB。原CSV、源均值、SNR曲线、selected/lineage、decision和资源账本齐全；两SVG完成XML检查，同源五指标PNG已视觉审阅。发布目录不可覆盖，权重、源图与大cache不上传。

根工程完成CPU/repository/release/fsck后正常提交推送，独立checkout重新验证全部文件、270000校准记录、90000源均值、完整决策、真实terminal CPU与热事件序列。成功以outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_m6_seed2026092504_20k_verification/receipt.json为准，成功回执前不宣称独立验收通过；CPU发布一致性不等于新增GPU质量。

本次未修改活动实验/worker源码、模型、loss、FEC、协议、硬件或队列。第三seed后续延长及m8/freshpure、正式selected评测/诊断/完整在线时间、四历史GPU worker与最终全方法配对报告尚待完成。新holdout和内容选择器继续暂缓，图像bootstrap与训练seed变异分开。
