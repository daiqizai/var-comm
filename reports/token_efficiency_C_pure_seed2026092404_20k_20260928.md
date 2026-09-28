# C 第二训练 seed 纯连续 P4084 初始 20k 校准发布

日期：2026-09-28。N4084 / seed2026092404，原 token_efficiency.C_train scoped_N4084 从零训练，parent=null、parent_updates=0。唯一 P4084 已完成 20000 更新；与第一 seed 继承历史 10k 的 pure 不同。原 at_20000 决策 extend=true，自动接续 30000；本阶段不是收敛或完整实验交付。

## 真实校准与原决策

原 1000 校准源、五个 SNR、三个噪声 seed，在 0 至 20000 每2500更新全校准，九轮共135000条完整记录及45000条逐源噪声均值。全部 source / preprocessing / SNR / noise / run 键、指标公式、均值和全历史选模已独立核验。selected=20000，utility=0.016495200846251102；末两区间改善 +1.248089% / +1.050833%，两段均满足原至少0.2%的条件，正式延长10000更新。六项原校准证据SHA通过，未使用development。

纯连续没有数字CRC：body_crc_ok=not_applicable，header_ok=1只是无头序列化约定，不能宣称数字头或CRC传输成功。九轮复用原F缓存，不是新增独立PHY传输。20k MSE=.005316410425906846、LPIPS-Alex=.08597435777485371、normalized_latent=.2581354541413486、U_image=.01391384636441556，均为calibration，不能当development质量。

## 原 stage 回执缺口与资产核验

本次原内外层收到thermal/requested_stop，训练仍完整写完20k全校准、terminal checkpoint、selected与completion；外层stopping分支退出前未归档原stage，原重启读取资产后登记延长。独立发布器显式allow-thermal-receipt-gap已验证原launch/实际退出身份、双方热标志、严格时间顺序、完整无未知失败console、原重启和正式decision。completion mtime=1790602534.5785973。原始launch/stop/console按SHA封存于thermal目录。

明确状态：REAL_C_REPEAT_INITIAL20K_CALIBRATION_VERIFIED_STAGE_RECEIPT_MISSING；complete_stage=false、terminal_artifacts_verified=true、process_returncode=null。不伪造原stage、exit0或finalization，不重训补回执，不写回原调度outputs。

- registration SHA：81ef88ebbadb9be16c17f5ff52ca2ad10dc2073a7c28a6a2df34093dea5c3dd0
- selected / terminal checkpoint SHA：afe13cd12777d54973a0db9b1e98d14727a898c5007cc36fbe0562ddb199a358
- 发布器 SHA：184d4a66c77eb4718202cebb0cb357bd41a30eadf14ed20043d6a99865d9cedf
- 发布 index SHA：6a008d87d1da679ac9617dd63709ab64d8d8c00839a806024745a2104a401535

真实20k CPU参数、已填充optimizer和恢复RNG/state、原GPU梯度/能量/optimizer隔离/bitwise resume qualification及probe丢弃身份通过。55项合并绑定、210份cache shard元数据、两role completion/registration及10份calibration tensor SHA通过。原F registration的历史source_snapshot依据第一seed完整发布索引核验，不能冒称当前运行源码绑定；当前training/runtime源码仍逐SHA核验。训练大tensor没有由发布器全部重新哈希，保留原loader实际验证范围。

## 资源、数据与审阅

N4084全部连续：NH=0、ND=0、NA=4084；source payload bits不适用，数字控制/FEC/coded slots均0，不等于零信道资源。参数522880、Decoder冻结。E8168仅原登记约束，校准CSV没有新的逐帧实测E；训练/校准秒数不是完整在线TX/RX时间。正式selected development、实际E、完整在线计时仍待原队列。

结果目录：[N4084_pure_seed2026092404_20k](../results/token_channel_efficiency_20260923/C_initial_milestones/N4084_pure_seed2026092404_20k/index.json)。47项索引文件共35771917字节，全部SHA/大小且每项<10MB通过；完整九轮CSV、逐源均值、SNR曲线、selected、decision、lineage、资源账本与回执缺口证据保留。两SVG完成XML检查，并用同源CSV渲染五指标PNG完成视觉审阅。发布目录不可覆盖。

根工程按规则完成CPU/repository/release/fsck后提交、正常推送；独立checkout复验全部SHA、135000条原CSV/45000源均值、公式/决策/真实terminal CPU及热事件序列。实际验收以outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_repeat_pure_seed2026092404_20k_verification/receipt.json为准，成功回执前不宣称独立验收通过；CPU发布一致性不等于新增GPU质量。

本次未改活动实验/worker源码、模型、loss、FEC、协议、硬件与队列。后续pure延长、第三seed、正式selected评测/诊断/完整在线计时、四历史GPU worker及最终全方法配对报告仍待完成。新holdout和内容选择器暂缓，图像bootstrap与训练seed变异分开。
