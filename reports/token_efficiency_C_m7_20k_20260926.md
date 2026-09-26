# C / m7 首轮20k真实校准里程碑

实际完成：2026-09-26（北京时间）。本阶段完成；合并实验尚未完整交付。

H7-V与H7-P各完成20,000次更新，保持原共同初始化、独立optimizer、有效batch16/micro4、FP32及冻结视觉模型/Decoder。N4084由header68、数字1968、连续2048组成；登记逐帧能量约束为E8168。每臂577,152个可训练参数，VAR臂仍额外使用冻结VAR模型和推理计算，不宣称整个系统同参数或同计算。

## 真实身份与校准核验

现有独立CPU发布器 tools/publish_c_initial_milestone.py --m 7 核验成功stage snapshot、56项源码/资产绑定、真实训练资格及共享cached/online/replay资格、cache注册/完成/全部分片元数据、10个校准tensor SHA和selected checkpoint SHA。训练cache大tensor没有在发布器中重复哈希，保留实际训练loader已核验的完成清单与分片身份。

0/2500/…/20000共9轮，每轮1000原校准源×五SNR×三noise×两臂，总计270,000条真实校准记录。逐条核对完整无重复source/SNR/noise/arm网格、指标有限性、配对失败状态、U_image及含header掩码的utility，并核对每轮均值和全历史最小值选模。两个selected均为20,000步。

| 臂 | 校准utility | 15000→17500改善 | 17500→20000改善 |
|---|---:|---:|---:|
| H7-V | 0.022774326757605497 | 0.974828% | 0.520596% |
| H7-P | 0.023068429467811558 | -0.145076% | 1.078607% |

checkpoint SHA：4cb17164a2bad5d56e6c9cf3a3023e66e72f32b306ca89ed26775b6c66bb44f1。
registration SHA：35a5ebd92b83b76daf70f11bdb8734bc69643ac6d3c577683dda4efa28b7150a2。

V臂满足原登记“至少一臂连续两段改善≥0.2%，V/P共同延长10k”条件，P臂上一段退步完整保留。正式决策由原C调度器在m6/m7/m8初始矩阵完成后执行；本发布器不登记决策或启动训练，不将20k称为收敛。

## 分项与资源

| SNR/dB | H7-V U_image | H7-P U_image | 每臂header失败/3000 | 每臂body CRC失败/3000 |
|---|---:|---:|---:|---:|
| 1 | 0.02639232 | 0.02703301 | 1 | 2979 |
| 4 | 0.01800300 | 0.01822579 | 0 | 13 |
| 7 | 0.01629110 | 0.01641069 | 0 | 0 |
| 13 | 0.01484068 | 0.01490852 | 0 | 0 |
| 19 | 0.01444477 | 0.01451290 | 0 | 0 |

20k每臂15,000条记录中header失败1、body CRC失败2992，失败可重叠且全部保留。两臂与九轮checkpoint复用相同数字cache；270,000条重评分记录不等于270,000次独立数字PHY传输，也不是独立源图样本。此校准表不能代替development配对结论、训练seed波动或与纯连续方法的正式比较。

source payload1860bit，body加CRC16/tail6后1882bit，母码3764bit，发送coded slots3936。header class10＋mode2bit，加CRC16/tail6后母码68bit，发送136coded slots/68QPSK uses。完整定义见resource_ledger.json。校准CSV未导出逐帧测量E和在线时间，登记E8168不是本轮新测量；实际E与端到端时间待selected评测。

## 发布与后续

[结果索引](../results/token_channel_efficiency_20260923/C_initial_milestones/m7_seed2026092304_20k/index.json)包含9份完整原CSV与receipt、每轮逐源逐SNR均值、分项/SNR曲线、两张SVG、source/预处理/noise/cache身份、selected/lineage、资源账本和独立核验记录。262项索引文件共65,244,159字节，逐项SHA/大小通过，均小于10MB；SVG语法与同数据PNG视觉检查通过。不上传权重、tensor或原图。

原队列已自动进入m8训练cache。后续仍需m8初始训练、C配对延长、N3060校准选m与同一P3060复用、额外两个训练seed及直接控制、selected质量/实际E/计时、无增强参考和跨噪声源知情诊断、四个历史GPU worker，以及最终全方法源图配对统计和发布。诊断不是部署选择器。新holdout和内容选择器仍暂缓，原completion中的new test为已被补充协议覆盖的旧模板。

工程CPU测试、实际GPU身份/校准结果和独立远端发布验证分别记录。独立远端验证回执目录为outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_m7_20k_verification；只有实际成功receipt才表示远端复核通过。

发布前201项CPU测试通过，仓库及release检查通过（GPU未运行）；新增发布文件另逐项完成SHA/大小核验。
