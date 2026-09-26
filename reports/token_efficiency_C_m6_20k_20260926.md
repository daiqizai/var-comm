# C / m6 首轮20k真实校准里程碑

实际完成：2026-09-26（北京时间）。本阶段已完成；合并实验尚未完整交付。

H6-V与H6-P各完成20,000次配对更新，保持原共同初始化、独立optimizer、有效batch16/micro4、FP32及冻结视觉模型/Decoder。主资源仍为N4084，header68＋数字1200＋连续2816，登记逐帧能量约束E8168。两臂各584,070个可训练参数；VAR臂额外使用冻结VAR预训练模型与推理计算，不能称整个系统同参数或同计算。

## 真实验收与可追溯记录

独立CPU发布器 `tools/publish_c_initial_milestone.py --m 6` 核验了原调度器成功退出的完成快照、56项合并源码/资产绑定、训练及共享cached/online/replay真实GPU资格、cache注册/完成/分片元数据、全部10个校准tensor的实际SHA、训练registration与selected checkpoint SHA。训练cache大tensor本轮不重复哈希，保留实际训练loader已核验的原完成清单与全部分片身份；不声称本发布器重新验算了这些大tensor。

0/2500/…/20000共9轮，每轮1000原校准源×五SNR×三noise×两臂，总计270,000条真实记录。逐条核对source顺序、完整无重复网格、有限指标、配对数字失败状态、U_image及含header掩码的utility，并重算全历史最小值选模。没有访问development或新holdout。

两个selected均为20,000步：

| 臂 | 全五SNR校准utility | 15000→17500改善 | 17500→20000改善 |
|---|---:|---:|---:|
| H6-V | 0.020069305233191698 | 0.879715% | 0.751515% |
| H6-P | 0.019924074921756983 | 1.268603% | 0.254928% |

checkpoint SHA：`14c8fa4ce75fb25fc09cbbfcc28a92f3309bd2af16d25ba2c3ddfa34c274238e`。
registration SHA：`25fcd28c9ec902a0e725bbca572ff17561ae014cdb4e6a201f2338ae9a257d11`。

两臂末两区间均满足原登记“至少一臂连续两段改善≥0.2%，V/P共同延长10k”条件。这里只发布条件核验；正式延长决策和启动由现有C调度器在初始m6/m7/m8矩阵完成后执行，不另启训练，不把20k称为收敛。

## 校准分项与失败

| SNR/dB | H6-V U_image | H6-P U_image | 每臂body CRC失败/3000 |
|---|---:|---:|---:|
| 1 | 0.02406917 | 0.02415245 | 2816 |
| 4 | 0.01656768 | 0.01652517 | 9 |
| 7 | 0.01454696 | 0.01448638 | 0 |
| 13 | 0.01287117 | 0.01276120 | 0 |
| 19 | 0.01244303 | 0.01229252 | 0 |

以上为20k校准输出；各臂共15,000条记录中header失败0、body CRC失败2825，失败候选全部保留。两臂共享同一数字接收输入，不能把两臂或9次checkpoint重评分当作独立PHY传输/源图样本。V/P在1dB及其他SNR上的方向不同，不根据这张校准表宣布接收VAR有益/无益或优于纯连续；正式development配对比较、重复训练seed和计时尚未执行。

源payload为1092bit，body含CRC16/tail6后1114bit，母码2228bit、发送coded slots2400；header class10＋mode2bit、CRC16/tail6，母码68bit，经重复发送到136coded slots/68QPSK uses。完整资源定义见resource_ledger.json。校准原CSV不导出逐帧实际E/在线时间，本轮不把登记能量约束伪装成新测量；selected在线评测将给出实际E及端到端计时。

## 发布索引与后续范围

结果：[C_initial_milestones/m6_seed2026092304_20k](../results/token_channel_efficiency_20260923/C_initial_milestones/m6_seed2026092304_20k/index.json)。

该目录保留9份原CSV与SHA/receipt、每轮逐源逐SNR均值、全部校准分项、两张校准SVG、source/预处理与cache/noise/源码身份、selected/lineage、资源账本及独立核验receipt；不上传tensor、权重或原图。图像bootstrap与训练seed波动留到正式development及重复seed结果，不将本校准均值称为它们的置信区间。

原主链已自动进入m7缓存准备。后续仍包括m7/m8初始训练、C配对延长与P4084控制、按校准选择m的N3060 V/P及同一P3060复用、额外两个训练seed、selected评测/计时、无增强参考和跨噪声源知情诊断、四个历史GPU worker、最终全方法配对报告及远端验证。诊断不是部署选择器。原completion中的“new test”是旧模板描述，最新补充协议明确继续暂缓新holdout与内容选择器。

本轮发布器第一次预检查误假定训练qualification含bindings字段，在创建发布目录前停止；已依据实际schema改为训练registration绑定并补验原qualification stage snapshot。原GPU输出/绑定源均未改变，此事不是训练失败。首次定向测试用模块路径受tests非包布局影响，改用unittest discover后四项回归通过。

发布前验证：201项CPU测试通过（GPU未运行），仓库2203文件及release检查通过；262项index文件SHA/大小逐项通过，两张SVG语法和同数据PNG视觉检查通过。实际GPU实验身份/校准审计与CPU工程测试分开记录。远端独立检出receipt存于outputs同实验/remote_C_m6_20k_verification。
