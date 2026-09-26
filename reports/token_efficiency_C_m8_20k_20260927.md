# C / m8 首轮20k真实校准里程碑

实际完成：2026-09-27（北京时间）。本阶段完成；合并实验尚未完整交付。

H8-V完成20,000次更新，selected为20,000步。本组仅登记H8-V，没有H8-P。保持原有效batch16/micro4、FP32、冻结视觉模型与Decoder及原loss。H8-V有567,928个可训练参数，另使用冻结VAR模型；不宣称全系统与其他方法同参数或同计算。

## 真实身份与校准核验

未改独立CPU发布器tools/publish_c_initial_milestone.py --m 8核验成功stage snapshot、56项源码/资产绑定、真实训练及共享cached/online/replay资格、cache注册/完成/210份分片元数据、10个校准tensor SHA、selected checkpoint身份。训练大tensor未由发布器重复哈希，保留实际训练loader核验的清单。

9轮0/2500/…/20000，每轮1000原校准源×五SNR×三noise×单臂，共135,000条真实校准记录。完整无重复source/SNR/noise/arm键、有限指标、U_image与含header掩码的utility公式、失败状态、每轮均值及全历史最小utility选模通过核验。

| 臂 | selected utility | 15000→17500改善 | 17500→20000改善 |
|---|---:|---:|---:|
| H8-V | 0.027237659834418445 | 0.388777% | 0.629655% |

checkpoint SHA：1cac15785f32521da2bb25dcc542ab322f7295754995b9ed727f62df8a51e3f4。
registration SHA：79a2e13d35416c2d7a84960ee70020bfb421bc5e67b589b3c722afd38b666407。

两段改善均≥0.2%，支持原规则的10k延长；正式决策由原调度器完成初始矩阵后执行，发布器不登记决策、不启动训练，不将20k称为收敛。原队列已自动进入C_initial_pure登记对照，尚未进入正式C延长阶段。

## 分项、失败与资源

| SNR/dB | H8-V U_image | header失败/3000 | body CRC失败/3000 |
|---|---:|---:|---:|
| 1 | 0.03256095 | 0 | 3000 |
| 4 | 0.02011522 | 0 | 55 |
| 7 | 0.01903818 | 0 | 0 |
| 13 | 0.01798901 | 0 | 0 |
| 19 | 0.01765782 | 0 | 0 |

20k的15,000条记录保留0 header失败、3055 body CRC失败，包括1dB全部3000条body失败。九轮checkpoint复用数字cache，135,000条重评分不是135,000次独立数字PHY传输，也不是独立源图样本。该校准结果不代替development比较或训练seed变异。

N4084=header68+数字2992+连续1024。source payload3060bit，body加CRC16/tail6后3082bit，母码6164bit，原puncture规则发送5984coded slots。header class10+mode2bit，加入CRC16/tail6后母码68bit，发送136coded slots/68 QPSK uses。资源字段详见resource_ledger.json，source bit不能代替N。

登记能量约束E8168不是本轮新测量值：原校准CSV未导出逐帧实测E或在线时间。selected development实际E与完整在线计时仍待执行，不能据此给出同质量N节省率。

## 发布与后续

[结果索引](../results/token_channel_efficiency_20260923/C_initial_milestones/m8_seed2026092304_20k/index.json)保存9份完整原CSV及receipt、逐源逐SNR均值、分项/SNR曲线、两张SVG、source/预处理/noise/cache身份、selected/lineage、资源账本与核验记录。262项索引文件共35,310,302字节，逐项SHA/大小通过，均小于10MB；两SVG语法及同数据PNG视觉检查通过；不上传权重、tensor或原图。

后续仍需初始pure对照、C原规则延长、N3060校准选m与复用同一P3060、额外两个训练seed和直接控制、selected质量/实际E/在线计时、无增强参考和跨噪声源知情诊断、四个历史GPU worker、最终全方法配对统计与发布。诊断不是部署选择器，新holdout和内容选择器继续暂缓。旧completion的new test模板由补充协议覆盖。

工程CPU测试不等于新GPU质量验收。独立远端发布验证入口为outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/remote_C_m8_20k_verification；只有实际成功receipt才代表远端复核通过。

发布前201项CPU测试、仓库/release及diff检查通过，均未执行GPU质量评测。
