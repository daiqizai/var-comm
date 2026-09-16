# 通信骨干2×2：完成结果与解读边界

本轮于2026-09-12 01:43:30（UTC+8）全部完成：训练01:29:06结束，16800行development评测01:43:11结束，随后统计完成。监测器01:44自动结束；本轮GPU占用已释放。
四臂均实际完成20000更新（10000 warmup+2000 prefix+8000 joint）、每臂80000图像曝光。旧B、视觉骨干、m8、3060 uses及原hard/ST规则保持不变；全程训练microbatch=2、有效batch=4，microbatch=4没有正式启用。

## 主要结论

**本次未证实增强通信骨干相对同历史小骨干带来稳定质量增益。** 两个next-scale都选中总16000步，增强版LPIPS均值略低，但PSNR/LPIPS/DINO三项配对区间均跨0。不能据此说两模型完全等效，也不能说所有更大网络无效。
固定数字m8的LPIPS和DINO仍更好；这轮没有解决残余通信映射差距。不自动将增强模型替换为主基线或追加训练。

## 主1/4/7 dB结果

原100张development图，每图三个噪声；先按源图平均噪声和三个SNR。均为3060 complex uses；Deep为0 header+3060 data，其他为68+2992。

| 模型 | PSNR ↑ | LPIPS ↓ | DINO ↑ |
|---|---:|---:|---:|
| 本轮小骨干next-scale | 18.584340 | 0.238414 | 0.831542 |
| 本轮增强next-scale | 18.617943 | 0.236827 | 0.830332 |
| 冻结旧B参考 | 18.505096 | 0.238479 | 0.832665 |
| 固定数字m8 | 18.653161 | 0.214927 | 0.844922 |
| 数字自适应 | 19.626654 | 0.183763 | 0.886743 |
| 感知DeepJSCC | 24.378339 | 0.205291 | 0.544427 |

旧B仅工程参考：它的预训练历史和TX SNR使用与本轮重建的四臂不同，不用它替代严格架构对照。

### 增强next-scale减小骨干next-scale

| 指标 | 配对差 | 95% CI |
|---|---:|---|
| PSNR | +0.033603 dB | [-0.045802, +0.097327] |
| LPIPS | -0.001586 | [-0.004106, +0.000725] |
| DINO | -0.001210 | [-0.005005, +0.002675] |

LPIPS描述性相对改善约0.67%，不能把跨0的区间包装成明确增益。
相对旧B，增强版LPIPS差-0.001652，CI [-0.004115,+0.000798]；相对固定数字m8，LPIPS仍差+0.021900，CI [+0.017094,+0.026567]。

## Parallel回退必须单独解释

| Parallel臂 | 实际训练到 | 按协议输出的checkpoint | 资格 |
|---|---:|---:|---|
| small_parallel | 20000 | 10000（warmup端点） | 不合格fallback |
| enhanced_parallel | 20000 | 10000（warmup端点） | 不合格fallback |

两条parallel所有完整校准候选均未通过预设的共同PSNR约束。warmup端点在多个SNR低于阈值；B阶段各端点主要是1 dB未达标。
到20000步，small/enhanced parallel的1 dB PSNR相对共同起点分别下降0.355/0.278 dB，仍超过允许的0.2 dB。因此按预注册规则回退，没有事后修改阈值或挑选另一checkpoint。
**原始主表中next-scale相对parallel的巨大差值，是相对warmup回退输出的差，不是充分经过B训练的parallel对照结果，不能直接作为B配方下next-scale结构增量。** 同样，包含回退项的交互差也须限定解释。
可报告本次parallel在这一共同约束下没有合格候选，但不能把它概括为parallel经过B训练没有改善或必然很弱。关于B配方下结构增量的证据仍不完整；原冻结主结果不改写。

## 成本、微批测试与复核

同机七SNR平均完整接收端时延：小next-scale约69.04 ms，增强next-scale约73.75 ms。训练进程观测10.49379 GPU小时（含校准），评测0.23300 GPU小时；另有开跑前/运行时无更新测速成本，非租赁账单。
微批4测试有吞吐优势，但增强parallel后缀出现差异，不通过当时登记的一致性检查；实际恢复并一直使用microbatch=2，prefix阶段也未切换。不要把候选优化写成已启用。
三阶段产物和源码SHA复核通过；16800行键唯一、100张源图、全为3060 uses；主表均值从逐帧记录重新汇总一致。
自动训练审计覆盖80000训练行、150000完整/子集校准行；统计生成192组配对差及24组交互差。没有另做全量独立模型推理或全量独立图像指标重算。
本轮仍是原development，不是新正式holdout。此完成记录不启动其他实验，也不更改另行授权的视觉骨干选型安排。

## 原始产物

相对`/workspace/projects/var-next-scale-comm`：
- 冻结原始报告及图表：`outputs/VAR-TOKEN-BACKBONE-20260911-ANALYSIS/report.md`、`development.png`、`calibration.png`。
- 完整均值/配对/交互/数字m8逐点差：同目录`primary.csv`、`summary.csv`、`paired.csv`、`interaction.csv`、`fixed_m8_gap.csv`。
- 逐帧质量及重建输出：`outputs/VAR-TOKEN-BACKBONE-20260911-EVALUATION/`。
- 训练及选模：`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING-R2/`。
- 原始报告、选模、checkpoint和receipt均未修改；本文件补充结果解读边界。
