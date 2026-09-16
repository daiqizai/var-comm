# 通信关键对照：校准与development结论（独立验证进行中）

2026-09-15。此报告只总结已完成校准/development与在线成本；不把正在运行的holdout当作已验证结果。

## 已完成

- R3按原10000 checkpoint补完2100新质量记录和480条自身计时，未训练或重选；见`r3_completion_result_20260915.md`。
- 同冻结VAR概率的真实整帧算术编码、单一data FEC块和全部开销已落地；不是旧分组熵编码。8项CPU算术/FEC检查、GPU概率/无损往返/独立渲染比对均通过。
- 1000源×六候选×七SNR×三噪声，共126000条校准；100源12600条development。全部PHY重新CPU回放，完整payload正确恢复时源codec反演均正确；独立PSNR/DINO代数核对通过。
- 模式表只在完整校准上一次固定。目标BLER10%最大源率、最大源索引goodput、最小LPIPS且PSNR最多退0.25dB，均使用同一m7/m8/m9集合和同一资源。

## 冻结后的模式表

SNR顺序：1 / 4 / 5 / 6 / 7 / 13 / 19 dB。

| 源编码/准则 | 模式m |
|---|---|
| raw，BLER目标 | 7 / 8 / 8 / 8 / 8 / 9 / 9 |
| raw，goodput | 7 / 8 / 8 / 8 / 9 / 9 / 9 |
| raw，图像质量 | 7 / 8 / 8 / 9 / 9 / 9 / 9 |
| arithmetic，BLER目标 | 7 / 8 / 8 / 9 / 9 / 9 / 9 |
| arithmetic，goodput | 7 / 8 / 9 / 9 / 9 / 9 / 9 |
| arithmetic，图像质量 | 7 / 8 / 8 / 9 / 9 / 9 / 9 |

**关键：arithmetic的质量与BLER策略在全部支持点完全相同。** 这是策略等价，不是样本不够而暂未看出差异；不能称质量目标在最强熵编码对照上产生独立新机制。新raw质量表也与旧历史自适应在这些点相同，新增的是独立校准来源，而非改写旧结果。

## Development主区间1/4/7

| 方法 | PSNR↑ | LPIPS↓ | DINO↑ |
|---|---:|---:|---:|
| raw BLER | 19.13397 | 0.196987 | 0.876053 |
| raw quality / goodput / old adaptive | 19.62665 | 0.183763 | 0.886743 |
| arithmetic quality / BLER / goodput | 19.66918 | 0.182580 | 0.887229 |

同源图配对，arithmetic quality相对raw quality的LPIPS差为−0.001183，95% CI [−0.002165,−0.000268]；幅度较小。raw quality对goodput、arithmetic quality对BLER/goodput的主区间差均为0，不能以只胜保守BLER规则宣布一般调度创新。

## 过渡区间与高SNR

- 6dB：raw quality LPIPS0.154269，比raw goodput/BLER的0.177141降低0.022872，95% CI [−0.026954,−0.018886]。但arithmetic quality/BLER达到0.141419，又比raw quality低0.012850，CI支持改善。
- 5dB：arithmetic goodput选择m9，LPIPS0.191472；quality/BLER选择m8，LPIPS0.177141，差−0.014331，95% CI [−0.027191,−0.002359]。恢复比特的goodput不能替代所有失败后的图像风险，但已有BLER规则在这里也作出同样合适选择。
- 13/19dB：所有这些VAR策略均选m9且图像一致，LPIPS0.135896。R3及同WeTok数字有各自更好的高SNR表现，不能说所有学习方法输给数字VAR。

## 资源与计算不是免费的

- 每次传输N3060/E6120。raw为68header+2992data，arithmetic为94header+2966data；实际CRC/tail/rate matching均计费。
- 算术族在1000校准图的平均实际payload：m7约1540.1bit、m8约2484.8bit、m9约3988.7bit；对应raw为1860/3060/5088bit。这是索引payload减少，不是物理带宽减少。
- 新算术链2016条CPU端点计时全部波形/图像与development归档差0，模型不变；旧七系统未重跑。主1/4/7的arithmetic quality约TX60.28ms、RX75.05ms、处理和135.33ms；raw quality由旧固定模式计时复用为约10.07/62.36/72.43ms。
- 新熵编码的主要额外负担在发送端VAR概率计算。不能只报码长或恢复率而隐去这个代价；处理和不含空口/排队。各次计时是同端点不同session，不用微小差值作纯机制速度归因。

## 当前收敛判断与独立验证

目前支持的是固定一次资源下源压缩、保护强度和失败图像后果的具体取舍，**尚不支持“图像质量导向模式选择全面超过常规适配及熵编码”的主张**。不能把前缀生成、next-scale压缩或源率/FEC适配称为首次贡献，参见`communication_related_work_recheck_20260915.md`。

按预定校准主LPIPS选择的候选仍是arithmetic_quality（与普通arithmetic BLER等价）；没有为了创新性或速度改选方法。15:59先冻结方法、指标、SNR、失败规则和按类排序的路径清单，随后才访问新的holdout像素作完整性筛查。排除1个历史内容重复，最终1000图/1000类；未根据模型质量替换图像。

独立验证在`outputs/COMMUNICATION-CONVERGENCE-20260915/HOLDOUT_001/`运行，四个冻结参考随后补齐。其结果尚未进入本报告，方法不因holdout修改，不启动新模型搜索。
