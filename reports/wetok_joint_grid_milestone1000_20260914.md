# Joint网格控制首1000：校准审计与继续决定

2026-09-14 17:55两臂各1000更新、完整1000图×五SNR校准及CPU审计通过。实际E/R、Adam全参数覆盖、数据/增强/SNR/标准噪声、能量和完整校准选模均PASS。

## 相同1000更新机会

| 结构/政策 | 选中step | 校准LPIPS ↓ | 校准PSNR ↑ |
|---|---:|---:|---:|
| Joint普通full-grid state | 1000 | 0.200014 | 21.18531 |
| Joint full-grid innovation | 1000 | 0.202011 | 21.19071 |
| 原Joint single | 0 | 0.200233 | 21.07275 |
| 原Joint no-history | 0 | 0.200233 | 21.07275 |
| 原Joint multiscale state | 1000 | 0.201303 | 21.19964 |
| R-only full-grid innovation | 1000 | 0.199371 | 21.22714 |

新两臂起点LPIPS均为0.202972。普通full-grid降至0.200014，innovation降至0.202011；此时Joint innovation仍不如匹配R-only，不能预设让内部E也更新一定有利。

这些只是同机会的校准，不是新development结果，也不与原5000选择点混排。普通full-grid和multiscale的初始函数不同，不能由这一短阶段证明空间尺度或普通迭代的普遍优劣。

## 继续

两臂从**实际1000末端模型/Adam**共同恢复至2500，global data 8000→9500。原参数、接口、loss、学习率、N/E和配对采样全部不变。继续完成原5000对齐计划，不因一个短里程碑决定机制有效，也不增加新模块。

此段已观察训练/校准墙钟约0.46838 GPU小时；profile的纯前反向估算不含完整校准等成本，不当完整工期。

里程碑SHA `9b67e8354a03bfc50a4798f8a2c267e6f2b410e0a9e41cd08f4cf770e5dbc330`，原始曲线/选择见`outputs/WETOK-JOINT-GRID-CONTROLS-R1-ANALYSIS/calibration_0001000/`。原所有模型与结果保留。
