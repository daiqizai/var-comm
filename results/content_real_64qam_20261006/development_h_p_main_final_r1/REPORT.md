# H/P/MAIN：同源开发集结果、尺度与组件成本

100 个原 development 源，各点先平均三次噪声，再等权平均源。固定展示全部24点，不按开发集挑选赢家。

H/MAIN 使用6201–6203，P保留2001–2003；P对照按源配对，不声称信道噪声逐帧相同。H/MAIN PSNR为float64，P保留原float32标量；各指标原模型、精度与批量准入记录原样保留。

58组固定对照共1044个区间来自原10000次源级bootstrap（seed2026100605）。均为未作多重比较校正的点级95%区间；本批不是holdout确认。

## 整尺度四臂：各自校准策略

| 冻结点 | SNR | PSNR | LPIPS | DINO-L | CLIP | DISTS | DreamSim | ResNet标签正确率 | ResNet源预测一致率 | ConvNeXt标签正确率 | ConvNeXt源预测一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H18_SLOT_00: H16-R | 13 | 20.30462 | 0.16629 | 0.78443 | 0.87560 | 0.14938 | 0.10597 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_01: H16-A | 13 | 20.34716 | 0.16600 | 0.78452 | 0.87562 | 0.14900 | 0.10572 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_02: H64-R | 13 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_03: H64-A | 13 | 20.34060 | 0.16615 | 0.78451 | 0.87557 | 0.14917 | 0.10578 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_04: H16-R | 19 | 20.30462 | 0.16629 | 0.78443 | 0.87560 | 0.14938 | 0.10597 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_05: H16-A | 19 | 20.34716 | 0.16600 | 0.78452 | 0.87562 | 0.14900 | 0.10572 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_06: H64-R | 19 | 20.30462 | 0.16629 | 0.78443 | 0.87560 | 0.14938 | 0.10597 | 0.73000 | 0.84000 | 0.74000 | 0.82000 |
| H18_SLOT_07: H64-A | 19 | 21.92322 | 0.12632 | 0.83183 | 0.89620 | 0.12908 | 0.08328 | 0.79000 | 0.84000 | 0.78000 | 0.86000 |

## 最强 raw64 部分尺度校准对照

| 冻结点 | SNR | PSNR | LPIPS | DINO-L | CLIP | DISTS | DreamSim | ResNet标签正确率 | ResNet源预测一致率 | ConvNeXt标签正确率 | ConvNeXt源预测一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H18_SLOT_08: H64-RAW-COMPLETE-STATE-CONTROL | 13 | 19.99615 | 0.17460 | 0.77160 | 0.86600 | 0.15219 | 0.11031 | 0.75000 | 0.83000 | 0.76000 | 0.80000 |
| H18_SLOT_09: H64-RAW-COMPLETE-STATE-CONTROL | 19 | 21.62640 | 0.13255 | 0.82286 | 0.89392 | 0.13182 | 0.08601 | 0.77000 | 0.85000 | 0.78000 | 0.84000 |

## 固定 m7/K0/rate1/2 归因对照

| 冻结点 | SNR | PSNR | LPIPS | DINO-L | CLIP | DISTS | DreamSim | ResNet标签正确率 | ResNet源预测一致率 | ConvNeXt标签正确率 | ConvNeXt源预测一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H18_SLOT_10: H16-R | 13 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_11: H16-A | 13 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_12: H64-R | 13 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_13: H64-A | 13 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_14: H16-R | 19 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_15: H16-A | 19 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_16: H64-R | 19 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |
| H18_SLOT_17: H64-A | 19 | 18.66614 | 0.22291 | 0.65588 | 0.83915 | 0.17451 | 0.15275 | 0.64000 | 0.76000 | 0.72000 | 0.78000 |

## MAIN 四个调制参考点

| 冻结点 | SNR | PSNR | LPIPS | DINO-L | CLIP | DISTS | DreamSim | ResNet标签正确率 | ResNet源预测一致率 | ConvNeXt标签正确率 | ConvNeXt源预测一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN_SLOT_18: QPSK | 13 | 18.05098 | 0.24758 | 0.61823 | 0.82435 | 0.18553 | 0.16816 | 0.66000 | 0.78000 | 0.69000 | 0.75000 |
| MAIN_SLOT_19: 16QAM | 13 | 20.35053 | 0.16471 | 0.78466 | 0.87453 | 0.14858 | 0.10525 | 0.73000 | 0.82000 | 0.76000 | 0.83000 |
| MAIN_SLOT_20: QPSK | 19 | 18.05098 | 0.24758 | 0.61823 | 0.82435 | 0.18553 | 0.16816 | 0.66000 | 0.78000 | 0.69000 | 0.75000 |
| MAIN_SLOT_21: 16QAM | 19 | 20.35053 | 0.16471 | 0.78466 | 0.87453 | 0.14858 | 0.10525 | 0.73000 | 0.82000 | 0.76000 | 0.83000 |

## P 两个原参考点

| 冻结点 | SNR | PSNR | LPIPS | DINO-L | CLIP | DISTS | DreamSim | ResNet标签正确率 | ResNet源预测一致率 | ConvNeXt标签正确率 | ConvNeXt源预测一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1024_SNR_13 | 13 | 20.72845 | 0.18247 | 0.58344 | 0.77456 | 0.20819 | 0.19157 | 0.56667 | 0.63333 | 0.58333 | 0.62667 |
| P1024_SNR_19 | 19 | 20.90626 | 0.17622 | 0.60112 | 0.78162 | 0.20624 | 0.18485 | 0.56667 | 0.64333 | 0.59333 | 0.64000 |

## 原 H 成功门槛：逐 SNR、逐 MAIN 参考判断

PSNR差值区间下界>0；LPIPS上界≤0；CLIP与ConvNeXt源预测一致率下界≥0。零非劣效界沿用原协议。跨零表示未解决，不能写成“无恶化”。DINO提升本身不证明内容更正确。

| H点 | MAIN参考 | SNR | PSNR | LPIPS | CLIP | ConvNeXt一致率 | 联合门槛 |
|---|---|---:|---|---|---|---|---|
| H18_SLOT_00 | MAIN_SLOT_18 | 13 | PASS [+2.08402, +2.42656] | PASS [-0.08654, -0.07611] | PASS [+0.03842, +0.06466] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_00 | MAIN_SLOT_19 | 13 | NOT_MET [-0.06245, -0.02954] | DETERIORATED [+0.00102, +0.00212] | UNRESOLVED [-0.00273, +0.00485] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_01 | MAIN_SLOT_18 | 13 | PASS [+2.11456, +2.49187] | PASS [-0.08679, -0.07646] | PASS [+0.03845, +0.06468] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_01 | MAIN_SLOT_19 | 13 | UNRESOLVED [-0.05083, +0.06015] | DETERIORATED [+0.00059, +0.00195] | UNRESOLVED [-0.00271, +0.00492] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_02 | MAIN_SLOT_18 | 13 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_02 | MAIN_SLOT_19 | 13 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_03 | MAIN_SLOT_18 | 13 | PASS [+2.10942, +2.48570] | PASS [-0.08664, -0.07629] | PASS [+0.03841, +0.06461] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_03 | MAIN_SLOT_19 | 13 | UNRESOLVED [-0.05413, +0.05231] | DETERIORATED [+0.00084, +0.00204] | UNRESOLVED [-0.00277, +0.00485] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_04 | MAIN_SLOT_20 | 19 | PASS [+2.08402, +2.42656] | PASS [-0.08654, -0.07611] | PASS [+0.03842, +0.06466] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_04 | MAIN_SLOT_21 | 19 | NOT_MET [-0.06245, -0.02954] | DETERIORATED [+0.00102, +0.00212] | UNRESOLVED [-0.00273, +0.00485] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_05 | MAIN_SLOT_20 | 19 | PASS [+2.11456, +2.49187] | PASS [-0.08679, -0.07646] | PASS [+0.03845, +0.06468] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_05 | MAIN_SLOT_21 | 19 | UNRESOLVED [-0.05083, +0.06015] | DETERIORATED [+0.00059, +0.00195] | UNRESOLVED [-0.00271, +0.00492] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_06 | MAIN_SLOT_20 | 19 | PASS [+2.08402, +2.42656] | PASS [-0.08654, -0.07611] | PASS [+0.03842, +0.06466] | PASS [+0.00000, +0.14000] | PASS |
| H18_SLOT_06 | MAIN_SLOT_21 | 19 | NOT_MET [-0.06245, -0.02954] | DETERIORATED [+0.00102, +0.00212] | UNRESOLVED [-0.00273, +0.00485] | UNRESOLVED [-0.06000, +0.04000] | NOT_MET |
| H18_SLOT_07 | MAIN_SLOT_20 | 19 | PASS [+3.60810, +4.14492] | PASS [-0.12829, -0.11418] | PASS [+0.05796, +0.08694] | PASS [+0.03000, +0.19000] | PASS |
| H18_SLOT_07 | MAIN_SLOT_21 | 19 | PASS [+1.44178, +1.70755] | PASS [-0.04130, -0.03537] | PASS [+0.01371, +0.03017] | UNRESOLVED [-0.02000, +0.08000] | UNRESOLVED |
| H18_SLOT_08 | MAIN_SLOT_18 | 13 | PASS [+1.78871, +2.11106] | PASS [-0.07746, -0.06852] | PASS [+0.02880, +0.05520] | UNRESOLVED [-0.02000, +0.12000] | UNRESOLVED |
| H18_SLOT_08 | MAIN_SLOT_19 | 13 | NOT_MET [-0.42225, -0.28828] | DETERIORATED [+0.00811, +0.01170] | DETERIORATED [-0.01502, -0.00246] | UNRESOLVED [-0.08000, +0.02000] | NOT_MET |
| H18_SLOT_09 | MAIN_SLOT_20 | 19 | PASS [+3.33065, +3.82164] | PASS [-0.12175, -0.10835] | PASS [+0.05538, +0.08463] | PASS [+0.02000, +0.17000] | PASS |
| H18_SLOT_09 | MAIN_SLOT_21 | 19 | PASS [+1.16879, +1.38706] | PASS [-0.03476, -0.02960] | PASS [+0.01065, +0.02886] | UNRESOLVED [-0.04000, +0.06000] | UNRESOLVED |
| H18_SLOT_10 | MAIN_SLOT_18 | 13 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_10 | MAIN_SLOT_19 | 13 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_11 | MAIN_SLOT_18 | 13 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_11 | MAIN_SLOT_19 | 13 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_12 | MAIN_SLOT_18 | 13 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_12 | MAIN_SLOT_19 | 13 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_13 | MAIN_SLOT_18 | 13 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_13 | MAIN_SLOT_19 | 13 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_14 | MAIN_SLOT_20 | 19 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_14 | MAIN_SLOT_21 | 19 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_15 | MAIN_SLOT_20 | 19 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_15 | MAIN_SLOT_21 | 19 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_16 | MAIN_SLOT_20 | 19 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_16 | MAIN_SLOT_21 | 19 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |
| H18_SLOT_17 | MAIN_SLOT_20 | 19 | PASS [+0.51698, +0.72102] | PASS [-0.02814, -0.02137] | PASS [+0.00575, +0.02394] | UNRESOLVED [-0.04000, +0.10025] | UNRESOLVED |
| H18_SLOT_17 | MAIN_SLOT_21 | 19 | NOT_MET [-1.82375, -1.54360] | DETERIORATED [+0.05438, +0.06210] | DETERIORATED [-0.04535, -0.02589] | UNRESOLVED [-0.11000, +0.01000] | NOT_MET |

| 整尺度臂 | 同时覆盖13/19及两个MAIN参考 |
|---|---|
| H16-R | NOT_MET |
| H16-A | NOT_MET |
| H64-R | NOT_MET |
| H64-A | NOT_MET |

缩小P差距与H−MAIN是同一差值，不重复计为独立证据。整尺度H64-R不替代部分尺度raw64对照；固定m7是归因口径，不升级为重新选出的系统。

## 完整指标与尺度分布

完整共同指标表保留24×18均值/区间及58×18配对区间。原H全部21项指标和26组因子对照另表原样保留。P缺MSE、ResNet概率与confidently_wrong，不填零、不补算。F恢复误差未测；FID/KID保留holdout待办。

H与MAIN的TX尺度按每点100源，RX按每点300帧统计。MAIN的KEEP硬token保留CRC失败状态；gray独立列出，不算m0、不删分母。MAIN能量只报实际header、body+idle和total三列，每列包含300帧的均值、总体标准差、p05/median/p95和极值，不推算body与idle能量拆分。

### MAIN TX源尺度与实际RX状态

| slot | 方向 | 结局 | m | K | public ID | body CRC接受 | 计数/分母 |
|---:|---|---|---:|---:|---:|---|---:|
| 18 | TX | 冻结传输尺度 | 6 | 40 | 88 | N/A | 100/100 |
| 18 | RX | TOKENS | 6 | 40 | 88 | True | 300/300 |
| 19 | TX | 冻结传输尺度 | 8 | 9 | 243 | N/A | 100/100 |
| 19 | RX | TOKENS | 8 | 9 | 243 | True | 300/300 |
| 20 | TX | 冻结传输尺度 | 6 | 40 | 88 | N/A | 100/100 |
| 20 | RX | TOKENS | 6 | 40 | 88 | True | 300/300 |
| 21 | TX | 冻结传输尺度 | 8 | 9 | 243 | N/A | 100/100 |
| 21 | RX | TOKENS | 8 | 9 | 243 | True | 300/300 |

### MAIN实际能量（每点全部300帧）

| slot | 原能量列 | 均值 | 总体标准差 | 最小 | p05 | median | p95 | 最大 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 18 | header_energy | 136.000000 | 0.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 |
| 18 | body_and_idle_energy | 1912.000000 | 0.000000 | 1912.000000 | 1912.000000 | 1912.000000 | 1912.000000 | 1912.000000 |
| 18 | total_energy | 2048.000000 | 0.000000 | 2048.000000 | 2048.000000 | 2048.000000 | 2048.000000 | 2048.000000 |
| 19 | header_energy | 136.000000 | 0.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 |
| 19 | body_and_idle_energy | 1911.983801 | 35.494878 | 1827.199812 | 1851.119809 | 1910.399802 | 1969.759794 | 2036.799786 |
| 19 | total_energy | 2047.983801 | 35.494878 | 1963.199812 | 1987.119809 | 2046.399802 | 2105.759794 | 2172.799786 |
| 20 | header_energy | 136.000000 | 0.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 |
| 20 | body_and_idle_energy | 1912.000000 | 0.000000 | 1912.000000 | 1912.000000 | 1912.000000 | 1912.000000 | 1912.000000 |
| 20 | total_energy | 2048.000000 | 0.000000 | 2048.000000 | 2048.000000 | 2048.000000 | 2048.000000 | 2048.000000 |
| 21 | header_energy | 136.000000 | 0.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 | 136.000000 |
| 21 | body_and_idle_energy | 1910.895801 | 34.166667 | 1827.199812 | 1855.999808 | 1911.999801 | 1969.679794 | 2004.799790 |
| 21 | total_energy | 2046.895801 | 34.166667 | 1963.199812 | 1991.999808 | 2047.999801 | 2105.679794 | 2140.799790 |

## H在线组件成本

固定原16源、noise6201、1次预热+3次实测、B1、每次fresh计算与精确TX/RX parity。下列总项包含子项，不与子项相加。

| slot | 方向 | 组件 | 源均值平均/秒 | 源均值中位数/秒 | 包含子项 |
|---:|---|---|---:|---:|---|
| 0 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 0 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 0 | RX | RX_source_and_image_total | 0.133138 | 0.133607 | True |
| 0 | RX | RX_suffix_VAR_and_Dc | 0.126545 | 0.127042 | False |
| 0 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 0 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 0 | TX | TX_source_total | 0.002560 | 0.002493 | True |
| 0 | TX | TX_visual_encoding | 0.010135 | 0.009018 | False |
| 1 | RX | RX_CDF_construction | 0.014792 | 0.014790 | False |
| 1 | RX | RX_VAR_probability | 0.102101 | 0.101378 | False |
| 1 | RX | RX_integer_decode | 0.005782 | 0.005832 | False |
| 1 | RX | RX_integer_encode | 0.003741 | 0.003772 | False |
| 1 | RX | RX_integer_finish | 0.000084 | 0.000084 | False |
| 1 | RX | RX_probability_setup | 0.000386 | 0.000382 | False |
| 1 | RX | RX_probability_teardown | 0.000139 | 0.000139 | False |
| 1 | RX | RX_source_and_image_total | 0.258659 | 0.258325 | True |
| 1 | RX | RX_suffix_VAR_and_Dc | 0.125830 | 0.126277 | False |
| 1 | TX | TX_CDF_construction | 0.039201 | 0.039084 | False |
| 1 | TX | TX_VAR_probability | 0.218683 | 0.216313 | False |
| 1 | TX | TX_integer_encode | 0.009873 | 0.009886 | False |
| 1 | TX | TX_integer_finish | 0.000191 | 0.000193 | False |
| 1 | TX | TX_probability_setup | 0.000582 | 0.000580 | False |
| 1 | TX | TX_probability_teardown | 0.000252 | 0.000252 | False |
| 1 | TX | TX_source_total | 0.279679 | 0.277183 | True |
| 1 | TX | TX_visual_encoding | 0.010125 | 0.009018 | False |
| 2 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 2 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 2 | RX | RX_source_and_image_total | 0.131599 | 0.131939 | True |
| 2 | RX | RX_suffix_VAR_and_Dc | 0.126836 | 0.127142 | False |
| 2 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 2 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 2 | TX | TX_source_total | 0.002192 | 0.002211 | True |
| 2 | TX | TX_visual_encoding | 0.010836 | 0.009024 | False |
| 3 | RX | RX_CDF_construction | 0.014788 | 0.014784 | False |
| 3 | RX | RX_VAR_probability | 0.103220 | 0.101596 | False |
| 3 | RX | RX_integer_decode | 0.005761 | 0.005778 | False |
| 3 | RX | RX_integer_encode | 0.003730 | 0.003763 | False |
| 3 | RX | RX_integer_finish | 0.000084 | 0.000084 | False |
| 3 | RX | RX_probability_setup | 0.000386 | 0.000387 | False |
| 3 | RX | RX_probability_teardown | 0.000141 | 0.000139 | False |
| 3 | RX | RX_source_and_image_total | 0.260236 | 0.258868 | True |
| 3 | RX | RX_suffix_VAR_and_Dc | 0.126436 | 0.126644 | False |
| 3 | TX | TX_CDF_construction | 0.039578 | 0.039128 | False |
| 3 | TX | TX_VAR_probability | 0.223063 | 0.215699 | False |
| 3 | TX | TX_integer_encode | 0.009854 | 0.009926 | False |
| 3 | TX | TX_integer_finish | 0.000190 | 0.000191 | False |
| 3 | TX | TX_probability_setup | 0.000577 | 0.000576 | False |
| 3 | TX | TX_probability_teardown | 0.000251 | 0.000251 | False |
| 3 | TX | TX_source_total | 0.284993 | 0.276927 | True |
| 3 | TX | TX_visual_encoding | 0.010779 | 0.009006 | False |
| 4 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 4 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 4 | RX | RX_source_and_image_total | 0.133005 | 0.133168 | True |
| 4 | RX | RX_suffix_VAR_and_Dc | 0.126439 | 0.126608 | False |
| 4 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 4 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 4 | TX | TX_source_total | 0.002418 | 0.002419 | True |
| 4 | TX | TX_visual_encoding | 0.009728 | 0.009028 | False |
| 5 | RX | RX_CDF_construction | 0.014880 | 0.014790 | False |
| 5 | RX | RX_VAR_probability | 0.104964 | 0.101432 | False |
| 5 | RX | RX_integer_decode | 0.005836 | 0.005839 | False |
| 5 | RX | RX_integer_encode | 0.003732 | 0.003773 | False |
| 5 | RX | RX_integer_finish | 0.000083 | 0.000084 | False |
| 5 | RX | RX_probability_setup | 0.000386 | 0.000385 | False |
| 5 | RX | RX_probability_teardown | 0.000139 | 0.000139 | False |
| 5 | RX | RX_source_and_image_total | 0.261753 | 0.258138 | True |
| 5 | RX | RX_suffix_VAR_and_Dc | 0.125782 | 0.125962 | False |
| 5 | TX | TX_CDF_construction | 0.039153 | 0.039079 | False |
| 5 | TX | TX_VAR_probability | 0.219441 | 0.216219 | False |
| 5 | TX | TX_integer_encode | 0.009969 | 0.009970 | False |
| 5 | TX | TX_integer_finish | 0.000191 | 0.000193 | False |
| 5 | TX | TX_probability_setup | 0.000581 | 0.000576 | False |
| 5 | TX | TX_probability_teardown | 0.000254 | 0.000251 | False |
| 5 | TX | TX_source_total | 0.280493 | 0.277134 | True |
| 5 | TX | TX_visual_encoding | 0.010273 | 0.009018 | False |
| 6 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 6 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 6 | RX | RX_source_and_image_total | 0.133782 | 0.133877 | True |
| 6 | RX | RX_suffix_VAR_and_Dc | 0.126999 | 0.127062 | False |
| 6 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 6 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 6 | TX | TX_source_total | 0.002594 | 0.002535 | True |
| 6 | TX | TX_visual_encoding | 0.011088 | 0.009027 | False |
| 7 | RX | RX_CDF_construction | 0.024518 | 0.024376 | False |
| 7 | RX | RX_VAR_probability | 0.116960 | 0.114345 | False |
| 7 | RX | RX_integer_decode | 0.008832 | 0.008792 | False |
| 7 | RX | RX_integer_encode | 0.005683 | 0.005698 | False |
| 7 | RX | RX_integer_finish | 0.000109 | 0.000110 | False |
| 7 | RX | RX_probability_setup | 0.000388 | 0.000386 | False |
| 7 | RX | RX_probability_teardown | 0.000116 | 0.000115 | False |
| 7 | RX | RX_source_and_image_total | 0.290849 | 0.288685 | True |
| 7 | RX | RX_suffix_VAR_and_Dc | 0.126022 | 0.126630 | False |
| 7 | TX | TX_CDF_construction | 0.024354 | 0.024375 | False |
| 7 | TX | TX_VAR_probability | 0.118529 | 0.115061 | False |
| 7 | TX | TX_integer_encode | 0.006155 | 0.006091 | False |
| 7 | TX | TX_integer_finish | 0.000109 | 0.000110 | False |
| 7 | TX | TX_probability_setup | 0.000341 | 0.000340 | False |
| 7 | TX | TX_probability_teardown | 0.000115 | 0.000115 | False |
| 7 | TX | TX_source_total | 0.157165 | 0.153647 | True |
| 7 | TX | TX_visual_encoding | 0.012370 | 0.009021 | False |
| 8 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 8 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 8 | RX | RX_source_and_image_total | 0.132499 | 0.132665 | True |
| 8 | RX | RX_suffix_VAR_and_Dc | 0.126378 | 0.126538 | False |
| 8 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 8 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 8 | TX | TX_source_total | 0.000840 | 0.000838 | True |
| 8 | TX | TX_visual_encoding | 0.009875 | 0.009024 | False |
| 9 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 9 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 9 | RX | RX_source_and_image_total | 0.136021 | 0.136422 | True |
| 9 | RX | RX_suffix_VAR_and_Dc | 0.126513 | 0.126879 | False |
| 9 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 9 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 9 | TX | TX_source_total | 0.001328 | 0.001328 | True |
| 9 | TX | TX_visual_encoding | 0.009034 | 0.009024 | False |
| 10 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 10 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 10 | RX | RX_source_and_image_total | 0.130975 | 0.131220 | True |
| 10 | RX | RX_suffix_VAR_and_Dc | 0.126564 | 0.126799 | False |
| 10 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 10 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 10 | TX | TX_source_total | 0.002093 | 0.002102 | True |
| 10 | TX | TX_visual_encoding | 0.009018 | 0.009009 | False |
| 11 | RX | RX_CDF_construction | 0.009196 | 0.009196 | False |
| 11 | RX | RX_VAR_probability | 0.088347 | 0.088538 | False |
| 11 | RX | RX_integer_decode | 0.003625 | 0.003638 | False |
| 11 | RX | RX_integer_encode | 0.002379 | 0.002390 | False |
| 11 | RX | RX_integer_finish | 0.000058 | 0.000058 | False |
| 11 | RX | RX_probability_setup | 0.000385 | 0.000386 | False |
| 11 | RX | RX_probability_teardown | 0.000130 | 0.000130 | False |
| 11 | RX | RX_source_and_image_total | 0.234978 | 0.235674 | True |
| 11 | RX | RX_suffix_VAR_and_Dc | 0.126675 | 0.127196 | False |
| 11 | TX | TX_CDF_construction | 0.009208 | 0.009195 | False |
| 11 | TX | TX_VAR_probability | 0.088718 | 0.088777 | False |
| 11 | TX | TX_integer_encode | 0.002434 | 0.002442 | False |
| 11 | TX | TX_integer_finish | 0.000056 | 0.000056 | False |
| 11 | TX | TX_probability_setup | 0.000340 | 0.000339 | False |
| 11 | TX | TX_probability_teardown | 0.000128 | 0.000127 | False |
| 11 | TX | TX_source_total | 0.105186 | 0.105140 | True |
| 11 | TX | TX_visual_encoding | 0.009015 | 0.009007 | False |
| 12 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 12 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 12 | RX | RX_source_and_image_total | 0.131539 | 0.132138 | True |
| 12 | RX | RX_suffix_VAR_and_Dc | 0.126688 | 0.127246 | False |
| 12 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 12 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 12 | TX | TX_source_total | 0.002219 | 0.002230 | True |
| 12 | TX | TX_visual_encoding | 0.009018 | 0.009025 | False |
| 13 | RX | RX_CDF_construction | 0.009189 | 0.009177 | False |
| 13 | RX | RX_VAR_probability | 0.089536 | 0.088559 | False |
| 13 | RX | RX_integer_decode | 0.003636 | 0.003659 | False |
| 13 | RX | RX_integer_encode | 0.002370 | 0.002382 | False |
| 13 | RX | RX_integer_finish | 0.000058 | 0.000058 | False |
| 13 | RX | RX_probability_setup | 0.000386 | 0.000385 | False |
| 13 | RX | RX_probability_teardown | 0.000131 | 0.000129 | False |
| 13 | RX | RX_source_and_image_total | 0.236308 | 0.235701 | True |
| 13 | RX | RX_suffix_VAR_and_Dc | 0.126485 | 0.126574 | False |
| 13 | TX | TX_CDF_construction | 0.009214 | 0.009206 | False |
| 13 | TX | TX_VAR_probability | 0.089615 | 0.088565 | False |
| 13 | TX | TX_integer_encode | 0.002434 | 0.002438 | False |
| 13 | TX | TX_integer_finish | 0.000057 | 0.000057 | False |
| 13 | TX | TX_probability_setup | 0.000335 | 0.000336 | False |
| 13 | TX | TX_probability_teardown | 0.000128 | 0.000128 | False |
| 13 | TX | TX_source_total | 0.106205 | 0.105164 | True |
| 13 | TX | TX_visual_encoding | 0.009497 | 0.009030 | False |
| 14 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 14 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 14 | RX | RX_source_and_image_total | 0.131037 | 0.131560 | True |
| 14 | RX | RX_suffix_VAR_and_Dc | 0.126533 | 0.126980 | False |
| 14 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 14 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 14 | TX | TX_source_total | 0.002093 | 0.002094 | True |
| 14 | TX | TX_visual_encoding | 0.010550 | 0.009033 | False |
| 15 | RX | RX_CDF_construction | 0.009205 | 0.009200 | False |
| 15 | RX | RX_VAR_probability | 0.088534 | 0.088458 | False |
| 15 | RX | RX_integer_decode | 0.003632 | 0.003630 | False |
| 15 | RX | RX_integer_encode | 0.002370 | 0.002374 | False |
| 15 | RX | RX_integer_finish | 0.000058 | 0.000058 | False |
| 15 | RX | RX_probability_setup | 0.000388 | 0.000385 | False |
| 15 | RX | RX_probability_teardown | 0.000131 | 0.000132 | False |
| 15 | RX | RX_source_and_image_total | 0.234926 | 0.235293 | True |
| 15 | RX | RX_suffix_VAR_and_Dc | 0.126427 | 0.126679 | False |
| 15 | TX | TX_CDF_construction | 0.009219 | 0.009198 | False |
| 15 | TX | TX_VAR_probability | 0.089040 | 0.088975 | False |
| 15 | TX | TX_integer_encode | 0.002433 | 0.002451 | False |
| 15 | TX | TX_integer_finish | 0.000056 | 0.000056 | False |
| 15 | TX | TX_probability_setup | 0.000340 | 0.000339 | False |
| 15 | TX | TX_probability_teardown | 0.000128 | 0.000128 | False |
| 15 | TX | TX_source_total | 0.105538 | 0.105447 | True |
| 15 | TX | TX_visual_encoding | 0.009032 | 0.009027 | False |
| 16 | RX | RX_CDF_construction | 0.000000 | 0.000000 | False |
| 16 | RX | RX_VAR_probability | 0.000000 | 0.000000 | False |
| 16 | RX | RX_source_and_image_total | 0.130862 | 0.131195 | True |
| 16 | RX | RX_suffix_VAR_and_Dc | 0.126123 | 0.126438 | False |
| 16 | TX | TX_CDF_construction | 0.000000 | 0.000000 | False |
| 16 | TX | TX_VAR_probability | 0.000000 | 0.000000 | False |
| 16 | TX | TX_source_total | 0.002213 | 0.002215 | True |
| 16 | TX | TX_visual_encoding | 0.009421 | 0.009031 | False |
| 17 | RX | RX_CDF_construction | 0.009216 | 0.009202 | False |
| 17 | RX | RX_VAR_probability | 0.089136 | 0.088496 | False |
| 17 | RX | RX_integer_decode | 0.003636 | 0.003637 | False |
| 17 | RX | RX_integer_encode | 0.002376 | 0.002387 | False |
| 17 | RX | RX_integer_finish | 0.000058 | 0.000058 | False |
| 17 | RX | RX_probability_setup | 0.000387 | 0.000385 | False |
| 17 | RX | RX_probability_teardown | 0.000131 | 0.000131 | False |
| 17 | RX | RX_source_and_image_total | 0.236210 | 0.235838 | True |
| 17 | RX | RX_suffix_VAR_and_Dc | 0.126733 | 0.127049 | False |
| 17 | TX | TX_CDF_construction | 0.009175 | 0.009168 | False |
| 17 | TX | TX_VAR_probability | 0.088785 | 0.088599 | False |
| 17 | TX | TX_integer_encode | 0.002428 | 0.002430 | False |
| 17 | TX | TX_integer_finish | 0.000056 | 0.000056 | False |
| 17 | TX | TX_probability_setup | 0.000338 | 0.000338 | False |
| 17 | TX | TX_probability_teardown | 0.000127 | 0.000126 | False |
| 17 | TX | TX_source_total | 0.105338 | 0.105170 | True |
| 17 | TX | TX_visual_encoding | 0.009820 | 0.009037 | False |

H的TX_source_total为源编码组件，fresh Encoder/VQ另列；不含CRC/FEC、调制和信道。该名称不表示完整发送端耗时。

## MAIN在线组件成本

固定原16源、noise6201、1次预热+3次实测、B1、每次fresh计算与精确TX/RX parity。下列总项包含子项，不与子项相加。

| slot | 方向 | 组件 | 源均值平均/秒 | 源均值中位数/秒 | 包含子项 |
|---:|---|---|---:|---:|---|
| 18 | RX | RX_source_and_image_total | 0.137285 | 0.137560 | True |
| 18 | RX | RX_suffix_VAR_and_Dc | 0.127489 | 0.127777 | False |
| 18 | TX | TX_source_total | 0.000081 | 0.000080 | True |
| 18 | TX | TX_visual_encoding | 0.009005 | 0.009004 | False |
| 19 | RX | RX_source_and_image_total | 0.138842 | 0.138871 | True |
| 19 | RX | RX_suffix_VAR_and_Dc | 0.127712 | 0.127815 | False |
| 19 | TX | TX_source_total | 0.000095 | 0.000094 | True |
| 19 | TX | TX_visual_encoding | 0.008990 | 0.008988 | False |
| 20 | RX | RX_source_and_image_total | 0.137381 | 0.137557 | True |
| 20 | RX | RX_suffix_VAR_and_Dc | 0.127599 | 0.127695 | False |
| 20 | TX | TX_source_total | 0.000080 | 0.000080 | True |
| 20 | TX | TX_visual_encoding | 0.008998 | 0.008990 | False |
| 21 | RX | RX_source_and_image_total | 0.138911 | 0.138637 | True |
| 21 | RX | RX_suffix_VAR_and_Dc | 0.127804 | 0.127562 | False |
| 21 | TX | TX_source_total | 0.000094 | 0.000093 | True |
| 21 | TX | TX_visual_encoding | 0.008991 | 0.008994 | False |

MAIN的TX_source_total仅为raw12 raster序列化；fresh Encoder/VQ单列。RX总项包含实际KEEP接收解析与后缀VAR/Dc（或实际拒收gray），不含PHY。

P在线成本、PHY编码、独占PHY译码与端到端耗时均为N/A。历史PHY事件窗含并发及记账开销，只作独立诊断，不与GPU组件拼成端到端延迟，也不计算缺测P的加速比。

本报告是已封结果的纯投影；正式调用方须先验证各阶段正常退出和完整输入输出SHA。
