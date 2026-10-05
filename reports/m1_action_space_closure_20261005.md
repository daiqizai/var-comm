# M1 动作空间补测：N1024 与 N2048

N1024/16QAM、N2048/QPSK、N2048/16QAM；1/4/7/13/19dB。分别73、73、98个登记动作，在1000张校准图×3个校准噪声上选择，100张development图×3个噪声上评测。未训练新模型。

**扩范围是在观察旧development结果后提出。同一development上的配对区间用于描述本组数据，不是独立holdout或未经选择的确认性显著性证据。**

## 范围与复用

补齐审计登记的m8部分第9尺度及N2048原m9范围的低SNR覆盖。完整是指本次登记的四分位K加最大合法K网格；不是逐一枚举每个整数K。新增头部mode7表示m8的K≥128，仍在原12源bit/68信道使用内。旧D_U/D_C的两位模式字段未改变。

3660000校准逻辑帧中2187000帧来自已核验旧结果，新增1473000帧。N1024从原200图筛选加短名单，改为73动作全1000×3校准；策略变化同时含候选扩展与完整校准，不能全部归因新增4个动作。本轮没有另跑旧69动作全校准反事实的development。

实际development方法行34800，理论上限36000。付费oracle同K不可行时保留N/A，不降K、不放宽码率、不生成替代灰图。

## 选中的动作

| N | 调制 | SNR | 方法 | m | K | 解释 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 16QAM | 1 | whole_policy | 5 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 1 | entropy_policy | 5 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 4 | whole_policy | 6 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 4 | entropy_policy | 6 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 7 | whole_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 7 | entropy_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 13 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 13 | entropy_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 19 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 1024 | 16QAM | 19 | entropy_policy | 8 | 27 | partial scale 9 |
| 2048 | QPSK | 1 | whole_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 1 | entropy_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 4 | whole_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 4 | entropy_policy | 7 | 25 | partial scale 8 |
| 2048 | QPSK | 7 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 7 | entropy_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 13 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 13 | entropy_policy | 8 | 38 | partial scale 9 |
| 2048 | QPSK | 19 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | QPSK | 19 | entropy_policy | 8 | 38 | partial scale 9 |
| 2048 | 16QAM | 1 | whole_policy | 6 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 1 | entropy_policy | 6 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 4 | whole_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 4 | entropy_policy | 7 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 7 | whole_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 7 | entropy_policy | 8 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 13 | whole_policy | 9 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 13 | entropy_policy | 9 | 64 | partial scale 10 |
| 2048 | 16QAM | 19 | whole_policy | 9 | 0 | whole scale; no partial-token or ordering contribution |
| 2048 | 16QAM | 19 | entropy_policy | 9 | 166 | partial scale 10 |

K0只传完整尺度，不能当作部分尺度或排序收益。部分尺度看entropy相对whole；排序看相同m/K下的raster/random/oracle。

## 主表与源配对差值

[全部主表](../results/m1_action_space_closure_20261005/MAIN_TABLE.md) · [完整CSV](../results/m1_action_space_closure_20261005/MAIN_TABLE.csv) · [配对区间](../results/m1_action_space_closure_20261005/metrics_paired_intervals.csv) · [全部登记动作的校准结果](../results/m1_action_space_closure_20261005/calibration_summary.csv)

先对每张图的3次噪声平均，再以100张源图配对bootstrap10000次，seed20261002。差值A−B，PSNR/DINO/CLIP正值更好，LPIPS/DISTS负值更好。

| N | 调制 | SNR | 比较 | ΔPSNR [95%CI] | ΔLPIPS [95%CI] | ΔDINOv2-L [95%CI] | ΔCLIP [95%CI] | ΔDISTS [95%CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 16QAM | 1 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 1024 | 16QAM | 1 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 1 | 新entropy−P | -3.8273 [-4.0810, -3.5865] | 0.1222 [0.1124, 0.1324] | 0.0045 [-0.0308, 0.0390] | 0.0476 [0.0354, 0.0598] | 0.0004 [-0.0056, 0.0065] |
| 1024 | 16QAM | 4 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] |
| 1024 | 16QAM | 4 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 4 | 新entropy−P | -2.9587 [-3.1773, -2.7434] | 0.0714 [0.0637, 0.0794] | 0.0688 [0.0327, 0.1037] | 0.0697 [0.0566, 0.0830] | -0.0207 [-0.0260, -0.0155] |
| 1024 | 16QAM | 7 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 1024 | 16QAM | 7 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 7 | 新entropy−P | -1.6937 [-1.8783, -1.5074] | 0.0235 [0.0181, 0.0288] | 0.1391 [0.1024, 0.1772] | 0.0823 [0.0676, 0.0967] | -0.0417 [-0.0458, -0.0378] |
| 1024 | 16QAM | 13 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 1024 | 16QAM | 13 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 13 | 新entropy−P | -0.5982 [-0.8199, -0.3806] | -0.0118 [-0.0163, -0.0069] | 0.1934 [0.1614, 0.2274] | 0.0981 [0.0843, 0.1123] | -0.0569 [-0.0614, -0.0523] |
| 1024 | 16QAM | 19 | 新entropy−历史entropy | 0.3264 [0.2790, 0.3758] | -0.0072 [-0.0083, -0.0061] | 0.0072 [-0.0023, 0.0169] | 0.0019 [-0.0033, 0.0072] | -0.0037 [-0.0050, -0.0023] |
| 1024 | 16QAM | 19 | 新entropy−新whole | 0.3264 [0.2790, 0.3758] | -0.0072 [-0.0083, -0.0061] | 0.0072 [-0.0023, 0.0169] | 0.0019 [-0.0033, 0.0072] | -0.0037 [-0.0050, -0.0023] |
| 1024 | 16QAM | 19 | 新entropy−P | -0.2753 [-0.4739, -0.0578] | -0.0171 [-0.0211, -0.0132] | 0.1905 [0.1576, 0.2245] | 0.0959 [0.0822, 0.1098] | -0.0605 [-0.0648, -0.0564] |
| 2048 | QPSK | 1 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 2048 | QPSK | 1 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 1 | 新entropy−P | -2.7261 [-2.9930, -2.4711] | 0.0681 [0.0609, 0.0754] | -0.0247 [-0.0568, 0.0094] | 0.0192 [0.0070, 0.0321] | 0.0036 [-0.0005, 0.0075] |
| 2048 | QPSK | 4 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| 2048 | QPSK | 4 | 新entropy−新whole | 0.4737 [0.3994, 0.5544] | -0.0163 [-0.0183, -0.0142] | 0.0410 [0.0243, 0.0584] | 0.0102 [0.0039, 0.0168] | -0.0054 [-0.0071, -0.0037] |
| 2048 | QPSK | 4 | 新entropy−P | -2.0615 [-2.2394, -1.8853] | 0.0463 [0.0416, 0.0510] | -0.0234 [-0.0490, 0.0015] | 0.0156 [0.0054, 0.0257] | 0.0030 [-0.0002, 0.0062] |
| 2048 | QPSK | 7 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| 2048 | QPSK | 7 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 7 | 新entropy−P | -1.7278 [-1.8849, -1.5763] | 0.0324 [0.0291, 0.0356] | 0.0066 [-0.0110, 0.0256] | 0.0127 [0.0034, 0.0218] | -0.0010 [-0.0038, 0.0018] |
| 2048 | QPSK | 13 | 新entropy−历史entropy | 0.4177 [0.3643, 0.4749] | -0.0094 [-0.0107, -0.0082] | 0.0083 [-0.0012, 0.0177] | 0.0039 [-0.0024, 0.0110] | -0.0041 [-0.0055, -0.0027] |
| 2048 | QPSK | 13 | 新entropy−新whole | 0.4177 [0.3643, 0.4749] | -0.0094 [-0.0107, -0.0082] | 0.0083 [-0.0012, 0.0177] | 0.0039 [-0.0024, 0.0110] | -0.0041 [-0.0055, -0.0027] |
| 2048 | QPSK | 13 | 新entropy−P | -2.0280 [-2.1974, -1.8596] | 0.0423 [0.0385, 0.0462] | -0.0181 [-0.0344, -0.0023] | -0.0039 [-0.0118, 0.0040] | 0.0064 [0.0036, 0.0092] |
| 2048 | QPSK | 19 | 新entropy−历史entropy | 0.4177 [0.3643, 0.4749] | -0.0094 [-0.0107, -0.0082] | 0.0083 [-0.0012, 0.0177] | 0.0039 [-0.0024, 0.0110] | -0.0041 [-0.0055, -0.0027] |
| 2048 | QPSK | 19 | 新entropy−新whole | 0.4177 [0.3643, 0.4749] | -0.0094 [-0.0107, -0.0082] | 0.0083 [-0.0012, 0.0177] | 0.0039 [-0.0024, 0.0110] | -0.0041 [-0.0055, -0.0027] |
| 2048 | QPSK | 19 | 新entropy−P | -2.2342 [-2.4115, -2.0580] | 0.0473 [0.0433, 0.0514] | -0.0266 [-0.0425, -0.0116] | -0.0092 [-0.0174, -0.0009] | 0.0101 [0.0073, 0.0129] |
| 2048 | 16QAM | 1 | 新entropy−历史entropy | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| 2048 | 16QAM | 1 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 1 | 新entropy−P | -3.0952 [-3.3008, -2.8935] | 0.0884 [0.0824, 0.0944] | -0.0650 [-0.1013, -0.0289] | 0.0161 [0.0016, 0.0308] | 0.0146 [0.0107, 0.0184] |
| 2048 | 16QAM | 4 | 新entropy−历史entropy | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| 2048 | 16QAM | 4 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 4 | 新entropy−P | -2.5570 [-2.7644, -2.3552] | 0.0635 [0.0585, 0.0686] | -0.0612 [-0.0893, -0.0340] | 0.0030 [-0.0080, 0.0139] | 0.0084 [0.0053, 0.0116] |
| 2048 | 16QAM | 7 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 2048 | 16QAM | 7 | 新entropy−新whole | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 7 | 新entropy−P | -1.7392 [-1.8981, -1.5849] | 0.0326 [0.0294, 0.0359] | 0.0052 [-0.0126, 0.0242] | 0.0123 [0.0032, 0.0212] | -0.0010 [-0.0039, 0.0019] |
| 2048 | 16QAM | 13 | 新entropy−历史entropy | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| 2048 | 16QAM | 13 | 新entropy−新whole | 0.3325 [0.2958, 0.3729] | -0.0063 [-0.0071, -0.0054] | 0.0066 [-0.0007, 0.0137] | 0.0041 [0.0005, 0.0080] | -0.0030 [-0.0039, -0.0022] |
| 2048 | 16QAM | 13 | 新entropy−P | -0.4949 [-0.6099, -0.3733] | 0.0056 [0.0035, 0.0075] | 0.0278 [0.0158, 0.0404] | 0.0169 [0.0090, 0.0254] | -0.0128 [-0.0154, -0.0103] |
| 2048 | 16QAM | 19 | 新entropy−历史entropy | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| 2048 | 16QAM | 19 | 新entropy−新whole | 1.0327 [0.9619, 1.1080] | -0.0191 [-0.0204, -0.0177] | 0.0230 [0.0135, 0.0325] | 0.0115 [0.0052, 0.0181] | -0.0083 [-0.0096, -0.0070] |
| 2048 | 16QAM | 19 | 新entropy−P | -0.0006 [-0.1211, 0.1294] | -0.0023 [-0.0043, -0.0003] | 0.0355 [0.0229, 0.0481] | 0.0191 [0.0098, 0.0287] | -0.0143 [-0.0170, -0.0117] |

## 同m/K排序消融

| N | 调制 | SNR | m | K | 控制 | 范围 | ΔPSNR [95%CI] | ΔLPIPS [95%CI] | ΔDINOv2-L [95%CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 16QAM | 1 | 5 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 1 | 5 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 1 | 5 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 4 | 6 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 4 | 6 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 4 | 6 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 7 | 7 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 7 | 7 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 7 | 7 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 13 | 8 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 13 | 8 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 13 | 8 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 1024 | 16QAM | 19 | 8 | 27 | raster_at_entropy | same m/K; oracle mask charged | 0.1230 [0.0634, 0.1858] | -0.0015 [-0.0029, -0.0002] | 0.0063 [-0.0048, 0.0174] |
| 1024 | 16QAM | 19 | 8 | 27 | random_at_entropy | same m/K; oracle mask charged | 0.1037 [0.0508, 0.1558] | -0.0014 [-0.0026, -0.0003] | -0.0005 [-0.0113, 0.0103] |
| 1024 | 16QAM | 19 | 8 | 27 | oracle_at_entropy | N/A: paid mask exceeds budget | N/A | N/A | N/A |
| 2048 | QPSK | 1 | 7 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 1 | 7 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 1 | 7 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 4 | 7 | 25 | raster_at_entropy | same m/K; oracle mask charged | 0.1690 [0.0870, 0.2600] | -0.0044 [-0.0064, -0.0022] | 0.0170 [0.0014, 0.0330] |
| 2048 | QPSK | 4 | 7 | 25 | random_at_entropy | same m/K; oracle mask charged | 0.1965 [0.1181, 0.2835] | -0.0047 [-0.0067, -0.0027] | 0.0045 [-0.0148, 0.0225] |
| 2048 | QPSK | 4 | 7 | 25 | oracle_at_entropy | same m/K; oracle mask charged | 0.0356 [-0.0026, 0.0751] | -0.0016 [-0.0028, -0.0005] | 0.0033 [-0.0008, 0.0073] |
| 2048 | QPSK | 7 | 8 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 7 | 8 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 7 | 8 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | QPSK | 13 | 8 | 38 | raster_at_entropy | same m/K; oracle mask charged | 0.1186 [0.0475, 0.1891] | -0.0012 [-0.0027, 0.0003] | -0.0022 [-0.0118, 0.0071] |
| 2048 | QPSK | 13 | 8 | 38 | random_at_entropy | same m/K; oracle mask charged | 0.0764 [0.0187, 0.1330] | -0.0007 [-0.0020, 0.0006] | -0.0081 [-0.0196, 0.0024] |
| 2048 | QPSK | 13 | 8 | 38 | oracle_at_entropy | N/A: paid mask exceeds budget | N/A | N/A | N/A |
| 2048 | QPSK | 19 | 8 | 38 | raster_at_entropy | same m/K; oracle mask charged | 0.1186 [0.0475, 0.1891] | -0.0012 [-0.0027, 0.0003] | -0.0022 [-0.0118, 0.0071] |
| 2048 | QPSK | 19 | 8 | 38 | random_at_entropy | same m/K; oracle mask charged | 0.0764 [0.0187, 0.1330] | -0.0007 [-0.0020, 0.0006] | -0.0081 [-0.0196, 0.0024] |
| 2048 | QPSK | 19 | 8 | 38 | oracle_at_entropy | N/A: paid mask exceeds budget | N/A | N/A | N/A |
| 2048 | 16QAM | 1 | 6 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 1 | 6 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 1 | 6 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 4 | 7 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 4 | 7 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 4 | 7 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 7 | 8 | 0 | raster_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 7 | 8 | 0 | random_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 7 | 8 | 0 | oracle_at_entropy | K0: identical whole action | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 2048 | 16QAM | 13 | 9 | 64 | raster_at_entropy | same m/K; oracle mask charged | 0.0549 [0.0158, 0.0962] | -0.0015 [-0.0024, -0.0007] | 0.0022 [-0.0041, 0.0083] |
| 2048 | 16QAM | 13 | 9 | 64 | random_at_entropy | same m/K; oracle mask charged | 0.0048 [-0.0282, 0.0379] | -0.0003 [-0.0011, 0.0005] | 0.0019 [-0.0057, 0.0097] |
| 2048 | 16QAM | 13 | 9 | 64 | oracle_at_entropy | same m/K; oracle mask charged | 0.0265 [-0.0134, 0.0688] | -0.0011 [-0.0023, -0.0002] | 0.0011 [-0.0014, 0.0036] |
| 2048 | 16QAM | 19 | 9 | 166 | raster_at_entropy | same m/K; oracle mask charged | 0.0249 [-0.0357, 0.0873] | -0.0004 [-0.0013, 0.0006] | 0.0026 [-0.0056, 0.0115] |
| 2048 | 16QAM | 19 | 9 | 166 | random_at_entropy | same m/K; oracle mask charged | 0.0497 [0.0067, 0.0952] | -0.0011 [-0.0020, -0.0002] | 0.0036 [-0.0030, 0.0104] |
| 2048 | 16QAM | 19 | 9 | 166 | oracle_at_entropy | N/A: paid mask exceeds budget | N/A | N/A | N/A |

## 公平性与指标

原始DINO为DINOv2 ViT-S/14，另列ViT-L/14；CLIP、DISTS、DreamSim、MS-SSIM、ResNet50真实标签准确率与原图预测一致率均在完整表中。骨干和权重见[模型元数据](../results/m1_action_space_closure_20261005/MODEL_METADATA.json)。新指标没有参与策略选择。分类器一致率是自动代理，不等同人工语义正确性。

QPSK和连续链逐帧E=2N；16QAM保留固定星座，实际逐帧能量分别记录，不能称逐帧同能量。P使用各自历史最终模型与训练seed；只按源图及名义噪声种子配对，噪声命名空间不同。

历史指标从完成凭证复制，缺少的后加诊断显式N/A；不把N/A补成0。F误差的无有效latent分支用登记零擦除代理。已保存的批处理耗时不等于在线单帧接收成本；见[COSTS](../results/m1_action_space_closure_20261005/COSTS.json)。

## 固定16张图

每个N/调制/SNR四页，列为原图/P/新whole/新entropy/新oracle，seed2001；只辅助看图，不替代100×3统计。

- N1024 16QAM 1dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr1_page4.pdf)
- N1024 16QAM 4dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr4_page4.pdf)
- N1024 16QAM 7dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr7_page4.pdf)
- N1024 16QAM 13dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr13_page4.pdf)
- N1024 16QAM 19dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N1024_16QAM_snr19_page4.pdf)
- N2048 QPSK 1dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr1_page4.pdf)
- N2048 QPSK 4dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr4_page4.pdf)
- N2048 QPSK 7dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr7_page4.pdf)
- N2048 QPSK 13dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr13_page4.pdf)
- N2048 QPSK 19dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_QPSK_snr19_page4.pdf)
- N2048 16QAM 1dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr1_page4.pdf)
- N2048 16QAM 4dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr4_page4.pdf)
- N2048 16QAM 7dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr7_page4.pdf)
- N2048 16QAM 13dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr13_page4.pdf)
- N2048 16QAM 19dB：[PNG 1](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page1.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page1.pdf), [PNG 2](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page2.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page2.pdf), [PNG 3](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page3.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page3.pdf), [PNG 4](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page4.png) / [PDF](../results/m1_action_space_closure_20261005/figures/fixed16_N2048_16QAM_snr19_page4.pdf)

结论应分别看语义、像素和感知指标及区间。单项获胜不等于全面优势；候选空间扩大也不保证所选策略更好。
