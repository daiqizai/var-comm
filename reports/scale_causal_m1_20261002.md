# 方法一：尺度因果的部分尺度传输

## 主要结果

统计单位为100个源图像。每源先平均三次登记噪声，再对源做10,000次共享配对bootstrap。区间为重复使用development上的描述性95%区间，未包含训练种子不确定性。

熵排序的主要机制比较使用相同m、K的raster/random对照。与历史legacy/P仅按源及名义seed配对，历史物理噪声namespace不同。QPSK满足E=2N；16QAM沿用原符号映射，实际E可随bit变化，其对照仅同N/调制，能量差另列。

| N | PHY | SNR | 对照 | ΔLPIPS [95% CI] | ΔPSNR [95% CI] | 判定 |
|---|---|---:|---|---|---|---|
| 512 | 16QAM | 7 | P512 | 0.0717443 [0.0635642, 0.0798221] | -3.23237 [-3.4704, -3.00001] | 该对照未建立此结论 |
| 512 | 16QAM | 7 | legacy_policy | -0.0101902 [-0.0142196, -0.0062268] | 0.126087 [0.0114427, 0.237981] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 7 | random_at_entropy | 0.00124235 [-0.00168184, 0.00414503] | 0.0234859 [-0.0558291, 0.106744] | 该对照未建立此结论 |
| 512 | 16QAM | 7 | raster_at_entropy | -0.000659706 [-0.00355475, 0.00225343] | -0.0131964 [-0.0928766, 0.0675107] | 该对照未建立此结论 |
| 512 | 16QAM | 7 | whole_policy | -0.01121 [-0.0150998, -0.00728173] | 0.132692 [0.0240103, 0.239937] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 13 | P512 | 0.0261369 [0.0196693, 0.0326666] | -2.2949 [-2.52653, -2.05877] | 该对照未建立此结论 |
| 512 | 16QAM | 13 | legacy_policy | -0.0171509 [-0.0201514, -0.0142254] | 0.367169 [0.278386, 0.453494] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 13 | random_at_entropy | -0.00211944 [-0.00504911, 0.000665675] | 0.0971216 [0.0224159, 0.177285] | 该对照未建立此结论 |
| 512 | 16QAM | 13 | raster_at_entropy | -0.00492573 [-0.0081461, -0.00181902] | 0.0771203 [-0.0254141, 0.180423] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 13 | whole_policy | -0.0171509 [-0.0201515, -0.0142254] | 0.367169 [0.278386, 0.453494] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 19 | P512 | 0.0072597 [0.00114098, 0.0132399] | -1.79852 [-2.02048, -1.56983] | 该对照未建立此结论 |
| 512 | 16QAM | 19 | legacy_policy | -0.0425788 [-0.0465136, -0.0388578] | 1.00715 [0.882787, 1.14302] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 19 | random_at_entropy | -0.0040599 [-0.0071593, -0.000822727] | 0.132034 [0.022521, 0.244461] | LPIPS下降且PSNR保护成立 |
| 512 | 16QAM | 19 | raster_at_entropy | -0.00153242 [-0.00458688, 0.00148416] | 0.000974884 [-0.0855497, 0.087813] | 该对照未建立此结论 |
| 512 | 16QAM | 19 | whole_policy | -0.0425788 [-0.0465136, -0.0388578] | 1.00715 [0.882787, 1.14302] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 7 | P512 | 0.0721554 [0.0631784, 0.0810992] | -3.2889 [-3.54376, -3.04075] | 该对照未建立此结论 |
| 512 | QPSK | 7 | legacy_policy | -0.00844628 [-0.0124368, -0.00422202] | 0.0278915 [-0.0796641, 0.127105] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 7 | random_at_entropy | -0.00329767 [-0.00625009, -0.000413822] | -0.00600266 [-0.0941363, 0.0885842] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 7 | raster_at_entropy | -0.0040935 [-0.00727284, -0.00100746] | -0.0192104 [-0.105059, 0.0652276] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 7 | whole_policy | -0.0083514 [-0.0123381, -0.0041085] | 0.0294678 [-0.0784779, 0.128606] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 13 | P512 | 0.0921151 [0.0831593, 0.100909] | -3.68372 [-3.9445, -3.42183] | 该对照未建立此结论 |
| 512 | QPSK | 13 | legacy_policy | -0.0146445 [-0.0176155, -0.0116775] | 0.149654 [0.0675787, 0.233601] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 13 | random_at_entropy | -0.00326548 [-0.0066712, 5.14974e-05] | -0.0161604 [-0.11059, 0.0853635] | 该对照未建立此结论 |
| 512 | QPSK | 13 | raster_at_entropy | -0.00393956 [-0.00737548, -0.000558665] | -0.0316801 [-0.131553, 0.0647927] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 13 | whole_policy | -0.0146445 [-0.0176155, -0.0116775] | 0.149654 [0.0675786, 0.233601] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 19 | P512 | 0.0986658 [0.0895391, 0.107617] | -3.82732 [-4.09557, -3.55922] | 该对照未建立此结论 |
| 512 | QPSK | 19 | legacy_policy | -0.0146445 [-0.0176155, -0.0116775] | 0.149654 [0.0675787, 0.233601] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 19 | random_at_entropy | -0.00326548 [-0.0066712, 5.14974e-05] | -0.0161604 [-0.11059, 0.0853635] | 该对照未建立此结论 |
| 512 | QPSK | 19 | raster_at_entropy | -0.00393956 [-0.00737548, -0.000558665] | -0.0316801 [-0.131553, 0.0647927] | LPIPS下降且PSNR保护成立 |
| 512 | QPSK | 19 | whole_policy | -0.0146445 [-0.0176155, -0.0116775] | 0.149654 [0.0675786, 0.233601] | LPIPS下降且PSNR保护成立 |
| 1024 | 16QAM | 7 | P1024 | 0.0234867 [0.0180501, 0.0287739] | -1.69371 [-1.87831, -1.50742] | 该对照未建立此结论 |
| 1024 | 16QAM | 7 | legacy_policy | -0.00112971 [-0.00431129, 0.00190996] | 0.0877048 [-0.0337418, 0.223891] | 该对照未建立此结论 |
| 1024 | 16QAM | 7 | random_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 7 | raster_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 7 | whole_policy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 13 | P1024 | -0.0117698 [-0.0163198, -0.00687524] | -0.598209 [-0.819905, -0.38065] | 该对照未建立此结论 |
| 1024 | 16QAM | 13 | legacy_policy | -0.0522134 [-0.0565967, -0.0475076] | 1.4641 [1.29616, 1.6273] | LPIPS下降且PSNR保护成立 |
| 1024 | 16QAM | 13 | random_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 13 | raster_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 13 | whole_policy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 19 | P1024 | -0.00993635 [-0.0140635, -0.00588567] | -0.601641 [-0.788982, -0.397452] | 该对照未建立此结论 |
| 1024 | 16QAM | 19 | legacy_policy | -0.0566246 [-0.0604511, -0.052816] | 1.63849 [1.50054, 1.78121] | LPIPS下降且PSNR保护成立 |
| 1024 | 16QAM | 19 | random_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 19 | raster_at_entropy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | 16QAM | 19 | whole_policy | 0 [0, 0] | 0 [0, 0] | 该对照未建立此结论 |
| 1024 | QPSK | 7 | P1024 | 0.0482511 [0.0420245, 0.0542936] | -2.33134 [-2.54035, -2.11912] | 该对照未建立此结论 |
| 1024 | QPSK | 7 | legacy_policy | -0.0339214 [-0.037588, -0.0304333] | 0.798581 [0.677861, 0.928715] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 7 | random_at_entropy | -0.0045821 [-0.00740917, -0.00181808] | 0.169274 [0.072272, 0.275819] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 7 | raster_at_entropy | -0.00404308 [-0.00729554, -0.000983841] | 0.0500479 [-0.0474626, 0.153282] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 7 | whole_policy | -0.0339214 [-0.037588, -0.0304333] | 0.798581 [0.677861, 0.928715] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 13 | P1024 | 0.0522966 [0.0462635, 0.0582655] | -2.47545 [-2.69892, -2.24901] | 该对照未建立此结论 |
| 1024 | QPSK | 13 | legacy_policy | -0.0544099 [-0.0593168, -0.0498199] | 1.24164 [1.1141, 1.37716] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 13 | random_at_entropy | -0.00610412 [-0.00869768, -0.0035566] | 0.0883308 [-0.00688774, 0.188415] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 13 | raster_at_entropy | -0.00444941 [-0.00764314, -0.00134921] | 0.0240119 [-0.0754875, 0.12225] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 13 | whole_policy | -0.0544099 [-0.0593169, -0.0498199] | 1.24164 [1.1141, 1.37716] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 19 | P1024 | 0.0585413 [0.0522696, 0.0647706] | -2.65327 [-2.88454, -2.41772] | 该对照未建立此结论 |
| 1024 | QPSK | 19 | legacy_policy | -0.0544099 [-0.0593168, -0.0498199] | 1.24164 [1.1141, 1.37716] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 19 | random_at_entropy | -0.00610412 [-0.00869768, -0.0035566] | 0.0883308 [-0.00688774, 0.188415] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 19 | raster_at_entropy | -0.00444941 [-0.00764314, -0.00134921] | 0.0240119 [-0.0754875, 0.12225] | LPIPS下降且PSNR保护成立 |
| 1024 | QPSK | 19 | whole_policy | -0.0544099 [-0.0593169, -0.0498199] | 1.24164 [1.1141, 1.37716] | LPIPS下降且PSNR保护成立 |

上述判定逐N/PHY/SNR/对照报告，条件为LPIPS差区间上界<0且PSNR差区间下界≥−0.25dB。它不单独证明总体最优或高SNR已全面解除饱和。熵机制须同时查看两个同K对照；16QAM还须查看ΔE。

## 冻结的校准选择

校准policy SHA256：`785e2223d1126bfcc4de78cb2c752d565090d5d7537d9e681879f029f4f76053`。policy在development读取前封存，analysis不重选K。

| N | PHY | SNR | 方法 | m | K | 顺序 |
|---|---|---:|---|---:|---:|---|
| 512 | 16QAM | 1 | entropy_policy | 4 | 0 | whole |
| 512 | 16QAM | 1 | oracle_policy | 4 | 0 | whole |
| 512 | 16QAM | 1 | random_policy | 4 | 0 | whole |
| 512 | 16QAM | 1 | raster_policy | 4 | 0 | whole |
| 512 | 16QAM | 1 | whole_policy | 4 | 0 | whole |
| 512 | 16QAM | 4 | entropy_policy | 4 | 6 | entropy |
| 512 | 16QAM | 4 | oracle_policy | 4 | 6 | oracle |
| 512 | 16QAM | 4 | random_policy | 4 | 6 | random |
| 512 | 16QAM | 4 | raster_policy | 4 | 6 | raster |
| 512 | 16QAM | 4 | whole_policy | 4 | 0 | whole |
| 512 | 16QAM | 7 | entropy_policy | 5 | 9 | entropy |
| 512 | 16QAM | 7 | oracle_policy | 5 | 9 | oracle |
| 512 | 16QAM | 7 | random_policy | 5 | 9 | random |
| 512 | 16QAM | 7 | raster_policy | 5 | 9 | raster |
| 512 | 16QAM | 7 | whole_policy | 5 | 0 | whole |
| 512 | 16QAM | 13 | entropy_policy | 6 | 16 | entropy |
| 512 | 16QAM | 13 | oracle_policy | 6 | 16 | oracle |
| 512 | 16QAM | 13 | random_policy | 6 | 16 | random |
| 512 | 16QAM | 13 | raster_policy | 6 | 16 | raster |
| 512 | 16QAM | 13 | whole_policy | 6 | 0 | whole |
| 512 | 16QAM | 19 | entropy_policy | 6 | 38 | entropy |
| 512 | 16QAM | 19 | oracle_policy | 6 | 33 | oracle |
| 512 | 16QAM | 19 | random_policy | 6 | 38 | random |
| 512 | 16QAM | 19 | raster_policy | 6 | 38 | raster |
| 512 | 16QAM | 19 | whole_policy | 6 | 0 | whole |
| 512 | QPSK | 1 | entropy_policy | 4 | 0 | whole |
| 512 | QPSK | 1 | oracle_policy | 4 | 0 | whole |
| 512 | QPSK | 1 | random_policy | 4 | 0 | whole |
| 512 | QPSK | 1 | raster_policy | 4 | 0 | whole |
| 512 | QPSK | 1 | whole_policy | 4 | 0 | whole |
| 512 | QPSK | 4 | entropy_policy | 4 | 12 | entropy |
| 512 | QPSK | 4 | oracle_policy | 4 | 12 | oracle |
| 512 | QPSK | 4 | random_policy | 4 | 12 | random |
| 512 | QPSK | 4 | raster_policy | 4 | 12 | raster |
| 512 | QPSK | 4 | whole_policy | 4 | 0 | whole |
| 512 | QPSK | 7 | entropy_policy | 5 | 7 | entropy |
| 512 | QPSK | 7 | oracle_policy | 5 | 4 | oracle |
| 512 | QPSK | 7 | random_policy | 5 | 7 | random |
| 512 | QPSK | 7 | raster_policy | 5 | 7 | raster |
| 512 | QPSK | 7 | whole_policy | 5 | 0 | whole |
| 512 | QPSK | 13 | entropy_policy | 5 | 7 | entropy |
| 512 | QPSK | 13 | oracle_policy | 5 | 4 | oracle |
| 512 | QPSK | 13 | random_policy | 5 | 7 | random |
| 512 | QPSK | 13 | raster_policy | 5 | 7 | raster |
| 512 | QPSK | 13 | whole_policy | 5 | 0 | whole |
| 512 | QPSK | 19 | entropy_policy | 5 | 7 | entropy |
| 512 | QPSK | 19 | oracle_policy | 5 | 4 | oracle |
| 512 | QPSK | 19 | random_policy | 5 | 7 | random |
| 512 | QPSK | 19 | raster_policy | 5 | 7 | raster |
| 512 | QPSK | 19 | whole_policy | 5 | 0 | whole |
| 1024 | 16QAM | 1 | entropy_policy | 5 | 0 | whole |
| 1024 | 16QAM | 1 | oracle_policy | 5 | 0 | whole |
| 1024 | 16QAM | 1 | random_policy | 5 | 0 | whole |
| 1024 | 16QAM | 1 | raster_policy | 5 | 0 | whole |
| 1024 | 16QAM | 1 | whole_policy | 5 | 0 | whole |
| 1024 | 16QAM | 4 | entropy_policy | 6 | 0 | whole |
| 1024 | 16QAM | 4 | oracle_policy | 6 | 0 | whole |
| 1024 | 16QAM | 4 | random_policy | 6 | 0 | whole |
| 1024 | 16QAM | 4 | raster_policy | 6 | 0 | whole |
| 1024 | 16QAM | 4 | whole_policy | 6 | 0 | whole |
| 1024 | 16QAM | 7 | entropy_policy | 7 | 0 | whole |
| 1024 | 16QAM | 7 | oracle_policy | 7 | 0 | whole |
| 1024 | 16QAM | 7 | random_policy | 7 | 0 | whole |
| 1024 | 16QAM | 7 | raster_policy | 7 | 0 | whole |
| 1024 | 16QAM | 7 | whole_policy | 7 | 0 | whole |
| 1024 | 16QAM | 13 | entropy_policy | 8 | 0 | whole |
| 1024 | 16QAM | 13 | oracle_policy | 8 | 0 | whole |
| 1024 | 16QAM | 13 | random_policy | 8 | 0 | whole |
| 1024 | 16QAM | 13 | raster_policy | 8 | 0 | whole |
| 1024 | 16QAM | 13 | whole_policy | 8 | 0 | whole |
| 1024 | 16QAM | 19 | entropy_policy | 8 | 0 | whole |
| 1024 | 16QAM | 19 | oracle_policy | 8 | 0 | whole |
| 1024 | 16QAM | 19 | random_policy | 8 | 0 | whole |
| 1024 | 16QAM | 19 | raster_policy | 8 | 0 | whole |
| 1024 | 16QAM | 19 | whole_policy | 8 | 0 | whole |
| 1024 | QPSK | 1 | entropy_policy | 5 | 0 | whole |
| 1024 | QPSK | 1 | oracle_policy | 5 | 0 | whole |
| 1024 | QPSK | 1 | random_policy | 5 | 0 | whole |
| 1024 | QPSK | 1 | raster_policy | 5 | 0 | whole |
| 1024 | QPSK | 1 | whole_policy | 5 | 0 | whole |
| 1024 | QPSK | 4 | entropy_policy | 6 | 0 | whole |
| 1024 | QPSK | 4 | oracle_policy | 6 | 0 | whole |
| 1024 | QPSK | 4 | random_policy | 6 | 0 | whole |
| 1024 | QPSK | 4 | raster_policy | 6 | 0 | whole |
| 1024 | QPSK | 4 | whole_policy | 6 | 0 | whole |
| 1024 | QPSK | 7 | entropy_policy | 6 | 32 | entropy |
| 1024 | QPSK | 7 | oracle_policy | 6 | 32 | oracle |
| 1024 | QPSK | 7 | random_policy | 6 | 32 | random |
| 1024 | QPSK | 7 | raster_policy | 6 | 32 | raster |
| 1024 | QPSK | 7 | whole_policy | 6 | 0 | whole |
| 1024 | QPSK | 13 | entropy_policy | 6 | 48 | entropy |
| 1024 | QPSK | 13 | oracle_policy | 6 | 43 | oracle |
| 1024 | QPSK | 13 | random_policy | 6 | 48 | random |
| 1024 | QPSK | 13 | raster_policy | 6 | 48 | raster |
| 1024 | QPSK | 13 | whole_policy | 6 | 0 | whole |
| 1024 | QPSK | 19 | entropy_policy | 6 | 48 | entropy |
| 1024 | QPSK | 19 | oracle_policy | 6 | 43 | oracle |
| 1024 | QPSK | 19 | random_policy | 6 | 48 | random |
| 1024 | QPSK | 19 | raster_policy | 6 | 48 | raster |
| 1024 | QPSK | 19 | whole_policy | 6 | 0 | whole |

## 指标、有效分母和失败

`m1_summary.csv`保留PSNR、LPIPS、DINO、DINO specificity、DINO<0.6与LPIPS>0.35事件、E、latent有效率和各CRC指标。图像质量包含全部300帧/组，包括灰色erasure。

F平方误差仅在实际latent有效的帧计算，不以零填补失败。汇总先对每源可用噪声平均再平均源；`n_eligible_frames/n_sources`注明分母。F配对差只使用双方同一source/noise均有效的交集，并注明其源与帧数。它不能代替全失败率。

prefix CRC失败率以header有效为条件；partial CRC失败率以K>0且header/prefix均有效为条件。缺少字段的历史项保持不可用，不当作零失败。oracle按显式mask收费，是错误位置优先的诊断参考，未称全局质量上界。

## 证据与图

- [完整汇总](../results/scale_causal_partial_residual_20261002/m1_summary.csv)、[逐源均值](../results/scale_causal_partial_residual_20261002/m1_table_shards/m1_source_means.csv.index.md)、[全部配对区间](../results/scale_causal_partial_residual_20261002/m1_paired_intervals.csv)、[高SNR对照](../results/scale_causal_partial_residual_20261002/m1_high_snr_evidence.csv)。
- [N512质量](../results/scale_causal_partial_residual_20261002/figures/m1_quality_N512.png)、[N1024质量](../results/scale_causal_partial_residual_20261002/figures/m1_quality_N1024.png)、[校准K](../results/scale_causal_partial_residual_20261002/figures/m1_K_vs_SNR.png)。
- `m1_rate_curve.csv`与对应图是zero-noise source诊断，只连接实测点，不作为实际付费链路性能。
- 样例固定source_index=0/25/50/75、SNR=4/13、seed=2001，位于`examples/m1/`，不按结果挑图。

## 接收与发送时间

计时来自独立uncached路径；实际header失败/erasure路径保留，不由质量CSV中的缓存运行时间代替。

| N | PHY | SNR | 方法 | 阶段 | 均值ms | 样本数 |
|---|---|---:|---|---|---:|---:|
| 1024 | 16QAM | 1 | entropy_policy | tx_ms | 9.80141 | 10 |
| 1024 | 16QAM | 1 | entropy_policy | rx_ms | 116.208 | 10 |
| 1024 | 16QAM | 1 | oracle_policy | tx_ms | 9.84257 | 10 |
| 1024 | 16QAM | 1 | oracle_policy | rx_ms | 116.839 | 10 |
| 1024 | 16QAM | 1 | random_policy | tx_ms | 9.82305 | 10 |
| 1024 | 16QAM | 1 | random_policy | rx_ms | 116.124 | 10 |
| 1024 | 16QAM | 1 | raster_policy | tx_ms | 9.89534 | 10 |
| 1024 | 16QAM | 1 | raster_policy | rx_ms | 119.142 | 10 |
| 1024 | 16QAM | 1 | whole_policy | tx_ms | 9.80637 | 10 |
| 1024 | 16QAM | 1 | whole_policy | rx_ms | 117.57 | 10 |
| 1024 | 16QAM | 13 | entropy_policy | tx_ms | 11.0783 | 10 |
| 1024 | 16QAM | 13 | entropy_policy | rx_ms | 117.644 | 10 |
| 1024 | 16QAM | 13 | oracle_policy | tx_ms | 10.9441 | 10 |
| 1024 | 16QAM | 13 | oracle_policy | rx_ms | 116.742 | 10 |
| 1024 | 16QAM | 13 | random_policy | tx_ms | 11.1763 | 10 |
| 1024 | 16QAM | 13 | random_policy | rx_ms | 117.78 | 10 |
| 1024 | 16QAM | 13 | raster_policy | tx_ms | 11.0942 | 10 |
| 1024 | 16QAM | 13 | raster_policy | rx_ms | 117.917 | 10 |
| 1024 | 16QAM | 13 | whole_policy | tx_ms | 10.993 | 10 |
| 1024 | 16QAM | 13 | whole_policy | rx_ms | 117.623 | 10 |
| 1024 | 16QAM | 19 | entropy_policy | tx_ms | 10.9339 | 10 |
| 1024 | 16QAM | 19 | entropy_policy | rx_ms | 116.748 | 10 |
| 1024 | 16QAM | 19 | oracle_policy | tx_ms | 10.9298 | 10 |
| 1024 | 16QAM | 19 | oracle_policy | rx_ms | 117.04 | 10 |
| 1024 | 16QAM | 19 | random_policy | tx_ms | 10.9487 | 10 |
| 1024 | 16QAM | 19 | random_policy | rx_ms | 117.096 | 10 |
| 1024 | 16QAM | 19 | raster_policy | tx_ms | 10.9891 | 10 |
| 1024 | 16QAM | 19 | raster_policy | rx_ms | 117.321 | 10 |
| 1024 | 16QAM | 19 | whole_policy | tx_ms | 11.02 | 10 |
| 1024 | 16QAM | 19 | whole_policy | rx_ms | 116.807 | 10 |
| 1024 | 16QAM | 4 | entropy_policy | tx_ms | 10.1325 | 10 |
| 1024 | 16QAM | 4 | entropy_policy | rx_ms | 116.649 | 10 |
| 1024 | 16QAM | 4 | oracle_policy | tx_ms | 10.0027 | 10 |
| 1024 | 16QAM | 4 | oracle_policy | rx_ms | 116.259 | 10 |
| 1024 | 16QAM | 4 | random_policy | tx_ms | 10.0372 | 10 |
| 1024 | 16QAM | 4 | random_policy | rx_ms | 117.872 | 10 |
| 1024 | 16QAM | 4 | raster_policy | tx_ms | 10.1133 | 10 |
| 1024 | 16QAM | 4 | raster_policy | rx_ms | 116.331 | 10 |
| 1024 | 16QAM | 4 | whole_policy | tx_ms | 10.0116 | 10 |
| 1024 | 16QAM | 4 | whole_policy | rx_ms | 116.65 | 10 |
| 1024 | 16QAM | 7 | entropy_policy | tx_ms | 10.391 | 10 |
| 1024 | 16QAM | 7 | entropy_policy | rx_ms | 117.752 | 10 |
| 1024 | 16QAM | 7 | oracle_policy | tx_ms | 10.3652 | 10 |
| 1024 | 16QAM | 7 | oracle_policy | rx_ms | 117.409 | 10 |
| 1024 | 16QAM | 7 | random_policy | tx_ms | 10.4404 | 10 |
| 1024 | 16QAM | 7 | random_policy | rx_ms | 117.432 | 10 |
| 1024 | 16QAM | 7 | raster_policy | tx_ms | 10.3818 | 10 |
| 1024 | 16QAM | 7 | raster_policy | rx_ms | 116.743 | 10 |
| 1024 | 16QAM | 7 | whole_policy | tx_ms | 10.3863 | 10 |
| 1024 | 16QAM | 7 | whole_policy | rx_ms | 116.85 | 10 |
| 1024 | QPSK | 1 | entropy_policy | tx_ms | 9.80584 | 10 |
| 1024 | QPSK | 1 | entropy_policy | rx_ms | 116.109 | 10 |
| 1024 | QPSK | 1 | oracle_policy | tx_ms | 9.78561 | 10 |
| 1024 | QPSK | 1 | oracle_policy | rx_ms | 116.05 | 10 |
| 1024 | QPSK | 1 | random_policy | tx_ms | 9.78635 | 10 |
| 1024 | QPSK | 1 | random_policy | rx_ms | 115.702 | 10 |
| 1024 | QPSK | 1 | raster_policy | tx_ms | 9.78741 | 10 |
| 1024 | QPSK | 1 | raster_policy | rx_ms | 115.69 | 10 |
| 1024 | QPSK | 1 | whole_policy | tx_ms | 9.79718 | 10 |
| 1024 | QPSK | 1 | whole_policy | rx_ms | 115.884 | 10 |
| 1024 | QPSK | 13 | entropy_policy | tx_ms | 84.594 | 10 |
| 1024 | QPSK | 13 | entropy_policy | rx_ms | 118.801 | 10 |
| 1024 | QPSK | 13 | oracle_policy | tx_ms | 85.4155 | 10 |
| 1024 | QPSK | 13 | oracle_policy | rx_ms | 116.229 | 10 |
| 1024 | QPSK | 13 | random_policy | tx_ms | 10.5754 | 10 |
| 1024 | QPSK | 13 | random_policy | rx_ms | 116.62 | 10 |
| 1024 | QPSK | 13 | raster_policy | tx_ms | 10.4678 | 10 |
| 1024 | QPSK | 13 | raster_policy | rx_ms | 117.275 | 10 |
| 1024 | QPSK | 13 | whole_policy | tx_ms | 10.007 | 10 |
| 1024 | QPSK | 13 | whole_policy | rx_ms | 116.224 | 10 |
| 1024 | QPSK | 19 | entropy_policy | tx_ms | 85.0282 | 10 |
| 1024 | QPSK | 19 | entropy_policy | rx_ms | 120.197 | 10 |
| 1024 | QPSK | 19 | oracle_policy | tx_ms | 85.1549 | 10 |
| 1024 | QPSK | 19 | oracle_policy | rx_ms | 118.282 | 10 |
| 1024 | QPSK | 19 | random_policy | tx_ms | 10.5244 | 10 |
| 1024 | QPSK | 19 | random_policy | rx_ms | 116.489 | 10 |
| 1024 | QPSK | 19 | raster_policy | tx_ms | 10.6002 | 10 |
| 1024 | QPSK | 19 | raster_policy | rx_ms | 116.647 | 10 |
| 1024 | QPSK | 19 | whole_policy | tx_ms | 10.0216 | 10 |
| 1024 | QPSK | 19 | whole_policy | rx_ms | 116.192 | 10 |
| 1024 | QPSK | 4 | entropy_policy | tx_ms | 9.98247 | 10 |
| 1024 | QPSK | 4 | entropy_policy | rx_ms | 115.955 | 10 |
| 1024 | QPSK | 4 | oracle_policy | tx_ms | 9.97031 | 10 |
| 1024 | QPSK | 4 | oracle_policy | rx_ms | 115.805 | 10 |
| 1024 | QPSK | 4 | random_policy | tx_ms | 9.99774 | 10 |
| 1024 | QPSK | 4 | random_policy | rx_ms | 115.938 | 10 |
| 1024 | QPSK | 4 | raster_policy | tx_ms | 10.0002 | 10 |
| 1024 | QPSK | 4 | raster_policy | rx_ms | 115.922 | 10 |
| 1024 | QPSK | 4 | whole_policy | tx_ms | 9.98218 | 10 |
| 1024 | QPSK | 4 | whole_policy | rx_ms | 115.858 | 10 |
| 1024 | QPSK | 7 | entropy_policy | tx_ms | 84.7357 | 10 |
| 1024 | QPSK | 7 | entropy_policy | rx_ms | 118.914 | 10 |
| 1024 | QPSK | 7 | oracle_policy | tx_ms | 85.0298 | 10 |
| 1024 | QPSK | 7 | oracle_policy | rx_ms | 116.447 | 10 |
| 1024 | QPSK | 7 | random_policy | tx_ms | 10.4424 | 10 |
| 1024 | QPSK | 7 | random_policy | rx_ms | 117.084 | 10 |
| 1024 | QPSK | 7 | raster_policy | tx_ms | 10.3534 | 10 |
| 1024 | QPSK | 7 | raster_policy | rx_ms | 116.484 | 10 |
| 1024 | QPSK | 7 | whole_policy | tx_ms | 10.0662 | 10 |
| 1024 | QPSK | 7 | whole_policy | rx_ms | 116.192 | 10 |
| 512 | 16QAM | 1 | entropy_policy | tx_ms | 9.62738 | 10 |
| 512 | 16QAM | 1 | entropy_policy | rx_ms | 116.217 | 10 |
| 512 | 16QAM | 1 | oracle_policy | tx_ms | 9.73026 | 10 |
| 512 | 16QAM | 1 | oracle_policy | rx_ms | 116.003 | 10 |
| 512 | 16QAM | 1 | random_policy | tx_ms | 9.61693 | 10 |
| 512 | 16QAM | 1 | random_policy | rx_ms | 116.521 | 10 |
| 512 | 16QAM | 1 | raster_policy | tx_ms | 9.60842 | 10 |
| 512 | 16QAM | 1 | raster_policy | rx_ms | 115.765 | 10 |
| 512 | 16QAM | 1 | whole_policy | tx_ms | 9.65994 | 10 |
| 512 | 16QAM | 1 | whole_policy | rx_ms | 115.793 | 10 |
| 512 | 16QAM | 13 | entropy_policy | tx_ms | 84.4817 | 10 |
| 512 | 16QAM | 13 | entropy_policy | rx_ms | 118.489 | 10 |
| 512 | 16QAM | 13 | oracle_policy | tx_ms | 84.6325 | 10 |
| 512 | 16QAM | 13 | oracle_policy | rx_ms | 116.202 | 10 |
| 512 | 16QAM | 13 | random_policy | tx_ms | 10.3479 | 10 |
| 512 | 16QAM | 13 | random_policy | rx_ms | 116.692 | 10 |
| 512 | 16QAM | 13 | raster_policy | tx_ms | 10.2789 | 10 |
| 512 | 16QAM | 13 | raster_policy | rx_ms | 116.566 | 10 |
| 512 | 16QAM | 13 | whole_policy | tx_ms | 10.0272 | 10 |
| 512 | 16QAM | 13 | whole_policy | rx_ms | 116.035 | 10 |
| 512 | 16QAM | 19 | entropy_policy | tx_ms | 85.1138 | 10 |
| 512 | 16QAM | 19 | entropy_policy | rx_ms | 118.959 | 10 |
| 512 | 16QAM | 19 | oracle_policy | tx_ms | 84.8022 | 10 |
| 512 | 16QAM | 19 | oracle_policy | rx_ms | 116.05 | 10 |
| 512 | 16QAM | 19 | random_policy | tx_ms | 10.5105 | 10 |
| 512 | 16QAM | 19 | random_policy | rx_ms | 116.615 | 10 |
| 512 | 16QAM | 19 | raster_policy | tx_ms | 10.3862 | 10 |
| 512 | 16QAM | 19 | raster_policy | rx_ms | 116.333 | 10 |
| 512 | 16QAM | 19 | whole_policy | tx_ms | 10.1265 | 10 |
| 512 | 16QAM | 19 | whole_policy | rx_ms | 116.378 | 10 |
| 512 | 16QAM | 4 | entropy_policy | tx_ms | 62.2292 | 10 |
| 512 | 16QAM | 4 | entropy_policy | rx_ms | 116.728 | 10 |
| 512 | 16QAM | 4 | oracle_policy | tx_ms | 62.2439 | 10 |
| 512 | 16QAM | 4 | oracle_policy | rx_ms | 115.792 | 10 |
| 512 | 16QAM | 4 | random_policy | tx_ms | 9.89886 | 10 |
| 512 | 16QAM | 4 | random_policy | rx_ms | 116.269 | 10 |
| 512 | 16QAM | 4 | raster_policy | tx_ms | 9.83451 | 10 |
| 512 | 16QAM | 4 | raster_policy | rx_ms | 116.756 | 10 |
| 512 | 16QAM | 4 | whole_policy | tx_ms | 9.62862 | 10 |
| 512 | 16QAM | 4 | whole_policy | rx_ms | 115.848 | 10 |
| 512 | 16QAM | 7 | entropy_policy | tx_ms | 73.1259 | 10 |
| 512 | 16QAM | 7 | entropy_policy | rx_ms | 117.229 | 10 |
| 512 | 16QAM | 7 | oracle_policy | tx_ms | 73.2054 | 10 |
| 512 | 16QAM | 7 | oracle_policy | rx_ms | 116.103 | 10 |
| 512 | 16QAM | 7 | random_policy | tx_ms | 10.0837 | 10 |
| 512 | 16QAM | 7 | random_policy | rx_ms | 116.395 | 10 |
| 512 | 16QAM | 7 | raster_policy | tx_ms | 10.0344 | 10 |
| 512 | 16QAM | 7 | raster_policy | rx_ms | 115.932 | 10 |
| 512 | 16QAM | 7 | whole_policy | tx_ms | 9.82684 | 10 |
| 512 | 16QAM | 7 | whole_policy | rx_ms | 115.784 | 10 |
| 512 | QPSK | 1 | entropy_policy | tx_ms | 9.61898 | 10 |
| 512 | QPSK | 1 | entropy_policy | rx_ms | 115.76 | 10 |
| 512 | QPSK | 1 | oracle_policy | tx_ms | 9.63076 | 10 |
| 512 | QPSK | 1 | oracle_policy | rx_ms | 115.693 | 10 |
| 512 | QPSK | 1 | random_policy | tx_ms | 9.79521 | 10 |
| 512 | QPSK | 1 | random_policy | rx_ms | 116.917 | 10 |
| 512 | QPSK | 1 | raster_policy | tx_ms | 9.6691 | 10 |
| 512 | QPSK | 1 | raster_policy | rx_ms | 116.592 | 10 |
| 512 | QPSK | 1 | whole_policy | tx_ms | 9.62403 | 10 |
| 512 | QPSK | 1 | whole_policy | rx_ms | 115.791 | 10 |
| 512 | QPSK | 13 | entropy_policy | tx_ms | 73.0471 | 10 |
| 512 | QPSK | 13 | entropy_policy | rx_ms | 117.437 | 10 |
| 512 | QPSK | 13 | oracle_policy | tx_ms | 72.9681 | 10 |
| 512 | QPSK | 13 | oracle_policy | rx_ms | 116.086 | 10 |
| 512 | QPSK | 13 | random_policy | tx_ms | 10.0965 | 10 |
| 512 | QPSK | 13 | random_policy | rx_ms | 116.601 | 10 |
| 512 | QPSK | 13 | raster_policy | tx_ms | 10.0468 | 10 |
| 512 | QPSK | 13 | raster_policy | rx_ms | 117.218 | 10 |
| 512 | QPSK | 13 | whole_policy | tx_ms | 9.92669 | 10 |
| 512 | QPSK | 13 | whole_policy | rx_ms | 117.719 | 10 |
| 512 | QPSK | 19 | entropy_policy | tx_ms | 72.8312 | 10 |
| 512 | QPSK | 19 | entropy_policy | rx_ms | 117.027 | 10 |
| 512 | QPSK | 19 | oracle_policy | tx_ms | 73.0178 | 10 |
| 512 | QPSK | 19 | oracle_policy | rx_ms | 115.663 | 10 |
| 512 | QPSK | 19 | random_policy | tx_ms | 10.0412 | 10 |
| 512 | QPSK | 19 | random_policy | rx_ms | 116.065 | 10 |
| 512 | QPSK | 19 | raster_policy | tx_ms | 10.0053 | 10 |
| 512 | QPSK | 19 | raster_policy | rx_ms | 115.947 | 10 |
| 512 | QPSK | 19 | whole_policy | tx_ms | 9.80905 | 10 |
| 512 | QPSK | 19 | whole_policy | rx_ms | 115.787 | 10 |
| 512 | QPSK | 4 | entropy_policy | tx_ms | 62.5514 | 10 |
| 512 | QPSK | 4 | entropy_policy | rx_ms | 116.626 | 10 |
| 512 | QPSK | 4 | oracle_policy | tx_ms | 62.698 | 10 |
| 512 | QPSK | 4 | oracle_policy | rx_ms | 115.862 | 10 |
| 512 | QPSK | 4 | random_policy | tx_ms | 9.91334 | 10 |
| 512 | QPSK | 4 | random_policy | rx_ms | 116.25 | 10 |
| 512 | QPSK | 4 | raster_policy | tx_ms | 9.85194 | 10 |
| 512 | QPSK | 4 | raster_policy | rx_ms | 116.094 | 10 |
| 512 | QPSK | 4 | whole_policy | tx_ms | 9.60052 | 10 |
| 512 | QPSK | 4 | whole_policy | rx_ms | 115.723 | 10 |
| 512 | QPSK | 7 | entropy_policy | tx_ms | 73.2081 | 10 |
| 512 | QPSK | 7 | entropy_policy | rx_ms | 116.835 | 10 |
| 512 | QPSK | 7 | oracle_policy | tx_ms | 72.874 | 10 |
| 512 | QPSK | 7 | oracle_policy | rx_ms | 115.794 | 10 |
| 512 | QPSK | 7 | random_policy | tx_ms | 10.054 | 10 |
| 512 | QPSK | 7 | random_policy | rx_ms | 115.923 | 10 |
| 512 | QPSK | 7 | raster_policy | tx_ms | 10.0637 | 10 |
| 512 | QPSK | 7 | raster_policy | rx_ms | 118.805 | 10 |
| 512 | QPSK | 7 | whole_policy | tx_ms | 9.80747 | 10 |
| 512 | QPSK | 7 | whole_policy | rx_ms | 116.005 | 10 |

## 范围

本报告完成方法一统计。方法二及其条件实际链路分支的状态由各自证据决定；本报告不写整个研究已完成。新颖性与已有先例见`literature_notes.md`。
