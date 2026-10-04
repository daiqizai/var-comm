# 同 K 排序对照

差值为 entropy − comparator。PSNR、DINO 正值更好；LPIPS 负值更好。

| PHY | SNR | K | 对照 | 解释 | ΔPSNR [95%CI] | ΔLPIPS [95%CI] | ΔDINOv2-L [95%CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QPSK | 1 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 1 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 1 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 4 | 25 | Raster @ entropy K | same m/K; oracle pays its mask | 0.1690 [0.0870, 0.2600] | -0.0044 [-0.0064, -0.0022] | 0.0170 [0.0014, 0.0330] |
| QPSK | 4 | 25 | Random @ entropy K | same m/K; oracle pays its mask | 0.1965 [0.1181, 0.2835] | -0.0047 [-0.0067, -0.0027] | 0.0045 [-0.0148, 0.0225] |
| QPSK | 4 | 25 | Paid oracle @ entropy K | same m/K; oracle pays its mask | 0.0356 [-0.0026, 0.0751] | -0.0016 [-0.0028, -0.0005] | 0.0033 [-0.0008, 0.0073] |
| QPSK | 7 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 7 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 7 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 13 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 13 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 13 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 19 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 19 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| QPSK | 19 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 1 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 1 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 1 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 4 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 4 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 4 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 7 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 7 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 7 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 13 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 13 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 13 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 19 | 0 | Raster @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 19 | 0 | Random @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 16QAM | 19 | 0 | Paid oracle @ entropy K | K0: no ordering contrast | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
