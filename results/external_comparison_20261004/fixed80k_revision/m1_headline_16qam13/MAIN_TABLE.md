# N1024，13 dB：固定16图五列比较

Original仅作视觉参照；下表为四种传输方法在相同16张源图、名义噪声种子2001上的均值。该集合用于展示，不代表完整开发集。

| 方法 | PSNR ↑ | LPIPS ↓ | DINOv2-L ↑ | CLIP ↑ | DISTS ↓ |
| --- | --- | --- | --- | --- | --- |
| SwinJSCC | 21.76 | 0.523 | 0.388 | 0.620 | 0.325 |
| HiFi-DiffCom | 21.40 | 0.460 | 0.470 | 0.741 | 0.267 |
| P | 19.49 | 0.189 | 0.628 | 0.760 | 0.204 |
| M1-selected (m8, K=0, 16QAM) | 18.61 | 0.184 | 0.763 | 0.842 | 0.159 |

**M1-selected本工作点为m8、K=0、whole、16QAM；不涉及部分尺度token传输，不能据此证明熵序有效。**

总N相同。连续链Swin/HiFi/P采用逐帧E=2048约束；M1保留原16QAM实际能量：均值2057.700000，范围[1980.800000, 2124.800000]。这不是严格同逐帧能量比较。

Swin/HiFi共享实际接收波形；P/M1的噪声命名空间不同。配对只在相同源图上进行。Swin为指定80k模型，训练预算截断、未证明收敛。

[13项指标、95%区间及配对差值](COMPARISON_REPORT.md) · [主表CSV](main_comparison.csv) · [逐帧指标](metrics_per_frame.csv)

[16图总览](figures/comparison_N1024_SNR13_all16.png) · [总览PDF](figures/comparison_N1024_SNR13_all16.pdf) · [四页近景PDF](figures/comparison_N1024_SNR13_closeups.pdf)
