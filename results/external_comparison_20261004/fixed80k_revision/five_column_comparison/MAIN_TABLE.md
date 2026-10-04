# 四种传输方法：固定16图对比

各已测行均为相同16张固定样例、名义噪声种子2001的均值。Original仅作图中视觉参照，不列入传输方法排名。该小样本用于展示，不代表完整开发集。

| N | SNR/dB | 方法 | PSNR ↑ | LPIPS ↓ | DINOv2-L ↑ | CLIP ↑ | DISTS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 7 | SwinJSCC | 20.90 | 0.561 | 0.271 | 0.586 | 0.339 |
| 1024 | 7 | HiFi-DiffCom | 20.43 | 0.502 | 0.397 | 0.700 | 0.287 |
| 1024 | 7 | P | 18.94 | 0.213 | 0.537 | 0.712 | 0.214 |
| 1024 | 7 | M1: entropy order | 16.34 | 0.275 | 0.534 | 0.800 | 0.201 |
| 1024 | 13 | SwinJSCC | 21.76 | 0.523 | 0.388 | 0.620 | 0.325 |
| 1024 | 13 | HiFi-DiffCom | 21.40 | 0.460 | 0.470 | 0.741 | 0.267 |
| 1024 | 13 | P | 19.49 | 0.189 | 0.628 | 0.760 | 0.204 |
| 1024 | 13 | M1: entropy order | 16.62 | 0.261 | 0.584 | 0.805 | 0.193 |
| 2048 | 7 | SwinJSCC | 22.32 | 0.448 | 0.521 | 0.682 | 0.296 |
| 2048 | 7 | HiFi-DiffCom | 21.78 | 0.397 | 0.601 | 0.783 | 0.246 |
| 2048 | 7 | P | 20.68 | 0.145 | 0.798 | 0.849 | 0.154 |
| 2048 | 7 | M1: entropy order | 未测 | 未测 | 未测 | 未测 | 未测 |
| 2048 | 13 | SwinJSCC | 23.21 | 0.416 | 0.616 | 0.731 | 0.280 |
| 2048 | 13 | HiFi-DiffCom | 22.86 | 0.345 | 0.713 | 0.830 | 0.219 |
| 2048 | 13 | P | 21.36 | 0.124 | 0.832 | 0.880 | 0.143 |
| 2048 | 13 | M1: entropy order | 未测 | 未测 | 未测 | 未测 | 未测 |

↑越大越好，↓越小越好。N2048的M1未正式运行，不能据空缺推断优劣。各链噪声命名空间不同，并非逐元素相同的信道噪声。Swin使用指定80k模型，训练预算截断，未证明收敛。

[完整13指标、95%区间及配对差值](COMPARISON_REPORT.md) · [本表CSV](main_comparison.csv) · [逐帧指标](metrics_per_frame.csv)

- N1024 / 7 dB：[16图总览](figures/comparison_N1024_SNR7_all16.png) · [总览PDF](figures/comparison_N1024_SNR7_all16.pdf) · [近景PDF](figures/comparison_N1024_SNR7_closeups.pdf)
- N1024 / 13 dB：[16图总览](figures/comparison_N1024_SNR13_all16.png) · [总览PDF](figures/comparison_N1024_SNR13_all16.pdf) · [近景PDF](figures/comparison_N1024_SNR13_closeups.pdf)
- N2048 / 7 dB：[16图总览](figures/comparison_N2048_SNR7_all16.png) · [总览PDF](figures/comparison_N2048_SNR7_all16.pdf) · [近景PDF](figures/comparison_N2048_SNR7_closeups.pdf)
- N2048 / 13 dB：[16图总览](figures/comparison_N2048_SNR13_all16.png) · [总览PDF](figures/comparison_N2048_SNR13_all16.pdf) · [近景PDF](figures/comparison_N2048_SNR13_closeups.pdf)
