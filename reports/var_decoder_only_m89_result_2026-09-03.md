# VAR Decoder-only full / m=8 / m=9 微调结果

日期：2026-09-03

实验：`CACHE-VAR-DECODER-FT-IMAGENET20K-CAL1K-001` / `EXP-VAR-DECODER-ONLY-M89-001`

分析：`ANALYSIS-VAR-DECODER-ONLY-IMAGENET100-001`

## 结论

Decoder-only 微调明确提高了 full-VQ、m=8 和 m=9 的像素失真上限，但没有保住原有感知质量：

- full/m8/m9 的平均 PSNR 分别提高 `+0.9931/+1.2267/+1.1713 dB`，SSIM 分别提高 `+0.03406/+0.04447/+0.04031`；
- LPIPS 分别恶化 `+0.01529/+0.02286/+0.01997`，DINO cosine 分别下降 `-0.00836/-0.01197/-0.00654`；
- PSNR 在 full 为 `99/100` 张更高，在 m8/m9 均为 `100/100`；SSIM 三个条件均为 `100/100`；
- LPIPS 只有 `5/100、7/100、6/100` 张改善，DINO 只有 `23/100、31/100、32/100` 张改善；
- 所有上述主要 paired 95% CI 都不跨 0。

因此答案是：**能提高 full-VQ、m=8、m=9 的 PSNR/SSIM 重建上限，但会显著损害 LPIPS 和 DINO，不能称为无损提升。** 代表图也显示微调输出整体更平滑，局部纹理和小目标边界被削弱，这与指标分裂一致。

## 冻结边界与训练

- 没有重训 VQ tokenizer、Encoder、codebook、多尺度量化器、任何 `phi_k` 或 VAR。
- 只训练 `vae.decoder.*`，精确参数量 `64,653,923`；`post_quant_conv` 的 `9,248` 个参数冻结。
- 非 Decoder state 的训练前后 SHA-256 都是 `6075d38f0ffa4910df3432d7ec8c75c48fd9059013ff0558061f26a5800e0d57`，冻结审计通过。
- 训练数据为 ImageNet-train 预冻结的 20,000 张；独立 ImageNet calibration-1,000 只用于 checkpoint 选择；最终测试为原冻结 ImageNet-100。
- 每张训练图同时使用 `full`、`m8`、`m9` 三种 latent，条件等权；其中 m8/m9 的后续尺度由冻结 VAR 闭环 argmax 补全。
- 训练 3 epoch、15,000 optimizer steps，耗时 `5,657.2 s`；latent cache 耗时约 `744.6 s`。
- loss 为 `MSE + 0.02 LPIPS-Alex + 0.01 (1-DINO cosine)`。DINO 反传条件按 full→m8→m9 循环，三条件长期等频。

预注册 selection gate 要求每个条件 LPIPS 恶化不超过 `0.003`、DINO 下降不超过 `0.002`。三个 epoch 都没有通过；按预注册 fallback 选择三条件平均 PSNR 最高的 epoch 3，不能包装成 gate 成功。

最终 checkpoint：

`outputs/train/EXP-VAR-DECODER-ONLY-M89-001/checkpoints/best.pt`

SHA-256：`7b79b4b96f0b13ba5f77cf3cd4038ac1f0fe216e2b516e6a3a3ba57fb20d2802`

## ImageNet-100 主结果

同一 latent、同一图片、同一指标实现下的均值：

| 条件 | Decoder | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO cosine ↑ |
|---|---|---:|---:|---:|---:|
| full | 官方 | 22.8357 | 0.63801 | 0.10569 | 0.93836 |
| full | 微调 | 23.8288 | 0.67207 | 0.12099 | 0.93000 |
| m8 | 官方 | 19.6805 | 0.51191 | 0.17760 | 0.89060 |
| m8 | 微调 | 20.9072 | 0.55637 | 0.20046 | 0.87863 |
| m9 | 官方 | 21.2316 | 0.56976 | 0.13589 | 0.92317 |
| m9 | 微调 | 22.4029 | 0.61007 | 0.15586 | 0.91663 |

逐图 paired 差值定义为 `微调 − 官方`；LPIPS 的正差表示恶化：

| 条件 | 指标 | 平均差值 | 95% CI | 微调胜/平/负 |
|---|---|---:|---:|---:|
| full | PSNR | +0.99310 dB | [+0.92614, +1.06069] | 99/0/1 |
| full | SSIM | +0.034057 | [+0.031254, +0.036867] | 100/0/0 |
| full | LPIPS | +0.015294 | [+0.012387, +0.018641] | 5/0/95 |
| full | DINO | -0.008359 | [-0.011507, -0.005387] | 23/0/77 |
| m8 | PSNR | +1.22673 dB | [+1.17097, +1.28271] | 100/0/0 |
| m8 | SSIM | +0.044466 | [+0.041301, +0.047596] | 100/0/0 |
| m8 | LPIPS | +0.022860 | [+0.018602, +0.027432] | 7/0/93 |
| m8 | DINO | -0.011971 | [-0.017073, -0.007169] | 31/0/69 |
| m9 | PSNR | +1.17128 dB | [+1.11667, +1.22477] | 100/0/0 |
| m9 | SSIM | +0.040306 | [+0.037173, +0.043460] | 100/0/0 |
| m9 | LPIPS | +0.019969 | [+0.016434, +0.023680] | 6/0/94 |
| m9 | DINO | -0.006537 | [-0.009637, -0.003414] | 32/0/68 |

## 已有 SGD-JSCC 复现审计

工程内已经存在可运行的 SGD-JSCC 发布模型复现，已直接复用，没有重复训练：

- 源码：`third_party/SGDJSCC`，commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`；
- 原生资产配置：`configs/external_sgdjscc_native_smoke.yaml`；
- 本次评测脚本：`scripts/evaluate_sgdjscc_imagenet100.py`；
- 发布 checkpoint：`JSCC_model.pth`、`muge-epoch-19-checkpoint.pth`、`diffusion_backbone.pth`、`diffusion_controlnet.pth`，四个 SHA 均与冻结配置一致；
- 图像分辨率为 `256×256`，按作者路径拆成四个 `128×128` patch；
- main image latent 为 `16,384 real`，active edge 为 `3,328 real`，合计 `19,712 real = 9,856 complex channel uses`；
- 项目口径 CBR 为 `9856 / (3×256×256) = 0.0501302`；
- 本次测试 SNR 为 `[1,4,7,13,19] dB`；
- 发布 JSCC checkpoint 的训练条件为 `ImageNet fixed 10 dB AWGN`，diffusion/ControlNet 训练数据约为 `14M text-image pairs`；
- 每张 256 图由发送端 BLIP2 生成四条 patch caption，评测中 caption 完美送达且通信开销记为 0。active edge 坐标已计费，但 caption 没有计费。

所以 SGD-JSCC 只能标为 **released paper-protocol upper bound**。它比 m8/DeepJSCC-3060 使用约 `3.22×` 的 complex uses，并额外获得免费发送端语义文本，不能进入 exact-rate 公平排名。

本次正式 SGD-JSCC 评测完成 100 张 × 5 SNR，共 500 行，耗时 `962.2 s`，峰值 GPU memory `9,928.8 MiB`。

## 外部定位结果

DeepJSCC-3060 使用 `3,060 complex channel uses`，与 m=8 名义信息预算对应；物理信道按标准复 AWGN 口径运行。其 checkpoint 训练时仍使用旧 repository half-variance 约定，因此测试虽已修正，训练分布错配必须保留披露。

SGD-JSCC 使用 `9,856 complex uses` 加免费 sender caption。以下均值用于定位，不代表四方法等码率：

| 方法 | SNR dB | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ |
|---|---:|---:|---:|---:|---:|
| DeepJSCC-3060 | 1 | 22.8294 | 0.61537 | 0.42286 | 0.31059 |
| DeepJSCC-3060 | 4 | 24.4311 | 0.68039 | 0.34782 | 0.46995 |
| DeepJSCC-3060 | 7 | 25.5909 | 0.72118 | 0.30734 | 0.57976 |
| DeepJSCC-3060 | 13 | 26.8745 | 0.75977 | 0.27595 | 0.69174 |
| DeepJSCC-3060 | 19 | 27.3497 | 0.77178 | 0.26649 | 0.72449 |
| SGD-JSCC paper protocol | 1 | 24.8025 | 0.75334 | 0.11336 | 0.91192 |
| SGD-JSCC paper protocol | 4 | 25.4469 | 0.77325 | 0.10736 | 0.92421 |
| SGD-JSCC paper protocol | 7 | 26.3055 | 0.79546 | 0.08624 | 0.93782 |
| SGD-JSCC paper protocol | 13 | 26.8466 | 0.80665 | 0.08166 | 0.94370 |
| SGD-JSCC paper protocol | 19 | 27.3312 | 0.81490 | 0.07178 | 0.94854 |

作为无物理信道噪声的 source-reconstruction 参考，微调 m8 为 `20.9072 dB / 0.55637 SSIM / 0.20046 LPIPS / 0.87863 DINO`。它和随 SNR 变化的 JSCC 通信系统不是同一种实验条件；联合图只用于展示位置。

## 输出与核验

VAR 主评测：`outputs/analysis/ANALYSIS-VAR-DECODER-ONLY-IMAGENET100-001/`

- `per_image_metrics.csv`：官方/微调 × full/m8/m9 的 600 行逐图指标；
- `mean_metrics.csv`：六个条件的均值与 bootstrap CI；
- `paired_differences_and_win_rates.csv`：12 个严格 paired 主比较；
- `paired_decoder_differences.png`、`representative_grid.png`：差值与代表样例。

DeepJSCC：`outputs/analysis/ANALYSIS-VAR-DECODER-ONLY-IMAGENET100-001/deepjscc_3060/`

SGD-JSCC：`outputs/external_baselines/ANALYSIS-SGDJSCC-IMAGENET100-AUTHOR-PROTOCOL-001/`

联合定位：`outputs/analysis/ANALYSIS-VAR-DECODER-ONLY-IMAGENET100-001/combined_comparison/`

- `all_per_image_metrics.csv`：总计 1,600 行；
- `all_mean_metrics.csv`：全部均值；
- `paired_differences_and_win_rates.csv`：严格 Decoder 对照和外部描述性 paired 差值/win rate，共 64 行；
- `comparison_scope.json`：明确禁止“all four methods are exact-rate equivalents”；
- `all_methods_positioning.png`：四指标联合定位图。

独立核验结果：三个来源均为相同 100 个唯一 image ID；VAR/DeepJSCC/SGD-JSCC 分别为 `600/500/500` 行，无重复 arm key；全部均值从逐图 CSV 重算误差为 0；联合表行数严格为 `1,600`。
