# VAR receiver-prior 跨数据集正式结果（2026-09-04）

分析 ID：`ANALYSIS-VAR-RECEIVER-PRIOR-CROSS-DATASET-002`

正式输出：
`outputs/analysis/ANALYSIS-VAR-RECEIVER-PRIOR-CROSS-DATASET-002/`

汇总图表：
`outputs/analysis/ANALYSIS-VAR-RECEIVER-PRIOR-CROSS-DATASET-002/combined_report/`

## 结论

本轮在 ImageNetV2、CLIC2020 test 和 Kodak 上完整保留了
`prefix-only / prefix+VAR completion / full-VQ / fixed m7,m8,m9 / adaptive m /
class-only`，并统一计算 PSNR、SSIM、LPIPS-Alex、DINO cosine、DINO instance
retrieval Recall@1、分类一致率、BER、BLER、总 TER、各尺度 TER、首错尺度和
BLER 分层质量。总计 `68,656` 行逐图结果。

结果支持一个边界明确的主张：

> 在传输真实多尺度 prefix 并提供共享类别条件的前提下，冻结 VAR receiver prior
> 能补出对感知质量、语义和实例可识别性有价值的后续尺度；固定 3,060 complex-use
> 带宽下，根据 SNR 联合改变真实尺度数和 FEC 强度，可以让该价值穿过 AWGN 链路。

最强证据来自未参与当前策略形成的 2,000 张 ImageNetV2 query 与完整 10,000 张
gallery：

- m7 只使用 full-VQ `22.8%` 的 raw token payload，completion 相对 prefix-only
  改善 `+0.869 dB PSNR / -0.1804 LPIPS / +0.5216 DINO`，full-gallery R@1
  从 `23.35%` 提高到 `98.85%`；
- m8 使用 `37.5%` payload，改善
  `+1.148 dB / -0.1308 / +0.3426`，R@1 从 `72.20%` 到 `99.95%`；
- m9 使用 `62.4%` payload，改善
  `+0.980 dB / -0.0677 / +0.1307`，R@1 从 `97.90%` 到 `99.95%`。

但这不等于恢复了所有未传细节。ImageNetV2 completion m7/m8/m9 相对 full-VQ
仍低约 `4.93/3.29/1.66 dB` PSNR；m9 在三个数据集仍低约
`1.50--1.71 dB`，LPIPS 高约 `0.030`。因此可声称 receiver prior 替代了一部分
source payload，并保留实例身份；不能声称它恢复了未发送的精确纹理。

## 数据总体与角色

| 数据集 | query | retrieval gallery | 逐图结果行 | 类别条件 | 角色 |
|---|---:|---:|---:|---|---|
| ImageNetV2 matched-frequency | 2,000 | 10,000 | 56,000 | 真实 ImageNet-1K 类别 | 主要新测试总体 |
| CLIC2020 test | 428 | 428 | 11,984 | 冻结 ResNet-50 pseudo class | OOD 补充 |
| Kodak | 24 | 24 | 672 | 冻结 ResNet-50 pseudo class | 小样本 OOD 补充 |

ImageNetV2 query 是四个既有项目已用 8,000 张的固定 complement，每类 2 张；gallery
为完整 10,000 张、每类 10 张。CLIC/Kodak 使用冻结的完整 benchmark population。

本轮按预注册选择 CLIC 而不是 COCO：COCO 已被现有 DeepJSCC 和多个模块用于训练、
选模与反复诊断，CLIC 能提供更干净的跨域证据。用户要求的“CLIC 或 COCO”由 CLIC
满足。

所有数据统一按 VAR 协议转为 RGB，短边 bicubic resize 到 256 后做 256×256 center
crop。CLIC/Kodak 结果因此只能称为 `256×256 center-crop protocol`，不是原生整图
压缩 benchmark。

## 冻结方法与通信账本

无信道 arms：

1. `full_vq`：十尺度全部为真实 token；
2. `prefix_only_m7/m8/m9`：只累积真实前缀的 latent residual contribution，后续尺度
   contribution 严格置零；
3. `var_completion_m7/m8/m9`：使用同一个真实 prefix，由冻结 VAR closed-loop greedy
   argmax 补全后续尺度；
4. `class_only`：不发送真实 prefix，只由同一类别条件生成全部尺度。

raw VQ payload 为：

| cutoff | tokens | information bits | full-VQ payload 比例 | 加 10-bit class 后 |
|---:|---:|---:|---:|---:|
| m7 | 155 | 1,860 | 22.794% | 1,870 bit |
| m8 | 255 | 3,060 | 37.500% | 3,070 bit |
| m9 | 424 | 5,088 | 62.353% | 5,098 bit |
| full | 680 | 8,160 | 100% | 不需要 class condition |

无线 arms 在每图都固定：

```text
6,120 coded-bit slots = 3,060 QPSK complex channel uses
I,Q in {-1,+1}, Es=2
SNR = E[|X|^2] / E[|W|^2]
Var(W_I) = Var(W_Q) = 1/gamma
```

K=7、`(171,133)` octal rate-1/2 mother convolutional code 使用 known-zero start、
open final、无 tail、无 CRC、soft Viterbi。m7 在 mother code 后均匀 repetition 并
soft combine；m8 原生占满 6,120 slots；m9 deterministic uniform puncturing，punctured
bit 使用 zero evidence。同一个 `(dataset,image,SNR)` 的 m7/m8/m9 共用完全相同的
6,120-D standard-normal realization。

冻结 adaptive policy 未在新数据上重选：

```text
1 dB -> m7
4 dB -> m8
7/13/19 dB -> m9
```

adaptive 行是相应 fixed-digital 行与 embedding 的精确复制，不产生第二次信道噪声或
VAR 推理。

## ImageNetV2 无信道主结果

| 路径 | m | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ | full R@1 ↑ | same-class R@1 ↑ | Alex consistency ↑ | Mobile consistency ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-VQ | 10 | 22.711 | 0.6518 | 0.1047 | 0.9305 | 100.00% | 100.00% | 72.50% | 75.30% |
| Class-only | 0 | 8.241 | 0.1884 | 0.8188 | 0.2755 | 5.15% | 10.35% | 30.00% | 35.90% |
| Prefix-only | 7 | 16.913 | 0.4265 | 0.4155 | 0.2886 | 23.35% | 69.05% | 15.90% | 16.20% |
| VAR completion | 7 | 17.782 | 0.4559 | 0.2350 | 0.8102 | 98.85% | 99.45% | 55.90% | 63.70% |
| Prefix-only | 8 | 18.269 | 0.4770 | 0.3073 | 0.5253 | 72.20% | 93.15% | 31.00% | 34.25% |
| VAR completion | 8 | 19.417 | 0.5186 | 0.1766 | 0.8679 | 99.95% | 99.95% | 62.25% | 67.90% |
| Prefix-only | 9 | 20.074 | 0.5444 | 0.2022 | 0.7755 | 97.90% | 99.55% | 49.50% | 56.65% |
| VAR completion | 9 | 21.054 | 0.5804 | 0.1345 | 0.9062 | 99.95% | 99.95% | 67.25% | 71.75% |

completion 的 DINO 胜图数为 `2000/2000`、`1995/2000`、`1978/2000`；LPIPS
胜图数三个 m 都是 `1999/2000`。m7 的 full-R@1 paired 增量为 `+75.50 pp`
`[+73.60,+77.35] pp`，说明 completion 不只是提高平均 embedding cosine，也显著提高
实例可找回性。

## 跨数据集 completion 边际贡献

差值均为同图 `VAR completion - prefix-only`：

| 数据集 | m | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | ΔDINO ↑ | full-R@1 增量 |
|---|---:|---:|---:|---:|---:|---:|
| ImageNetV2 | 7 | +0.869 | +0.0295 | -0.1804 | +0.5216 | +75.50 pp |
| ImageNetV2 | 8 | +1.148 | +0.0416 | -0.1308 | +0.3426 | +27.75 pp |
| ImageNetV2 | 9 | +0.980 | +0.0359 | -0.0677 | +0.1307 | +2.05 pp |
| CLIC2020 test | 7 | +0.843 | +0.0312 | -0.1697 | +0.4168 | +52.34 pp |
| CLIC2020 test | 8 | +1.142 | +0.0439 | -0.1260 | +0.2793 | +12.85 pp |
| CLIC2020 test | 9 | +1.023 | +0.0381 | -0.0681 | +0.1140 | 0 pp，prefix 已饱和 |
| Kodak | 7 | +0.705 | +0.0183 | -0.1843 | +0.4792 | +41.67 pp |
| Kodak | 8 | +0.795 | +0.0329 | -0.1375 | +0.3103 | 0 pp，prefix 已饱和 |
| Kodak | 9 | +0.807 | +0.0342 | -0.0733 | +0.1178 | 0 pp，prefix 已饱和 |

三个数据集、三个 m 的 PSNR、LPIPS、DINO paired 95% CI 全部不跨零。LPIPS/DINO
逐图方向也极稳定：ImageNetV2 LPIPS 三个 m 均 `1999/2000` 胜，DINO 为
`2000/1995/1978` 胜；CLIC LPIPS 均 `428/428` 胜，DINO 为
`428/427/417` 胜；Kodak 两项三个 m 均 `24/24` 胜。

分类一致率总体同向改善，但 Kodak 只有 24 张，部分分类差值 CI 接触或跨零；因此跨域
最稳的主统计证据是 LPIPS、DINO 和 ImageNetV2/CLIC 的实例检索，而不是把每个小样本
分类差值都写成显著。

## Class-only：类别原型不等于实例保留

ImageNetV2 class-only 的 aligned DINO 为 `0.2755`，full-gallery R@1 为 `5.15%`，
same-class R@1 为 `10.35% [9.05%,11.70%]`。每类 gallery 恰好 10 张，因此同类随机
机会为 `10%`。相反，completion m7 的 full/same-class R@1 已达
`98.85%/99.45%`。

CLIC/Kodak 的 class-only same-condition R@1 也分别为 `31.60%/50.00%`，等于各自
冻结 condition group 的平均随机机会。类别条件可以生成“合理的同类图”，但不能指出
是哪一张源图；真实 prefix 才提供实例锚点。

普通 GT 分类准确率还存在“类别原型奖励”。ImageNetV2 class-only 直接拿到 GT class，
其 AlexNet/MobileNet GT top-1 为 `56.90%/57.80%`，甚至高于 full-VQ 的
`42.60%/53.05%`；但 clean-source-correct retention 只有 `66.48%/64.09%`，低于
full-VQ 的 `88.29%/89.78%`。因此普通 GT accuracy 不能单独证明 source semantics
保留，instance retrieval 与 clean-correct retention 更可信。

## 固定模式与 adaptive 无线结果

ImageNetV2 的关键 fixed-mode 结果如下：

| SNR | m | BLER | TER | PSNR ↑ | LPIPS ↓ | DINO ↑ | full R@1 ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 17.70% | 0.1826% | 17.756 | 0.2357 | 0.8096 | 99.00% |
| 1 | 8 | 100% | 10.5808% | 17.069 | 0.2569 | 0.7734 | 96.90% |
| 1 | 9 | 100% | 94.6165% | 9.988 | 0.6935 | 0.0437 | 0.05% |
| 4 | 7 | 0.65% | 0.0042% | 17.782 | 0.2350 | 0.8103 | 98.85% |
| 4 | 8 | 3.60% | 0.0165% | 19.415 | 0.1766 | 0.8680 | 99.95% |
| 4 | 9 | 100% | 41.6317% | 13.498 | 0.4730 | 0.3134 | 29.35% |
| 7 | 7 | 0% | 0% | 17.782 | 0.2350 | 0.8102 | 98.85% |
| 7 | 8 | 0.10% | 0.0004% | 19.417 | 0.1766 | 0.8679 | 99.95% |
| 7 | 9 | 36.55% | 0.2208% | 21.000 | 0.1358 | 0.9050 | 99.95% |
| 13/19 | 7 | 0% | 0% | 17.782 | 0.2350 | 0.8102 | 98.85% |
| 13/19 | 8 | 0% | 0% | 19.417 | 0.1766 | 0.8679 | 99.95% |
| 13/19 | 9 | 0% | 0% | 21.054 | 0.1345 | 0.9062 | 99.95% |

冻结 adaptive 策略除 CLIC 1 dB 的 SSIM 比 fixed-m envelope 低 `2.5928e-5`
外，在 ImageNetV2 和 CLIC 的 PSNR、SSIM、LPIPS、DINO 和 R@1 上都落在 fixed-m
均值 envelope；该唯一偏差数值上可忽略，但这里不把它写成严格并列最优。三数据集
adaptive 的可靠性为：

| 数据集 | 1 dB / m7 BLER | 4 dB / m8 BLER | 7 dB / m9 BLER | 13/19 dB |
|---|---:|---:|---:|---:|
| ImageNetV2 | 17.70% | 3.60% | 36.55% | 0%，严格等于 oracle |
| CLIC2020 test | 17.06% | 4.21% | 37.38% | 0%，严格等于 oracle |
| Kodak | 12.50% | 4.17% | 41.67% | 0%，严格等于 oracle |

7 dB m9 的 BLER 虽高，但 ImageNetV2 TER 只有 `0.2208%`；1,872 个错误 token 中
876 个在 scale 9、425 个在 scale 8，错误集中在后部尺度，所以 PSNR/LPIPS/DINO 仍
接近 oracle。m7@1 的 566 个错误中 358 个在 scale 7；m8@4 的 84 个错误中 67 个在
scale 8。CLIC/Kodak 方向一致。只看 BLER 会过度悲观，必须结合 BER、TER、首错尺度
和质量后果。

13/19 dB 的三个 m 在三个数据集都零误码。所有 BLER=0 digital reconstruction 与对应
no-channel completion 的像素最大差严格为 `0.0`，且 PSNR、SSIM、LPIPS、DINO、分类
字段、retrieval hit 和 FP32 query embedding 全部规范化为 bit-exact 相等。

## 对 claim 的严格解释

### 可以支持

- 同一真实 prefix 下，接收端 completion stack 对 LPIPS、DINO 和实例检索有稳定边际
  价值，且不是官方 VQ Decoder 单独造成；
- class-only 无法保留源实例，真实 prefix 是实例身份锚点；
- 用 `22.8%--62.4%` full-VQ raw payload 可以保留很高的 DINO 和 instance R@1；
- 固定 3,060 complex uses 下，坏信道用较小 m/强保护、好信道用较大 m/弱保护的机制
  在三个新数据集上成立；
- 高 SNR 无误码时数字链严格回到同 m oracle，低 SNR 损失来自 PHY/rate matching，
  不是 bit-token replay bug。

### 不能扩张

- `prefix-only -> completion` 严格隔离的是“冻结 VAR + shared class condition”的接收端
  补全栈，不是 class-free VAR 单项。class-only 排除了“类别本身足以恢复实例”，但本轮
  没有再运行无类别 VAR completion；
- 10-bit class side information 是 error-free separate diagnostic，未进入 6,120 coded
  slots。尤其 ImageNetV2 使用 GT class，当前不是包含 class/header/CRC 的严格端到端
  同总预算实现；
- full-VQ 是无信道重建上限，不是 3,060-use 等预算方法；
- R@1 在 m8/m9 已接近饱和，只能证明实例可识别性，不能证明像素级细节已恢复；
- CLIC/Kodak 使用 pseudo class 和 256 center crop，不能冒充真实标签分类或原生整图
  压缩结果；
- m7/m9 仍是临时 repetition/puncturing，不是优化的 rate-compatible LDPC/Polar；
  BLER/TER 是事后 oracle 诊断，当前接收端没有 CRC 错误检测；
- Kodak same-condition 每 arm 只有 4 个 eligible source，实例检索主证据应以
  ImageNetV2，其次 CLIC 为准。

## 独立复核

独立只读审计完成并通过：

- 2,000 ImageNetV2 query、10,000 gallery、56,000 行，严格每图 28 arms，无重复键；
- 20 个 query shards、100 个 gallery shards complete，merged CSV/embedding 与 shard
  bit-exact；
- 三数据集合计 `31,671` 条 BLER=0 digital/adaptive 行的图像指标、分类、retrieval 和
  384-D query embedding 全部与 oracle bit-exact；
- 13/19 dB 的 72 个 paired comparison 全部均值差 0、100% tie、0 win/loss；
- 独立重放 285 组固定种子的 10,000 次 source bootstrap，最大差 `3.6e-15`；
- 独立 CUDA IEEE-FP32 分块复算 `56,000 x 10,000` retrieval，hit、eligibility 和
  group size 全部一致；
- combined report 的 36 个输入 SHA 和 16 个输出 SHA 全部复算通过，7 张 PNG 可读。

存在两个不改变结论的数值边界：相同 crop 的 source/gallery DINO 因独立 GPU 前向，
原始向量最大差约 `3.33e-5`，但所有 retrieval hit 一致；另有一条 class-only
same-class margin 约 `3.38e-6`，最多影响 56,000 行中的一条，不影响 completion 结论。

## 产物

每个数据集正式目录包含：

- `per_image_metrics.csv`；
- `mean_metrics.csv`；
- `paired_differences_and_win_rates.csv`；
- `classification_summary.csv`；
- `reliability_by_snr_mode_scale.csv`；
- `first_error_scale_histogram.csv`；
- `quality_by_bler_stratum.csv`；
- `query_dino_embeddings_f32.npy` 与 `gallery_dino_embeddings_f32.npy`；
- `retrieval/retrieval_summary.csv` 与 `retrieval/retrieval_hits.csv`；
- `completion.json`、配置/脚本 snapshot、代表图和 provenance。

`combined_report/` 额外提供 7 个跨数据集 CSV、`fig1`--`fig7`、`REPORT.md`、
`summary.json` 和包含全部输入/输出 SHA 的 `artifact_manifest.json`。

汇总脚本：`scripts/report_var_receiver_prior_cross_dataset.py`

汇总脚本 SHA-256：
`6b52bcf2ca4e507aa7a0887a068ecd83415fb4aa0b3cb12ecc21d71c5e4ad2dc`

combined `REPORT.md` SHA-256：
`0ce7f706bc4f3de8d49482b6353df5a8993f290053c30dbca17e8598946c1da9`

combined `summary.json` SHA-256：
`9f52b9b63fa218c33a83e35f921dca17a8b226c007258c40ebdd392b2e5994fe`

combined `artifact_manifest.json` SHA-256：
`abb8307c56480542962833b797b0118ab8abda104a9fcfb85530c631a4e8dddb`

旧 `ANALYSIS-VAR-RECEIVER-PRIOR-CROSS-DATASET-001` 因像素相同的 BLER=0 行在不同
metric batch 中产生约 `1e-6` 浮点抖动，污染 exact win/tie/loss，已通过
`ABORTED_DO_NOT_USE.json` 明确作废并保留为失败证据；只有 `-002` 可用于科学结论。
