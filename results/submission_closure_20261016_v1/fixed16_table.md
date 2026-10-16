# Fixed16：N1024，13 dB，原生浮点输出

| 方法 | PSNR ↑ | DISTS ↓ | DreamSim ↓ | MS-SSIM ↑ |
| --- | --- | --- | --- | --- |
| NeST-Com (raw) | 18.678887 | 0.156836 | 0.115557 | 0.687997 |
| Latent JSCC (100k) | 19.844035 | 0.194109 | 0.191163 | 0.755072 |
| SwinJSCC (adapted,200k) | 22.072844 | 0.316594 | 0.354706 | 0.838368 |
| HiFi (Swin200k) | 21.588449 | 0.245843 | 0.218072 | 0.821280 |

全部原16来源进入统计；上表为均值，完整中位数、逐源值和直接同源差值另附。raw 的 DISTS / DreamSim 较低，但 PSNR / MS-SSIM 低于其余三方法；不据此宣布总体优于 HiFi。

数字 raw 使用原6201信道种子，P/Swin使用2001，HiFi后验23。只有 Swin 与 HiFi 共享实际接收观测。source pairing 不等于共享信道噪声。这里没有13 dB熵码实验，也不能并入common500。

新增48个raw三指标分数和64个PSNR；原144个分数以相同像素SHA核验后逐字文本复用。RGB256原生float32；PSNR采用float64 MSE。DISTS原256RGB，DreamSim官方ensemble及原224 bicubic antialias浮点预处理，MS-SSIM原256RGB、11窗口、sigma1.5、data_range1。精确版本/权重/参数见metric_registration.json和fixed16/runtime_identity.json。没有新bootstrap，表内不提供总体显著性。

可用英文图注：Descriptive comparison on all 16 predeclared development sources at N=1024 and 13 dB, using archived native float32 reconstructions. Raw uses channel seed6201, continuous methods seed2001, and HiFi posterior seed23. Only SwinJSCC and HiFi share the actual received observation. We report all-source means, medians and direct source-paired differences without a new bootstrap. No inference is made about common500 or population-level superiority.
