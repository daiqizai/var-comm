# Fixed16 final-link supplementary quality

| Method | DISTS ↓ | DreamSim ↓ | MS-SSIM ↑ |
|---|---:|---:|---:|
| HiFiSwin200000 | 0.245843 | 0.218072 | 0.821280 |
| Swin200000 | 0.316594 | 0.354706 | 0.838368 |
| P100000 | 0.194109 | 0.191163 | 0.755072 |

All16 HiFi outputs and their saved final Swin200k and P100k counterparts were scored at N1024/13dB, channel seed2001 and frozen HiFi posterior seed23. Native float32 arrays were used, with original file and pixel hashes verified. This is a single-noise, originally selected development display set; its descriptive means and directly source-paired differences are not common500 results or population confidence intervals. No image generation, receiver/channel replay, training, checkpoint/policy selection, bootstrap or main-figure change occurred. Every outcome remains. DISTS and DreamSim are lower-is-better; MS-SSIM is higher-is-better. Full-precision frame, distribution and signed paired tables are supplied. DreamSim uses the fixed official ensemble and square224 bicubic antialiased tensor preprocessing, preserving native floats without PIL re-quantization; official per-backbone normalization is unchanged. DISTS/MS-SSIM consume original256 RGB with frozen settings in registration.json. Metrics share learned backbones and are not independent evidence.

The common500 six-method addition is WAITING_FOR_ORIGINAL_P100k_Swin200k_RECONSTRUCTIONS by explicit user choice; it is not scored or merged here. Old quality scores and main figures are unchanged.
