# ARPC decoder sensitivity — completed
The original frozen sampling result is retained. This matched100 analysis changes only generation of untransmitted tokens; it does not replace common500 curves or select a new main checkpoint/policy.
## Fixed scope and decode semantics
N=1024; SNR 4, 10 and 19 dB; the original three noise realizations 6201/6202/6203; identical paid caption, MCS, actual received prefixes, weights and source identity. Original sampling uses τ=0.5, CFG=3, top-k=900, top-p=0.97 and the frozen seed 0. The low-temperature value τ=0.25 was fixed before inference. Temperature is passed as a seven-scale vector: 0.5 on received scales, 0.25 only on untransmitted scales. Greedy is an explicit argmax at untransmitted scales; it is not an approximation by an arbitrarily low temperature. Native received-scale sampling/overwrite and state advancement are retained. No TX truth, hidden CDF or unsent token is supplied to RX.
## Actual work and reuse
The 900 original events contain 296 distinct received prefixes. The original setting reuses 900 logical results and 296 reconstructions. Two new settings produced 592 unique reconstructions, representing 1800 logical setting-frames with 1208 exact reconstruction reuses. All 592 render calls and all four new score sets completed. There were zero new source encodes, captions, channel events, channel decodes, tokenizer calls, training updates or bootstrap index draws. One preparation syntax error occurred before registration/model calls; it remains recorded. No successful scientific call was replayed.
RX/CLIP ran on the registered RTX PRO 6000 GPU1 with Python3.12.3/Torch2.10.0+cu130. The native ARPC model/text precision was retained. PSNR, LPIPS-Alex and DreamSim used the frozen H800 metric environment (Python3.10/Torch2.11.0+cu128), original float preprocessing and FP32 metric inference with TF32 disabled. CLIP-I is OpenAI ViT-L/14, frozen 224-pixel float preprocessing and FP32 image–image cosine; 100 exact source embeddings were reused. DreamSim ensemble v0.2.0 LoRA uses the original 224-pixel bicubic/antialiased float preprocessing. Scores come from native reconstruction arrays, not presentation PNGs.
## Statistics
Average the three original noises within each source and condition. Each setting/SNR has 100 sources and 300 logical outcomes. Reuse the original 10000×100 bank, seed 2026100701, SHA256 `5e11cd2ee931a2de9f22f8e42034a71ef610eb83a48bac9e53224e666f5a447d`. Form same-source differences first, then resample by the frozen source order. Intervals below are pointwise 95% intervals, with no multiplicity correction. PSNR and CLIP-I favor positive changes; LPIPS and DreamSim favor negative changes. All prespecified settings, including unfavorable and crossing-zero results, are retained.
## Mean scores
| Setting | SNR | PSNR | LPIPS-Alex | DreamSim | CLIP-I |
|---|---:|---:|---:|---:|---:|
| original | 4 | 13.655857 | 0.519334 | 0.503266 | 0.636886 |
| original | 10 | 16.126803 | 0.350717 | 0.316453 | 0.699748 |
| original | 19 | 17.819087 | 0.256681 | 0.201582 | 0.782594 |
| greedy | 4 | 13.593929 | 0.532229 | 0.516108 | 0.643507 |
| greedy | 10 | 16.189665 | 0.349466 | 0.310989 | 0.707433 |
| greedy | 19 | 17.969469 | 0.249665 | 0.187271 | 0.796468 |
| low_tau025 | 4 | 13.602286 | 0.525956 | 0.518332 | 0.641852 |
| low_tau025 | 10 | 16.195242 | 0.347061 | 0.311045 | 0.704242 |
| low_tau025 | 19 | 17.940805 | 0.251359 | 0.191997 | 0.791495 |

## All direct paired differences
Δ is the first setting minus the second setting. Full-precision values remain in CSV.
| Comparison | SNR | Metric | Mean | 95% CI |
|---|---:|---|---:|---|
| greedy_minus_original | 4 | psnr_db | -0.061928 | [-0.149654, +0.026323] |
| greedy_minus_original | 4 | lpips_alex | +0.012895 | [+0.008383, +0.017656] |
| greedy_minus_original | 4 | dreamsim | +0.012842 | [+0.002939, +0.022723] |
| greedy_minus_original | 4 | clip_image_cosine | +0.006621 | [-0.003135, +0.016447] |
| greedy_minus_original | 10 | psnr_db | +0.062862 | [-0.002841, +0.129304] |
| greedy_minus_original | 10 | lpips_alex | -0.001251 | [-0.004263, +0.001717] |
| greedy_minus_original | 10 | dreamsim | -0.005464 | [-0.012742, +0.001576] |
| greedy_minus_original | 10 | clip_image_cosine | +0.007685 | [-0.002395, +0.017435] |
| greedy_minus_original | 19 | psnr_db | +0.150381 | [+0.086670, +0.207008] |
| greedy_minus_original | 19 | lpips_alex | -0.007017 | [-0.009968, -0.004281] |
| greedy_minus_original | 19 | dreamsim | -0.014311 | [-0.020556, -0.008422] |
| greedy_minus_original | 19 | clip_image_cosine | +0.013875 | [+0.005955, +0.021965] |
| low_tau025_minus_original | 4 | psnr_db | -0.053572 | [-0.112489, +0.004726] |
| low_tau025_minus_original | 4 | lpips_alex | +0.006623 | [+0.003420, +0.009824] |
| low_tau025_minus_original | 4 | dreamsim | +0.015066 | [+0.007373, +0.022800] |
| low_tau025_minus_original | 4 | clip_image_cosine | +0.004966 | [-0.003672, +0.013382] |
| low_tau025_minus_original | 10 | psnr_db | +0.068439 | [+0.019912, +0.118137] |
| low_tau025_minus_original | 10 | lpips_alex | -0.003655 | [-0.005949, -0.001375] |
| low_tau025_minus_original | 10 | dreamsim | -0.005407 | [-0.011454, +0.000531] |
| low_tau025_minus_original | 10 | clip_image_cosine | +0.004494 | [-0.003201, +0.012429] |
| low_tau025_minus_original | 19 | psnr_db | +0.121718 | [+0.083386, +0.156192] |
| low_tau025_minus_original | 19 | lpips_alex | -0.005322 | [-0.007038, -0.003679] |
| low_tau025_minus_original | 19 | dreamsim | -0.009584 | [-0.013827, -0.005655] |
| low_tau025_minus_original | 19 | clip_image_cosine | +0.008902 | [+0.002280, +0.015588] |
| greedy_minus_low_tau025 | 4 | psnr_db | -0.008357 | [-0.073954, +0.056601] |
| greedy_minus_low_tau025 | 4 | lpips_alex | +0.006273 | [+0.003001, +0.009636] |
| greedy_minus_low_tau025 | 4 | dreamsim | -0.002224 | [-0.012010, +0.007263] |
| greedy_minus_low_tau025 | 4 | clip_image_cosine | +0.001655 | [-0.006421, +0.009610] |
| greedy_minus_low_tau025 | 10 | psnr_db | -0.005577 | [-0.057040, +0.044837] |
| greedy_minus_low_tau025 | 10 | lpips_alex | +0.002404 | [-0.000216, +0.004965] |
| greedy_minus_low_tau025 | 10 | dreamsim | -0.000056 | [-0.006030, +0.006089] |
| greedy_minus_low_tau025 | 10 | clip_image_cosine | +0.003191 | [-0.005700, +0.011957] |
| greedy_minus_low_tau025 | 19 | psnr_db | +0.028663 | [-0.007399, +0.061920] |
| greedy_minus_low_tau025 | 19 | lpips_alex | -0.001695 | [-0.003767, +0.000347] |
| greedy_minus_low_tau025 | 19 | dreamsim | -0.004726 | [-0.008661, -0.000771] |
| greedy_minus_low_tau025 | 19 | clip_image_cosine | +0.004973 | [-0.001183, +0.011326] |

## Interpretation for III.B
At 19 dB, both greedy and τ=0.25 improve all four metrics relative to the original setting, with favorable paired intervals excluding zero. At 4 dB, both alternatives worsen LPIPS and DreamSim; PSNR and CLIP-I intervals cross zero. At 10 dB, low temperature improves PSNR and LPIPS, while its DreamSim and CLIP-I intervals cross zero; all four greedy intervals cross zero. Thus ARPC performance depends on generation decoding and operating point. Do not characterize the original sampler as a universal optimum, and do not silently substitute a subset-selected decoder into the common500 comparison.
## Evidence and status
`registration.json` freezes received inputs, settings and RX code; `metric_registration.json` freezes scoring dependencies; `execution_receipts/` preserves actual launches, child waits and completion metadata. The raw closed run is under `liberai-tmp/liulu/liulu-var-comm/arpc_decode_sensitivity_20261021_v1`. RX, three-metric scoring, CLIP scoring and statistics all returned zero; verified science/controller processes are gone and no scientific failure is present. Original arrays, ledgers and complete logs remain on PFS, outside Git. N2048 CLIP was not started.
