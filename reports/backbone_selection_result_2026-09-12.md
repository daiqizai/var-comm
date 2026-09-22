# 运行产物位置 / Artifact location

`/home/liulu/projects/VAR_COMM/outputs/ei-liulu-xqvar-eval-20260912-v1/analysis`

下列CSV与图片文件名均相对于上述分析目录。

# Bounded no-training backbone evaluation

**Scope:** the original 100-image development population; noiseless source indices, not a channel experiment. Ground-truth ImageNet class is pre-shared for completion (optional 10-bit label shown separately). The XQ and original VAR scale numbers denote different token counts and are never rate-equated.

## Primary paired-image means

| Model | Natural point | Raw bits | PSNR | SSIM | LPIPS-Alex | DINO cosine | Encode ms | Decode/completion ms | Peak codec allocated MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old-fidelity | full | 8160 | 23.8304 | 0.67212 | 0.12089 | 0.92988 | 7.64 | 11.21 | 591.8 |
| old-official | full | 8160 | 22.8359 | 0.63820 | 0.10575 | 0.93768 | 7.56 | 11.15 | 1777.5 |
| old-official | p8_argmax | 3060 | 19.6916 | 0.51246 | 0.17714 | 0.88973 | 7.56 | 56.04 | 1781.1 |
| old-official | p8_direct | 3060 | 18.5004 | 0.47488 | 0.30558 | 0.54887 | 7.56 | 11.22 | 1777.5 |
| old-official | p9_argmax | 5088 | 21.2437 | 0.56982 | 0.13590 | 0.92386 | 7.56 | 55.92 | 1781.1 |
| old-official | p9_direct | 5088 | 20.2680 | 0.53711 | 0.20239 | 0.79286 | 7.56 | 11.30 | 1777.5 |
| wetok | full | 8192 | 23.7393 | 0.67351 | 0.08681 | 0.95440 | 28.46 | 34.62 | 2172.4 |
| xq | full | 6864 | 22.1412 | 0.59123 | 0.13194 | 0.93010 | 6.38 | 5.88 | 2465.4 |
| xq | p8_argmax | 2424 | 19.7391 | 0.49054 | 0.20490 | 0.87364 | 6.38 | 50.13 | 2520.6 |
| xq | p8_direct | 2424 | 20.0174 | 0.48854 | 0.21358 | 0.85683 | 6.38 | 5.45 | 2465.4 |
| xq | p9_argmax | 3960 | 20.9408 | 0.53678 | 0.16594 | 0.90525 | 6.38 | 50.10 | 2520.6 |
| xq | p9_direct | 3960 | 20.8616 | 0.52842 | 0.17681 | 0.89036 | 6.38 | 5.68 | 2465.4 |

## Replay of the known original-VAR reference

These are new minus archived per-image differences, not a model improvement. Values may shift slightly under batch-one deterministic/no-TF32 arithmetic; all comparisons above use the new unified evaluator. Large differences require investigation before interpreting migration.

| Model | Condition | Metric | Mean delta | Maximum absolute per-image delta |
|---|---|---|---:|---:|
| old-official | full | psnr_db | +0.00023357391 | 0.2022438 |
| old-official | full | ssim | +0.00018605709 | 0.007892549 |
| old-official | full | lpips_alex | +5.7591703e-05 | 0.0054136366 |
| old-official | full | dino_cosine | -0.00067548335 | 0.059035957 |
| old-official | p8_direct | psnr_db | -0.00041341782 | 0.36004829 |
| old-official | p8_direct | ssim | +7.9687983e-05 | 0.010302901 |
| old-official | p8_direct | lpips_alex | -7.260181e-05 | 0.017989606 |
| old-official | p8_direct | dino_cosine | +0.00056266628 | 0.12942547 |
| old-official | p8_argmax | psnr_db | +0.011167145 | 0.6534481 |
| old-official | p8_argmax | ssim | +0.0005556421 | 0.024664134 |
| old-official | p8_argmax | lpips_alex | -0.00046134071 | 0.012510151 |
| old-official | p8_argmax | dino_cosine | -0.00086852193 | 0.032604992 |
| old-official | p9_direct | psnr_db | +0.028971672 | 1.6552086 |
| old-official | p9_direct | ssim | +0.0002609776 | 0.0056387782 |
| old-official | p9_direct | lpips_alex | -0.00047089431 | 0.0216887 |
| old-official | p9_direct | dino_cosine | +0.0011096659 | 0.066519737 |
| old-official | p9_argmax | psnr_db | +0.012158928 | 0.54181862 |
| old-official | p9_argmax | ssim | +5.4006726e-05 | 0.0052189231 |
| old-official | p9_argmax | lpips_alex | +1.2842193e-06 | 0.013370514 |
| old-official | p9_argmax | dino_cosine | +0.00068672657 | 0.046633065 |
| old-fidelity | full | psnr_db | +0.0016422081 | 0.21109962 |
| old-fidelity | full | ssim | +5.2489042e-05 | 0.0062478781 |
| old-fidelity | full | lpips_alex | -9.3086548e-05 | 0.0043008029 |
| old-fidelity | full | dino_cosine | -0.0001157701 | 0.020120502 |

## Interpretation boundaries

- Full reconstruction is a codec result, not sufficient evidence for a next-scale migration.
- Primary completions use true transmitted prefixes and deterministic class-conditioned argmax (no unconditional CFG mixing).
- XQ official-style sampling is a sensitivity analysis only, not a per-image output selector; seed metrics are averaged within image before bootstrapping.
- Compare `paired_deltas.csv` and `per_image_deltas.csv`: deltas are candidate minus reference. Positive LPIPS is worse.
- Source-rate mismatches are explicit. No FEC/header/pilots/noise/complex-channel-use claim is made.
- XQ uses DINO-related training; DINO similarity is reported, not independent evidence of recovered semantic information.
- Lack of significant LPIPS degradation does not establish perceptual equivalence. Inspect paired intervals and difficult source-instance details.
- Codec timing excludes file I/O, preprocessing and metric inference. Completion timing includes its final RGB decode. Sender and receiver model weights are resident together. Metrics are loaded only after releasing the codec.
- A formal migration recommendation requires examining stable PSNR/LPIPS improvements at prefix+completion points and whether the actual rate advantage warrants adaptation cost. These descriptive flags are not a new arbitrary threshold.

## Descriptive evidence flags

```json
[
  {
    "candidate": "xq/full",
    "reference": "old-official/full",
    "raw_bit_delta": -1296,
    "psnr_paired_ci_strictly_positive": false,
    "lpips_paired_ci_strictly_worse": true,
    "lpips_paired_ci_no_worse_upper_bound": false,
    "note": "CI flags are descriptive; a CI crossing zero is not proof of equivalence. Migration also needs instance inspection and meaningful rate/quality gain."
  },
  {
    "candidate": "xq/p8_argmax",
    "reference": "old-official/p8_argmax",
    "raw_bit_delta": -636,
    "psnr_paired_ci_strictly_positive": false,
    "lpips_paired_ci_strictly_worse": true,
    "lpips_paired_ci_no_worse_upper_bound": false,
    "note": "CI flags are descriptive; a CI crossing zero is not proof of equivalence. Migration also needs instance inspection and meaningful rate/quality gain."
  },
  {
    "candidate": "xq/p9_argmax",
    "reference": "old-official/p8_argmax",
    "raw_bit_delta": 900,
    "psnr_paired_ci_strictly_positive": true,
    "lpips_paired_ci_strictly_worse": false,
    "lpips_paired_ci_no_worse_upper_bound": true,
    "note": "CI flags are descriptive; a CI crossing zero is not proof of equivalence. Migration also needs instance inspection and meaningful rate/quality gain."
  },
  {
    "candidate": "xq/p9_argmax",
    "reference": "old-official/p9_argmax",
    "raw_bit_delta": -1128,
    "psnr_paired_ci_strictly_positive": false,
    "lpips_paired_ci_strictly_worse": true,
    "lpips_paired_ci_no_worse_upper_bound": false,
    "note": "CI flags are descriptive; a CI crossing zero is not proof of equivalence. Migration also needs instance inspection and meaningful rate/quality gain."
  }
]
```

## Artifacts

- `summary.csv`: per-condition image means/95% bootstrap intervals, batch-one timings and measured GPU peaks.
- `paired_deltas.csv`, `per_image_deltas.csv`: source-paired comparisons and all per-image changes.
- `rate_quality_discrete.png`: natural rates plotted as points only; no false equal-rate interpolation.
- `*_difficult_cases.png`, `difficult_case_selection.json`: fixed worst-case selection rules, not best-looking examples.
- `inputs.json`: immutable result hashes and per-model metadata.
- `historical_replay_summary.csv` and `historical_replay_per_image.csv`: new-vs-archived numerical baseline audit.

**Communication attribution remains open:** any adopted visual backbone needs its own digital baseline and re-adapted communication E/D; a better visual codec is not itself a JSCC contribution.
