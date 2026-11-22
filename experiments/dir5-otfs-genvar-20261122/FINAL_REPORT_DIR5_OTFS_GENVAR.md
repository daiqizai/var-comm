# Direction 5 Final Experimental Report (`codex/dir5-otfs-genvar-v1`)

## 1. Executive Summary

Direction 5 establishes two major zero-pilot, zero-CSI wireless semantic communication breakthroughs directly aligned with 2025–2026 IEEE Transactions frontiers and the thesis **《资源受限下的语义通信自适应传输方法研究》**:

1. **Dir-5A (`Symp-OTFS-JSCC` — High-Mobility Doubly-Selective Delay-Doppler Zero-Pilot Blind Semantic Communications, `CBR = 1/32`, `K = 12,288`)**:
   - Solves severe fast-time Jakes Doppler rotation + physical intra-symbol Inter-Carrier Interference (`ICI`) + multipath delay spread (`P = 6` scatterers) across `50 km/h` to unseen **`900 km/h` (`k_d = f_{d,\max}/\Delta f = 0.18`)** with **`0` pilot symbols, `0` Delay-Doppler guard zones, and `0` prior CSI**.
   - **Core Physical-Layer Mechanism**: Combines a `24 × 32` micro-tile (`2 × 8 = 16` REs/tile) orthonormal `14/16` Hadamard subspace decomposition, CP-forbidden non-causal delay-bin (`m ∈ [9..24]`) blind noise-plus-ICI floor sensing, sparse Delay-Doppler Empirical-Bayes Wiener shrinkage, exact complex `2 × 8` boxcar-deconvolved `48 × 256` Fourier per-RE CSI upsampling, per-RE Decision-Free Probe Cancellation, and Unitary `24 × 32` Symplectic (`ISFFT / SFFT`) spreading.
   - **Headline Held-Out Results (`ImageNet-500` & `Kodak24`, 3 Paired Seeds)**:
     - At **`500 km/h` (`k_d = 0.10`, `10 / 14 dB`)**: **`28.168 / 28.716 dB`** on `common500` (`Kodak24`: **`28.727 / 29.228 dB`**), beating generous unpunctured `1,024`-pilot 5G 4-Symbol DMRS Interpolation (`12.615 / 12.529 dB`) by **`+15.553 / +16.187 dB`**, static-RB `V4 U-SPH` (`10.476 / 10.507 dB`) by **`+17.691 / +18.209 dB`**, and `BlackBox_OTFS_Blind` (`17.186 / 17.276 dB`) by **`+10.982 / +11.441 dB`**.
     - At **Unseen OOD Hyper-Mobility `~900 km/h` (`k_d = 0.18`, `10 / 14 dB`)**: **`27.419 / 27.970 dB`** on `common500` (`Kodak24`: **`27.794 / 28.498 dB`**, blind CSI `NMSE = -8.00 / -8.27 dB` vs `+1.34 dB` for 1,024-pilot DMRS!), outperforming `1,024`-pilot 5G DMRS by **`+16.540 / +17.099 dB`** and our controlled non-symplectic waveform ablation `Symp_OFDM_2DSubspace` (`26.907 / 27.480 dB`) by **`+0.512 / +0.490 dB`** (`95%` paired bootstrap CI `[+0.477, +0.549] dB`).

2. **Dir-5B (`GenHolo-VAR-JSCC` — Ultra-Low-Bandwidth `CBR = 1/128` [`K = 1,536`] & `CBR = 1/256` [`K = 768` = `16×` Compression] Generative Perceptual Blind Semantic Communications)**:
   - Breaks the severe perceptual blur cliff of pixel-MSE discriminative JSCC under extreme narrowband IoT / tactical satellite budgets (`8` narrowband RBs) with **`0` pilots and `0` CSI**, while keeping both pretrained `VAR VQ-VAE` (`V = 4,096`, `10`-scale residual VQ) and `SwinJSCC` backbones **100% frozen**.
   - **Core Generative-Physical Co-Design**: Couples a `VAR`-matched **Multi-Scale `8×8` Semantic Pyramid + `16×16` Edge Residual Bottleneck** (preserving `75%–100%` of `VAR`'s orthogonal 32-D VQ channel space), **256-point Stride-8 Walsh-Hadamard Holographic Dispersion** across `8` RBs (`4`-Walsh-vector zero-pilot blind subspace + automatic erased-RB null sensing), and **`SoftVQManifoldProjector`** (SNR-and-reliability gated convex fusion of continuous features, soft cosine codebook attention, and straight-through 10-scale `VAR` VQ snapping).
   - **Headline Held-Out Results (`ImageNet-500` & `Kodak24`, 3 Paired Seeds)**:
     - **`CBR = 1/128` (`K = 1,536` symbols, `10 / 13 dB`)**: Achieves **`DISTS = 0.1710 / 0.1560`** and **`LPIPS = 0.2082 / 0.1841`** (`PSNR = 20.199 / 20.782 dB`; `Kodak24`: **`PSNR = 21.147 / 21.652 dB`, `DISTS = 0.1726 / 0.1567`, `LPIPS = 0.2061 / 0.1844`**), slashing perceptual distortion by **`-49.9%` in `DISTS` and `-62.0%` in `LPIPS`** compared to **Oracle Exact-CSI Discriminative `Swin_Genie_CSI`** (`DISTS = 0.3114`, `LPIPS = 0.4851`), while beating `BlackBox_VAR_Blind` by **`+3.472 dB` PSNR** (`20.782 vs 17.309 dB`), **`-41.3%` `DISTS`**, and **`-47.8%` `LPIPS`**.
     - **Extreme `CBR = 1/256` (`K = 768` symbols = `16×` compression, `10 / 13 dB`)**: Achieves **`DISTS = 0.2338 / 0.2206`** and **`LPIPS = 0.3088 / 0.2846`** (`PSNR = 18.197 / 18.618 dB`; `Kodak24`: **`PSNR = 19.260 / 19.657 dB`, `DISTS = 0.2362 / 0.2210`, `LPIPS = 0.3123 / 0.2872`**), reducing `LPIPS` by **`-53.5%`** and `DISTS` by **`-39.7%`** below Oracle `Swin_Genie_CSI`, and beating `BlackBox_VAR_Blind` by **`+1.013 dB` PSNR** across `common500`.
     - **Unseen OOD `50%` Contiguous Bandwidth Wipeout (`RBs {2,3,4,5}` zeroed out $\implies$ ONLY `768` surviving symbols at `CBR=1/128` and `384` surviving symbols [`CBR_eff = 1/512`!] at `CBR=1/256`)**: `GenHolo_VAR_JSCC` maintains **`DISTS = 0.2656` (`768` syms) / `0.3051` (`384` syms)** and beats `BlackBox_VAR_Blind` by **`+2.138 dB` / `+1.006 dB` PSNR**.

---

## 2. Quantitative Tables (`ImageNet common500`, 500 Images × 3 Paired Seeds)

### Table 1 — Dir-5A High-Mobility Doubly-Selective Delay-Doppler Sweep (`CBR = 1/32`, `K = 12,288`)

| Condition | SNR (dB) | $k_d$ ($\approx$ Velocity) | `V4_USPH_OFDM` (0-P) | `5G 4-Sym DMRS` (1,024-P) | `BlackBox_OTFS` (0-P) | `Symp_OFDM_2D` (No ISFFT) | **`Symp_OTFS_JSCC` (Ours, 0-P)** | Blind CSI NMSE (Ours vs DMRS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vel_50kmh` | 10 | `0.01` (`50 km/h`) | 20.231 dB | **29.295 dB** | 17.632 dB | 28.057 dB | 28.394 dB | `-10.87` vs `-21.14 dB` |
| `vel_250kmh` | 10 | `0.05` (`250 km/h`) | 12.225 dB | 18.045 dB | 17.536 dB | 28.051 dB | **28.384 dB** | **`-10.73`** vs `-3.49 dB` |
| `snr_1dB_500kmh` | 1 | `0.10` (`500 km/h`) | 9.965 dB | 12.651 dB | 16.232 dB | 24.087 dB | **24.349 dB** | **`-8.22`** vs `+0.33 dB` |
| `snr_4dB_500kmh` | 4 | `0.10` (`500 km/h`) | 10.421 dB | 12.879 dB | 16.721 dB | 25.819 dB | **26.178 dB** | **`-9.09`** vs `+0.26 dB` |
| `snr_7dB_500kmh` | 7 | `0.10` (`500 km/h`) | 10.263 dB | 12.570 dB | 17.036 dB | 27.021 dB | **27.433 dB** | **`-9.63`** vs `+0.30 dB` |
| `snr_10dB_500kmh` | 10 | `0.10` (`500 km/h`) | 10.476 dB | 12.615 dB | 17.186 dB | 27.773 dB | **28.168 dB** | **`-9.94`** vs `+0.24 dB` |
| `snr_14dB_500kmh` | 14 | `0.10` (`500 km/h`) | 10.507 dB | 12.529 dB | 17.276 dB | 28.354 dB | **28.716 dB** | **`-10.14`** vs `+0.26 dB` |
| **`OOD 700 km/h`** | 10 | `0.14` (`700 km/h`) | 10.087 dB | 11.341 dB | 16.672 dB | 27.417 dB | **27.857 dB** | **`-9.01`** vs `+1.24 dB` |
| **`OOD 900 km/h`** | 10 | `0.18` (`900 km/h`) | 9.804 dB | 10.879 dB | 16.207 dB | 26.907 dB | **27.419 dB** | **`-8.00`** vs `+1.34 dB` |
| **`OOD 900 km/h`** | 14 | `0.18` (`900 km/h`) | 9.837 dB | 10.872 dB | 16.308 dB | 27.480 dB | **27.970 dB** | **`-8.27`** vs `+1.34 dB` |

### Table 2 — Dir-5B Ultra-Low-Bandwidth Generative Perceptual Blind JSCC (`8`-RB Rayleigh OFDM)

| CBR ($K_{\text{surv}}$) | Condition | `Swin_Genie_CSI` (Oracle CSI: DISTS / LPIPS / PSNR) | `Swin_Holo_Blind` (0-P: DISTS / LPIPS / PSNR) | `BlackBox_VAR_Blind` (0-P: DISTS / LPIPS / PSNR) | **`GenHolo_VAR_JSCC` (Ours, 0-P: DISTS ↓ / LPIPS ↓ / PSNR ↑)** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `1/128` (`1,536`) | `4 dB` | 0.3423 / 0.5526 / 20.89 dB | 0.3463 / 0.5555 / 21.06 dB | 0.3147 / 0.4329 / 16.09 dB | **0.2194 / 0.2926** / 18.45 dB |
| `1/128` (`1,536`) | `10 dB` | 0.3183 / 0.5014 / 21.97 dB | 0.3232 / 0.5082 / 22.39 dB | 0.2753 / 0.3699 / 17.03 dB | **0.1710 / 0.2082** / 20.20 dB |
| `1/128` (`1,536`) | `13 dB` | 0.3114 / 0.4851 / 22.34 dB | 0.3153 / 0.4920 / 22.83 dB | 0.2658 / 0.3528 / 17.31 dB | **0.1560 / 0.1841** / 20.78 dB |
| `1/128` (`768` OOD) | **`50%` Erase (`10 dB`)** | 0.3683 / 0.6099 / 18.96 dB | 0.3642 / 0.5973 / 19.76 dB | 0.3703 / 0.5120 / 14.89 dB | **0.2656 / 0.3732** / 17.03 dB |
| `1/256` (`768`) | `4 dB` | 0.3900 / 0.6600 / 19.63 dB | 0.3919 / 0.6579 / 19.64 dB | 0.3214 / 0.4490 / 16.40 dB | **0.2718 / 0.3840** / 16.97 dB |
| `1/256` (`768`) | `10 dB` | 0.3720 / 0.6242 / 20.57 dB | 0.3710 / 0.6117 / 20.95 dB | 0.2853 / 0.3824 / 17.34 dB | **0.2338 / 0.3088** / 18.20 dB |
| `1/256` (`768`) | `13 dB` | 0.3657 / 0.6118 / 20.91 dB | 0.3645 / 0.5966 / 21.35 dB | 0.2776 / 0.3666 / 17.61 dB | **0.2206 / 0.2846** / 18.62 dB |
| `1/256` (`384` OOD) | **`50%` Erase (`10 dB`)** | 0.4147 / 0.7035 / 18.00 dB | 0.4088 / 0.6943 / 18.42 dB | 0.3636 / 0.5100 / 15.08 dB | **0.3051 / 0.4438** / 16.09 dB |

---

## 3. Reproducibility & Hygiene Audit

- **Worktree**: `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1` (Branch `codex/dir5-otfs-genvar-v1`)
- **Checkpoints**: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir5_otfs_genvar_v1`
- **Logs / Runtime Cache**: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1`
- **Link Hygiene**: Verified `symlink_count == 0` and `hardlink_count == 0` across all `liulu` PFS directories.
