# Direction 2 — V3 Execution Report: Completely CSI-Free, Feedback-Free & Pilot-Free Self-Probing Hierarchical JSCC (`SPH-ZPAC`)

## 1. Motivation & 2024–2026 CSI-Free Literature Taxonomy

Following V2 (`MS-ZPAC` and `TR-ZPAC`, branch `codex/dir2-swinjscc-mismatch-v2`), which resolved imperfect/delayed CSI mismatch under pilot-aided closed-loop transmission, V3 investigates a stricter physical question: **Can an adaptive semantic transceiver achieve near-oracle rate-distortion-perception performance over both unknown-SNR AWGN and 32-RB frequency-selective complex OFDM fading with ZERO prior CSI, ZERO feedback, ZERO dedicated pilot symbols, and ZERO per-frame scalar side-channels?**

Recent 2024–2026 IEEE literature on CSI-free adaptive JSCC falls into three paradigms, each facing a fundamental bottleneck:
1. **Open-Loop Universal / Query-Based JSCC** (*SwinJSCC w/o SA*, Yang et al., *IEEE TCCN* 2024; *SNR-EQ-JSCC*, Zhang & Tao, *IEEE WCL* Mar. 2025; *DeepJSCC-l++*, Bian, Shao & Gündüz, *GLOBECOM* 2023): removes explicit SNR tokens at the encoder, yet implicitly assumes phase-aligned real AWGN or genie receiver CSI + genie per-image power scalar `pwr`, collapsing to `~8.9 dB` under pilot-free complex Rayleigh/OFDM fading due to $U(1)$ rotational phase unidentifiability.
2. **End-to-End Black-Box Blind JSCC Training** (Oh, Park, Choi, Park & Jeon, *"Blind Training for Channel-Adaptive Digital Semantic Communications,"* *IEEE TCOM*, Nov. 2025): jointly trains TX and RX over random channels without pilots so the encoder implicitly breaks rotational symmetry. However, implicit pilot features share the exact same linear span as semantic symbols, creating severe self-interference that caps high-SNR reconstruction (`~20.7 dB` on 32-RB OFDM).
3. **Iterative Diffusion / Generative Blind Receivers** (*CDDM*, Wu, Tao et al., *IEEE TWC* Sept. 2024; *SGD-JSCC*, Zhang, Wu & Gündüz, 2025): leverages score priors for blind restoration at the cost of `50–200x` iterative sampling latency.

## 2. Proposed Architecture: `SPH-ZPAC` (Single-Pass, `1.27M` Trainable Params / `4.32%`)

`SPH-ZPAC` keeps `95.68%` of the official `SwinJSCC_w_SA` backbone strictly frozen and introduces three lightweight, zero-overhead physical-semantic modules:
1. **Open-Loop Hierarchical TX (`H-TX`, image-only input, zero CSI)**: predicts content-adaptive unequal-error-protection (UEP) channel scaling and residual shaping from the frozen Swin encoder stem alone, protecting coarse semantic channels against harsh unknown fading while preserving fine details at high SNR.
2. **3-Vector Orthonormal Walsh-Hadamard Self-Probing Subspace Modem (`0` extra symbols, exact `K = 12,288` complex symbols, exact unit per-frame transmit power $\frac{1}{K}\|\mathbf{s}_{\text{tx}}\|_2^2 \equiv 1$)**:
   Within each $N_b = 384$-symbol resource block ($M = 32$ RBs), semantic symbols are projected onto the $381$-dimensional orthogonal complement ($99.22\%$ DoF) of three deterministic orthonormal Walsh vectors $(\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3)$:
   - $\mathbf{u}_2$ (**Orthogonal Null Subspace**): zero transmitted energy $\implies \mathbf{u}_2^T \mathbf{y}_m \sim \mathcal{CN}(0, \sigma_n^2)$ yields an exact, signal-interference-free MVUE of noise variance $\hat{\sigma}_n^2$ under arbitrary unknown fading;
   - $\mathbf{u}_1$ (**Coherent Spread-Spectrum Sensing Subspace**): learned power fraction $\eta_1 = 0.0523$ ($\eta_1 N_b = 20.08$ coherent pilot-equivalent energy units per RB — `5.02x` stronger than 4-pilot LMMSE without spending any extra symbols!), refined by a 5-tap analytical Wiener + 1D Conv RB denoiser;
   - $\mathbf{u}_3$ (**Constant-Modulus Frame-Norm Subspace**, $\eta_3 = 0.005$): encodes per-image latent RMS $p_{\text{img}}$ on the unit circle $(\phi_p + j\sqrt{1-\phi_p^2})\mathbf{u}_3$, eliminating the upstream `pwr` side-channel completely while keeping frame power identically `1.000000`.
3. **Blind Multi-Scale Decoder Calibrator**: feeds the blindly equalized latent, spatial posterior noise map, I/Q cross-half conjugate residual, and blind effective SINR $\hat{c}_R$ into zero-locked adapters across all 4 Swin Decoder stages ($16^2 \to 32^2 \to 64^2 \to 128^2$).

## 3. Held-Out `ImageNet common500` (`500` Images × `3` Seeds) & `Kodak24` Main Results

### 3.1 Track 2: Completely Pilot-Free & Feedback-Free 32-RB Complex OFDM Fading (`common500`)

| Condition (`common500`, `p10` in parens) | `Frozen_Blind` (`0` Pilots) | `Blind_RX_ZPAC` (`0` Pilots) | `Blind_BlackBox` (*TCOM'25*, `0` Pilots) | `Pilot4_Frozen_MMSE` (`128` Pilots + `pwr`) | `Pilot4_TR_ZPAC` (`128` Pilots + `pwr`) | `SPH_ZPAC` w/o Hier-TX (`0` Pilots) | **`SPH-ZPAC` (Ours: `0` Pilots, `0` CSI)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 dB ($\rho_f=0.85$)** | `8.781` (`6.07`) | `12.323` (`9.93`) | `18.901` (`15.73`) | `15.070` (`11.11`) | `21.278` (`16.74`) | `21.198` (`16.28`) | **`23.453` (`18.80`)** |
| **4 dB ($\rho_f=0.85$)** | `8.879` (`6.00`) | `12.558` (`10.10`) | `19.670` (`16.41`) | `18.284` (`13.17`) | `23.746` (`19.04`) | `23.966` (`18.87`) | **`25.464` (`20.74`)** |
| **7 dB ($\rho_f=0.85$)** | `8.949` (`6.01`) | `12.704` (`10.22`) | `20.190` (`16.87`) | `21.513` (`15.82`) | `25.892` (`21.10`) | `26.289` (`21.32`) | **`27.199` (`22.42`)** |
| **10 dB ($\rho_f=0.85$)** | `8.993` (`6.06`) | `12.789` (`10.26`) | `20.503` (`17.14`) | `24.540` (`18.55`) | `27.782` (`23.05`) | `28.157` (`23.13`) | **`28.665` (`23.93`)** |
| **13 dB ($\rho_f=0.85$)** | `9.018` (`6.08`) | `12.830` (`10.30`) | `20.674` (`17.29`) | `27.108` (`21.29`) | `29.284` (`24.48`) | `29.575` (`24.75`) | **`29.834` (`25.18`)** |
| **OOD 7 dB ($\rho_f=0.55$)** | `8.739` (`6.54`) | `12.474` (`10.23`) | `20.022` (`16.74`) | `21.450` (`16.45`) | `25.893` (`21.38`) | `26.219` (`21.46`) | **`27.125` (`22.52`)** |
| **OOD 10 dB ($\rho_f=0.55$)** | `8.799` (`6.61`) | `12.551` (`10.26`) | `20.300` (`16.96`) | `24.446` (`19.16`) | `27.746` (`23.02`) | `28.040` (`23.04`) | **`28.555` (`23.79`)** |

- With **0 dedicated pilots and 0 side-channels**, `SPH-ZPAC` improves over `Frozen_Blind` by **`+14.67 ~ +20.82 dB`**, beats *IEEE TCOM'25*-style E2E `Blind_BlackBox` by **`+4.55 ~ +9.16 dB`** (reducing LPIPS from `0.472` to `0.115` and DISTS from `0.337` to `0.143` at 10 dB), outperforms 128-pilot `Pilot4_Frozen_MMSE` by **`+2.73 ~ +8.38 dB`**, and even surpasses 128-pilot closed-loop `Pilot4_TR_ZPAC` by **`+0.55 ~ +2.18 dB`**!

### 3.2 Track 1: Unknown-SNR Blind AWGN (`common500`, `-2` & `15 dB` Unseen OOD)

| True AWGN SNR | `Frozen_Blind` | `Blind_BlackBox` (*TCOM'25*) | `Blind_RX_ZPAC` (Frozen TX) | `SPH_ZPAC` w/o Hier-TX | **`SPH-ZPAC` (Ours, Blind MAE)** | `AvgSNR` (Genie `pwr`) | `Oracle` (Matched CSI + `pwr`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`-2 dB` (OOD)** | `17.965` | `22.921` | `20.356` | `21.128` | **`24.655` (`1.24 dB`)** | `21.024` | `24.466` |
| **`1 dB`** | `23.249` | `25.678` | `25.114` | `25.672` | **`27.016` (`1.10 dB`)** | `26.175` | `27.630` |
| **`4 dB`** | `26.620` | `27.634` | `28.066` | `28.586` | **`28.677` (`1.00 dB`)** | `29.207` | `29.368` |
| **`7 dB`** | `27.947` | `28.979` | `29.448` | **`30.174`** | `29.962` (`0.94 dB`) | `30.710` | `30.710` |
| **`10 dB`** | `28.530` | `29.895` | `30.203` | **`31.207`** | `30.935` (`0.88 dB`) | `31.692` | `31.696` |
| **`13 dB`** | `28.852` | `30.475` | `30.641` | **`31.892`** | `31.602` (`0.83 dB`) | `32.342` | `32.367` |
| **`15 dB` (OOD)** | `28.984` | `30.722` | `30.819` | **`32.194`** | `31.895` (`0.81 dB`) | `32.628` | `32.673` |

## 4. Reproducibility & Hygiene Audit
- Unit tests: `12 passed` (`test_dir2_zpac_cpu.py`, `test_dir2_zpac_v2.py`, `test_dir2_blind_v3.py`).
- Zero symlinks (`symlink_count == 0`) & zero multi-hardlink files (`hardlink_count == 0`) across all PFS code, checkpoint, and runtime directories.
