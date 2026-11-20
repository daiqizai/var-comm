# Direction 2 Complete Experimental Report: Zero-Locked Physics-Guided Asymmetric Calibration (ZPAC) for SwinJSCC Under Transceiver SNR Mismatch

- **Branch**: `codex/dir2-swinjscc-mismatch-v1` (branched cleanly from `origin/main` `564c18059dbe6c80ada4c111df01bdd8320fc52b`)
- **Hardware**: 4× NVIDIA A800-SXM4-80GB (`FP32`, `TF32=False`, deterministic algorithms)
- **PFS Storage Layout & Link Hygiene**:
  - Code & lightweight results: `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1`
  - Model checkpoints: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1`
  - Runtime caches: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_swinjscc_mismatch_v1`
  - Verified `symlink_count = 0` and `hardlink_count = 0` across all PFS directories.

---

## 1. Stage 0 Discovery: Decoder ModNet Degeneration in Pretrained SwinJSCC

Evaluating the official `SwinJSCC_w/_SA` (`base`, $C=96$, $\text{CBR}=1/16$, AWGN, MSE) checkpoint on sealed `dev64` (`64` calibration images $\times 3$ paired noise realizations) established two key findings:
1. **Exact Split Equivalence**: Splitting `(c_T, Gamma, c_R)` matched upstream `SwinJSCC` forward outputs to **`0.0` maximum absolute error** across $\{1, 4, 7, 10, 13\}\text{ dB}$, and `P-ZPAC`'s zero-locked identity gate exhibited **`0.0` leakage** at $c_T = c_R$.
2. **Decoder Modulation Collapse**: Under standard matched end-to-end training ($c_T = c_R = \Gamma$), the 7-layer multiplicative `ChannelModNet` in the `SwinJSCC` decoder (`temp = sm_list[i](temp) * sigmoid(...)` without residual skips) suffers $(0.5 \times 0.6)^6 \approx 10^{-3}$ Jacobian attenuation and collapses to an SNR-insensitive constant gate (`(4, 4, 13)` vs `(4, 4, 4)` gives identical `28.824 dB`; feeding accurate $c_R = 4\text{ dB}$ to `(13, 4, 13)` yields `0.000 dB` gain). Consequently:
   - **TX Overestimation** (`c_T = 13 -> Gamma = 1 dB`): PSNR drops **`-1.981 dB`** (`95% CI [-2.267, -1.698] dB`), MS-SSIM drops `-0.0412`.
   - **TX Underestimation** (`c_T = 1 -> Gamma = 13 dB`): PSNR drops **`-0.585 dB`**, while perceptual quality degrades sharply (**DISTS `+0.0284`, LPIPS `+75%`**).

---

## 2. Main Held-Out Results on ImageNet `common500` & `Kodak24` (`C = 96`, `CBR = 1/16`)

All trainable models were trained on `train20k` (`4,000` steps, batch `16`, identical paired batch/noise seed `20261120`) using **only** $\Delta c \in \{0, \pm 3, \pm 6\}\text{ dB}$, selected on `val256`, and evaluated once on sealed `common500` (`500` images $\times 3$ seeds = `1,500` frames/cond) and `kodak24` (`24` images $\times 3$ seeds = `72` frames/cond).

### Key Paired Differences (`P_ZPAC_Hier` minus Baseline on `common500`, 10,000-Resample Source Bootstrap 95% CI)

| Regime | vs `Frozen` (`SwinJSCC`) | vs `B1` (Dual Dec `ModNet`) | vs `B0` (Single $c_R$) | vs `P_Raw` ($[\bar{c}_R, \bar{c}_T, 0]$) | vs `B2` (Dual Front+Dec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Matched ($\Delta c = 0$)** | **`0.000 dB`** (`[0, 0]`) | **`0.000 dB`** (`[0, 0]`) | `-0.008 dB` (DISTS **`-0.0017`**) | `-0.011 dB` (DISTS **`-0.0007`**) | `-0.012 dB` (DISTS **`-0.0007`**) |
| **Seen Underestimation** | **`+0.052 dB`** (`[+0.047, +0.057]`) | **`+0.052 dB`** | **`+0.055 dB`** (DISTS **`-0.0062`**) | **`+0.045 dB`** (DISTS **`-0.0017`**) | **`+0.043 dB`** (DISTS **`-0.0019`**) |
| **Unseen Interpolation ($\pm 2, \pm 4\text{ dB}$)** | **`+0.029 dB`** (`[+0.026, +0.032]`) | **`+0.029 dB`** | **`+0.014 dB`** (`[+0.013, +0.016]`) | **`+0.012 dB`** (`[+0.010, +0.013]`) | **`+0.010 dB`** (`[+0.008, +0.011]`) |
| **Unseen Extrapolation ($\pm 8, \pm 9, \pm 12\text{ dB}$)** | **`+0.220 dB`** (`[+0.209, +0.232]`) | **`+0.220 dB`** | **`+0.132 dB`** (DISTS **`-0.0045`**) | **`+0.086 dB`** (DISTS **`-0.0051`**) | **`+0.084 dB`** (DISTS **`-0.0052`**) |
| **Imperfect RX CSI ($c_R \neq \Gamma$)** | **`+0.118 dB`** (`[+0.108, +0.129]`) | **`+0.118 dB`** | **`+0.078 dB`** (`[+0.071, +0.087]`) | **`+0.049 dB`** (DISTS **`-0.0029`**) | **`+0.047 dB`** (DISTS **`-0.0032`**) |

- On extreme unseen extrapolation: `P_ZPAC_Hier` achieves **`+0.545 dB`** at `(13, 1, 1)` (`25.323 -> 25.868 dB`) and **`+0.248 dB` PSNR + `-0.0142` DISTS** at `(1, 13, 13)` (`31.654 -> 31.902 dB`, DISTS `0.1340 -> 0.1198`) on `common500`, and **`+0.327 dB` / `+0.292 dB` (DISTS `-0.0172`)** zero-shot on out-of-domain `Kodak24`.
- Compared to `AvgSNR` fallback (`c_T = 7 dB`), `P_ZPAC_Hier` avoids `AvgSNR`'s severe **`-0.324 dB`** matched-condition penalty (`95% CI [-0.344, -0.304] dB` on `common500`) while outperforming `AvgSNR` on perceptual DISTS across both underestimation (`-0.0044`) and unseen extrapolation (`-0.0056`).

---

## 3. Time-Correlated AR(1) Rayleigh Block Fading & Feedback Delay (`common500` & `Kodak24`)

Zero-shot transfer of AWGN-trained calibrators to AR(1) Gauss-Markov flat Rayleigh fading with pilot LMMSE equalization ($N_p = 4$) and 5-bit delayed feedback ($d \in \{0, 1, 2, 3\}$):
- **`common500` ($\bar{\gamma}_0 = 10\text{ dB}, \rho_c = 0.90, d = 2$, mean $|c_T - c_R| = 3.74\text{ dB}$)**:
  `P_ZPAC_Hier` achieves **`25.502 dB`** mean PSNR (vs `Frozen` `25.256`, `B1` `25.256`, `P_Raw` `25.331`, `B2` `25.336`, `B0` `25.374`) — leading all adaptive-TX methods by **`+0.13 ~ +0.25 dB`** and lowest perceptual DISTS (`0.2139`).
- **Fast fading (`common500`, $\bar{\gamma}_0 = 10\text{ dB}, \rho_c = 0.70, d = 2$, mean $|c_T - c_R| = 5.24\text{ dB}$)**:
  `P_ZPAC_Hier` achieves **`25.041 dB`** (`+0.311 dB` over `Frozen`, **`+0.270 dB` over `P_Raw`**, **`+0.241 dB` over `B2`**) and boosts Worst-10% tail `p10` PSNR from `17.768 dB` (`Frozen`) / `17.461 dB` (`P_Raw`) to **`18.381 dB` (`+0.613 dB` / `+0.919 dB` tail gain)**!
- **`Kodak24` ($\bar{\gamma}_0 = 10\text{ dB}, \rho_c = 0.90, d = 2$)**: `P_ZPAC_Hier` ranks #1 across PSNR (`28.103 dB`), `p10` PSNR (`24.333 dB`), MS-SSIM (`0.9146`), and DISTS (`0.1728`).

---

## 4. Internal Structural Ablations & Complexity Accounting

1. **Ablation Verification (`common500` Unseen Extrapolation)**:
   - Removing Zero-Locked Gate (`P_NoGate`): loses `-0.030 dB` PSNR and degrades DISTS by `+0.0073` vs `P_ZPAC` (`-0.097 dB` vs `P_ZPAC_Hier`), while introducing matched-condition drift.
   - Removing `+ / -` Asymmetry (`P_Sym`): loses `-0.025 dB` PSNR and degrades DISTS by `+0.0029` vs `P_ZPAC` (`-0.092 dB` vs `P_ZPAC_Hier`).
   - Removing Posterior SNR & Wiener Prior (`P_NoXi`): loses `-0.010 dB` PSNR (`95% CI [-0.014, -0.007]`) on extrapolation and `-0.005 dB` on seen overestimation vs `P_ZPAC`.
   - Removing Hierarchical Post-Head Delta (`P_ZPAC` vs `P_ZPAC_Hier`): `P_ZPAC_Hier` adds **`+0.067 dB`** (`[+0.062, +0.073]`) on extrapolation and **`+0.347 dB`** on fast Rayleigh fading.
2. **Complexity (`A800` Single-Image RX)**:
   - `Frozen SwinJSCC`: `14.10M` RX params (`28.20M` total), `18.30 ms` median RX latency.
   - `P_ZPAC` (Front): `+0.46M` params (`+3.26%` RX), `22.46 ms` (`+4.16 ms`).
   - `P_ZPAC_Hier` (Full): `+0.77M` params (`+5.44%` RX, `6.96x` fewer trainable params than `B2`'s `5.34M` when opening `ChannelModNet`), `28.29 ms`.
   - Signaling overhead: `0` forward bits under acknowledged delayed feedback history; even if explicitly signaled forward via a rate-`1/2` protected 5-bit index (`5` BPSK/QPSK symbols out of `12,288` symbols = `0.041%` bandwidth), channel capacity impact is `< 0.002 dB`.

---

## 5. Stage 3 Cross-Chapter Bridge (`N = 1024` Symbol Budget Aligned with Chapter 3 `NeST-Com`)

Trained and evaluated `P_ZPAC_Hier` on the exact `N = 1024` (`rate_channels = 6`, `768` data + `256` paid header symbols) `SwinJSCC_w/_SAandRA` (`Swin_1to19` `model_200000.pt`) backbone on `common500`:
- **Matched conditions**: **`0.000 dB`** exact zero drift across `{1, 4, 7, 10, 13} dB`.
- **Seen Underestimation**: **`+0.129 dB`** average (`+0.255 dB` at `(1, 7, 7)`, recovering **`71%–94%`** of the full oracle gap!), DISTS **`-0.0077`**, LPIPS **`-0.0113`**.
- **Unseen Extrapolation**: **`+0.293 dB`** average (`[+0.280, +0.307] dB`), reaching **`+0.571 dB` (DISTS `-0.0194`)** at `(1, 13, 13)` and **`+0.230 dB`** at `(13, 1, 1)`.
