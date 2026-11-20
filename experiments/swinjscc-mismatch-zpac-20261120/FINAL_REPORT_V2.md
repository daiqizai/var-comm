# Direction 2 V2 Complete Experimental Report: Multi-Scale CSI-Physics-Guided & Closed-Loop Uncertainty-Aware Transceiver Calibration (MS-ZPAC & TR-ZPAC)

- **Branch**: `codex/dir2-swinjscc-mismatch-v2` (built cleanly upon `codex/dir2-swinjscc-mismatch-v1` `f5bb7c4d`)
- **Hardware**: 4× NVIDIA A800-SXM4-80GB (`FP32`, `TF32=False`, deterministic algorithms)
- **PFS Storage Layout & Zero-Link Hygiene**:
  - Code, CSVs & figures: `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1`
  - V2 checkpoints (12 models): `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v2`
  - Runtime caches: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_swinjscc_mismatch_v2`
  - Verified `symlink_count = 0` and `hardlink_count = 0` across all PFS directories.

---

## 1. Root-Cause Diagnosis of V1 Ceiling & Three Architectural Breakthroughs in V2

1. **Universal Decoder Modulation Collapse + CSI Blindness**:
   Both official `SwinJSCC_w_SA_AWGN` and `SwinJSCC_w_SA_Rayleigh` (`C=96`) exhibit **`0.0000 dB`** sensitivity to receiver SNR $c_R$ in their 7-layer multiplicative `ChannelModNet`, while throwing away per-subblock pilot CSI $\hat{\mathbf{h}}$.
2. **I/Q Complex Packing Ghosting & Frozen Swin Stages**:
   `SwinJSCC` pairs the top half (`rows 0..7`) and bottom half (`rows 8..15`) of the $16\times 16$ latent grid into complex symbols $\mathbf{s}_{\text{re}} + j\mathbf{s}_{\text{im}}$. Under 32-RB OFDM frequency-selective fading and finite-pilot phase error, local $3\times 3$ convolutions on $16\times 16$ cannot see paired rows 8 units apart, nor can a frozen 4-stage Swin Decoder suppress multi-scale spatial artifacts. **`MS-ZPAC` (RX-Only, 100% Frozen Encoder)** introduces an **I/Q Conjugate-Paired Cross-Half Block** + **Per-Symbol Unbiased-MMSE Sufficient Statistics** + **4-Stage Swin Decoder Zero-Locked Spatial-Channel AdaLN** (`1.67M` params, `5.58%`).
3. **Closed-Loop Uncertainty-Aware Transmitter Adaptation (`TR-ZPAC`, Zero Extra Feedback Bits)**:
   Under delayed feedback ($d \ge 1$ blocks with temporal correlation $\rho_c$), feeding raw stale CQI into `SwinJSCC` causes `-4.8 to -6.6 dB` out-of-domain encoder collapse when $\hat{c}_{t-d} < 1\text{ dB}$ and severe overestimation cliffs when $\hat{c}_{t-d}$ was a transient peak. **`TR-ZPAC`** (`1.75M` params, `5.84%`, zero forward overhead bits) conditions a zero-locked risk-averse effective SNR + latent superposition power-shaping head on known feedback innovation uncertainty $\tau_T = 1 - \rho_c^{2d}$, preserving **exact `0.000000 dB` bit-exact equivalence** whenever $\tau_T = 0, \Delta c = 0$ on AWGN.

---

## 2. Held-Out Results on Suite B: Official `SwinJSCC_w_SA_AWGN` (`C=96`) Over 32-RB 5G/OFDM Delayed Fading & Wide AWGN (`ImageNet common500` × 3 Seeds)

| Scenario (`common500`, 1,500 frames/pt) | `Frozen` (Unbiased MMSE) | `AvgSNR` | `B1` (`4.94M`) | `P-Raw` (`[c_R,c_T,0]`) | **`MS-ZPAC` (v2 RX-Only)** | **`TR-ZPAC` (v2 Joint TX-RX)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matched AWGN ($\Delta c=0$)** | `30.899 dB` | `30.569` (`-0.330`) | `30.749` (`-0.150`) | `30.606` (`-0.293`) | **`30.899 dB` (`0.000`)** | **`30.899 dB` (`0.000`)** |
| **32-RB OFDM (10 dB, $\rho=0.90, d=0$)** | `24.767` (`p10: 18.92`) | `24.106` | `24.767` | `26.042` | **`27.237` (`+2.470 dB`)** | **`27.205` (`+2.438 dB`)** |
| **32-RB OFDM (10 dB, $\rho=0.90, d=2$)** | `24.764` (`p10: 18.96`) | `24.185` (`p10: 18.24`) | `24.764` | `26.009` (`p10: 21.16`) | **`27.349` (`+2.585 dB`, `p10: 22.48`)** | **`27.694` (`+2.930 dB`, `p10: 22.91`)** |
| **32-RB Fast OFDM (10 dB, $\rho=0.70, d=2$)** | `24.839` (`p10: 18.91`) | `24.294` (`p10: 18.24`) | `24.839` | `26.105` (`p10: 20.98`) | **`27.406` (`+2.567 dB`, `p10: 22.36`)** | **`27.740` (`+2.901 dB`, `p10: 22.89`)** |
| **32-RB Mid-SNR (7 dB, $\rho=0.85, d=2$)** | `22.413` (`p10: 16.78`) | `21.513` (`p10: 15.94`) | `22.413` | `24.109` (`p10: 19.32`) | **`25.580` (`+3.167 dB`, `p10: 20.72`)** | **`25.998` (`+3.585 dB`, `p10: 21.22`)** |
| **32-RB Low-SNR (4 dB, $\rho=0.85, d=2$)** | `19.178` (`p10: 14.03`) | `18.752` (`p10: 13.77`) | `19.178` | `21.263` (`p10: 16.65`) | **`22.692` (`+3.514 dB`, `p10: 18.07`)** | **`23.654` (`+4.476 dB`, `p10: 19.23`)** |

- **Perceptual Quality Simultaneous Win**: At 10 dB ($\rho=0.90, d=2$), `TR-ZPAC` improves MS-SSIM from `0.9181 -> 0.9505`, reduces DISTS from `0.1622 -> 0.1480` (`-0.0142`), and reduces LPIPS from `0.1568 -> 0.1244` (`-20.7%`)!
- **Zero-Shot Cross-Domain `Kodak24`**: `TR-ZPAC` gains **`+2.825 dB`** at 10 dB ($\rho=0.90, d=2$), **`+3.212 dB`** (`p10` **`+4.652 dB`**) under fast fading ($\rho=0.70, d=2$), and **`+4.695 dB`** (`20.086 -> 24.781 dB`) at 4 dB!
- **Static AWGN Mismatch (`common500`)**: `TR-ZPAC` achieves **`+0.732 dB`** (`[+0.678, +0.786] dB`, `+0.929 dB` over `P-Raw`) under Uncertain CSI and **`+0.355 dB`** (`MS-ZPAC`, `+0.451 dB` over `P-Raw`) on Unseen Extrapolation.

---

## 3. Held-Out Results on Suite A: Official `SwinJSCC_w_SA_Rayleigh` (`C=96`) (`ImageNet common500` × 3 Seeds)

| Scenario (`common500`, 1,500 frames/pt) | `Frozen` (`SA_Rayleigh`) | `AvgSNR` | `P-Raw` | `MS-ZPAC-ZeroMis` | **`MS-ZPAC` (v2 RX-Only)** | **`TR-ZPAC` (v2 Joint TX-RX)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **32-RB Rayleigh (10 dB, $\rho=0.90, d=0$)** | `23.945` (`p10: 18.85`) | `23.465` | `25.256` | **`23.945` (`0.000`)** | **`26.800` (`+2.855 dB`)** | **`27.296` (`+3.351 dB`)** |
| **32-RB Rayleigh (10 dB, $\rho=0.90, d=2$)** | `23.850` (`p10: 18.72`) | `23.460` | `25.250` | `26.232` (`+2.382`) | **`27.083` (`+3.233 dB`)** | **`27.588` (`+3.738 dB`, `p10: 22.87`)** |
| **32-RB Fast Rayleigh (10 dB, $\rho=0.70, d=2$)** | `23.893` (`p10: 18.59`) | `23.517` | `25.280` | `26.326` (`+2.433`) | **`27.146` (`+3.253 dB`)** | **`27.601` (`+3.708 dB`, `p10: 22.85`)** |
| **32-RB Mid-SNR (7 dB, $\rho=0.85, d=2$)** | `22.898` (`p10: 17.56`) | `22.420` | `24.235` | `24.992` (`+2.094`) | **`25.665` (`+2.767 dB`)** | **`26.170` (`+3.272 dB`, `p10: 21.42`)** |
| **32-RB Low-SNR (4 dB, $\rho=0.85, d=2$)** | `18.257` (`p10: 8.46`) | `20.802` | `20.420` | `20.758` (`+2.501`) | `21.403` (`+3.146 dB`) | **`24.041` (`+5.784 dB`, `p10: 19.59`)** |
| **Native i.i.d. Rayleigh Unseen Extrap** | Baseline | `+1.286` | `+0.233` | `+0.602` | **`+0.932 dB`** (DISTS `-0.0101`) | **`+1.068 dB`** (`[+1.001, +1.135]`) |
| **Native i.i.d. Rayleigh Uncertain CSI** | Baseline | `+1.297` | `+0.062` | `+0.511` | **`+0.733 dB`** (DISTS `-0.0115`) | **`+1.848 dB`** (`[+1.751, +1.944]`) |

---

## 4. Component Ablation & Cross-Chapter Multi-Budget `SA&RA` Bridge (`common500`)

1. **Monotonic Component Ablation (`awgn_common500`, 32-RB Fast OFDM $\rho_c=0.70, d=2$)**:
   - `Frozen`: `24.839 dB` | `B1` (`4.94M`): `24.839 dB` | `P-Raw` (`0.53M`): `26.105 dB` (`-0.293 dB` matched AWGN loss!)
   - `MS-ZPAC w/o 4-Stage Swin AdaLN` (`1.07M`): `26.498 dB` (`+0.393 dB` over `P-Raw`, **`0.000 dB` matched loss**)
   - `MS-ZPAC w/o I/Q Pair` (`1.62M`): `27.405 dB` (**`+0.907 dB` from 4-Stage Swin AdaLN, DISTS `-0.0184`, LPIPS `-0.0331`!**)
   - `MS-ZPAC` (Full RX-Only, `1.67M`): `27.406 dB` (`p10`: `22.283 -> 22.356 dB`, **`+0.073 dB` tail gain from I/Q pairing**)
   - `TR-ZPAC` (Full Joint TX-RX, `1.75M`): **`27.740 dB`** (`+0.334 dB` mean, **`+0.538 dB` `p10` tail gain** from `UA-TX`)!
2. **Multi-Budget `SwinJSCC_w/_SAandRA` Bridge (`N = 640, 1024, 2048` Total Complex Symbols)**:
   - Exact **`0.000 dB`** matched AWGN drift across all three budgets.
   - `N = 1024` (`k = 6`, aligned with Chapter 3 `NeST-Com`): **`+0.674 dB`** over `Frozen` (`+0.967 dB` over `AvgSNR`) on 4 dB OFDM fading, **`+0.423 dB`** on Uncertain CSI, **`+0.307 dB`** on Unseen Extrapolation.
   - `N = 2048` (`k = 14`): **`+0.864 dB`** over `Frozen` (`+1.225 dB` over `AvgSNR`) on 4 dB OFDM fading, **`+0.668 dB`** on Uncertain CSI, **`+0.609 dB`** on Unseen Extrapolation.
   - `N = 640` (`k = 3`, zero-shot cross-bandwidth transfer of `k=6` checkpoint): **`+1.220 dB`** on 32-RB fast OFDM fading (`p10` `+1.272 dB`, DISTS `-0.0396`, LPIPS `-0.0517`).
