# Direction 7 (`codex/dir7-fas-persist-impulse-v1`): Zero-Pilot Fluid/Pixel Antenna Kriging, Non-Stationary Outage-Constrained VAR AMC, and Null-Gated Fast-Time Impulse Cancellation

## 1. Executive Motivation & Scientific Scope

Building upon both **2025–2026 6G Wireless & Semantic Communication literature** (Fluid/Movable/Pixel Antenna Systems, Adaptive Conformal Delayed-Feedback Control, and Industrial/Vehicular Non-Gaussian Impulsive Coexistence) and the newly landed `origin/main` roadmap (`commit 582038a`: *"Evaluate delayed channel feedback and outage-constrained image transmission"*), **Direction 7 (`experiments/dir7-fas-persist-impulse-20261124`)** investigates three strictly controlled, audit-hardened wireless semantic communication frontiers:

1. **`Dir-7A` (`blind_fas_jscc.py`) — Zero-Pilot 32-Port Fluid / Reconfigurable Pixel Antenna JSCC (`N_rf = 1`, `W = 2.5λ`, `CBR = 1/16`)**:
   - **Bottleneck**: A single-RF-chain (`N_rf = 1`) mobile receiver equipped with a 32-port sub-wavelength fluid or electronic pixel aperture (`W = 2.5λ`) can escape deep spatial standing-wave fades without multi-RF hardware cost—but conventional orthogonal port probing consumes `32/48 = 66.7%` of a short frame's symbol budget.
   - **Proposed Solution (`fas_kriging`)**: Unitary `24×32` 2D ISFFT spreads every image token across all `48` symbols × `256` subcarriers (`K = 12,288` active REs, `CBR = 1/16`). During the first `6` time tiles (`12/48 = 25%` of the frame, carrying **100% live semantic payload and 0 dedicated pilot slots**), the single RF chain visits `6` anchor ports `{0, 6, 12, 19, 25, 31}`, extracts blind Walsh-probe channel samples + intra-subband noise variance, reconstructs the entire `32`-port complex spatial profile via closed-form **Bessel-$J_0$ Wiener-Kriging**, and locks the remaining `36/48` (`75%`) symbols onto the optimal port $p^\star$.
   - **Strict Matched Controls**: `fas_kriging`, `fixed_port` (Port `0`), and `random_hop` (uniform cycle over the same `6` anchors without Kriging selection) share **byte-for-byte identical parameter counts, `state_dict` keys, initial weights, and paired mini-batch/channel draws**, with pure in-domain `val256` selection (`best_indist.pt`) + fixed final step (`last.pt`).

2. **`Dir-7B` (`persist_outage_amc.py`) — Directly Answering `origin/main` `582038a`'s #1 Standalone Paper Direction (`N = 1,024` Coded QAM Symbols, `CBR = 1/192`)**:
   - **Bottleneck**: Under delayed 2-bit CQI feedback (`2` bits every `2` frames = `1` bit/frame average, `delay = 2..3` slots), a stationary linear-program policy calibrated at slow persistence (`p_design = 0.95`, `Slow_LP_582038a`) suffers a severe `7.69% – 12.90%` grey-screen frame-rejection cliff when channel persistence drops (`p = 0.45..0.50`).
   - **Proposed Solutions** (zero neural retraining; frozen `VQVAE + VAR-d16`, calibrated strictly on `val256` ONLY):
     - **`PersistGuard_SinglePacket`**: Pure controller switch over the **exact same single-packet MCS × Prefix catalog** (`QPSK r=1/3..2/3`, `16-QAM r=1/2..3/4`, `64-QAM r=2/3..4/5`), combining an **Online Joint CQI + 1-Bit HARQ-NACK BOCPD/Wonham Persistence Filter** (`h = 0.025`) with **Adaptive Conformal Inference (`ACI`)** over pre-solved `val256` HiGHS LP mixtures.
     - **`PersistGuard_DualLayer_Anchor`**: Splits the exact same `960` payload symbols (`+64` header = `1,024` total) into a `384`-symbol `QPSK r=1/4` Base chunk (`m = 4`, `30` coarse tokens + `VAR-d16` causal completion) + `576`-symbol opportunistic Enhancement chunk (`scales 5..10`), eliminating grey-screen NULLs (`0.01%`) while boosting delivered semantic `DINOv2` across all regimes.

3. **`Dir-7C` (`impulse_shield_blind.py`) — Null-Gated Decision-Directed Fast-Time Impulsive Interference Cancellation (`CBR = 1/16`)**:
   - **Bottleneck**: Pre-FFT fast-time electromagnetic bursts smear non-Gaussian interference across all `256` subcarriers, while classical time-domain blanking clips legitimate high-PAPR OFDM peaks on clean frames (`-1.61 dB` penalty on `common500`!).
   - **Proposed Solution (`sem_impulse_2round`)**: Uses `2` orthogonal quiet Walsh null modes (`w_2, w_3`) per OFDM symbol as an unbiased $4.8\sigma$ non-Gaussian detector (`~1.5%` gate activation on clean AWGN -> zero PAPR self-clipping!) coupled with `2` fast-time `SafeSemanticLatentPriorNet -> IFFT re-modulation -> signal-cancelled residual soft-thresholding -> FFT` rounds + `1` post-peel polish, compared against **byte-identical parameter & 3-call submodule control `huber_owndenoise`**, end-to-end trained `classical_blanking`, `standard_blind`, and `oracle_genie_csi_ofdm`.

---

## 2. Quantitative Results (`common500` `N = 500` & `kodak24` `N = 24`, 3 Paired Seeds `1001, 2002, 3003`)

### 2.1 `Dir-7A`: Zero-Pilot 32-Port Fluid / Pixel Antenna JSCC (`common500`, `N_rf = 1`, `CBR = 1/16`)

| Condition (`common500`) | Method | PSNR (dB) ↑ | p10 Outage PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | DISTS ↓ | DINOv2 ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rayleigh `0 dB`** (`n_sb=1`) | **Blind-FAS Kriging (Ours)** | **27.286 ± 0.112** | **21.836** | **0.9370** | **0.1632** | **0.1746** | **0.8244** |
| | Random 6-Hop (Byte-Id.) | 24.203 ± 0.050 | 18.795 | 0.8738 | 0.2614 | 0.2278 | 0.6469 |
| | Fixed Port 0 (Byte-Id.) | 23.540 ± 0.096 | 16.353 | 0.8225 | 0.3143 | 0.2552 | 0.5603 |
| | 32-Pilot Exhaustive FAS | 18.206 ± 0.001 | 13.965 | 0.6827 | 0.5341 | 0.3840 | 0.3424 |
| | *Oracle Genie-Port+CSI* | *29.345 ± 0.009* | *23.957* | *0.9604* | *0.1072* | *0.1422* | *0.8952* |
| **Rayleigh `3 dB`** (`n_sb=1`) | **Blind-FAS Kriging (Ours)** | **28.721 ± 0.073** | **23.575** | **0.9586** | **0.1262** | **0.1535** | **0.8806** |
| | Random 6-Hop (Byte-Id.) | 26.224 ± 0.050 | 21.122 | 0.9215 | 0.1981 | 0.1943 | 0.7723 |
| | Fixed Port 0 (Byte-Id.) | 25.357 ± 0.238 | 18.065 | 0.8750 | 0.2474 | 0.2197 | 0.6751 |
| | 32-Pilot Exhaustive FAS | 17.311 ± 0.056 | 13.083 | 0.6489 | 0.5144 | 0.3849 | 0.3689 |
| **Rayleigh `6 dB`** (`n_sb=1`) | **Blind-FAS Kriging (Ours)** | **29.716 ± 0.034** | **24.746** | **0.9702** | **0.1039** | **0.1395** | **0.9098** |
| | Fixed Port 0 (Byte-Id.) | 27.296 ± 0.236 | 21.332 | 0.9245 | 0.1792 | 0.1827 | 0.7947 |
| **Rayleigh `10 dB`** (`n_sb=1`) | **Blind-FAS Kriging (Ours)** | **30.529 ± 0.019** | **25.800** | **0.9775** | **0.0870** | **0.1283** | **0.9292** |
| | Fixed Port 0 (Byte-Id.) | 28.678 ± 0.101 | 23.101 | 0.9498 | 0.1390 | 0.1591 | 0.8545 |
| **OOD Wideband Pixel-FAS** (`n_sb=4, 4 dB`) | **Blind-FAS Kriging (Ours)** | **28.976 ± 0.028** | **24.094** | **0.9631** | **0.1199** | **0.1499** | **0.8912** |
| | Random 6-Hop (Byte-Id.) | 26.666 ± 0.037 | 21.651 | 0.9318 | 0.1866 | 0.1888 | 0.8025 |
| | Fixed Port 0 (Byte-Id.) | 26.190 ± 0.058 | 21.110 | 0.9210 | 0.2046 | 0.1982 | 0.7536 |
| **OOD Rician `K=2`** (`n_sb=1, 4 dB`) | **Blind-FAS Kriging (Ours)** | **29.163 ± 0.008** | **24.225** | **0.9651** | **0.1151** | **0.1475** | **0.8968** |
| | Random 6-Hop (Byte-Id.) | 27.430 ± 0.011 | 22.556 | 0.9442 | 0.1625 | 0.1750 | 0.8329 |
| | Fixed Port 0 (Byte-Id.) | 26.894 ± 0.052 | 21.215 | 0.9241 | 0.1878 | 0.1882 | 0.7766 |

- **`3×3` Weight × Port-Policy Cross-Switch Audit (`common500`, `3 dB`)**: Taking `fixed_port`-trained weights (`25.357 dB`, `p10 = 18.065 dB`) and switching **only** the physical port policy to `fas_kriging` at inference immediately jumps `PSNR` to **`28.137 dB` (`+2.780 dB` zero-shot!)** and `p10` outage PSNR to **`23.388 dB` (`+5.323 dB` zero-shot!)**, proving the gain originates directly from Bessel-$J_0$ spatial Kriging rather than checkpoint or training confounders.
- **Fixed Final Step (`last.pt`) Audit**: `fas_kriging` beats `fixed_port` by **`+3.364 dB` (`p10 +5.510 dB`)** at `3 dB` and **`+2.786 dB` (`p10 +2.984 dB`)** on unseen 4-subband wideband pixel-FAS OOD.

---

### 2.2 `Dir-7B`: Delayed-Feedback Outage-Constrained Digital Multi-Scale VAR AMC (`common500`, `N = 1,024`, `CBR = 1/192`)

| Regime (`common500`, `18,000` frames/regime) | Policy | Frame Rejection (%) ↓ | PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ | DISTS ↓ | DINOv2 ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stationary Slow (`p=0.95`, Cap `2%`)** | Static Safe | 0.16 ± 0.03% | 15.116 | 0.4369 | 0.3708 | 0.2355 | 0.4229 |
| | Slow_LP (`582038a`, `p_des=0.95`) | 2.21 ± 0.24% | 17.354 | 0.5864 | 0.2760 | 0.1907 | 0.6175 |
| | **PersistGuard_Single (Ours)** | **1.49 ± 0.15%** | 16.828 | 0.5530 | 0.2946 | 0.1988 | 0.5754 |
| | **PersistGuard_DualLayer (Ours)** | **0.01 ± 0.01%** | **18.261** | **0.6387** | **0.2352** | **0.1642** | **0.6805** |
| **Slow->Fast Mismatch (`p=0.50`, Cap `2%`)** | Stale Greedy | 25.83 ± 0.54% | 16.278 | 0.5104 | 0.4291 | 0.3280 | 0.4817 |
| | Fixed-Margin AMC (`-3.5 dB`) | 21.96 ± 0.54% | 16.000 | 0.4961 | 0.4204 | 0.3140 | 0.4726 |
| | Slow_LP (`582038a`, `p_des=0.95`) | **12.90 ± 0.49%** *(6.5x violation!)* | 16.504 | 0.5319 | 0.3572 | 0.2597 | 0.5340 |
| | **PersistGuard_Single (Ours)** | **2.59 ± 0.31%** *(5.0x reduction!)* | 15.497 | 0.4653 | 0.3582 | 0.2346 | 0.4612 |
| | **PersistGuard_DualLayer (Ours)** | **0.02 ± 0.00%** | **17.129** | **0.5684** | **0.2850** | **0.1914** | **0.5934** |
| **Non-Stationary Switch (`0.95->0.45->0.95->0.50`, Cap `2%`)** | Stale Greedy | 15.56 ± 0.48% | 17.443 | 0.5785 | 0.3419 | 0.2567 | 0.5780 |
| | Slow_LP (`582038a`, `p_des=0.95`) | **7.69 ± 0.22%** *(3.8x violation!)* | 17.005 | 0.5631 | 0.3150 | 0.2246 | 0.5827 |
| | **PersistGuard_Single (Ours)** | **2.99 ± 0.13%** | 16.089 | 0.5035 | 0.3345 | 0.2230 | 0.5135 |
| | **PersistGuard_DualLayer (Ours)** | **0.01 ± 0.01%** | **17.710** | **0.6033** | **0.2603** | **0.1779** | **0.6381** |
| **Strict `1.0%` Cap Regime Switch** | Slow_LP (`582038a`, `p_des=0.95`) | **4.07 ± 0.21%** *(4.1x violation!)* | 16.172 | 0.5067 | 0.3386 | 0.2279 | 0.5135 |
| | **PersistGuard_Single (Ours)** | **1.56 ± 0.01%** | 15.629 | 0.4715 | 0.3517 | 0.2284 | 0.4710 |
| | **PersistGuard_DualLayer (Ours)** | **0.01 ± 0.01%** | **17.702** | **0.6030** | **0.2604** | **0.1779** | **0.6379** |

---

### 2.3 `Dir-7C`: Null-Gated Fast-Time Impulsive Interference Cancellation (`common500`, `CBR = 1/16`)

| Condition (`common500`) | SemImpulse-2Round (Ours) | Huber-OwnDenoise (Byte-Id.) | Trained Classical Blanking | Trained Standard Blind | Oracle Genie-CSI OFDM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean AWGN (`0%`, `14 dB`)** PSNR / DINOv2 | **28.070** / **0.8729** | 27.872 / 0.8692 | 26.457 / 0.8239 *(-1.61 dB!)* | 28.077 / 0.8655 | *31.096 / 0.9409* |
| **`4%` Imp (`SIR=-6 dB, 12 dB`)** PSNR / DINOv2 | **26.938** / **0.8204** | 26.641 / 0.8082 | 25.955 / 0.7945 *(-0.98 dB)* | 26.699 / 0.7970 | *27.205 / 0.8678* |
| **`6%` Imp (`SIR=-7 dB, 12 dB`)** PSNR / DINOv2 | **26.333** / **0.7912** | 25.901 / 0.7737 | 25.776 / 0.7833 *(-0.56 dB)* | 25.867 / 0.7492 | 25.129 / 0.8101 |
| **`8%` Imp (`SIR=-8 dB, 10 dB`)** PSNR / DINOv2 | **25.328** / **0.7386** | 24.827 / 0.7141 | 25.380 / 0.7569 | 24.688 / 0.6755 | 22.704 / 0.7119 |
| **`8%` Imp (`SIR=-9 dB, 14 dB`)** PSNR / DINOv2 | **25.598** / **0.7512** | 24.893 / 0.7180 | 25.678 / 0.7785 | 24.330 / 0.6485 | 21.888 / 0.6775 |
| **`12%` OOD (`SIR=-10 dB, 12 dB`)** PSNR / DINOv2 | **23.791** / **0.6369** | 23.027 / 0.5904 | 25.161 / 0.7419 | 22.331 / 0.4949 | 18.513 / 0.4671 |

- **Strict Matched-Control Verdict (`SemImpulse-2Round` vs Byte-Identical `Huber-OwnDenoise`)**: Across **every single condition (`6/6` on `common500` and `6/6` on `kodak24`, in both `best_indist.pt` and `last.pt`)**, enabling the physical fast-time `IFFT -> residual impulse subtraction -> FFT` loop improves PSNR monotonically as impulse severity rises: **`+0.198 dB -> +0.297 dB -> +0.432 dB -> +0.501 dB -> +0.705 dB -> +0.764 dB` (`+0.0465` DINOv2)** on `common500`, and surpasses `Oracle Genie-CSI Standard OFDM` by **`+1.20 dB` (`6%` imp) to `+5.28 dB` (`12%` OOD)**.
- **Transparent Physical Crossover vs `Trained Classical Blanking`**: On clean AWGN and medium-SIR overlapped impulses (`SIR = -6 ~ -7 dB`, where impulse amplitudes sit inside the OFDM PAPR envelope), `SemImpulse-2Round` beats `Classical Blanking` by **`+1.613 dB` (clean), `+0.982 dB` (`4%`), and `+0.557 dB` (`6%`)** because the quiet Walsh null gate completely avoids PAPR peak clipping. Conversely, at extreme `-10 dB` SIR (`12%` OOD giant spikes `>>` signal envelope), hard memoryless zeroing (`r -> 0`) removes giant spikes more aggressively than conservative soft-shrinkage—pointing directly to a Null-Gated Hard-Blanking + Residual Soft-Peeling hybrid for extreme industrial spikes.

---

## 3. Reproducibility, Storage Hygiene & Verification

- **Checkpoints**: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir7_fas_persist_impulse_v1`
- **Temp/Cache/Logs**: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir7_fas_persist_impulse_v1`
- **Unit Tests**: `29/29` passed across `Dir-2` through `Dir-7` (`pytest`).
- **Zero-Link Audit**: `symlink_count == 0` and `hardlink_count == 0` across worktree, checkpoints, and temp roots.
