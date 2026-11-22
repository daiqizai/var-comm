# Direction 4 Technical Report: RF Hardware-Impaired 1-Bit/2-Bit Blind Semantic Transceiver (`RF-Bussgang-JSCC`) & Grant-Free Overloaded Multi-User Semantic SIC (`SemSIC-NOMA-JSCC`)

- **Branch**: `codex/dir4-rf1bit-noma-v1` (branched from `codex/dir3-mimo-holorate-v1`)
- **Backbone**: Frozen official `SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model` (`28,187,746` frozen parameters; `0%` backbone weight modification)
- **Hardware**: 4× `NVIDIA A800-SXM4-80GB` (`FP32`, TF32 disabled, deterministic kernels)
- **Data Hygiene**: `train20k` (`20,000` images) for training; `val256` (`256` images) for checkpoint selection; `common500` (`500` ImageNet images × 3 paired seeds) & `kodak24` (`24` Kodak images × 3 paired seeds) strictly sealed (`holdout_touched = False`).
- **Storage Hygiene**: 100% under `liulu` personal PFS directories (`symlink_count = 0`, `hardlink_count = 0`).

---

## 1. Motivation & 2025–2026 IEEE Literature Positioning

1. **Dir-4A (`RF-Bussgang-JSCC`) — Ultra-Low-Resolution `1-Bit / 2-Bit` ADCs + Nonlinear Power Amplifier Saturation + Wiener Phase Noise**:
   - In edge IoT, UAV, and mmWave transceivers, analog OFDM semantic waveforms suffer high Peak-to-Average Power Ratio (`PAPR ~ 10.6 dB` in `V4 U-SPH`), triggering severe nonlinear clipping in low-cost solid-state power amplifiers (Rapp HPA, `IBO = 0..1 dB`), while receiver power budgets motivate replacing power-hungry high-bit ADCs and analog AGC loops with **1-Bit Zero-AGC Sign Comparators** (`{±1 ± j}/√2`) or **2-Bit Mid-Rise ADCs**, under intra-slot local-oscillator Wiener phase noise (`3.5°–5.5°` per OFDM symbol).
   - **Our Solution (`RF-Bussgang-JSCC`)**: Combines (i) **Newman quadratic phase coding** across 32 RBs + **differentiable time-domain PAPR companding with exact post-compand `u_2` null restoration** (cutting TX PAPR from `10.59 dB` to `7.37–7.44 dB`), (ii) **32-RB coherent Common-Phase-Error (CPE) trajectory tracking** across `48` OFDM symbols (`4` temporal sub-blocks per RB), and (iii) **1-bit-invariant `atan2` phase-angle scale recovery + Gain-Unbiased Bussgang MMSE equalization** (automatically cancelling unknown Bussgang attenuation $\alpha_{\text{ADC}}\alpha_{\text{PA}}$ with `0` pilots and `0` AGC).

2. **Dir-4B (`SemSIC-NOMA-JSCC`) — Grant-Free Zero-Pilot Near-Far & Overloaded `3-User -> 2-BS-Antenna` Non-Orthogonal Uplink**:
   - In multi-agent/multi-camera edge networks, $U$ single-antenna terminals transmit independent images simultaneously on shared `100%` spectrum without scheduling grants, power control, or dedicated orthogonal pilot slots. Under **150% spatial overloading ($U=3 > N_r=2$)** and severe Near-Far disparity (`12–14 dB` vs `2–4 dB`), even **Exact Oracle `Genie-CSI Linear MMSE`** hits a rank-deficiency interference ceiling (`13.90 dB` on the weak `4 dB` user and `12.57 dB` on the OOD `2 dB` user).
   - **Our Solution (`SemSIC-NOMA-JSCC`)**: Embeds a **collision-free 8-dimensional Walsh signature bank** (`u_0, u_7` universal quiet nulls + `u_{1+u}` orthogonal per-user channel probes + `u_{4+u}` per-user `atan2` scale probes; `97.92%` shared non-orthogonal payload DoF) paired with **2-Round Physical-Semantic Iterative Interference Cancellation (`SemanticLatentPriorNet` + Covariance-Updated Soft I/Q Subtraction)**.

---

## 2. Sealed Held-Out Quantitative Results (`ImageNet-500` & `Kodak24`)

### 2.1 Track A (`Dir-4A`): 1-Bit & 2-Bit Hardware-Impaired OFDM (`common500`)

| ADC & Hardware Regime | `Frozen_RF` (0-P) | `Pilot4_Bussgang` (128-P) | `V4_USPH_ZeroShot` (0-P) | `BlackBox_RF` (0-P E2E) | **`RF_Bussgang_JSCC` (Ours, 0-P)** | Gain vs `V4_USPH` | Gain vs `BlackBox_RF` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Bit**, `1 dB` (`1 dB` IBO, `3.5°` PN) | 8.552 | 11.936 | 13.894 | 20.161 | **21.105** | **+7.211 dB** | **+0.944 dB** |
| **1-Bit**, `4 dB` (`1 dB` IBO, `3.5°` PN) | 8.638 | 12.255 | 15.391 | 21.398 | **22.453** | **+7.063 dB** | **+1.056 dB** |
| **1-Bit**, `7 dB` (`1 dB` IBO, `3.5°` PN) | 8.660 | 12.389 | 16.672 | 22.242 | **23.322** | **+6.649 dB** | **+1.079 dB** |
| **1-Bit**, `10 dB` (`1 dB` IBO, `3.5°` PN) | 8.671 | 12.525 | 17.613 | 22.707 | **23.841** | **+6.228 dB** | **+1.134 dB** |
| **1-Bit**, `14 dB` (`1 dB` IBO, `3.5°` PN) | 8.690 | 12.689 | 18.283 | 23.013 | **24.209** | **+5.926 dB** | **+1.196 dB** |
| **1-Bit OOD Extreme** (`10 dB`, `0 dB` IBO, `5.5°` PN!) | 8.644 | 12.192 | 16.458 | 22.324 | **23.808** | **+7.349 dB** | **+1.483 dB** |
| **2-Bit**, `4 dB` (`1 dB` IBO, `3.5°` PN) | 8.718 | 14.455 | 18.862 | 23.378 | **24.283** | **+5.420 dB** | **+0.904 dB** |
| **2-Bit**, `10 dB` (`1 dB` IBO, `3.5°` PN) | 8.817 | 15.453 | 21.738 | 25.543 | **26.466** | **+4.729 dB** | **+0.924 dB** |
| **2-Bit**, `14 dB` (`1 dB` IBO, `3.5°` PN) | 8.825 | 15.915 | 22.469 | 26.200 | **27.166** | **+4.697 dB** | **+0.966 dB** |
| **2-Bit OOD Extreme** (`10 dB`, `0 dB` IBO, `5.5°` PN!) | 8.833 | 14.582 | 20.288 | 25.106 | **26.364** | **+6.075 dB** | **+1.257 dB** |

*(On `Kodak24`, `RF-Bussgang-JSCC` reaches `25.209 dB` on 1-Bit `14 dB` / `24.798 dB` on 1-Bit OOD Extreme, and `28.016 dB` on 2-Bit `14 dB` / `27.213 dB` on 2-Bit OOD Extreme.)*

### 2.2 Track B (`Dir-4B`): Grant-Free `2×2` & Overloaded `3×2` Non-Orthogonal Uplink (`common500`)

| Uplink Scenario (`100%` Shared Spectrum) | Evaluated Role | `Uncoord_V4` | `BlackBox_NOMA` | `Pilot4_MMSE` | `Genie_Linear_MMSE` | `Stage1_Only` (No SIC) | **`SemSIC_NOMA` (Ours)** | Gain vs `Genie_Linear` | Pure `Sem-SIC` Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2×2` Balanced (`10/10 dB`)** | 2-User Mean | 12.373 | 15.283 | 20.161 | 19.810 | 29.249 | **29.415** | **+9.604 dB** | +0.166 dB |
| **`2×2` Near-Far (`12/3 dB`)** | **Weak User (`3 dB`)** | 9.277 | 9.951 | 18.507 | 20.506 | 24.883 | **25.194** | **+4.688 dB** | +0.312 dB |
| **`2×2` OOD Near-Far (`14/2 dB`)** | **Weak User (`2 dB`)** | 8.978 | 9.367 | 17.453 | 19.735 | 23.850 | **24.117** | **+4.382 dB** | +0.267 dB |
| **Overloaded `3×2` (`13/8/4 dB`)** | Strong User 1 (`13 dB`) | 15.199 | 18.682 | 19.769 | 19.342 | 28.935 | **29.328** | **+9.986 dB** | +0.393 dB |
| **Overloaded `3×2` (`13/8/4 dB`)** | Medium User 2 (`8 dB`) | 10.051 | 11.573 | 17.461 | 17.798 | 24.292 | **25.420** | **+7.622 dB** | **+1.128 dB** |
| **Overloaded `3×2` (`13/8/4 dB`)** | **Weak User 3 (`4 dB`)** | 9.340 | 10.260 | 13.501 | 13.903 | 18.523 | **21.238** | **+7.335 dB** | **+2.715 dB** |
| **Overloaded `3×2` (`13/8/4 dB`)** | **3-User Mean** | 11.530 | 13.505 | 16.910 | 17.014 | 23.917 | **25.329** | **+8.314 dB** | **+1.412 dB** |
| **Overloaded `3×2` (`12/9/6 dB`)** | **Weak User 3 (`6 dB`)** | 9.708 | 10.878 | 14.570 | 14.973 | 20.090 | **22.644** | **+7.671 dB** | **+2.554 dB** |
| **Overloaded `3×2` OOD (`14/8/2 dB`)** | **Weak User 3 (`2 dB`!)** | 9.107 | 9.840 | 12.202 | 12.566 | 16.465 | **19.369** | **+6.803 dB** | **+2.904 dB** |
| **Overloaded `3×2` OOD (`10/10/10 dB`)** | **3-User Mean** | 10.923 | 13.065 | 17.716 | 17.891 | 24.271 | **25.853** | **+7.963 dB** | **+1.582 dB** |
