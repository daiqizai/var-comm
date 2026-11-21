# Direction 3 Technical Report: Zero-Pilot Spatial-Multiplexed MIMO-OFDM (`OS-MIMO-JSCC`) & Holographic Rateless / Anti-Jamming Semantic Communications (`HoloRate-JSCC`)

- **Branch**: `codex/dir3-mimo-holorate-v1` (branched from `codex/dir2-universal-blind-v4`)
- **Backbone**: Frozen official `SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model` (`28,187,746` frozen parameters; `0%` backbone weight modification)
- **Hardware**: 4× `NVIDIA A800-SXM4-80GB` (`FP32`, TF32 disabled, deterministic kernels)
- **Data Hygiene**: `train20k` (`20,000` images) for training; `val256` (`256` images) for checkpoint selection; `common500` (`500` ImageNet images × 3 paired seeds) & `kodak24` (`24` Kodak images × 3 paired seeds) strictly sealed (`holdout_touched = False` in all training manifests).
- **Storage Hygiene**: 100% under `liulu` personal PFS directories (`symlink_count = 0`, `hardlink_count = 0`).

---

## 1. Motivation & 2025–2026 Literature Survey

Following Directions 2 V1–V4 (closed-loop ZPAC $\to$ pilot-free SISO-OFDM `SPH-ZPAC` $\to$ universal residual-free `U-SPH-ZPAC`, 1-to-4 wireless broadcast, and discrete 16/64-QAM `DD-QAM-SPH`), two major physical-layer bottlenecks remain open in 2025–2026 semantic communications under severe spectrum and link constraints:

1. **Multi-Antenna Spatial Multiplexing Without Pilot Inflation (`Dir-3A`)**:
   - Recent MIMO-JSCC systems (*HANA-JSCC*, Gong et al., *IEEE TVT* Jul. 2026; Wu et al., *IEEE TWC* Oct. 2024; *Blind-MIMOSC* 2025) either require $N_t$ orthogonal pilot symbols per coherence block (whose overhead grows as $\mathcal{O}(N_t M)$) + closed-loop SVD precoding CSI, or suffer severe inter-stream interference (`~15–17 dB` ceiling) when attempting pure neural blind MIMO demixing due to unitary phase/stream ambiguity.
   - **Our Solution (`OS-MIMO-JSCC`)**: Embeds an **8-dimensional deterministic Walsh-Hadamard subspace** (`N_b = 384` REs per 5G NR PRB; `98.96%` semantic DoF for $2\times 2$ MIMO at `CBR=1/32` and `98.44%` DoF for $4\times 4$ MIMO at `CBR=1/64`) + **Real Ring Isomorphism $\Phi(\mathbf{H}_m) \in \mathbb{R}^{2N_r \times 2N_t}$ Gain-Unbiased Blind LMMSE** + **Per-Stream Post-Equalization SINR Calibration**, achieving $2\times$ (`50%` spectrum saving) and $4\times$ (`75%` spectrum saving) spatial multiplexing with **zero dedicated pilot slots and zero CSI feedback**.

2. **Anytime-Stopping Rateless Transmission, Contiguous Burst Erasures & Hostile Partial-Band Jamming (`Dir-3B`)**:
   - Recent resilient JSCC works (*ResiComp*, Wang, Dai, Qin et al., *IEEE TCSVT* Jul. 2025; *Every Packet Counts*, Wei et al., *ACM MM* 2026; *ResiGLC*, *IEEE TMC* 2026) highlight that standard SwinJSCC maps spatial tokens `[L=256, C=96]` sequentially into physical resource blocks (RBs). Consequently, rateless prefix truncation, contiguous deep-fade packet loss, or partial-band jamming wipes out entire spatial bands of the reconstructed image, and high-power jammers poison blind noise estimators.
   - **Our Solution (`HoloRate-JSCC`)**: Applies an **exact orthonormal 256-point Walsh-Hadamard spatial transform $\mathbf{H}_{256}$** with deterministic sign dithering across all $L=256$ spatial tokens and interleaves $3$ hierarchical channels (`3 × 256 = 768 reals = 384 complex REs`) into each RB, paired with **Blind Median-Null + Excess-Null + Kurtosis Jammer Excision (`w_jam`)** and **`HoloChannelCompletionBlock`**. Every single RB carries holographic global support over the entire image while maintaining exact unitarity (`atol < 1e-6`) at full rate.

---

## 2. Held-Out Quantitative Results (`ImageNet-500` & `Kodak24`)

### 2.1 Track A (`Dir-3A`): Zero-Pilot $2\times 2$ (`CBR=1/32`) & $4\times 4$ (`CBR=1/64`) MIMO-OFDM (`common500`)

| MIMO Config | Condition | `Frozen_Blind` (0-P) | `BlackBox_Blind` (0-P) | `Pilot4_LMMSE` (128-P) | `Genie_SVD` (TX+RX CSI) | **`OS-MIMO-JSCC` (Ours, 0-P)** | Gain vs `BlackBox` | Gain vs `Pilot4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$2\times 2$ (`CBR=1/32`)** | i.i.d. `1 dB` | 8.718 | 16.162 | 15.406 | 16.165 | **22.631** | **+6.469 dB** | **+7.226 dB** |
| **$2\times 2$ (`CBR=1/32`)** | i.i.d. `4 dB` | 8.805 | 16.469 | 18.346 | 17.859 | **24.801** | **+8.332 dB** | **+6.455 dB** |
| **$2\times 2$ (`CBR=1/32`)** | i.i.d. `7 dB` | 8.874 | 16.627 | 21.390 | 20.064 | **26.585** | **+9.958 dB** | **+5.195 dB** |
| **$2\times 2$ (`CBR=1/32`)** | i.i.d. `10 dB` | 8.913 | 16.699 | 24.258 | 22.877 | **28.063** | **+11.364 dB** | **+3.805 dB** |
| **$2\times 2$ (`CBR=1/32`)** | i.i.d. `13 dB` | 8.940 | 16.728 | 26.686 | 25.699 | **29.256** | **+12.528 dB** | **+2.571 dB** |
| **$2\times 2$ (`CBR=1/32`)** | **OOD $\rho_{\text{ant}}=0.60$, `7 dB`** | 8.785 | 16.425 | 18.843 | 16.219 | **24.157** | **+7.731 dB** | **+5.314 dB** |
| **$2\times 2$ (`CBR=1/32`)** | **OOD $\rho_{\text{ant}}=0.60$, `10 dB`** | 8.827 | 16.474 | 21.478 | 18.694 | **26.019** | **+9.546 dB** | **+4.542 dB** |
| **$4\times 4$ (`CBR=1/64`)** | i.i.d. `1 dB` | 8.773 | 14.965 | 15.937 | 15.042 | **22.597** | **+7.632 dB** | **+6.660 dB** |
| **$4\times 4$ (`CBR=1/64`)** | i.i.d. `4 dB` | 8.896 | 15.047 | 18.824 | 16.261 | **24.788** | **+9.741 dB** | **+5.964 dB** |
| **$4\times 4$ (`CBR=1/64`)** | i.i.d. `7 dB` | 8.970 | 15.087 | 21.742 | 17.788 | **26.508** | **+11.421 dB** | **+4.766 dB** |
| **$4\times 4$ (`CBR=1/64`)** | i.i.d. `10 dB` | 9.016 | 15.105 | 24.470 | 19.892 | **27.839** | **+12.733 dB** | **+3.368 dB** |
| **$4\times 4$ (`CBR=1/64`)** | i.i.d. `13 dB` | 9.040 | 15.114 | 26.639 | 22.454 | **28.902** | **+13.788 dB** | **+2.263 dB** |
| **$4\times 4$ (`CBR=1/64`)** | **OOD $\rho_{\text{ant}}=0.60$, `7 dB`** | 8.816 | 15.314 | 18.399 | 14.559 | **23.498** | **+8.184 dB** | **+5.099 dB** |
| **$4\times 4$ (`CBR=1/64`)** | **OOD $\rho_{\text{ant}}=0.60$, `10 dB`** | 8.861 | 15.324 | 20.817 | 16.087 | **25.226** | **+9.902 dB** | **+4.410 dB** |

*(On `Kodak24`, `OS-MIMO-JSCC` achieves `30.157 dB` on $2\times 2$ 13 dB MIMO and `29.470 dB` on $4\times 4$ 13 dB MIMO at `CBR=1/64`.)*

### 2.2 Track B (`Dir-3B`): Rateless Anytime-Stopping, Burst-Erasure & Partial-Band Anti-Jamming (`common500`, 10 dB 0-Pilot OFDM)

| Category | Condition (`10 dB` 32-RB OFDM) | `V4_U_SPH_ZeroShot` | `Raster_Blind_Resilient` | **`HoloRate_JSCC` (Ours)** | Gain vs `V4_U_SPH` | Gain vs `Raster_Resilient` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rateless** | `25%` Prefix (`8/32` RBs, `CBR=1/64`) | 12.095 | 15.597 | **22.298** | **+10.203 dB** | **+6.701 dB** |
| **Rateless** | `50%` Prefix (`16/32` RBs, `CBR=1/32`) | 16.117 | 18.022 | **25.184** | **+9.067 dB** | **+7.162 dB** |
| **Rateless** | `75%` Prefix (`24/32` RBs, `CBR=3/64`) | 20.850 | 22.130 | **27.244** | **+6.394 dB** | **+5.114 dB** |
| **Rateless** | `100%` Full (`32/32` RBs, `CBR=1/16`) | 28.816 | 28.694 | **29.256** | **+0.441 dB** | **+0.562 dB** |
| **Erasure** | `25%` Contiguous Burst (`8` RBs Lost) | 21.991 | 22.973 | **26.925** | **+4.934 dB** | **+3.952 dB** |
| **Erasure** | `37.5%` Random RB Drop (`12` RBs Lost) | 20.445 | 21.584 | **25.480** | **+5.035 dB** | **+3.895 dB** |
| **Erasure** | `50%` Contiguous Burst (`16` RBs Lost) | 16.889 | 18.996 | **24.582** | **+7.693 dB** | **+5.586 dB** |
| **Jamming** | `25%` Band Jammed (`+15 dB` INR) | 13.458 | 23.508 | **26.835** | **+13.377 dB** | **+3.326 dB** |
| **Jamming** | `37.5%` Band Jammed (`+15 dB` INR) | 12.304 | 21.694 | **25.517** | **+13.213 dB** | **+3.823 dB** |
| **Jamming** | **OOD `31%` Band Jammed (`+18 dB` INR!)** | 12.442 | 22.561 | **26.172** | **+13.729 dB** | **+3.610 dB** |

*(All paired 95% bootstrap confidence intervals across all 24 comparisons in Track A & Track B exclude zero with $p < 0.0002$.)*
