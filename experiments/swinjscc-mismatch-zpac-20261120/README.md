# Direction 2: Zero-Locked Physics-Guided Asymmetric Calibration (ZPAC) for SwinJSCC Under Transceiver SNR Mismatch

This study implements **Direction 2** of the Master's thesis (*Robust Adaptive Visual Semantic Transmission Under Transceiver Channel Condition Mismatch*) on top of the official `SwinJSCC` (`SwinJSCC_w/_SA`, `base`, $C=96$, $\text{CBR}=1/16$, AWGN/Rayleigh) architecture.

## Core Scientific Design

1. **Non-Mutating $(c_T, \Gamma, c_R)$ Split Operator**:
   Separates transmitter channel condition $c_T$, physical wireless SNR $\Gamma$, and receiver condition estimate $c_R$, preserving exact numerical equivalence (`< 1e-6`) with upstream `SwinJSCC` when $c_T = \Gamma = c_R$.
2. **Zero-Locked Identity Gating**:
   $$\tilde{\mathbf{u}} = \mathbf{u} + \tanh(\gamma_+ \Delta c_+) R_\psi^+(\mathbf{u}, \bar{c}_R, \Delta c_+, \hat{\boldsymbol{\Xi}}, \mathbf{W}^{\text{Wiener}}) + \tanh(\gamma_- \Delta c_-) R_\psi^-(\mathbf{u}, \bar{c}_R, \Delta c_-)$$
   Guarantees strict mathematical identity ($\tilde{\mathbf{u}} \equiv \mathbf{u}$, `0.0000 dB` matched-condition drift) whenever $c_T = c_R$.
3. **Frequency-Decoupled Physics-Guided Asymmetric Branches**:
   - **Overestimation ($\Delta c_+ > 0, c_T > c_R$)**: Combines $3\times 3$ low/high spatial-frequency latent decomposition with scale-free posterior local SNR $\hat{\boldsymbol{\Xi}}$ and empirical Wiener shrinkage $\mathbf{W}^{\text{Wiener}}$ to suppress Gaussian noise on fragile detail channels while preserving structural edges.
   - **Underestimation ($\Delta c_- > 0, c_T < c_R$)**: Re-excites attenuated detail channels and boosts high-frequency spatial residuals via channel-SE + FiLM conditioning.

## Strict Data & Hygiene Boundaries

- **Training**: `train20k` (`20,000` ImageNet $256\times 256$ RGB images), sampling ONLY $\Delta c \in \{0, \pm 3, \pm 6\}\text{ dB}$.
- **Validation / Selection**: `dev64` (`calibration[0:64]` for Stage 0 diagnostics) and `val256` (`calibration[100:356]` for checkpoint selection).
- **Held-Out Test Sets (`holdout_for_selection = False`)**: `common500` (`500` ImageNet holdout images $\times 3$ paired noise seeds) and `kodak24` (`24` Kodak $256\times 256$ crops $\times 3$ paired noise seeds), evaluated across seen mismatch, **unseen interpolation ($\pm 2, \pm 4\text{ dB}$)**, **unseen extrapolation ($\pm 8, \pm 9, \pm 12\text{ dB}$)**, imperfect RX CSI, and **time-correlated AR(1) Rayleigh block fading with feedback delay $d \in \{0, 1, 2, 3\}$**.
- **PFS Hygiene**: Verified `0` symbolic links and `0` multi-hardlink files across code, checkpoints, and runtime directories.
