"""Direction 6: GPT-Audit-Grade Matched-Control High-Gain Semantic Communication Breakthroughs:
  - Dir-6A (burst_otfs_matched): Intermittent Time-Blockage + High-Doppler Doubly-Selective Symp-OTFS vs Byte-Identical Symp-OFDM-2D
  - Dir-6B (bayes_var_ultralow): Wireless-Posterior Guided VAR-d16 Next-Scale Autoregressive JSCC vs Loss-Matched Perceptual SwinJSCC at CBR=1/512 (384 syms) & 1/256 (768 syms)
  - Dir-6C (overloaded_noma_4x2): 200% Overloaded 4-User -> 2-Antenna Grant-Free Uplink with Blind CFAR Active-User Detection vs Byte-Identical OwnDenoise & Tuned Conventional PIC
"""
from .burst_otfs_matched import (
    BurstMatchedTransceiver,
    forward_genie_csi_ofdm_burst,
    physical_burst_doubly_selective_channel,
    sample_blockage_envelope,
)
from .bayes_var_ultralow import (
    BayesVARTransceiver,
    PerceptualSwinUltraLowTransceiver,
    load_frozen_vae_and_var_d16,
    ultralow_dir6_geometry,
)
from .overloaded_noma_4x2 import (
    OverloadedNOMA4x2Transceiver,
    build_walsh_16_basis,
    forward_genie_linear_noma_4x2,
    physical_grantfree_4x2_mac,
    tune_pic_damping_val256,
)

__all__ = [
    "BurstMatchedTransceiver",
    "physical_burst_doubly_selective_channel",
    "sample_blockage_envelope",
    "forward_genie_csi_ofdm_burst",
    "load_frozen_vae_and_var_d16",
    "ultralow_dir6_geometry",
    "BayesVARTransceiver",
    "PerceptualSwinUltraLowTransceiver",
    "build_walsh_16_basis",
    "physical_grantfree_4x2_mac",
    "OverloadedNOMA4x2Transceiver",
    "tune_pic_damping_val256",
    "forward_genie_linear_noma_4x2",
]
