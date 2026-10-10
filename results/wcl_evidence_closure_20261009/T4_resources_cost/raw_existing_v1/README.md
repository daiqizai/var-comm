# Existing RAW / latent JSCC / adapted Swin resources

This export reuses actual A4 v2 tables from the published common500 evaluation. All six source CSV hashes match the old completion receipt. No transmission, model reconstruction, metric evaluation or bootstrap was repeated.

`energy_summary.csv` retains the saved E_frame mean, standard deviation, min, max and5/50/95 percentiles. Its rho columns divide those same statistics by2N. This deterministic unit conversion does not establish strict equal-energy waveforms, shift the old quality curves or create new SNR measurements.

`resource_summary.csv`, `state_quality.csv` and `token_error_summary.csv` are exact original CSV bytes. `failure_breakdown.csv` groups the existing frame diagnostics by method/SNR/state. All frames, including rejected headers and CRC-rejected KEEP, remain represented. State subsets are descriptive; no new confidence intervals are calculated.

The scope is original raw whole/partial, P1024 and adapted Swin,500 sources×3 noises at six SNRs. BPG and new entropy families require their own explicitly bound records. This export does not claim those T4 additions or new entropy timing are complete. The selected19dB raw partial configuration remains m8,K142.
