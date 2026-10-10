# Existing adaptive BPG resource evidence

All9000 actual holdout rows (500 sources x6SNR x3noise) were SHA-verified against their original completed resource and metric exports, joined by exact frame ID, and reconciled for source, workpoint, status, image/reference SHA, PSNR and actual waveform energy. All source/noise and failure rows remain. The frozen adaptive source/MCS rule is unchanged; MCS selection used100 calibration sources x3noise, not the holdout.

original/ contains unchanged source resource/failure CSV copies and the original completion receipts. per_frame.csv preserves raw CSV precision and adds the four existing metrics, E_frame and rho=E_frame/(2N). No model, codec, channel, decoder, metric network or bootstrap ran.

energy_summary.csv reports actual saved waveform E and rho: mean, population standard deviation(ddof=0), min/max and linear empirical5/50/95percentiles. SOURCE_UNFIT has no waveform; its energy/rho is NA and its count remains explicit. No ideal2N energy is inserted. Zero padding information bits are encoded inside the body and are distinct from physical unused symbols. Existing exact header-energy derivations are retained as such.

state_quality.csv gives all seven mutually exclusive final statuses, including zero-count states, and ALL_FRAMES. It includes frame counts, source counts, observed metric counts, frame means and source-balanced conditional means. Conditional means first average the selected noise outcomes within each represented source; varying noise counts are reported. These are diagnostic conditions, not fairer replacement test sets. ALL_FRAMES reuses the original published confidence intervals after reconciling the existing means. Conditional intervals remain empty: no resampling. ConvNeXt is source-prediction agreement, stored as a proportion, not classification accuracy. Original overlapping failure diagnostics remain in failure_breakdown.csv and must not be added together.

Observed final states: {'BPG_DECODED': 8985, 'BODY_CRC_REJECT': 15}. SOURCE_UNFIT rows: 0. No scientific input was modified.
