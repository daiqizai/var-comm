# A4: resources, failures and state-conditioned quality

Original published result commit: `252176e041758ecb2d3e81b6fde5b587e7e17bb7`. All 500 sources and three noises per SNR are retained.
No inference, channel simulation, packet decoding, policy selection or bootstrap was performed.

The 16,500 original RAW physical frames expand to 18,000 WHOLE/PARTIAL memberships because the 7 dB policy is shared. They have two reconstruction arms. The continuous and adapted Swin baselines contribute 9,000 original frames each.

`resource_summary.csv` contains source bits, CRC/control fields, actual LDPC k/n, nominal/full-budget allocation, header/body/padding, frame-use efficiency and measured energy quantiles. WHOLE retains the same legal MCS/allocation permission as PARTIAL. Its selected configuration and padding do not establish that extra protection was prohibited.

Known RAW padding is energized QPSK carrying no image information; frame energy is the saved actual waveform energy, not an imposed 2048 value.

`token_error_fraction_compared` excludes absent token positions; no-overlap frames are missing for that diagnostic. `wrong_or_missing_planned_fraction` separately counts wrong or absent transmitted-prefix tokens. Neither is the transmitter planned prefix itself. All truth comparisons are offline diagnostics and do not change receiver decisions.

Body CRC rejection under KEEP is not image failure. Header rejection, wrong accepted header, CRC rejected KEEP, wrong accepted body and correct accepted body are retained separately. Zero-count states remain in the CSVs.

State-quality tables contain both frame-weighted subset means and means balanced over sources represented in that state. They are descriptive, have varying subset membership, and have no newly computed intervals. ALL_FRAMES retains the original published source-level intervals. ConvNeXt agreement is source-prediction agreement, not label accuracy; plotted values multiply the stored fractions by100. LPIPS is not sign-flipped.

The source-information/quality figure changes SNR along each line as well as the selected policy. It is descriptive and does not isolate a causal bit-allocation effect. Its error bars are the original published intervals. The state-quality figure uses frame-weighted subset means and omits only undefined zero-count means, preserving the zero-count CSV rows.

SwinJSCC is an adapted, budget-truncated 80k checkpoint. 19 dB remains outside its training and calibration range.

## Selected 10/19 dB facts

- Full-scale digital transmission, 10 dB: m=7, K=0, 64QAM, actual LDPC 1876/5628, header/body/padding=68/938/18; CRC-rejected KEEP=0/1500, gray=0/1500, mean actual energy=2048.447593.
- Partial-scale digital transmission, 10 dB: m=7, K=75, 16QAM, actual LDPC 2776/3824, header/body/padding=68/956/0; CRC-rejected KEEP=66/1500, gray=0/1500, mean actual energy=2048.705935.
- Full-scale digital transmission, 19 dB: m=8, K=0, 64QAM, actual LDPC 3076/3696, header/body/padding=68/616/340; CRC-rejected KEEP=0/1500, gray=0/1500, mean actual energy=2048.841804.
- Partial-scale digital transmission, 19 dB: m=8, K=142, 64QAM, actual LDPC 4780/5736, header/body/padding=68/956/0; CRC-rejected KEEP=0/1500, gray=0/1500, mean actual energy=2049.107399.
