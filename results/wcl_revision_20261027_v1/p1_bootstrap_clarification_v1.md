# P1 bootstrap metadata clarification

This companion note clarifies inherited description fields without changing the original registration, statistical metadata, resampling bank, scores or confidence intervals. It applies to the completed TX-policy SNR mismatch diagnostic, not to a new statistical run.

P1 compares **PARTIAL minus WHOLE**, with both TX policies fixed at 10 dB and true SNRs of 8, 10 and 12 dB. Each of the existing 100 sources first averages three receptions, including gray failures. The source-level difference is formed before resampling. WHOLE and PARTIAL share the **same actual AWGN vector** for each source/repeat, scaled to each true SNR; RX uses the true noise variance.

Only the existing 10,000 × 100 index bank and its matching source order were reused from the earlier task. Its seed is 2026100701 and SHA256 is `5e11cd2ee931a2de9f22f8e42034a71ef610eb83a48bac9e53224e666f5a447d`. No indices were drawn for P1, and this clarification performs no scoring or statistics.

The `bootstrap` object copied into `p1_registration.json` and `p1_delivery/statistics_identity.json` retains historical `paired_delta`, `noise`, `NeST19` and `old_method_scores_recomputed` descriptions. Those ARPC/other-baseline statements do not define P1. Likewise, `draw_matrix_calls: 1` describes the original bank creation; P1's own `new_random_draws: 0` is the current count. The actual P1 pairing is recorded by all 12 rows of `p1_delivery/paired_differences.csv` as `PARTIAL_minus_WHOLE`. Its source order matches the bank exactly. The original metric compute/reuse counts remain in `p1_delivery/actual_compute_and_reuse.json`.

## Preserved original files

| File | Bytes | SHA256 |
|---|---:|---|
| `results/wcl_revision_20261027_v1/p1_registration.json` | 14073 | `aa4a02575d5aa12110ab7c264303cdc8f3456971a1522538462abfd0dd1e4d90` |
| `results/wcl_revision_20261027_v1/p1_delivery/statistics_identity.json` | 11631 | `94565d3d6bf39f67a924f94f152d2e2ce7e30cf23379d38bd956aa9c1f3b5593` |
| `results/wcl_revision_20261027_v1/p1_delivery/paired_differences.csv` | 1117 | `1fc7884941d1a6760de7d04dba287e90e75bda520630dbac52a4d12a7e4b7bba` |
| `results/wcl_revision_20261027_v1/p1_delivery/source_means.csv` | 85315 | `a5f711f369c38efe47ca25f8f01139f938dee4c46bff998ccb0692b03b3531e3` |
| `results/wcl_revision_20261027_v1/p1_delivery/noise_identity.csv` | 36572 | `6e4e49bcf62bacbab56543185e7f3c1b7b38d7b683f8cdf632b4c14d345bd417` |

`p1_bootstrap_clarification_v1.json` records the same clarification and the exact noise-file and array identities. The original files above remain byte-for-byte unchanged.
