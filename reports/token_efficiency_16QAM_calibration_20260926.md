# 16QAM calibration delivery — 2026-09-26

The complete registered 16QAM calibration has finished and the original delivery chain has automatically entered development. This publication audits 1,000 original calibration sources × 29 eligible candidates × five SNRs × three noise seeds: **435,000 real frames in 29,000 sealed cells**. Full study delivery remains pending.

## Calibration policy

All 30 family/budget/SNR choices were independently recomputed from source-averaged MSE + 0.1 LPIPS, using the registered deterministic method-name tie break. DINO remains report-only. Selected requested modes, in SNR order 1 / 4 / 7 / 13 / 19 dB:

| Family | N | Requested m |
|---|---:|---|
| Arithmetic | 2048 | 6 / 7 / 8 / 9 / 10 |
| Arithmetic | 3060 | 6 / 8 / 9 / 10 / 10 |
| Arithmetic | 4084 | 7 / 8 / 9 / 10 / 10 |
| Raw | 2048 | 6 / 7 / 8 / 9 / 9 |
| Raw | 3060 | 7 / 8 / 9 / 10 / 10 |
| Raw | 4084 | 7 / 8 / 10 / 10 / 10 |

Arithmetic requested m is a cap; each frame retains its actual fallback mode and charged bits. The registered raw N2048/m10 candidate is unencodable (8,160 payload bits plus CRC/tail exceed 7,912 coded slots) and remains explicitly excluded in the 30-candidate ledger. The other 29 candidates are complete.

Policy SHA256: `e4e652bb01e620a9cfbf7a890709154f7e072cb2a8102e9477e97c8760c80755`.
Canonical frame-table SHA256: `87a84ab2cf0f15d30d3bdd5d788c4df0014fe6b0dd3159b2b410095cc84e11a2`.

[Policy and candidate scores](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/policy.json), [quality versus SNR](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/quality_snr_db.svg), [quality versus N](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/quality_N.svg).

## Failures and actual resources

All failures are retained: **7 header failures, 127,392 body-CRC failures, zero source-overflow erasures**. Failure categories may overlap. These counts cover all eligible calibration candidates, not only policy-selected methods.

N is the actual complex channel-use count, including the paid QPSK header (70 uses for raw, 96 for arithmetic) and the 16QAM body. Coded slots equal four times body uses. The original convolutional mother code has 2 × (payload bits + 22) bits; registered rate matching can puncture or repeat it. Mother-code length need not be less than transmitted coded slots. No mother-code or receiver change is introduced.

The fixed 16QAM constellation uses an average-symbol energy constraint, **not a per-frame E = 2N constraint**. Actual E is retained for every frame and separately summarized by method/SNR with mean, min, 5th/95th percentiles and max. Across the three budgets, observed E spans 3,822.4 to 8,665.6. The [energy plot](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/actual_energy.svg) uses SNR 1 dB because transmitted energy is determined before channel noise; the [complete energy table](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/energy_summary.csv) retains every SNR.

## Audit and publication

The independent CPU publisher checks completion/stage snapshot identity, all 64 registered bindings, every cell/seal, registered source order and preprocessing, candidate resources, run/context/noise identities, complete frame population, original per-frame CSV and summary hashes, and all policy candidate scores. Quality plots, source means and energy/failure summaries derive from the same complete canonical frame object.

The [publication index](../results/token_channel_efficiency_20260923/QAM16_calibration_v1/index.json) records 94 artifact hashes, including full per-frame bit/control/FEC ledgers split into byte-preserving CSV parts. Concatenate parts in manifest order; only the first has the CSV header. The reconstructed 367,930,555 bytes have SHA256 `39b727b6da9e1ec82518a9232c1c8801ae7d5300fba46d122442877ec5c3920b`. No pixels, weights or large runtime caches are published.

The earlier interrupted prefix remains historical evidence; the completed original grid, not a partial prefix or stale RUNNING status, defines this milestone. The publisher's initial engineering check incorrectly assumed mother-code bits could not exceed transmitted slots; inspection of the unchanged registered rate-matching implementation corrected that assumption before publication. A regression now preserves valid puncturing while rejecting identity and resource-ledger mismatches. This did not affect the GPU queue or experimental outputs.

## Scope still pending

Calibration is not development quality acceptance. This stage contains no online timing calls. No minimum-N or same-quality savings conclusion is drawn by comparing these 1,000 calibration sources with the 100 development sources. Development will apply the frozen policy and frozen quality targets, retain negative savings/unreached targets, and report paired source-image uncertainty separately from training-seed variability.

16QAM development, combined B1/B2 statistics, C training/evaluation and seed controls, four registered historical GPU metric/compatibility/timing workers, and the final merged comparison/report remain outstanding. The existing GPU0 ownership and thermal gates remain in force.
