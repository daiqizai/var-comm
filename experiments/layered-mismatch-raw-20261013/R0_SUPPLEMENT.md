# Separately authorized finite R0 screening supplement

The user approved the previously reported R0 gap on 2026-10-13. This addendum
supersedes the *waiting for a budget decision* status in the immutable initial
inventory. It does not change the original R1 or R2 limits or the GO rule.

The additional workload is exactly the missing selection evidence:

| Evidence | New maximum | Existing evidence retained |
| --- | ---: | --- |
| Body packet trials | 1,389 layout/SNR cells × 64 = 88,896 | 672 exact layout/SNR cells |
| Clean unconditional VAR reconstructions and DINO-L scores | 27 raster endpoints × 200 calibration sources = 5,400 | 3,400 verified clean source/endpoint rows |
| Source DINO-L embeddings for the new scores | 200 | Separate preprocessing count, no extra VAR image |

Only SNR 4, 7 and 10 dB enter this screening. There are no additional header
trials, 13 dB screening trials, tokenizer runs, training updates or bootstrap
calls. Original results are preserved. No development or holdout image informs
the shortlist. The 44 complete terminal states over 200 sources are joined by
exact source identity, not by population means.

The DROP-mixture screening formula and deterministic tie order were frozen
before new outcomes in `r0_screening.py`. The complete legal catalogue retains
all 433 single-group candidates, 201 unequal-protection candidates and 55
equal-protection candidates. IID scrambled-payload BLER and the correct-input
quality mixture are approximations. They do not provide actual KEEP quality,
a guaranteed quality bound, a global optimum or the R1 GO decision. The frozen
nominal and backoff anchors remain eligible for the actual R1 comparison.

## Bound execution

`r0_bler.py` runs the original qualified q2/q4/q6 encoder, Es2 modulation,
AWGN and receiver on two CPU cores. Every trial is reserved before transmission;
software failure leaves an unresolved attempt rather than becoming a scientific
CRC rejection. Exactly 64 trials are required for a completed point. The actual
CRC outcomes include correct, rejected and undetected-error packets. Historical
points and headers are not called again.

`r0_clean.py` renders the authorized clean raster prefixes using the original
unconditional VAR and frozen decoder, then scores DINOv2-L in a separate
process. It retains the qualified numerical settings. The 200 reference
embeddings and 5,400 reconstruction forwards are separately counted. Original
4090D clean evidence is reused as historical screening data; it is not relabeled
as a new H800 run. This clean-prefix calculation is not a channel-receiver
isolation test.

Each owner has a finite six-hour execution ceiling. Clean GPU admission requires
18 GiB free, with a private 16 GiB allocator cap and resource checks between
calls. Existing shared-H800 timing v6 is protected by a shared/exclusive lock
held by owners and inherited by children until actual wait. No other users'
processes are displaced. Scientific failures stop without automatic replay.

The final registrations and independent 48-test CPU/interface review are under
`results/layered_mismatch_raw_20261013/r0_supplement_preparation/` and
`r0_supplement_review_v1/`. Registration, a successful CPU test, dispatch or
resource WAIT is not scientific completion. Only actual exits, complete grids
and closed ledgers permit the later pure-data shortlist calculation. R1 and R2
require their own bounded execution and actual decision evidence.

Logs, journals and images remain under the personal `liberai-tmp/liulu` root.
Publication retains only permitted code and lightweight evidence. Existing
model, data and environment backup dependencies remain necessary.
