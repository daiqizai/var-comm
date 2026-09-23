# N3060 reference waveform audit and real timing continuation

The actual archived transmissions and observations now pass a 6000-frame audit:
100 original development sources, five SNRs and three noise seeds for each of
raw adaptive VAR, arithmetic adaptive VAR, perceptual DeepJSCC and WeTok R3.
518 file bindings include original completion receipts, calibration-frozen policy,
selected weights, source files and per-source waveform archives. The complete
manifest stays server-local; source-level scalar/hash evidence is published in
`results/token_channel_efficiency_20260923/n3060_phy_audit_v1`.

Digital TX/observed hashes reproduce exactly. Raw and arithmetic protected headers
occupy 68 and 94 of N3060 uses; body CRC/tail/mother-code/rate-matched bits are checked
separately from source payload bits. The archive stores QPSK signs as int8; audit
restores the original float64 representation before comparing the historical SHA.
DeepJSCC and R3 use no header. Their actual stored FP32 observations match a single
registered AWGN application to within 5.852e-7 and 5.755e-7 respectively. Maximum
actual waveform energy deviations from E6120 are 0.001417 and 0.001413; digital
energy is exact. These are actual stored arrays, not synthetic replacements.
R3 thresholded bit metrics remain diagnostics, never a delivered bitstream claim.

Nine historical/current source identities differ, primarily repaired VAR helpers.
This is not proof that old model outputs are wrong. Exact archived-pixel and waveform
compatibility must be tested with the current inference before new timing is accepted.
The original quality tables and all failures are retained. Existing common metric
rescoring is already queued; do not duplicate those 6000 images' LPIPS/DINO work.

## Selected model loading and migrated source provenance

CPU loading with CUDA masked verifies the actual VAR/VQ, DeepJSCC, native WeTok and
selected R3 parameter/buffer hashes against their historical evaluation receipts.
It uses the existing backbone environment: torch2.11.0+cu128, einops0.8.1. The default
Python lacks einops, so no shared environment or ongoing task dependency was changed.
`tools/check_n3060_runtime_cpu.py` records the exact imported root module files.
This is strict real-weight loading, not a real image-forward or quality acceptance.

The legacy recursive training-settings loader rejects the pre-migration hash of
`wetok_comm/deep_support.py`. Commit e6fa036 only relocated that helper's dependency
lookup into the unique project root and retained explicit implementation SHA checks.
The new inference loader does not edit or relabel that historical training receipt.
Instead it verifies all ten exact completed R3 milestone source/config bindings,
both original parent asset SHAs, the selected checkpoint/step/variant, strict model
loading, and the complete resulting state hashes against the original quality run.
No new training, optimizer restoration or changed model selection occurs.

## Deferred real execution

`tools/run_n3060_reference_timing.sh` starts a separate process with an exclusive
worker lock. It waits for C/main, common reference metrics, N4084 replay and author
native/timing workers to complete AND exit; then the existing thermal-safe Runner
serially owns GPU0. PID/start ticks/command checks protect signals and recovery.
All active dependencies are bound, and unknown failures stop with preserved evidence.

The first 16 real checks compare two original development sources, SNR1/13 and all
four fixed methods against saved TX, observations and RGB. They perform compatibility
checks only, not development selection. RGB tolerance is 2e-5; digital waveforms
must match exactly and learned waveform differences must not exceed 2e-5.
Only after acceptance do ten predetermined sources (0,11,...,99), five SNRs,
seed2001, three warmups per source/method and two repetitions produce 400 measured
calls and 120 warmups. Every call also checks its saved counterpart.

The same full online TX/channel/RX functions serve qualification and timing. TX
includes source encoding, selected-mode arithmetic probabilities when applicable,
metadata and FEC. RX reads only the actual observed waveform and nominal SNR, with
paid digital headers providing its class/mode. R3 includes its full learned receiver
and native visual Decoder. Both ends use CPU RGB/IQ; one original-precision channel
noise application lies outside RX timing. Six CPU threads and FP32/TF32-off apply.
Per-source/method seals bind resume; final model-state hashes must remain unchanged.

CPU synthetic regressions validate audit rejection and queue gates. GPU acceptance
and timing remain NOT_RUN until actual receipts exist. This is a queued execution
milestone, not final study delivery. After all queued real stages, review deltas,
finish source-paired all-method statistics, natural-resource and matched-N plots,
quality targets/minimum-N savings, result index and final remote-verified report.
