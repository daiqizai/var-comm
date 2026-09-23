# Actual author parity and common-endpoint timing queue

The previous author qualification bound adapter 230ffbc; the audited development
used ec1be3a. This does not establish a model error, but the old PASS cannot certify
the later adapter. The new independent tools retain the actual development adapter,
weights, upstream source, paid protocol, sources and noise. No old adapter is edited.

`tools/run_author_reference_timing.sh` uses the pinned author Python environment
(torch 1.12.1+cu116), not current VAR's torch runtime. Its worker waits for actual
C/main-chain completion and exit, common archived metric scoring completion and
exit, and N4084 replay completion and exit. A process lock prevents duplicate
workers; PID/start ticks/command guard every child signal. Unknown failures stop.
The existing thermal thresholds apply: three hot samples request a safe boundary,
six cool samples at <=75C and no thermal flags permit resume. No hardware changes.

For each of Swin RA32/64/96 and ADJSCC C2/4/6, four real native checks use two
original calibration sources and SNR13/19 before that point's development timing.
The formal Swin RX uses decoded paid power/mask, never TX truth. Native execution
is only an offline parity diagnostic after checking actual metadata decoded
exactly. Legal CRC-rejected metadata remains usable under the original rule;
illegal metadata produces the original gray fallback. Bare ADJSCC has no header
and no transmitted power input to its RX. No diffusion inference is added.

Timing uses ten original development sources (0,11,...,99), all five SNRs, seed2001,
three full online warmups per source/method and two measured repetitions: 600
measured calls and 180 warmups. One shared TX/channel/RX function serves both parity
and timing. The CPU uint8 RGB and CPU I/Q endpoints include transfers and paid
metadata encoding/decoding. One whole-waveform noise realization is applied outside
RX timing. Every measured output is compared to the actual archived reconstruction
(max absolute error <=2e-5), with exact TX/data observation/standard noise hashes and
identical metadata/fallback decisions. Per-source/method seals bind resumptions.
All model parameters/buffers remain frozen; resource and energy assertions apply.

Natural resource labels stay unchanged: Swin total N4498/8754/12952 (metadata
402/562/664); ADJSCC N4096/8192/12288. These are not N3060/N4084 points. Six CPU
threads and three warmups align the endpoint procedure with current VAR, but the
older author runtime must remain disclosed, not described as runtime-controlled
speed superiority. The original 9000 frames' common LPIPS/DINO scoring stays with
the existing reference worker; this queue does not duplicate those quality metrics.

## Validation scope and incident

Five new synthetic CPU engineering checks cover single-noise ordering, decoded-only
RX information, CRC-rejected legal metadata, invalid metadata/NaN rejection, bare
ADJSCC information boundaries, natural resource accounting and predecessor release.
They are not quality metrics. Real CPU checks load all three ADJSCC models strictly
and deserialize/verify the Swin checkpoint on CPU. The Swin CPU constructor is not
supported: upstream `encoder.py:137` calls `.cuda()` unconditionally when updating
attention masks, even when the wrapper device is CPU.

The first CPU-constructor probe therefore briefly initialized CUDA. The budget
worker detected that process and safely saved P3060 at step11280; its checkpoint
hash was verified. The probe exited, the registered guard resumed, and training
progress beyond step11300 was verified. Original failure evidence is retained in
`outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/author_native_timing_v1/incidents/`.
The repeatable CPU tool now masks CUDA before importing torch; it never substitutes
mock weights or changes upstream code. Real GPU native parity and measured timing
remain NOT_RUN until the deferred stage actually produces verified receipts.

Runtime receipts: `outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/author_native_timing_v1/`.
After execution, review failure/delta receipts, natural-resource timing, source-paired
comparisons, and publish results. N3060 DeepJSCC/R3/digital full PHY/timing and the
final all-method report are still pending. This queue is not whole-study completion.
