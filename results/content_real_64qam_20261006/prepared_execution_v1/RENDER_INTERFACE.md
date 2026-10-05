# H initial_true200 receiver image stage — prepared implementation

This new entry reconstructs only the closed, paid CPU reception traces for the
original fixed 200 sources and at most 16 frozen candidate/SNR slots, with noise
seeds 6101, 6102 and 6103. It makes no packet-decoder calls, policy choices,
development reads, training updates or new source exports.

## Invocation and admission

`python -B h_payload_render_driver.py --config /absolute/render_config.json`

The template is `render_config.template.json`. All placeholders require actual
absolute paths. Its four-hour default work cap is an engineering proposal for
the future registration, not an already admitted GPU run. The original absolute
96-hour H deadline remains unchanged.

The separate execution registration must have status
`H_EXECUTION_REVISION_REGISTERED`, branch `H`, and exactly `allowed_stage_ids:
["render"]`. Bind the render config, owner config, all named files and all
actually used old/new source modules by their original SHA. In particular,
bind every `m1_*.py` in `native_runtime`, the visual runtime's complete
`native.driver_bindings`, and the arithmetic/RX sources. The model identity
must equal the original calibration registration. Original FP32 and visual
thread flags are retained; the driver does not change them.

Before constructing a visual model, the entry verifies:

- Original S1 root/export receipts, 200 source identities and target pixel SHA.
- The CPU execution/config/merge receipt and all sealed source traces, including
  a fresh trace-to-ledger result-SHA audit. The immutable H ledger must be
  quiescent, with the finite initial stage using at most 19,200 paid calls.
- The CPU owner launch identity and every closed worker/merge receipt, output
  and closed log. The original owner and all children must have exited and
  been reaped. A zombie does not release the gate.
- This process must be the exact registered GPU job of a live frozen
  `stage_owner.py`, with matching PID/start time/UID/argv, affinity, CUDA device,
  launch receipt and its exclusive `GPU_IDLE_CONFIRMED` admission. The old
  visual owner holds its visual lock and original GPU health guard throughout.

The parent launch receipt has fields `identity`, `argv`,
`registration_sha256`, and `owner_config_sha256`. The separate visual owner has
one GPU stage `render` with one job, this exact CLI, and completion status
`H_INITIAL_TRUE200_RX_COMPLETE`.

## Receiver and cache

The receiver reads only actual `rx` events and their redundant public profile
record. It reparses the actual decoded bits and chooses the received 12-bit
public ID. It never substitutes TX tokens/profile, target pixels or a clean
reconstruction for an accepted wrong message. Arithmetic decoding uses fresh
receiver-generated CDF state, the frozen finite-tail rule and exact canonical
re-encoding. Protocol rejection gives gray; software/model/resource failures
stop and preserve evidence.

Within one source, identical actual received profile and payload may reuse a
result already computed by this same receiver in this process. The key also
binds catalogue and frozen visual/receiver identity. Every frame still reparses
its actual received bits before lookup. No preexisting clean-image cache is
read. A cache hit points to its origin event, preserves the canonical result,
and records that no new canonical/renderer call occurred. Repeated accepted
noncanonical payloads may likewise reuse their real canonical rejection.

This is deterministic offline computation reuse. Its call counts are not an
online receiver latency measurement. Source ID, target pixels, noise and TX
correctness are absent from the receiver cache key. Image deduplication only
shares identical verified float32 pixel arrays; it does not change scoring.

## Outputs

- `images/NNNN.npz`: exact float32 CHW RGB, one copy per unique image in that
  source. Each array is reread, validated and checked against its image hash.
- `sources/NNNN.json`: direct list of that source's frame rows.
- `source_checkpoints/NNNN.json`: exact input/output hashes and receiver counts.
- `frame_metrics.json`: direct list of every frame; no wrapper object.
- `frame_metrics.csv`: compact scalar table with LF line endings.
- `receiver_cost_counts.json`: canonical/renderer/cache counts with the offline
  timing limitation stated explicitly.
- `completion.json`: sealed science outputs only. Open logs, status, launch
  records and exit records are not included as science-output hashes. The
  enclosing owner closes and hashes its own log after the process exits.

Each row retains candidate ID, slot, arm, target m/K, modulation q, nominal rate,
source index/ID, SNR, noise seed, event ID, MSE, PSNR, image archive/key/SHA,
receiver-view SHA, complete `rx_summary` and post-reconstruction wire diagnostics.
Target conversion and PSNR exactly follow the frozen H source driver.

The only final source statuses are `RAW_SOURCE_DECODED`,
`ARITHMETIC_SOURCE_DECODED`, `WIRE_REJECT_GRAY`, and
`ARITHMETIC_SOURCE_INVALID_GRAY`. The last two are gray. All frames remain in
the scored population. The separate prepared selector consumes the raw bytes
of `frame_metrics.json`, the original shortlist and this completion; selection
does not occur here.

Completion status is `H_INITIAL_TRUE200_RX_COMPLETE`, with
`images_scored=true`, `source_decode_complete=true`, `new_packet_decodes=0`,
`development_used=false`, `policy_selection=false`, `source_count=200`, exact
`frame_count`, `shortlist_sha256`, registration/config hashes, science output
hashes, predecessor closure and visual launch bindings. This is not full H
delivery or permission to start another phase.

## Failure and validation limits

STOP and the finite stage/deadline guard stop at frame/source boundaries; the
original GPU health checks remain active. Partial attempts and failures are
preserved. The same output cannot be blindly retried; recovery needs separate
registration. Already complete output may only be read and reverified.

Local tests use synthetic tokens, the frozen pure arithmetic implementation and
fake visual functions. They cover received-input cache identity, canonical
origin, corrupt trace/archive rejection, complete candidate/noise coverage,
unchanged PSNR math, propagated model errors, and exact live-owner admission.
**Actual GPU execution, model-bound parity and runtime timing have not been
performed by this preparation task.** At 9,600 frames the undeduplicated RGB
payload upper bound is 7,549,747,200 bytes, plus container/evidence overhead.
