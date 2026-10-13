# Independent CPU timing audit and manuscript tables

This entry reads saved receipts and ledgers. It never imports an experiment
owner, Torch, a model, a channel, or a bootstrap implementation. It must not
be used to restart timing. Original scientific files, registrations and
completed calls remain unchanged.

The implementation understands the pinned v5 registration
`44b14186a878621ea15e18da8cbf50e185927e6390826d019cff0865afc666f6`.
The actual v5 attempt stopped after 576 completed calls, with one reserved
entropy warmup. Therefore the default complete gate must reject that attempt.
The separately reviewed v6 successor is supported through an explicit
registration SHA and segment-binding audit. Inherited v5 calls are never
relabeled as v6; both actual waits and the diagnosed failure must remain.

## Commands and gates

Run in the repository root using the existing frozen visual Python interpreter
(NumPy is needed only for existing arrays and arithmetic summaries):

```text
python -B experiments/baseline-delivery-20261012/timing_postprocess_v1/postprocess.py partial-check
python -B experiments/baseline-delivery-20261012/timing_postprocess_v1/postprocess.py partial-report
python -B experiments/baseline-delivery-20261012/timing_postprocess_v1/postprocess.py check
python -B experiments/baseline-delivery-20261012/timing_postprocess_v1/postprocess.py run
```

For the v6 successor, use `check` or `run` with
`--input results/baseline_strength_20261011/timing/six_method_v6` and
`--expected-registration` followed by its independently reviewed actual
registration SHA. A v6 registration SHA is never inferred from a completed
result. Before actual successful completion, the check remains WAIT or STOP.

`check`/`run` require actual child exit 0, no timeout, the original owner gone,
the worker PID absent, the final shared-resource completion layer, and the
full 1,728-call grid. Missing closure returns `WAIT_ACTUAL_COMPLETION` (exit 2)
without writing. A failed receipt or mismatched evidence is an audit failure
(exit 1), never authorization to replay a case. The worker receipt does not
record a start tick, so an extant worker PID blocks conservatively even if it
may have been reused.

`partial-check`/`partial-report` instead require a real failed actual wait and
exited processes. They validate the immutable completed prefix and retain the
trailing reservation and failure scene. They do not produce six-method means.
A reservation's persisted packet count does not prove no in-flight work;
failure diagnosis and the original budget must account for any successor.

Read-only `check` commands never write. `run` writes only the new sibling
`results/baseline_strength_20261011/timing/six_method_v5_delivery_v1` (or the
corresponding `six_method_v6_delivery_v1` for v6) after all checks pass.
`partial-report` writes only the new sibling
`six_method_v5_partial_stop_v1`. Existing output directories are never replaced.

## Full audit and outputs

- Exact fixed 6-method x 16-source x 3-SNR x 6-repeat order, 864 measurement
  calls, no duplicate/reserved row, and digital packet sum no greater than 2,304.
- Actual source pixel/token hashes, frozen file/model/code hashes, all six
  fresh repeat identities, and per-call CSV equality with the persistent ledger.
- TX/RX/raw wall/audit subtraction, separately reported channel durations,
  measurement-only arithmetic means/medians/p95, and existing summary parity.
  No confidence intervals are created. Outer raw/corrected clocks are sampled
  separately; every signed residual is reported. A residual exceeding 1 ms is
  flagged for explanation, not treated as a protocol violation or replay trigger.
- Shared-resource admission and boundary snapshots, frozen FP32/TF32-off
  numerical settings, six CPU slots and 6/2 threads, same physical PCI device
  despite distinct CUDA/NVML UUID namespaces, and coordination-release hashes.
- For v6: exact inherited 576-row equality, reviewed recovery authorization,
  archived failed slot, new 1,152-row scope, the original 2,304-packet cap,
  and disclosure of 1,729 known fresh visual attempts/endpoint entries for
  1,728 completed calls. The failed warmup is not a completed warmup. Both
  segments' GPU/software identities are checked. v5 has 578 actual resource
  boundaries (including the failed case entry); v6's successful remaining
  loop has 1,154. The two histories are assessed separately.

Full outputs are `audit.json`, `qualification.csv`, `receiver_outcomes.csv`,
`timing_summary_all_components.csv`, `paper_timing_table.csv`, a LaTeX table
fragment, English and Chinese reports, a manuscript paragraph, and
`timing_boundaries.json`. The full summary includes the software channel as
a separate component and raw wall/provenance-exclusion components. Receiver
failures are retained. Component parameter counts do not claim to be per-arm
total-system counts.
For v6, `call_origins.csv` preserves each logical slot's v5/v6 segment and
original row hash. The reports explicitly disclose the stopped segment,
failed visual attempt and changing shared load across the two segments.

All times are **shared-environment latency**. Boundary samples are not continuous
resource surveillance. Fixed sequential method order and changing shared load
remain limitations. No speed ratio to another hardware platform, HiFi or BPG
is calculated, and no graph is generated by this entry.

## Model-free tests

```text
python -B -m unittest discover -s experiments/baseline-delivery-20261012/timing_postprocess_v1 -p test_postprocess.py -v
```

Synthetic fixtures exercise complete and partial grids, duplicate rows,
unresolved reservations, repeat/hash and packet failures, exact CSV parity,
warmup exclusion, timer accounting, failed/live-child gates, and shared-resource
device identity/guard violations. They do not simulate or execute a channel.
