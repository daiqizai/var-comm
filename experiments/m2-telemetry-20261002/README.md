# M2 telemetry query cache

This isolated startup extension reuses only two exact raw GPU telemetry query
outputs for at most one second from the original query's start. It changes no
model, inference function, policy, metric batch, precision flag or source record.
Original STOP, process admission and thermal checks still run at every boundary.
A newly appearing GPU process or changing temperature may be detected up to one
second later; the execution receipt states this bound.

Only cold temperatures below 75 degrees with `Not Active` hardware thermal
slowdown are cached. GPU process lists are cached only when every entry belongs
to this process or the live, registered R4 metric peer with matching identity and
parent/command/source proof. Each reuse checks that identity again. Unknown
processes, hot temperatures, malformed outputs and query errors invalidate all
entries. Exceptions are never replaced with stale data. Other subprocess calls
and keyword arguments retain their original behavior.

The hook applies only to the exact R4 M2 scheduled entry. It first runs the
separately pinned Boolean JSON recovery hook once. The controller and metric
scorer are inert. Formal timing leaves the raw subprocess function entirely
unwrapped and writes only disabled startup/exit receipts. A manifest or startup
failure exits 78 rather than being ignored by Python's site initialization.

`M2_TELEMETRY_MANIFEST` names the registered manifest; the original
`M2_JSON_RECOVERY_MANIFEST` remains required. All extension files, original M2,
R4, resource guards, Boolean recovery inputs and the real calibration benchmark
receipt are bound. Startup requires that benchmark to report exact equality and
at least 1.10 times faster execution for this complete source inventory.
Per-process execution
receipts record the source hashes, stage, PID/start ticks, raw query protocol and
fresh/hit/query-time statistics. The initial and first-query receipts are atomic;
counts also update every 30 seconds when queries occur, and final counts are
written at process exit. These are operational receipts,
not claims of scientific completion or future speedup. The independent
calibration-only benchmark must establish exact reconstruction agreement and
measured benefit before the operational extension is selected.
