# Shared-inference resource continuation

The first source owner stopped before native Popen: GPU1 utilization36 exceeded
the old25 percent startup threshold, despite88041MiB free. Its actual wait1 and
failure are preserved, with no native launch, tx directory, model or science call.

The root authorization shared_inference_resource_authorization.json explicitly
permits the exact still-unlaunched source and later RX to use utilization<=80,
for this shared inference task. Memory admission64GiB, runningfloor8GiB,
allocator70GiB, GPU1 identity, all precision/thread/native math and budgets remain
unchanged. This is not a timing experiment. Four independent synthetic checks
verify exact pinned-code override, native-call parity and source-unstarted guard.

resource_continuation.py binds the immutable original control bytes and root
failure evidence. It writes a separate source_unlaunched_control_v1 chain and
rx_resource_control_v1 chain. Original source_control wait1 is never changed or
represented as wait0. The successful source native wait is a first scientific
execution. Both new native chains actually exited 0. Original PHY RX later
stopped at its final denied-attempt assertion after all saved packets/decodes
were complete; its exit1 is preserved. The separate `phy_finalize.py` CPU audit
reused the already reviewed saved-packet closure with zero repeated decodes.
The final `rx_crosshost_finalize.py` also preserves and explicitly discloses
native RX's 100 denied reads of the exact frozen program-source path; it never
changes the original receiver allowlist or the recorded denied-path list.
Fresh process checks on each actual host feed the final CPU-only closure.
The full failure history and limits are in the result `delivery/REPORT.md`.
