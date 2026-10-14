# LE1 closed-prefix resource recovery

This entry is an append to the actual LE0/LE1 registration. The original entry,
registration, failed control, failure, streams and 24 independent RX proofs remain
unchanged. Its original deadline and scientific budgets remain unchanged.

The first segment ended because GPU3 free memory was 1959 MiB, below the 2048 MiB
floor. All 38 TX calls and all 24 independent RX proofs completed; no call is
unresolved. Model states and independent CDFs matched. This is a resource stop,
not a model/CDF/asset mismatch. The scope's completed-checkpoint recovery rule
allows only the 12 remaining TX sources and 450 missing clean endpoints.

The resume uses GPU1 on the same H800 host, an 8 GiB private allocator cap,
the unchanged startup floor 8192 MiB/utilization <=50%, running floor 2048 MiB,
and the original numeric flags, CPU affinity and frozen model mathematics.
This resource choice is not a timing or cross-device parity claim. The 24 source
proofs qualify the original GPU3 recipe. No additional source proof is run.

`resource_resume.py` modifies an in-memory copy of native.py at five exact,
inverse-AST checked sites: physical PCI metadata, allocator cap expression,
allocator metadata, and two ledger key namespaces. Original file bytes stay
fixed. A necessary second model load is separately disclosed as
`source_model_reload=1`; original model_load=1 stays in the global ledger.
All scientific counters remain cumulative in the original calls.sqlite.

First38 terminated streams were saved but the original driver had not flushed
per-endpoint finish increment metadata. Their full terminated stream lengths
and exact B+E-single segmentation cost remain recoverable. Individual finish
increments are left blank and explicitly labelled, never obtained by repeating
the successful 38 TX passes. New12 streams persist these increments immediately.

Run only the explicit registered control once. Qualification reads closed
bitvectors, metadata and AST; it does not load a model or run science. New unknown
failure must be preserved and diagnosed. Do not automatically retry.
