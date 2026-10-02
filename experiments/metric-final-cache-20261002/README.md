# Verified cache assembly

This separate extension removes duplicate final reconstruction for rows already
verified by the immutable R4 sidecar. It changes neither the scientific methods,
their models, their policies, nor the metric definitions.

For N512, N1024, M1 and the source-only M1 rate curve, the final assembler joins
the current original table rows to sealed cache rows by exact registered row
identity. It checks source, preprocessing, reference, model, runtime and full row
coverage. Cached scalar metrics and the recorded real replay parity are reused;
the original metadata is rebuilt from the current frozen tables. The output
states that these rows reuse earlier replay verification. It never fabricates
RGB arrays or describes cached verification as a fresh replay. M2 and any
uncached rows still use the original deterministic reconstruction and metric
evaluation. Clean source-only rows retain seed zero and their original scope.

The CPU guardian waits for all 100 R4 source checkpoints to be sealed and the R4
metric owner to report `WAITING_FOR_ORIGINAL_COMPLETION` with no worker. It
identity-checks that owner, stops it through a pidfd, checks the idle condition
again, and retires only that owner. Its idle cache check admits only the existing
M2 owner and current child whose PID, start ticks, source bindings, commands and
registered parent/lease all match. Unknown runtime processes still block it.
Heartbeat timestamps can change before the stop; the final status proof is frozen
after the identity-checked owner has stopped. A changed condition resumes the owner and
stops the handoff. The M2 scientific worker and its supervisor are never signalled.
Original registrations and retired owner records are preserved.

After M2 is complete, normally pushed and all original/R4 workers and owners have
exited, the guardian publishes the unchanged R4 extension, publishes this new
extension, and runs final assembly followed by the original paired analysis and
result publisher. No additional experiment or queue is started after delivery.
Unknown failures stop the guardian. Scientific GPU timing remains exclusive;
this extension starts GPU evaluation only after the scientific queue has exited.

The prior R4 extension has its own 93-test engineering suite. This extension has
separate controller, publisher and assembly tests. Real cached-row qualification
is recorded in `real_cached_row_qualification.json`, with its original row,
checkpoint and source bindings. Engineering fixtures do not count as scientific
outputs. These checks do not establish future full-output coverage or a total
speedup; final receipts and actual elapsed times establish those separately.
