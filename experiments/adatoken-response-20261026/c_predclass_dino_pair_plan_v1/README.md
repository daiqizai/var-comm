# Closed-calibration DINO pair join

This pure module joins already closed receiver results to the existing full
cal1000 source population. It performs no I/O, inference, strategy selection or
bootstrap. File/array checks, prior metric provenance and actual process closure
are mandatory in the separate admission preparer before calling build_plan.

Every source retains all six candidate/SNR choices and three noises. Exact pixel
identity can merge source-image pairs, including PHY-gray and source-parser-gray,
while every logical event retains its original status, frame pin and actual
output image pin. Missing output, unknown status or missing noise stops the join;
none is converted to gray. Pair reuse requires exact source archive, pixels and
an admitted frozen metric identity; the metric worker rechecks these bindings.

No source IDs are reindexed. The generated source plan contains only source
pixel/reference descriptors and received-output image descriptors, not tokens,
TX CDFs, class truth or source coding state. Originals are opened by the scorer
only after receiver output closure, for source-image distance computation.

The planned inference count is an upper bound before exact cross-source pair
reuse; the actual ledger is authoritative. Formal common500 and Kodak are not
part of this calibration plan. Do not run against active receiver arrays.
