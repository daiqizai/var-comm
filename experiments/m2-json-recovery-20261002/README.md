# M2 Boolean JSON serialization recovery

The frozen M2 lambda selector subtracts NumPy means. Its selected candidate
records therefore contain exact `numpy.bool_` feasibility flags. Python's normal
JSON encoder rejects these flags after all 200 calibration screen sources have
been computed. The selector, candidates, numerical values and winners remain
unchanged.

This separately registered startup extension permits only exact `numpy.bool_`
values in `JSONEncoder.default`, returning the equivalent Python boolean. Every
other type retains the original encoder behavior. Native booleans and supported
numbers are unchanged; NumPy arrays, integers and float32 scalars remain
unsupported. The adapter imports no Torch and starts no GPU work.

Add this immutable runtime directory to `PYTHONPATH` and set
`M2_JSON_RECOVERY_MANIFEST` to its registered output `manifest.json`. The hook is
inert for the M2 controller, metric scorer, and every other entry point. It acts
only on the exact registered R4 `scheduled_process.py` with its original root,
admission and one of the four original stage arguments. An invalid manifest or
scope exits with code 78, because ordinary Python startup would otherwise ignore
a failing `sitecustomize` hook.

The manifest binds all recovery Python/Markdown files, frozen common/M2/wrapper
sources, and the entire R4 runtime. Execution records bind PID, start ticks,
stage and source hashes, with conversion counts at the first conversion and
process exit. They contain no credentials or complete environment dump, and do
not claim scientific completion. Existing calibration checksums, source fields,
runtime profiles and checkpoints are not rewritten. Only the failed scientific
queue needs to restart; the metric owner and final cache guardian keep their
existing identities. New Python source is published after scientific completion
so the original tracked-source registration remains unchanged while resuming.

## Reuse and publication

`materialize_gate_reuse.py` verifies all 200 original screen checkpoints and
extracts the original selector and reuse expressions without importing model
code. Its dry run compares every derived gate row against the frozen original
expression. Apply writes only equivalent first-200 gate caches, with their
original registration. Existing files must match; no newly inferred result is
claimed. This avoids generating a baseline that the original reuse branch
would discard.

`publish_when_complete.py` waits for M2, the unified metrics report, and the
final cache guardian to finish publishing and exit. It then publishes only the
explicitly registered recovery source and fixed evidence after repository
checks. It does not launch further scientific work. The original failure and
all completed screen/metric checkpoints are retained.
