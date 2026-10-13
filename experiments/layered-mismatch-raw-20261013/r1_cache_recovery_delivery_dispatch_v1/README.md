# Once-only CPU delivery dispatcher for recovered R1

This dispatcher is separate from all scientific owners and from the original
CPU delivery dispatcher. It does not launch models, channels, training, selection
or R2. It refuses to create a control directory or a child until the reviewed
recovery delivery entry's full process preflight passes: old PHY/projection
exit zero, preserved old owner/visual exit one, new owner and all three new stages
exit zero, exact bindings, no new failure, all process generations gone, and
unheld original/new locks. It does not poll until success or automatically retry.

The recovery master is
`ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c`.
The CPU entry is `r1_cache_recovery_delivery_v1/deliver.py`, SHA
`f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01`.
The process/timeout/receipt implementation is reused from the reviewed original
dispatcher SHA `3f32be6e1d63205cf4d71d7e0cddc95d6f314f14628c8cf08c8a79b432a12748`.
Only the new fixed registration/source/schema/output/control names and the
mixed-segment preflight call differ. The original dispatcher remains unchanged.

After genuine scientific closure, the intended command is:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_dispatch_v1/dispatch_once.py --expected-registration ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c
```

Control is exclusively created at the personal temporary root's
`r1_cache_recovery_delivery_dispatch_v1`. Existing control, delivery output,
publication claim or publication failure is preserved and rejected. A lock is
held through the actual child wait and inherited by the child. The child runs
the original CPU Python with hidden CUDA, two CPU math threads, no user site and
bytecode disabled. Its 7,200-second wall limit is CPU audit/delivery time, not an
increase in any scientific budget. Timeout triggers terminate, a 15-second wait,
then kill if needed, with the real final wait retained.

`prepared.json`, `launch.json` and `actual_wait.json` use schema
`R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1`. Each binds the registration, delivery
entry, dispatcher, exact argv, owner PID/start, hidden CUDA, zero scientific
calls and finite wall limit. Launch and wait bind the child PID/start. Wait also
records real exit status, timeout/termination flags, metadata errors, elapsed
time and stdout/stderr SHA pins. A launch error is not labelled an actual wait.
Any failure is retained in `failure.json` and prevents replay.

Success requires `actual_wait: true`, a strict integer exit code zero, no timeout
or termination flags, `metadata_error: null`, no control/publication failure,
both dispatcher and child generations gone, the control lock released, and the
delivery marker/manifest/all-artifact checks described by the CPU entry. Neither
a prepared receipt nor a visible output directory is proof of completion.

## Synthetic tests

Use `python -B -m unittest discover -s <this-directory> -p test_dispatch.py -v`.
Fixtures use a fake child and temporary metadata only. They cover preflight
blocking, exact pins/argv/identity, held lock through waits, finite timeout,
terminate/kill, metadata failure, boolean exit rejection, interrupted waits,
duplicate refusal, no automatic retry and source changes before launch. These
tests do not start a real subprocess or read scientific results.
