# Finite R1 with the reviewed quality memory amendment

This independent entry preserves the frozen `r1_execution_v1` directory.
Its registration has not been created and no R1 scientific work is implied by
source preparation or CPU tests. Actual closed R0 evidence and a separate root
review are required before the single registration operation.

The user's request to use available shared-device capacity authorizes the
quality-only policy: 15 GiB free before a model stage, 14 GiB free plus verified
current-process allocator reservation while running, a real 12 GiB allocator
cap, and at least 2 GiB absolute free memory. The allocator is set before both
VAR/Dc construction and LPIPS/DINO construction. Current-process reservation
credit uses the minimum of observations around the device sample; foreign,
uninitialized and unknown memory contributes no credit. CPU, host-memory,
disk and utilization checks retain their original behavior.

The completed R0 resource revision supplies relevant engineering evidence but
does not prove that every R1 input fits, numerical bitwise identity, or timing
comparability. R1 retains FP32, TF32 off, six/two mathematical threads, batch one,
the original models, sources, noise, channel and packet semantics. There is no
new validation frame, metric, training or bootstrap. Its original maximums are
36,000 physical frames, 108,000 packet decodes and 46,800 logical VAR images.
The original DINO selection objective and GO gate remain unchanged. R2 is never
registered or launched automatically by this entry.

## Exact implementation changes

- `owner.py` is byte-identical to the original. Its five serial stages, fixed
  absolute deadline, child waits, no-retry behavior, STOP handling, original
  GPU0 timing priority and inherited locks are unchanged.
- `shared.py` changes only the declared resource thresholds and the matching
  private observer/assessor adapter. Unexpected source shapes fail closed.
- `receiver_visual.py` changes only allocator setup and its reported cap from
  16 to 12 GiB. Actual receiver inputs, source-free projection, file-access
  checks, fresh model path and exact-event reuse are unchanged.
- `score_selection.py` changes only its allocator setup from 16 to 12 GiB.
  Metric definitions, source averaging, candidate selection and GO are unchanged.
- `registration_builder.py` binds `r0_finalize_resource_v2`, the actual R0 clean
  resource registration and its authenticated closure. Both original and revised
  finalizer source pins remain in the code bindings. It additionally requires
  the new resource review and the inverse-AST proof before writing anything.
- `resource_revision.py` checks all five original source SHA256 values, allows
  only exact enumerated substitutions and reverses them to the identical AST.
  This proof is recorded in the new master registration. It is metadata proof,
  not scientific numerical qualification.

Both versions use the **same** authoritative `results/.../r1_execution_v1`
registration/result directory and personal temporary `.../r1_execution_v1`
runtime directory. This preserves the one-attempt namespace and owner lock.
The new resource policy and code pins reject using the old entry on the new
registration. No old file is rewritten and no new independent scientific budget
is created. The original six-method timing v6 policy is not changed.

## Future registration and persistent dispatch

After R0 actually closes, the root review must have status
`R1_RESOURCE_V2_IMPLEMENTATION_REVIEW_PASS`, `approved: true`, zero
`scientific_calls`, original `caps`, exact `resource_policy`, the full output
of `resource_revision.verify()` as `resource_revision_proof`, and `code_bindings`
for `shared.py`, `owner.py`, `registration_builder.py`, `receiver_visual.py`,
`score_selection.py`, `resource_revision.py` and `dispatch_once.py`.

Run the builder in the original CPU environment with the actual R0 closure
pin, new review pin, authoritative registration directory, eligible GPU and a
finite `--max-seconds`. It performs read-only authentication of already saved
R0 evidence, not repeated ranking, models, channels or bootstrap. The fixed
deadline includes every subsequent stage and wait; it is never reset by WAIT.

`dispatch_once.py --expected-registration SHA256` is the single persistent outer
parent after registration. It validates that registration and its own source
pin, refuses old launch/stage/failure markers, and records its identity, actual
owner PID/start ticks, stdout/stderr and the result of `Popen.wait()`. Its lock
is inherited by the owner. If control-record writing fails after a child exits,
the real child is still waited and an interrupted-wait receipt is retained.
It never retries, registers another attempt or declares scientific completion.
A successful zero-science WAIT exit is distinguished from an actual owner launch.
The coordinator must check real stage receipts before reporting results.

Run the synthetic tests with CUDA hidden in the original CPU environment:

```text
python -B -m unittest discover -s experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2 -p 'test_*.py' -v
```

These tests use fake children, source ASTs, synthetic receiver events and mock
resource observations. They do not import GPU models or execute a channel.
