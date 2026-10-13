# R1 missing-stage cache recovery

This is a separate, once-only recovery of the diagnosed receiver initialization
failure. The original master `70bb0747...`, failed model-load reservation, failed
owner exit and successful PHY/projection bytes remain unchanged. This directory
does not expose an entry to repeat PHY, projection, training or R2.

The private adapter uses the original receiver definitions and an inverse-AST
checked single cache initialization insertion after the new visual output is
created. It creates `runtime_cache/tmp`, `runtime_cache/inductor` and
`runtime_cache/triton` only under the new visual output. The original file
firewall, model bundle, numeric flags, actual inputs, candidate grid, resource
limits and absolute deadline are retained. This is an engineering repair, not
a claim of bitwise model qualification.

The new master has a complete CPU-only ancestry descriptor. The visual child
contains only the original source-free model registration plus its new runtime
and a fixed source-free projection wait/identity receipt; it does not receive
the source-bearing master or failure audit. CPU scoring adapts exactly three
provenance statements to the inherited projection. Mathematical scoring and
selection functions remain original.

Registration requires the diagnosed failure audit and every pinned evidence
file, a matching independent implementation review and the actual successful
zero-model cache/import/flags probe v2 with a real exit-zero wait. It also checks
that the failed visual segment still has zero images and an empty partial list.
All new source files must match the reviewed bindings.

The persistent dispatcher holds both original ancestor locks and its own lock
through the actual owner wait. The reused original owner coordinates the timing
lock and runs only visual, score and selection, with the original deadline
`1791954813.2221007`. Its completion describes actual stage exits only: the
owner's own successful exit must separately be proved by the outer dispatch
receipt and disappearance of the original process identity.

No registration or run is implicit. Root must first supply the exact authorization
pin to `registration.py --authorization PATH --authorization-sha256 SHA` and
then explicitly call `dispatch_once.py --expected-registration SHA` at most once.
Any unknown scientific failure remains stopped for diagnosis.

The reviewed commands, from the single remote Git root, are:

```sh
/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B experiments/layered-mismatch-raw-20261013/r1_cache_recovery_v1/registration.py --authorization /ABSOLUTE/REVIEWED/authorization.json --authorization-sha256 ACTUAL_AUTHORIZATION_SHA256
/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B experiments/layered-mismatch-raw-20261013/r1_cache_recovery_v1/dispatch_once.py --expected-registration ACTUAL_NEW_MASTER_SHA256
```

The placeholders are deliberately not runnable pins. Use the actual root-issued
authorization and the immutable registration returned by the first command;
never point this entry at the failed original master to attempt a replay.
The dispatcher must remain alive to record its real child wait, with its own
output redirected to a new personal temporary-log directory.

At implementation freeze, the 15 recovery and 11 score-provenance synthetic
tests passed both locally and in the original CPU interpreter. The original
visual environment also completed the separate zero-model cache/import/flags
probe v2 with actual exit zero. These are engineering checks only; they do not
mean that recovery registration, model rendering, scoring or R1 selection has
completed. Consult the actual bound launch/wait/completion receipts for state.

CPU tests `test_recovery.py` and `test_score_provenance.py` use synthetic fixtures
only. They do not initialize CUDA, load models, run channels or create a
scientific registration. Read-only CPU closure tools can use
`adapter.install(registration_pin, role='selection')` to obtain the private
`shared, receiver, score, owner` modules without invoking a scientific endpoint.
