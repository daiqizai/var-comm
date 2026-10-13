# Finite R1 calibration execution

This directory is an implementation prepared with synthetic CPU checks. It is
not an R1 registration, launch receipt, numerical qualification, or completed
experiment. The active R0 supplement must finish before any real registration.

## Frozen scope

The future registration authenticates the actual R0 closure audit and retains
the nominated rows unchanged. It uses the original calibration 200 sources,
registered noise namespace and physical layouts. The maximums remain 36,000
physical frames, 108,000 packet decoder calls and 46,800 logical VAR images.
Actual counts are calculated from the canonical nominated physical candidates
and receiver views. Aliased arms do not create duplicate receiver outputs.
There is no training, bootstrap, ConvNeXt scoring, extra qualification frame,
or automatic R2/development launch.

`registration_builder.py` is an explicit later operation. It verifies the root
implementation review and completed R0 receipts, including the saved actual
waits and complete scientific ledgers. These are read-only audits, not repeated
scoring or bootstrap. A single absolute deadline is fixed at registration;
resource waiting and all stages consume that same window. The future
coordinator must choose a finite wall limit explicitly. An expired registration
does not silently receive a new deadline.

## Execution and isolation

`owner.py` runs these five children serially:

| Stage | Entry and completion status |
| --- | --- |
| PHY | Frozen `../r1_phy.py run`; `COMPLETE_R1_PHY_ONLY` |
| RX projection | `receiver_visual.py project`; `COMPLETE_R1_RX_PROJECTION` |
| Reconstruction | `receiver_visual.py visual`; `COMPLETE_R1_VISUAL` |
| Three-metric scoring | `score_selection.py score`; `COMPLETE_R1_SCORE` |
| Calibration selection | `score_selection.py select`; `COMPLETE_R1_CALIBRATION_SELECTION` |

Projection authenticates the complete physical grid and ledgers. It produces
separate model inputs and statistical joins. The reconstruction child receives
actual receiver state plus a model/runtime-only bundle; source image and token
paths, transmitted candidate identity, and original source-bearing qualification
requests are excluded. Source images are available only to the separate scoring
child. The original mathematical model functions, checkpoint identities,
received-token semantics, numeric flags and batch size remain fixed.

The visual access firewall is an auditable Python file-access guard, not an OS
sandbox. Model/runtime metadata preparation and import probes do not qualify
the eventual scientific result. Real reconstruction and metrics must still
complete their registered calls and their actual ledger checks.

## Resource and failure ownership

Storage is under the existing personal `liberai-tmp/liulu` root. PHY uses the
original LDPC environment with CUDA hidden and two threads on CPU slots 2/3.
Reconstruction/scoring use the original visual environment, six intra-op and
two inter-op threads, FP32, TF32 off, and CPU slots 0/1. The registered shared
GPU must meet the declared 18 GiB admission boundary, 16 GiB allocator cap and
ongoing safety checks. No process is evicted and no global GPU setting changes.

Every child inherits the R1 owner lock and a shared lock on the existing v6
timing owner lock. These remain held through the actual wait, including if the
R1 parent dies. Between stages the timing lock is released. Pending v6 timing
gets priority only when its already registered **GPU 0** satisfies its original
free-memory/utilization limits; another GPU cannot make v6 eligible.

Before dispatch, a resource WAIT writes a control receipt with zero scientific
calls and no owner launch. After a real owner launch, any unknown failure,
STOP, incomplete ledger or timeout stops the chain; no automatic replay occurs.
Even a child that exits before a control-record error is still actually waited
and preserved in `interrupted_actual_wait.json`. A later recovery needs explicit
diagnosis against the immutable failed scene.

## Receipt schema

The owner writes `<runtime-root>/<stage>_launch.json` with `stage`, the actual
`process` (`pid`, `start_ticks`, `state`), `argv`, the master registration pin,
CPU affinity and deadline. PHY and reconstruction command lines separately
name their restricted subregistration pins.

Only the owner writes actual wait records, after `Popen.wait()`:
`<stage-directory>/actual_wait.json` and an identical
`<runtime-root>/<stage>_actual_wait.json`. They include `actual_wait`, `pid`,
`returncode`, `timed_out`, `stopped`, `terminated`, `killed`, `stage`, and elapsed
seconds. Downstream closure binds these to the actual dispatched identity and
checks the child is no longer live. Final completion requires all five stages;
it never treats a launch receipt or an ACTIVE automation as a result.

## CPU checks

From the repository root, the original CPU environment can run:

```text
/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B -m unittest discover -s experiments/layered-mismatch-raw-20261013/r1_execution_v1 -p 'test_*.py' -v
```

`test_owner.py` uses fake children only. It reads the actual worker ASTs to
check their command flags and completion statuses, exercises all five stages,
and tests immutable failure handling, exact subregistration use, actual wait
receipts, timing-device priority, STOP, expired deadlines and zero-launch WAIT.
It does not construct a scientific runtime, load models or execute packets.
