# Durable recovery and preservation

Current user pause: scalar step3809(next3810), progressive3815(next3816).
Both completed normal actual wait0 and have no unresolved calls. The current
verified checkpoints are in `interim_v1/checkpoint_recovery_manifest.json` and
`user_pause_20261028_verified.json`; the STOP file remains. Do not resume until
the user explicitly asks. The differing pause steps are not matched comparison
points; completed1000/2000/3000 evaluations are matched.

The user did not give a guaranteed GPU window. All checkpoints are on shared PFS,
under `liberai-checkpoint/liulu/var_comm/nested_residual_rd_20261028_v1/`, separate
from the Git checkout. The immutable registrations specify the exact paths.

Each arm completed updates 1–25 in one process. A new process loaded its saved
model, optimizer, CPU/CUDA RNG, source order and endpoint schedule, then started
update 26. `resume_proof.json` records matching next indices/endpoint and zero
repeated updates. The CPU persistence test is supporting engineering evidence;
the two actual process restarts are the real-weight recovery evidence.

`train_residual_v1.py` writes a complete temporary checkpoint, flushes/fsyncs it,
and atomically replaces `latest.pt` after each update. Only then does the update
ledger become COMPLETE. It also retains the five periodic evaluated checkpoints.
The file stores the model, optimizer, scheduler/scaler placeholders, step and
exposure count, RNG, schedule hash, next batch/endpoint, frozen model identity,
cache identity, scope and entropy-format version. Final selected and latest
hashes are listed in `b1_delivery/checkpoint_recovery_manifest.json` when closed.

For a future authorized continuation, first verify PID and start ticks, actual
wait, every ledger state, checkpoint hash and step. Preserve the old directory.
Use a new registration/output directory with the unchanged mathematics and
matching optimizer/RNG/schedule identity. Do not launch the old command again or
repeat completed updates. Reconcile any checkpoint/ledger disagreement before
continuing; an interrupted RESERVED call is not proof that no work occurred.

SIGTERM/SIGINT, the study STOP file and the finite wall-clock bound are checked
between training updates. A forced kill during a development evaluation needs a
separate missing-row continuation that retains completed source calls. The
current training entry does not automatically resume partial evaluations. New
diagnostic materialization and metric roles also preserve per-source/pair commits
and failures; they are not automatic-retry jobs.

No scheduler is allowed to widen the current stage. Second seed, longer training,
another lambda, wireless integration and formal evaluation remain conditional on
the predefined B1 evidence. A new checkpoint is not launch authorization.
