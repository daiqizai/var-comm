# Layered raw mismatch: R0 preparation and verified reuse

This separate raw study implements the user's 2026-10-13 execution package.
It does not alter the completed entropy-coded common500 results. The starting
repository commit is `2887b4ab40191e5a0eaa597181d002892aa76813`.

## What actually completed

- Source audit: original calibration1000 pixels, tokens and checkpoints were
  authenticated; 200 source IDs were selected by the fixed hash-only rule.
  Original development100 was restored from the existing verified backup,
  including its later fresh-encoding verification receipts. The common500-
  derived historical mismatch100 was correctly excluded from this population.
- Real CPU backend layout construction, with encoding/decoding disabled:
  433 original single-group, 201 unequal two-group and 55 equal-rate two-group
  profiles are legal. All 689 paid header IDs fit the 12-bit field.
- Exact old table coverage and incompatibilities were enumerated, preserving
  missing probabilities as unknown. The last targeted archive audit recovered
  17 compatible raster-prefix states on all 200 calibration sources, using
  actual received-token equality and identical repeated image/score evidence.
- A grouped receiver and parameterized R1 PHY worker were implemented and
  checked with synthetic CPU fixtures. These checks are not packet or model
  qualification. No R1 scientific registration or full visual/scoring owner
  has been created.

See [source and reuse inventory](r0_reuse/),
[actual CPU layout results](candidates_cpu_v1/),
[table coverage](candidates_cpu_v1/coverage_v2/SUPPLEMENT_DESIGN.md),
and [contract review](contract_review_v1.json). The old coverage design states
the earlier conditional clean-Q gap; the final raw-cache audit reduces the
remaining gap to 5,400 source/state reconstructions.

## Current boundary

R1 is **NOT_RUN**. There is no nominated shortlist, scientific GO decision,
new channel trial, reconstruction, real-image metric or bootstrap in this
milestone. Actual completed calls from older studies have not been repeated.

The available tables do not provide comparable screening over all single and
grouped permissions. Restricting nomination to known table entries would
exclude most grouped candidates and all 64QAM single candidates. The
[explicit gap and proposed finite addition](R0_GAP_DECISION.md) is awaiting
the user's decision because it exceeds the package's small-table-supplement
description. It is not a launched extension or part of R1's hidden budget.

Strong-single selection includes measured nominal/backoff KEEP outcomes even
if they are not nominated again. The source-average J objective and GO rule
are unchanged. Grouped DROP and legacy raw433 KEEP/DROP have distinct public
receive domains; this difference is explicit in the protocol.

## Existing timing

The six-method shared-H800 timing v6 registration remains unchanged and its
576 completed ancestor calls are preserved. The last actual resource check
found GPU0 free memory 10,197 MiB, below its registered 18,432 MiB admission
threshold; no v6 launch existed. This is a resource wait, not an exclusive-GPU
requirement or a timing result. The historical automations remain paused.

## Reproduce preparation checks

From the sole server Git root, using its original visual environment:

```sh
python -B -m unittest discover -s experiments/layered-mismatch-raw-20261013 -p 'test*.py'
```

This command uses synthetic CPU fixtures. Do not rerun `r0_reuse.py` or the
actual backend-layout build merely to reproduce a milestone that already has
authenticated output. No listed check launches an experiment automatically.

Models, crops, original archives and environments remain private dependencies;
they are not included in this repository or a lightweight backup increment.

## User-approved R0 supplement

The original inventory remains preserved. The human subsequently approved the finite missing-evidence supplement. Its reviewed CPU packet owner is now running; the clean GPU stage is registered and waiting for resource admission. R1/R2 have not started. See the independently registered supplement and actual execution milestone in `results/layered_mismatch_raw_20261013/r0_supplement_preparation/`.
