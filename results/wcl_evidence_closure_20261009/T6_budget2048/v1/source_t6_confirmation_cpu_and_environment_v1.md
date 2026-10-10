# T6 confirmation CPU cohort and source-environment admission

This adds two independent orchestration tools. The frozen scientific source,
PHY, render, scoring and statistical workers are unchanged.

## Current native inventory projection

The original native loader inventories tracked source files. The new-source
owner calls `t1_holdout_render.build_native`, which requires exact equality
between its current inventory and the registered environment inventory.
An old T2 environment therefore cannot be reused unchanged when newly tracked
files have entered that inventory.

```text
python scripts/t6_project_source_environment.py \
  --environment-request <original T2 environment request> \
  --out <WCL outputs>/T6_confirmation100_environment_projection_v1
```

The tool hashes every original `source_bindings` entry and requires all original
native entries to match the current inventory exactly. Additions must be newly
tracked paths returned by the same original `git ls-files` patterns. Changed or
missing old files and untracked runtime additions are rejected. It preserves
all other environment fields, including model, scorer and numerical identities,
and records the original request, the exact additions and all their SHA-256
values in a separate receipt. It imports only the pinned standard-library
inventory helper and performs no model, image, channel or bootstrap operation.

Use its `environment_request.json` for new100 source preparation and confirmation
render preparation. The source worker only requires the same repository root
and the supplied descriptor; it does not require the original T2 request path.
The render preparation similarly accepts the supplied path and recomputes the
inventory. Score/statistics/final export do not impose an old-T2 environment
path identity on source preparation. Keep the tracked inventory stable after
this projection and throughout the owned scientific stages.

## Raw-full to frozen confirmation CLI dependencies

1. Close actual raw full1000 and entropy full1000 calibration after their GPU
   owners have actually waited successfully. Both use original calibration IDs
   and noise seeds 4101/4102/4103.
2. `t6_freeze_policies.py --raw-completion ... --entropy-completion ...
   --entropy-family-selection ... --out <freeze.json>` writes all nine policies.
   It consumes sealed final winner tables; it never reads new100 images.
3. Complete `t6_content_reference_inventory.py` with all five registered old
   manifest roles. Its scope remains the available audited content hashes.
4. Prepare `t6_confirmation_sources.py` with the exact pre-fixed registration,
   freeze, selected family, reference inventory, full class mapping, original
   common500 source request, and the environment projection above. The source
   owner requires the original interpreter, affinity 4–9, nice level 15,
   `CUDA_VISIBLE_DEVICES=0`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
   It checks duplicates before Encoder access; no reselection is allowed.
   Root records its real launch and exit using the existing wait-only parent.
5. Prepare `t6_confirmation_render.py` against the completed new100 source
   artifact, same freeze and environment projection, explicitly using
   `--workers 4 --gpu-workers 2`. Its `workers` means workers **per branch**.
   The default 8 would mean 16 total CPU processes and is rejected by the new
   finite CPU owner.

## Exactly eight CPU workers

```text
python scripts/run_t6_confirmation_cpu_cohort.py prepare \
  --request <confirmation-render>/request.json \
  --record-dir <confirmation-render>/cpu_cohort_v1

python scripts/run_t6_confirmation_cpu_cohort.py run \
  --registration <confirmation-render>/cpu_cohort_v1/registration.json
```

Root runs the second command through the existing wait-only parent. The owner
first executes and waits for the exact `ledger-init` child, checks its admission
receipt, then launches raw indices 0–3 and entropy indices 0–3 in separate
processes, interleaved with 0.3-second launch spacing. The eight processes share
the same request and `packet_ledger.sqlite` with cap 5400. GPU visibility is
empty, CPU thread limits are two per worker, and no GPU successor is launched.

The registration seals the scientific request, runner, owner, complete source
dependency map, exact child argv, environment, deadline and ledger path. Each
child has a pre-launch reservation, argv/PID record and actual wait/exit receipt.
A global exclusive launch claim within the scientific output blocks duplicate
attempts even if another record-directory name is supplied. Failure stops later
launches and terminates only this owner's live children, then waits for every
actual child. An unresolved attempt is preserved for audit, never retried.

After a successful CPU cohort, root separately invokes the existing `plan-render`
and the frozen GPU owner. Source/CPU cohort completion is not render completion.

## Synthetic checks

Seven CPU-owner tests passed using real subprocesses with synthetic children:
eight distinct PIDs sharing one SQLite file; failure before later launch;
termination and actual waiting for a running sibling; pre-launch STOP; duplicate
global attempt rejection; fixed per-arm count and sealed dependencies; mutated
request rejection. No scientific runner was called.

Five environment-projection tests passed: exact added tracked files; unchanged
old fields; changed old source rejection; missing/changed native rejection;
untracked additions rejection; and the valid zero-addition case. No model or
image was loaded.
