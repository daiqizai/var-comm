# Leo shared-host preflight and CPU engineering entry v1

This is a preparation entry, not a production experiment owner or numerical qualification. It was implemented and checked locally; it has not connected to, installed software on, or executed on the shared server. No model, PHY, training, bootstrap or real source-selection calls were made.

## Scope and paths

The only admitted remote base is `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu`. The selected device is GPU UUID `GPU-bf8340be-bd73-7413-62d5-0537be7f30cb`. The tool queries that UUID's metadata only and does not identify or stop other users' processes. It makes no GPU allocation. A busy shared GPU does not forbid its CPU-only checks, and utilization zero does not imply exclusive ownership.

The CLI offers only `preflight` and `run-engineering`. The latter accepts three exact-SHA allowlisted tests: `plan`, `phy_format`, `source_fake_cdf`. It accepts no arbitrary command, no scientific task type, and no qualification command. The local engineering-venv preparation is separate from a frozen scientific runtime. Existing old production owners and result files remain unchanged.

The future project location can be `BASE/code/VAR_COMM`, with an engineering venv under `BASE/var_comm_runtime_20261010`. Each run needs a fresh `--out` below BASE. TMP, Python bytecode, Torch/HF/Matplotlib/Numba/CUDA/Triton/extension/pip/uv caches and logs point below that run directory. No HOME or global settings are modified. A personal venv's executable spelling is retained even when its `bin/python` symlink resolves to a system binary. This is needed to preserve venv activation semantics.

## Resource and process controls

Default two CPU threads, maximum four, one owned child. The child receives a matching CPU affinity subset, thread-library environment variables, increased nice level, and a CPU-time limit. Wall time is explicit and at most600 seconds. A personal file lock prevents overlapping invocations; output and receipts are write-once. The entry uses a fresh scoped GPU query and a short host CPU utilization sample, cgroup quota, process affinity, host/cgroup memory headroom and personal-filesystem free space before launch. High loadavg is recorded but is not mistaken for high CPU busy percentage.

CPU engineering is refused if sampled host CPU busy exceeds95%, fewer than2–4 requested CPUs are allowed, less than2GiB memory headroom is visible, or filesystem free space is below10GiB. A GPU estimate of at least20GiB free and utilization at most90% is reported but does not authorize GPU work or block CPU-only work. These are conservative admission estimates, not observed peak requirements, allocation guarantees or hard total-memory isolation. PFS filesystem free space is not a personal quota check. No per-process CUDA memory-fraction setting is used or represented as hard isolation.

Before a child starts, all admitted test/dependency bytes are checked again. Its exact argv, PID, environment overrides, affinity and real wait result are recorded. Nonzero exit, OOM-related termination or timeout stops the invocation and retains evidence. Timeout/interrupt cleanup targets only the Popen child created by this entry; no process group, foreign PID, GPU reset or global scheduler setting is touched. There is no automatic retry, alternative GPU, increased concurrency or next stage.

## Example commands for the root coordinator only

Use already prepared paths, a fresh output name for each invocation, and retain parent-level actual-wait evidence. These examples have not been executed on the server.

```sh
BASE=/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu
PY=$BASE/var_comm_runtime_20261010/engineering_venv/bin/python
TOOL=$BASE/code/VAR_COMM/experiments/entropy-partial-mismatch-20261010/scripts/leo_shared_preflight.py
$PY -B "$TOOL" preflight --out "$BASE/var_comm_runtime_20261010/preflight_v1" --threads 2
$PY -B "$TOOL" run-engineering --python "$PY" --project-root "$BASE/code/VAR_COMM" --check plan --out "$BASE/var_comm_runtime_20261010/plan_check_v1" --threads 2 --timeout 120
```

To check format or fake-CDF engineering, select `phy_format` or `source_fake_cdf` with another fresh output directory. A dependency SHA mismatch requires an explicitly reviewed new version; it is not bypassed. Metadata/engineering PASS never counts as the real32 source gate, 457-call PHY gate, 48-frame image link gate, new-GPU numerical qualification or timing admission.

## Local validation

Fourteen tests ran in0.181 seconds: thirteen passed; one Windows symlink-creation test was skipped because that local account lacks symlink privilege. Linux should run that path-escape test without the Windows limitation. Actual local owned subprocesses exercised exit0, exit7 and timeout termination/wait. Other checks cover GPU UUID/CSV validity, CPU admission, busy shared GPU, low resource/stale snapshot rejection, personal path traversal, venv spelling, duplicate lock rejection, private cache environment, immutable output and exact test-source binding. No SSH, GPU, image input or scientific runtime was used. The Linux resource query/affinity entry itself still requires a real remote engineering check by the root coordinator.

A separate independent review is pending; the companion preparation receipt records implementation/test SHA and explicit nonqualification status.
