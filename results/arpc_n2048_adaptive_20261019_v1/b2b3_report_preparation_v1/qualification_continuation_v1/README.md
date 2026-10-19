# Original-environment report qualification continuation v1

Status: author preparation only. Root and independent review are required before
deployment/execution. This is not a new production report version. No test was
executed by writing this entry; frozen report.py/test_report.py/README.md stay
unchanged.

## Initial actual failure, retained

The original frozen CPY run reached 15 completed tests, then the 16th fixture
`ReviewFixTests.test_changed_engineering_failure_sha_rejected` failed while
`TemporaryDirectory` was cleaning up. `shutil._rmtree_safe_fd` calls
`os.open(entry.name, ..., dir_fd=topfd)`, whereas the Python `open` audit event
does not expose dir_fd. The old guard resolved entry.name against cwd, rejected
its own private fixture, and later rejected a traceback source read from the
base interpreter. These are test-harness failures, not scientific output.

Initial private control directory:
`/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1/b2b3_report_root_v1/qualification`

- actual_wait.json: 452 bytes,
  `8d2b7e8c7c3002d426ef751d8af61b8d43d4e6c96703265032d65cb55b5b57d8`;
  returncode 1, 0.21732020378112793 s, child PID 3887446/start 209223394,
  parent PID 3887445/start 209223370, both subsequently verified gone.
- stderr: 7106 bytes,
  `a6078eb6f0a86bcbe28ce6283cbc7f3848b81af5ec9a6ce809b1757afbd03d07`;
  exactly 15 explicit `... ok` lines.
- failure_diagnosis.json: 10553 bytes,
  `1235b53e827dd058c5fa2672edab0d256c21a7733b234aef56badd3b137b0a07`.
- Original qualification_runner.py: 2256 bytes,
  `1e9dd85a38903c8720f74a5076b385e5d8d5aecfe2af2ee89d3bfa95752aa6bb`.

The exact initial logs/receipts and diagnosis also remain at
`results/arpc_n2048_adaptive_20261019_v1/b2b3_report_preparation_v1/original_environment_qualification_v1/`.
Neither evidence nor the failed temporary fixture is removed or overwritten.

## Narrow continuation and guard repair

The entry pins the 3 frozen sources, 7 initial receipt/log/diagnosis files and
old runner. It independently verifies the original child/parent are gone,
checks that the original wait remains actual exit1 and parses the original
stderr for the exact ordered 15 passing IDs. It loads, but does not execute,
the full suite to prove it is exactly those 15 plus the explicit remaining 11.
It executes only the failed 16th fixture plus the 10 fixtures not previously
reached. This is 11 total, not 12; 26 − 15 = 11. It does not rerun any of the
first 15 original-environment successes or either prior local qualification.

The original `os.open` is wrapped only in this new test process. For a relative
path with dir_fd, `/proc/self/fd/<fd>` and fstat establish the real directory;
canonical directory and target must both stay inside this continuation's
private `tmp/`. A ContextVar exposes that exact (raw path, fd, directory,
target) to the audit hook for the nested original os.open call. The hook
rechecks the actual fd and canonical target. The wrapper resets context in a
finally block; it does not substitute cwd or broaden a production whitelist.
Absolute opens use ordinary guard validation. Path traversal and symlink
escapes outside the private fixture remain denied.

Ordinary reads are restricted to the exact 3 frozen report files or the new
private continuation directory; writes are restricted to that new directory.
The old runtime, real R/L science/model/source assets and arbitrary interpreter
trees are not accessible after the guard is installed. Required stdlib modules
are imported before the guard; already-loaded stdlib traceback source and the
3 relevant Python source files are cached before the guard so reporting a test
failure does not trigger new base-interpreter reads. Scientific/model imports,
subprocesses and network calls remain forbidden. The guard is not installed
in any production worker, and no global configuration is changed.

The two immutable Python modules are loaded from their exact SHA-bound source
bytes with `ModuleType` plus `compile/exec`, in report→test_report order and
with their original absolute filenames. This avoids SourceFileLoader's
otherwise attempted `__pycache__` reads even under `-B`; no cache directory is
added to the guard whitelist. Loading the suite does not execute its tests.
The first reviewed author draft is retained in `before_source_only_review/`.

The receipt preserves original wait1, all initial evidence pins and 15 IDs,
the explicit 11 selected IDs, errors/failures/blocked/model imports, both
before/after immutable-source pins and actual dir_fd resolution records.
A successful continuation means 15 retained + 11 passed = 26 qualified tests;
it never changes the original actual wait to zero. Unexpected failures retain
the new directory and require diagnosis, never automatic retry.

## Intended root-reviewed execution, not yet authorized to invoke

Deploy this independent runner under
`.../b2b3_report_root_v1/qualification_continuation_v1_runner.py`.
Create only the new exclusive control directory
`.../b2b3_report_root_v1/qualification_continuation_v1/` and its empty `tmp/`.
Root's outer controller must bind the actual runner SHA, record exact argv,
child/parent PID/start/host and persistent stdout/stderr, then do a bounded
actual Popen.wait (180 s plus at most 30 s own-process-group termination).

Use original CPY `-B`, cwd equal to that control directory,
`CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONNOUSERSITE=1`, `TMPDIR=<control>/tmp`, OMP_NUM_THREADS=2,
OPENBLAS_NUM_THREADS=1; remove PYTHONPATH. Run once only after review.
The runner accepts no path overrides. On success verify true exit0,
all processes gone, no failure and the complete qualification receipt.

No production report invocation, rendering, model, score, reconstruction,
channel experiment, bootstrap or publication is part of this continuation.
