# VAR_COMM — single source worktree

The actual project root is now the sole Git worktree for `daiqizai/var-comm`.
Edit, run, test, commit and push here. `publish/var-comm` is retained locally as an ignored historical backup; no source-copy or separate publication workflow is required.

## Check a checkout

```bash
python tools/verify_repository.py
python tools/run_cpu_checks.py
```

CPU checks require Python 3.10+, numpy, torch, PyYAML and pytest (see requirements-cpu.txt). They use source from this checkout and include synthetic engineering tests and stored-result validation. They do not train models or recompute real image quality. External weights, datasets, caches and third-party implementations remain local and are not distributed. Historical GPU asset paths may need configuration on another host.

## Contents and provenance

- `src/`, `experiments/`, `scripts/`: actual execution code, configurations and tests.
- `src/cadsd_jscc/`: the small self-written historical runtime dependency closure, with source hashes in the migration records. External model libraries are not bundled.
- `tools/`, `tests/`: repository checks and lightweight result reproduction.
- `results/`: historical published data plus selected lightweight run artifacts, with run IDs and original hashes. `outputs/` remains in place and ignored.
- `reports/`, `docs/`: scientific evidence and scope limitations.
- [Migration record](reports/repository_migration_20260923.md).
- [Research status](RESEARCH_STATUS.md) and [historical result index](docs/RESULTS_INDEX.md).
- [Previous publication overview](docs/history/publication_README_before_20260923.md) and [previous worktree overview](docs/history/worktree_README_before_20260923.md).

Repository migration does not certify the previous repair task as complete. No original A/B training, new method, or GPU quality evaluation is part of this migration.
