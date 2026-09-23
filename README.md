# VAR_COMM — single source worktree

The actual project root is now the sole Git worktree for `daiqizai/var-comm`.
Edit, run, test, commit and push here. `publish/var-comm` is retained locally as an ignored historical backup; no source-copy or separate publication workflow is required.

## Completed review and bounded study

- [Phase1 repairs and affected real-weight results](reports/review_20260923_phase1.md).
- [Phase2 real experiments, numerical-protocol correction and measured tradeoffs](reports/review_20260923_phase2.md).
- [Phase2 evidence and run identities](results/review_20260923_phase2/README.md).
- [Actual execution commands and preserved launch history](experiments/var-latent-enhancement-20260917/research/EXECUTION.md).

These runs preserve original A/B and use the existing development population. They do not certify every historical issue, open a new holdout, or complete m10/new-Decoder digital adaptation.

## Check a checkout

```bash
python tools/verify_repository.py
python tools/run_cpu_checks.py
```

CPU checks require Python 3.10+, numpy, torch, PyYAML, LPIPS/torchvision and pytest (see requirements-cpu.txt). They use source from this checkout and include synthetic engineering tests and stored-result validation. They do not train models or recompute real image quality. External weights, datasets, caches and third-party implementations remain local and are not distributed. Historical GPU asset paths may need configuration on another host.

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
