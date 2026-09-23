# Actual execution and bounded reproduction

Run from the actual project root. All self-written Python dependencies are versioned here; external models, source data and tensor caches remain local and are not distributed.

Set PYTHONPATH to root src plus this experiment's src, phase_b/src, evaluation/src, followup/src, mechanisms/src and research/src. Actual runs used OMP_NUM_THREADS=6, OPENBLAS_NUM_THREADS=2, CUBLAS_WORKSPACE_CONFIG=:4096:8 and the repaired stage_B_v2_repaired_20260921/decoder_gate.json. Immutable registration receipts list exact source, configuration, model, Decoder and cache identities. Use fresh result paths for a new experiment; training/digital resume rejects changed identities.

Actual commands, with BASE=outputs/VAR-LATENT-ENHANCEMENT-20260917/followup:

- python3 -m latent_research.diagnostics --output $BASE/research_20260923_diagnostics
- python3 -m latent_research.pca_projection --output $BASE/research_20260923_pca_basis
- python3 -m latent_research.train --output $BASE/research_20260923_training_v3 --prefix-cache-from $BASE/research_20260923_training --qualification-only
- The same training command without --qualification-only, after real qualification passed.
- python3 -m latent_research.evaluate --training $BASE/research_20260923_training_v3 --output $BASE/research_20260923_evaluation
- python3 -m latent_research.digital_requalify --output $BASE/research_20260923_digital_strict
- python3 -m latent_research.system_policy --output $BASE/research_20260923_system_policy_v2
- python3 -m latent_mechanisms.linear_measurement --output $BASE/research_20260923_pca_evaluation --projection-from $BASE/research_20260923_pca_basis/A.pt

The original prefix cache path refers to valid real-data cache shards from the initial qualification attempt, independently verified by input/model/file hashes. It does not reuse failed probe optimizer updates. A new run can build its own prefix cache when no cache override is supplied.

launch_history/ preserves exact historical queue wrappers, including transient process IDs, as text evidence. They are not portable launchers. The first queue stopped on a digital precision mismatch; the later queue requires the strict digital matrix to complete. The legacy precision probe is a diagnostic snapshot of the earlier probe interface, not a current method or acceptance test.

After every required GPU job has a complete receipt:

1. python3 tools/collect_research_comparison.py
2. python3 tools/publish_research_results.py
3. python3 tools/write_research_report.py
4. Explicitly review and stage the source, reports and lightweight evidence; refresh the release manifest.
5. Run python3 tools/verify_repository.py, python3 tools/check_release.py and python3 tools/run_cpu_checks.py.

The result copier refuses to overwrite an existing published version and excludes weights, raw images, tensor caches and partial results. Published evidence verification is asset-free; it does not substitute for separately recorded real GPU execution.
