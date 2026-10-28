# Shared codeword and finer fit, 2026-10-28

The authorized 32-source experiment is complete; shared-codeword expansion is stopped. See `results/shared_codeword_fit_20261028_v1/REPORT_zh.md` and `closure_32.json`.

This experiment preserves the old NO_GO study and adds five actual-frame controls. It does not change the WCL main results, train a model, or expand common500. Launch registrations and original failures are immutable; do not rerun the one-shot source, PHY, receiver, metric or summary entries.

- `EXECUTION_PLAN.md`: scope, budget and gate.
- `protocol_spec.md`: actual information fields and receiver conditions.
- `source_fit.py`, `fit_codec.py`: independently terminated source endpoints and receiver.
- `qualify_fit.py`, `test_fit_protocol.py`: geometry and syntax qualification.
- `fit_phy_v1.py`: actual header/body encoding, channel and decoding.
- `replay_and_summarize_v1.py`: received-stream cache replay and source-level aggregation.
- `plot_results_v1.py`: CPU-only figure from frozen CSVs, with a new output directory per run.

The sole Git checkout is the project main directory on shared PFS. Native arrays, observations and logs remain under `liberai-tmp/liulu/liulu-var-comm/shared_codeword_fit_20261028_v1`.
