# VAR_COMM checkpoint relocation — 2026-10-28

Seven existing weight files (4,964,756,469 bytes, 4.624 GiB) were moved from
`modelTeam/code/liulu/var_comm_runtime_20261010/assets` to the dedicated
`modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028` tree.
All seven full SHA-256 hashes are unchanged; no checkpoint was discarded.
No training, reconstruction, metric inference or channel computation was run.

PFS returned EXDEV for rename despite equal device IDs. The initial attempt
moved zero files. The preserved follow-up journal records copy, flush, full
destination SHA verification and only then removal of each unchanged source.
The seven old files are absent; no symlink or duplicate weight remains there.

## Current consumers

Use `model_routes.json` for new NeST jobs and `metric_config.json` for new
metric jobs. Historical registrations remain byte-for-byte unchanged and
describe their original locations. They are not current runnable path lists.
`tools/checkpoint_relocation_20261028.py` resolves exactly these seven old paths,
checking the full content hash at the destination. Its `metric_assets` wrapper
preserves the original asset verification function and replaces only the
weight-path lookup; original metadata hashes and numeric rules stay enforced.
`verification.json` records successful model-file and metric-asset checks.
This is a storage relocation check, not new scientific/model qualification.

## Scope

VAR_COMM and its frozen runtime only. The concurrent robot-evaluation chat is
handling other liulu projects, so their files were not moved by this task.
The targeted inventory excludes environments, dataset/image/source caches and
the protected historical publish backup; it is not a total PFS usage claim.
Future weights belong under `liberai-checkpoint/liulu/var_comm`; code, small
metadata and environments remain in their existing locations.
