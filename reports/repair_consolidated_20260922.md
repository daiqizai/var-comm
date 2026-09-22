# VAR_COMM consolidated repair — 2026-09-22

The merged repair task was applied to the actual work tree `/home/liulu/projects/VAR_COMM`; the publication checkout was synchronized separately. The pre-repair source/config/results snapshot is retained under `_codex_sync/repair_20260922_pre/`. The attached `source_export_002` archive is treated as evidence, not as an overlay.

## Implementation status

R01–R05, R07–R11 have source-level fixes and focused regressions. R06 and R12 retain explicit partial/UNKNOWN states for receipts that lack complete scope or newer follow-up coverage. R13 is not promoted to a new broad quality result: existing real-weight qualification receipts are preserved, while a new broad GPU matrix remains blocked pending a scoped queue.

## Validation

The focused suite passed 22 tests. The broader suite passed 55 tests after refreshing the publication manifest for the synchronized `scale_channel.py`; release integrity then passed for 757 exported files and 779 checked files. Numeric guards cover sharp log priors, invalid bits/CDFs, and NaN enhancement inputs. No synthetic metric is promoted as a quality result.

## Results and lineage

Machine-readable issue, test, impact/rerun, and lineage records are in `results/repair_consolidated/`. Existing Stage A/B checkpoints, failure logs, historical reports, and old outputs remain unchanged. New source and report records are separate from historical evidence.

## Limitations

The requested repair archive was not present on the remote filesystem, so the available `source_export_002.tar.gz`, merged task document, current source, and recorded completion receipts were used. No credentials, weights, raw images, company data, or large caches were added to the publication checkout.
