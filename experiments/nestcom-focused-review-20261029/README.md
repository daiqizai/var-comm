# Focused V2 engineering and read-only audit

This directory supports the focused paper revision, not a new scientific grid.

- `n1024_null_v2.py` is an isolated, hash-bound paid-NULL extension of the immutable N1024 PHY and fallback policies. Load with `phy, plan = load(repository)`, construct the historical catalogue, append the twelve explicit NULL profiles using `extend_catalogue`, and call `choose` with the explicit historical policy version. Packet transmission/receive functions still pay the complete 68+956 symbols. Feed only the received packet result into `received_projection`; a fixed-gray projection materializes with `fixed_gray`. The source-decoder path is unchanged.
- `check_n1024_null_v2.py` executes the finite 18-frame/36-packet engineering check registered in the accompanying result directory. It intentionally refuses an existing output directory. Do not call it as an automatic retry or as a replacement for a full formal experiment qualification. Three normal historical payloads retain exact waveform/profile/RX identity; twelve no-fit cases exercise all original MCS layouts.
- Read-only audit scripts join current formal logical identities to their actual physical observations and aggregate energy, status and endpoints. They do not re-encode, decode, infer or resample.

Results and exact commands/identities are in `results/nestcom_focused_review_20261029_v2/`. Historical N1024 results continue to describe the former error-on-no-fit implementation. The new version has not replaced those frozen results. No source-model inference, image metrics, training or bootstrap are authorized here.


## Audit table publication packaging

`read_only_audit.py` produces the full original event mapping in PFS. Run `split_formal_event_mapping.py <original_csv> <output_directory>` once to partition its existing rows by budget for the 10 MB Git file limit. The two CSVs and index retain every field and the original zero-based row positions. This operation does not rerun an experiment or statistic. See the publication packaging receipt alongside the audit tables.
