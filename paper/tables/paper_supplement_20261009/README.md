# Paper supplement tables (2026-10-09)

Generated directly from the completed, hash-pinned CSVs listed below. No new
training, inference, source coding, channel simulation, metric computation,
bootstrap, significance test or PDF compilation was performed. Decimal values
are formatted with ROUND_HALF_UP; original CSV precision is unchanged. Timing
mean/P95 uses two decimal places (ms). No value is clamped: a sample mean can
exceed an empirical P95 when a small upper tail is sufficiently large.

`timing_single_output.tex` contains three booktabs table fragments: five-method
TX/RX/software-E2E timing at7/13/19 dB, a separate BPG source-unfit cost table,
and loaded/stored footprint. BPG link sample counts3/12/30 (out of48) are bold
and explicitly conditional on transmitting. Each SNR has48 measured attempts
plus16 excluded warmups (three measured repetitions per source); repetitions are not
independent source/noise draws. SOURCE_UNFIT has no RX/E2E. Its coding cost is
shown separately, not replaced with zero. No HiFi timing from another environment
is mixed into the table. The BPG timing is native256, not adaptive downsampling.

`selected_digital_configs.tex` contains all12 raw digital configurations from
the original500-source/three-noise result: actual k/n, m/K, modulation bits per
symbol, header/body/padding and nominal/full-budget construction class. Printed
names are Full-scale and Partial-scale; internal file IDs remain in the script.

Use `\usepackage{booktabs}` and `\input{paper/tables/paper_supplement_20261009/timing_single_output.tex}`
or the configuration fragment. Main timing/storage/configuration tables use
`table*` with `tabular*{\textwidth}` for a conventional double-column paper.
The four-column BPG failure-cost table is a normal single-column `table`.
No resizebox, landscape rotation, makecell, multirow or nonstandard font is needed.
These are LaTeX fragments, not standalone documents. They have not been compiled,
as requested; check final placement within the manuscript's class and margins.

Reproduce from the repository root:

    python experiments/paper_supplement_20261008/export_paper_tables.py --root .

Optional `--output NEW_DIRECTORY` selects a separate output location. Re-running
only overwrites these generated table fragments and metadata, never source CSVs.
Existing publication filters are unchanged.

## Input identities

- `results/paper_supplement_20261008/a3_timing/final_v1/timing_single_output.csv`: SHA256 `ff44c9352b1b8dbc68c62e24b1796ea2133136178c5f8c542f0776001e4bb435`.
- `results/paper_supplement_20261008/a3_timing/final_v1/model_storage.csv`: SHA256 `006f903d0f82ab629d46120cf70918257c226c1f50d2a07f88a808d6c6ff291d`.
- `results/paper_supplement_20261008/a4_resources/v2/resource_summary.csv`: SHA256 `3dd6b0de3661db6782ca78f54ea0bc2b75c60d7f45224ae64bc5ac1b6e22c08b`.
