# B2/B3 deterministic report entry — authoring only until root approval

This entry consumes the closed B2/B3 transport, native, DINO selection and
seven-metric delivery produced by the existing frozen implementations. It never
imports their workers. It contains no model, channel, metric, bootstrap or policy
selection call. The nine calibration values and four choices are copied from the
saved selection and its exact candidate evidence; selection is not rerun.

Do not execute while any prerequisite is active or absent. No invented preview,
zero-filled score or synthetic scientific result is permitted. The tests are small
metadata fixtures; they do not make a plot or open the real R/PFS data. Original
N1024 sources, records, plots and reports remain untouched.

## Inputs and closure

The root supplies one immutable JSON descriptor using:

```text
CPY -B E/b2b3_report_v1/report.py \
  --authorization /absolute/PFS/path/authorization.json \
  --authorization-sha256 ACTUAL_SHA --authorization-bytes ACTUAL_BYTES
```

`CPY` is the existing frozen LDPC Python 3.11, and `E` is
`R/experiments/arpc-n2048-adaptive-20261019`. All fresh stdout, stderr, launch,
actual wait and Matplotlib cache files belong in a new directory below
`L/arpc_n2048_adaptive_20261019_v1` (here `L` means the common runtime BASE).
Use `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and an existing
private `MPLCONFIGDIR` below that adaptive runtime root. Do not install or upgrade
an environment. The installed CPU Matplotlib is sufficient.

The root must use a bounded actual `Popen.wait` (suggested 600 seconds), recording
the real parent/child PID, start ticks and host, exact argv, stdout/stderr and exit
flags. Successful export does not imply actual visual review or process closure.
No polling launcher, automatic retry, overwrite or scientific calls are included.

Authorization schema:

```json
{
  "schema": "N2048_B2B3_REPORT_AUTHORIZATION_V1",
  "report_only": true,
  "science_calls": 0,
  "source_files": ["exact path/bytes/sha256 descriptors for report.py, test_report.py, README.md"],
  "inputs": {
    "B2_transport": "O/b2b3_revision_v1/B2/transport_completion.json descriptor",
    "B3_transport": "O/b2b3_revision_v1/B3/transport_completion.json descriptor",
    "B2_native": "O/b2b3_analysis_v1/B2/native_completion.json descriptor",
    "B3_native": "O/b2b3_analysis_v1/B3/native_completion.json descriptor",
    "B2_score": "O/b2b3_analysis_v1/B2/score_completion.json descriptor",
    "selection": "O/b2b3_analysis_v1/B2/selection.json descriptor",
    "B3_metrics": "O/b3_metrics_v1/completion.json descriptor",
    "B3_registration": "O/b3_metrics_v1/registration.json descriptor",
    "fixed_delivery": "R/results/arpc_n2048_validation_20261019_v1/final_delivery_manifest.json descriptor"
  },
  "outer_waits": {
    "B2_tx": "actual outer actual_wait.json descriptor",
    "B2_rx": "actual outer actual_wait.json descriptor",
    "B2_audit": "actual outer actual_wait.json descriptor",
    "B3_tx": "actual outer actual_wait.json descriptor",
    "B3_rx": "actual outer actual_wait.json descriptor",
    "B3_audit": "actual outer actual_wait.json descriptor",
    "B2_DINO": "actual outer actual_wait.json descriptor",
    "B3_score": "actual outer actual_wait.json descriptor",
    "B3_statistics": "actual outer actual_wait.json descriptor"
  },
  "additional_host_observations": [],
  "engineering_exceptions": [
    {
      "scope": "engineering_only",
      "science_calls": 0,
      "description": "Actual recorded engineering failure and bounded continuation, if any",
      "evidence": ["exact failure/diagnosis/continuation descriptors"],
      "allowed_failure_files": ["optional exact zero-science metadata failure descriptors, also present in evidence"]
    }
  ]
}
```

The quoted descriptor explanations are placeholders, not valid runnable input.
Every real descriptor is `{path, bytes, sha256}`. Input paths are fixed in the
report code; a descriptor cannot redirect it to an older scientific result.
The root can add further named outer waits and host observations. Each outer
wait requires a sibling launch and actual child/control-parent gone. Any non-native
wait executing on another host must have an `additional_host_observations` entry
using the frozen `CLOSED_ON_EXECUTION_HOST` schema. The report indexes this proof
by the exact actual-wait **and** launch path/bytes/SHA triples; a different path,
changed bytes or unmatched identity cannot close the requested wait. The proof
pins wait/launch/child/parent and asserts both gone on the execution host. Native
completion uses its already-pinned observations. The report never tests a remote
PID in the local host's `/proc`; same-host observations and waits always receive a
fresh process check. Transport and metric inner waits are independently checked
even if the root supplies an outer wait.

Native manifests and their member descriptors, code bindings, transport
registration, selected policies, all saved metric outputs, source identities,
and saved bootstrap SHA are verified. Closed arrays are not reconstructed or
rescored. The old fixed-reference final delivery, member receipts and original
host process closure are checked without relabelling historical failed receipts.
All exact current transport/native phase, control, RX, DINO-root/worker and B3
metric failure locations are checked. Only three metadata paths are eligible for
`allowed_failure_files`: `L/b2b3_analysis_v1/B2/failure.json`, the corresponding
B3 path, and `L/b2b3_analysis_v1/B2/dino/failure.json`, with L here the adaptive
runtime root. Root must explicitly bind each path/bytes/SHA under a zero-science
engineering exception and include the same descriptor in its evidence. A changed
SHA, unknown phase-root failure, or any scientific worker/control failure stops the
report. Root exception records cannot allowlist a worker/control failure. Allowed
engineering failures are retained verbatim in input provenance and disclosed in
`engineering_exceptions.json`; they are never relabelled as science success.

## Data products

New output only: `O/b2b3_report_v1`; existing output is an error, not permission
to overwrite. The report produces:

- `b2_all_candidates.csv`: all nine actual calibration DINO means.
- `b2_four_selections.csv`: four saved choices with original tie-break statement.
- `b2_source_DINO_means_9000.csv`: saved source means, no recalculation.
- `b3_frames_1200.csv`: actual TX endpoint, actual received endpoint, paid MCS,
  source bits/capacity, byte padding, failure/NULL/accepted-wrong status, native
  parser outcome, actual pixel hash and all seven saved scores for every frame.
- `b3_source_means_400.csv`: 400 source × method × SNR groups, seven columns;
  `b3_source_metric_means_2800.csv` preserves the original long-format values.
- `b3_summary_28.csv`: 14 means per SNR; `b3_direct_paired_14.csv`: seven saved
  direct paired intervals per SNR. No marginal-CI subtraction or new bootstrap.
- Transport correct-delivery and outcome/endpoint/MCS distributions with all
  300 frames per method/SNR, including failures. Correct delivery requires exact
  received source object and no source-parse failure. Accepted wrong objects
  remain wrong and retain their actual outputs/scores.
- Separate fixed-full 4/10 and 19 dB summary/paired tables and LaTeX. The low-SNR
  adaptive and fixed-full means can be inspected in one labelled CSV, but no
  unregistered adaptive-minus-fixed significance interval is computed.
- Actual budget/reuse counts and original scoring ledgers; explicit preserved
  engineering exceptions; Chinese report and concise English captions.

The pair pays `2×68` header symbols plus `2×956` body symbols, `2×13` length bits,
`2×16` CRC bits and `2×96` reassembly bits. Actual source occupancy uses the
terminated source bit length divided by the registered byte-rounded source
capacity. It is not spectral efficiency or the LDPC code rate. Accepted and
correctly decoded source objects are separate counts.

The seven-metric B3 bootstrap remains the already completed 10,000×100 source
bank with seed 2026101902. This entry hashes the saved file, never loads its
indices, draws a new bank or calculates a quantile. Every negative, zero and
cross-zero paired result is retained. The original 19 dB reference is separate,
not an endpoint of a uniformly adapted 4/10/19 dB curve. External100 is never
pooled with common500, new100 or calibration sources.

## Figures and actual review

Three small scientific figure groups: seven adaptive metrics (2×4 with one empty
axis), seven direct paired differences (same layout), and all nine calibration
means grouped into the four codec/SNR conditions. Adaptive/paired pages are
180×102 mm; calibration is 180×64 mm. White background, redundant line/marker
style, strong baseline contrast, 8.5 pt axis labels, 7.5 pt ticks, 4.5 pt markers.
The direct contrast is consistently NeST minus ARPC, so a negative value favors
NeST on a lower-is-better metric. It is not silently negated for display. Agreement
means are percentages; agreement differences are percentage points (pp). The
original CSV fraction values and confidence bounds are not modified.

Exports: true vector PDF, SVG with editable text, real 600 dpi PNG, and separately
rendered grayscale 300 dpi review PNG. No raster experiment curve in PDF, no font
files, no image generation. Root must actually open all color and gray exports,
inspect paper-size PDF rendering, check overlaps/CI visibility and verify PDF
MediaBox, embedded fonts/no image XObjects, SVG text/no images. If a real layout
defect occurs, preserve the first export and revise only this deterministic entry
under a reviewed new output namespace; do not alter scientific sources.

The initial completion deliberately says `VISUAL_REVIEW_PENDING`. After root's
actual review and report wait0/all processes gone, add a separate actual review
and final reviewed manifest; do not rewrite the original completion. This entry
does not commit/push or change shared root documents. Root owns normal exact-path
publication and final manuscript integration. Backups remain deferred.

Skills used for design and inspection: local
`C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md` with its actual
design-theory/common-patterns and trend demo; K-Dense `scientific-visualization`
source actually read at
`W/.research/figures_redesign_v1/references/kd_SKILL.md` and
`kd_publication_guidelines.md`. These are design references, not data sources.

The original 14 local metadata fixtures passed before independent review. Run
only `python -B -m unittest test_report.ReviewFixTests -v` from this source directory
for the R1/R2 continuation; this class covers exact cross-host closure, fresh
same-host checks, precise engineering-failure preservation, altered/unknown
failure rejection and agreement display units. It does not access production
paths or render a scientific preview. Do not rerun the original 14 tests.
