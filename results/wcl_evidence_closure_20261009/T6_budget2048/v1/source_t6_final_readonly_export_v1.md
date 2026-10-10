# T6 final read-only export

`scripts/t6_final_report.py` consumes completed T6 calibration, confirmation source,
rendering, four-metric and source-paired statistical artifacts. It starts no
model, source codec, channel, rendering, or bootstrap computation.

The exporter verifies the bound SHA-256 files and exact request identities,
actual child wait receipts and successful exits, all 2,700 method-frame rows,
the 100-source / three-noise population, all 36 summary and 36 paired rows,
and all 3,600 source means. Its arithmetic validation checks already published
means against the completed frame rows; it never constructs new intervals.
The original CSV bytes are copied unchanged.

The actual raw and entropy PHY catalogues, including all 48 entropy resource
queries and unsupported constructor cases, are retained. No unsupported k is
truncated. All nine policies come from the original calibration population;
the entropy family was fixed using N1024 calibration before T6 confirmation.

The report explicitly retains m=10, K=0; raw PARTIAL selecting WHOLE; exact
zero paired differences; negative results; rejected CRCs; entropy parsing and
canonical rejection; gray outputs; and accepted errors diagnosed offline.
RAW KEEP remains distinct from DROP. If the two raw branches select the same
physical action, every corresponding received state, image, metric and zero
source-paired interval is checked.

The content audit is limited to all available audited source-content hashes.
Missing hash domains remain visible; no global never-seen or globally blind
claim is made. N1024 common500 and N2048 confirmation100 are different source
populations. The exporter does not infer an equal-quality bandwidth saving.

Invocation after actual scientific closure:

```text
python experiments/wcl-evidence-closure-20261009/scripts/t6_final_report.py \
  --calibration-freeze <N2048 frozen-policy JSON> \
  --source-completion <confirmation100 source completion JSON> \
  --source-wait-record <actual source parent launch/exit directory> \
  --render-completion <confirmation100 render completion JSON> \
  --score-completion <closed four-metric completion JSON> \
  --statistics-completion <closed source-paired statistics JSON> \
  --protocol <registered T6 protocol Markdown> \
  --out <new immutable final-report directory>
```

Repeat `--protocol` for additional frozen protocols. A render owner receipt is
discovered from sealed rendering outputs; `--render-owner-receipt` can identify
the exact sealed receipt explicitly. Calibration receipts must also be present
in their sealed output maps. An absent or failed prerequisite stops this export;
it never triggers scientific repair or launches a successor.

Outputs include `N2048_protocol.md`, `legal_actions.csv`,
`entropy_candidate_admission.csv`, `frozen_policy.json`,
`selected_configurations.csv`, `per_frame.csv`, `summary.csv`, `paired.csv`,
`source_means.csv`, `failure_breakdown.csv`, `points.json`, `pairs.json`,
`mechanism_and_failure_summary.csv`, `calibration_execution.csv`,
`content_duplicate_check.json`, `REPORT_T6.md`, and provenance/completion seals.
Cross-budget mechanism plotting consumes these tables separately; it does not
turn the two distinct populations into paired measurements.

Local synthetic test command:

```text
python -m unittest discover -s experiments/wcl-evidence-closure-20261009/scripts -p test_t6_final_report.py -v
```

The full synthetic fixture contains all 2,700 rows, failure states and zero
increments. Additional checks reject missing frames/source means, rewritten
means or zero intervals, invalid LPIPS direction/percentage units, tampered
sealed files, false wait provenance, and different reconstructions under an
identical raw action. Synthetic artifacts are temporary and are never merged
with scientific result files.
