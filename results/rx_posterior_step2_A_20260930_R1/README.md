# RX-POSTERIOR-STEP2-A-20260930-R1

[Reviewed report](report_reviewed.md) answers the four study questions and includes the complete tables and figures. [Original generated report](report.md) and its numerical artifacts remain unchanged. [Execution/code guide](../../experiments/rx-posterior-step2-A-20260930/README.md) describes assets and scope.

The run completed 200-source calibration and 100-source development at -5/-2/1/4/7/13 dB with three fixed noises. It has 25,200 image metric rows, 108,000 token diagnostic rows, 360 timings, 40 fixed examples and eight figure files. Training updates are zero. -5/-2 dB pass G1 and common-weight content checks; 1/4/7/13 dB select true bypass. Low-SNR improvement is outside the original training range, and VAR costs about 9.7 times B2 receive time.

## Full tables in Git

The repository requires each tracked file to be at most 10,000,000 bytes. The unchanged full `per_frame.csv` and `token_accuracy.csv` exceed this limit; all of their rows are committed as CSV shards of at most 8,000,000 bytes. [The manifest](table_shards/manifest.json) records ordered row ranges, per-shard SHA256, and the original full-table bytes/SHA256/row counts. Each shard has the same original header. Keep the first header and concatenate shard bodies in manifest order to recover the exact original bytes.

From the repository root:

```bash
python3 tools/archive_rx_step2_tables.py verify
python3 tools/archive_rx_step2_tables.py restore
```

`verify` checks all shard and full reconstructed hashes without creating the full CSVs. `restore` writes the two exact originals and verifies them; if an existing full table differs, it stops without overwriting it. The two restored originals are narrowly ignored by Git. Report links to these CSVs work after restoration.

## Original execution evidence

[Original result SHA256](provenance/original_result_sha256.json) describes the initial 68 result files, including the two full tables recoverable from shards. It predates the reviewed report, README, table archive and provenance copies. It is not a manifest of all files introduced by this commit.

`provenance/` contains byte-preserved copies of the server's completion, four necessary selfchecks, batch4 GPU check, calibration seal, development identity and the two preserved preflight failures. [Provenance index](provenance/index.json) identifies their original paths and hashes. Failures are engineering history, not scientific observations. The original completion receipt field `files=25200` denotes metric rows, not a count of files.

The original config binds the model, selected checkpoint, data, statistics and 13 executed source files. Additional publication docs/export tooling are not part of those execution bindings. No model, dataset, tensor, cache, password or background download script is published here.

All methods share the same per-frame N/E, transmitted waveform and observed noise. Per-source statistics average three noises before the shared 10,000 bootstrap resamples. The development population has been used before; intervals omit training-seed variability and multiple-comparison correction. Token and common-weight outputs are diagnostics, not new transmissions or deployable policies. At 13 dB the unbypassed fusion candidates significantly worsen LPIPS; the final bypass preserves B2.
