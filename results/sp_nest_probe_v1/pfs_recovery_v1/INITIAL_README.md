# Superposition NeST-Com probe, SP0–SP2

This branch investigates whether *one* protected 1,024-symbol transmission can serve a weak and a strong receiver better than an optimised single codeword or time division. The gates and fixed grid come from the user work order (`protocol.json`). This directory presently delivers the SP0 radio/codec and a partial, source-identified pilot table. SP1 and SP2 have not run. No training or changes to the WCL manuscript are involved.

**Code.** `profiles.py` constructs the source-independent public directory (including paid NULL). A conservative analytic upper bound on independently terminated lengths removes impossible fallback states; there is no data-dependent filtering. `split_prefixes.py` starts a fresh arithmetic code at each layer boundary and preserves the coarser VAR context. `source_frame.py` is the TX packet interface; `radio.py` uses the archived header, actual 5G NR LDPC backend and distinct public layer scramblers. The RX interface takes the received waveform, receiver noise level, public frame counter and a decoder, not TX indices/bits. Failed B cannot enable E. Failed E retains the already verified B. An invalid model execution raises an error rather than becoming a gray observation. With complex noise variance N0, each real dimension has variance N0/2; the Gaussian log-weight is −distance²/N0. The returned log-odds are positive for bit one, as required by this LDPC decoder. Energy is recorded, not renormalised per frame.

**Evidence.** See `results/sp_nest_probe_v1/REPORT_zh.md`, `resource_ledger.json`, the CSV files and `manifest.json`. Archive identities for WCL, the original source manifest, LE length/quality tables, independent coding checks and the catalog are pinned. Test probabilities used by `qualify_split.py` and `qualify_source_frame.py` are synthetic. Pairing them with a real source's tokens validates the interface, not the source model's scientific output.

**Reproduce the specific checks in an unused output directory.** `REPO` is a checkout with the original dependencies at the pinned SHA. `T6_SOURCE` is the genuine T6 `sources/0008.npz` for the registered Pilot50 source; it is verified by token hash before use. The original LE implementation is under `experiments/layered-entropy-feasibility-20261014/`. Use Python with NumPy, SciPy, Torch, Sionna PHY and the archived header decoder environment. Exact tested backend and CPU precision are in `sp0_radio.json`. On Windows the codec calls the original header receiver via WSL; on Linux it calls it natively. All example commands are for a fresh replay, not for repeating completed results in a live run.

```sh
P=experiments/sp-nest-probe-v1
python "$P/run_sp0_radio.py" --repo "$REPO" --out "$NEW"
python "$P/qualify_split.py" --repo "$REPO" --legacy "$REPO/experiments/layered-entropy-feasibility-20261014/source_split_codec.py" --tokens "$T6_SOURCE" --out "$NEW/sp0_split_compatibility.json"
python "$P/qualify_source_frame.py" --repo "$REPO" --tokens "$T6_SOURCE" --out "$NEW/sp0_source_frame.json"
python "$P/audit_pilot.py" --metadata "$REPO" --csv-root "$REPO" --asset-tar "$ORIGINAL_SOURCE_TAR" --t6 "$T6_SOURCE_DIR" --out "$NEW"
python "$P/export_sp0_evidence.py" --repo "$REPO" --out "$NEW"
python "$P/export_gray_evidence.py" --repo "$REPO" --out "$NEW"
python "$P/screen_old_lengths.py" --repo "$REPO" --out "$NEW/historical_capacity_screen.csv"
```

`audit_pilot.py` accepts repeated `--asset-tar`, `--original-dir`, and `--t6` (native source archives, loose source roots, and T6 roots). Match raw SHA256, source ordering and preprocessing, not filenames. `sp0_source_length_evidence.csv` marks genuine missing `(10,0)` entries; they are not zero-length payloads. `sp0_clean_quality_evidence.csv` preserves missing metrics as empty cells. The original 1,200 rows do not include either perceptual metric. Verify a stored reconstruction against its registered array hash before scoring it, and register/check gray-image scores for all sources. Do not replace native inputs with display PNGs, whole-stream tails, test-fixture CDFs or other model versions.

**Continuation.** Restore access to the registered runtime, verify the original frozen model and cache files, complete outstanding SP0 evidence, then use the predeclared SP1 gate. Resolve the mislabeled 396-candidate count in the work order before scoring the single-codeword frontier. Gates are scientific, not a reason to treat missing inputs as a negative result. Do not submit a SP2 run or present a claim of Pareto dominance until both SP1 channel pairs meet all criteria.
