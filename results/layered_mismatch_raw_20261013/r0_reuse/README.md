# R0 reuse audit and exact source recovery

This is a read-only data/provenance audit. It performs **zero model, source-encoding, channel, packet-decoding, reconstruction, metric or bootstrap calls**. R1 is blocked; this directory does not authorize supplementary scientific work.

## Sources actually available

- All original calibration1000 archives and checkpoints were read. Every uint8 RGB CHW 256×256 pixel array and int64 680-token array was verified against its original SHA and manifest identity.
- `cal200_sources.json` uses the full, exact source ID and the contract's SHA ordering under `UEP_MISMATCH_CAL200_V1`. `source_index` is the original calibration1000 index; `selection_slot` is the new 0–199 order. Noise counters must use the latter.
- `development100_sources.json` restores the original development100, not the first100 common500 holdout used by the older mismatch diagnostic. All 202 original asset/metadata files and all 100 historical fresh-encoder verification checkpoints were SHA-checked. The existing fixed16 seed remains unchanged; 84 previously unavailable sources are now available. The original development100 has been used before and is not a new blind set.
- Cal1000 and development100 have zero source-ID or exact pixel-hash intersection. This is a split-identity check, not a new claim about every historical training corpus.
- Calibration arrays are reused in place under `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets/`. The restored development100 arrays and raw-calibration metadata are actually stored under `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/`; their exact relocated paths and original SHA bindings remain in the source manifests and hydration receipts. Existing baseline-strength/timing resources use the separate `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/` root. No array or model was moved as part of this documentation repair.

## Historical quality that can be reused for pre-screening

The old UEP table has 295,000 DINOv2-L rows for the original calibration1000. Source IDs, source pixels, VAE/VAR/Dc state identities, Dc gate and the DINOv2-L metric identity match. It provides m4–m8 complete-scale states and per-source gray quality. Positive-K entries use entropy-ranked positions and cannot stand in for raster partial prefixes.

The final directed lookup recovered only the 200 selected sources' original raw actual-calibration metadata and sealed receipts. Of 18,000 existing frames, 17,662 have the accepted correct header, accepted body CRC and a received complete prefix plus raster partial tokens that match the original source **token by token**. The other 338 frames are excluded from the clean-Q lookup. This yields **17 states × 200 sources = 3,400** historical clean values, including 12 positive-K states. Repeated same-source/state images and metric values are identical; the first original JSON record is used, never the best noise outcome. The maximum old-UEP versus raw coarse DINO difference is 6.556510925292969e-7.

`historical_raw_intact_cal200.csv` contains the actual original DINO precision, source/result hashes, original event identity and token-equality evidence. `historical_raw_clean_audit.json` records the complete provenance audit. These are RTX 4090 D historical values usable only as an explicitly disclosed pre-screen reference. They are not new H800 measurements, new R1 draws, or proof that a selected strategy has passed the real-image gate. The original reconstruction arrays were not restored/re-scored; their sealed image-hash receipts were retained.

## Exact remaining gaps

Across the full 689-candidate universe, **27 endpoint states × 200 sources = 5,400** clean qualities remain unavailable. The U/EEP subset lacks **5 states × 200 = 1,000**: m6 K32; m7 K25/K50; m8 K42/K84. `clean_quality_missing_pairs.json` lists every missing source/endpoint; no missing value is imputed.

The independent CPU catalogue audit still reports **1,852 missing exact-layout/SNR BLER points**. See `../candidates_cpu_v1/coverage_v2/` for the authoritative join. The old table covers q2/q4 only, assumes scrambled iid payload applicability and has not established real-source-payload probability validity. Unknown probabilities stay unknown. The small subset with existing BLER and clean Q cannot define a fair full-domain U-versus-S nomination.

The Rayleigh unknown-CSI and feedback studies were historical probability-table/integration diagnostics. They do not supply actual new AWGN mismatch frames. The previous mismatch100 is a post-hoc common500 subset with its original holdout namespace, not this experiment's original development100 or new noise namespace.

## Entries and frozen receipts

`experiments/layered-mismatch-raw-20261013/r0_reuse.py` generated `completion.json` and the source/historical inventory. It refuses to overwrite an existing result. The later exact-raw audit is a separate read-only addendum; `final_audit.json` binds both. The root's `../contract_review_v1.json` binds the preserved historical contract and proves that selection/noise functions were unchanged by subsequent shortlist-validator edits; original source manifests are not rewritten.

`audit_raw_intact_metadata.py` is the exact local CPU audit program used for the metadata addendum. It uses the shared local workspace, the restored evidence under `.research/layered_mismatch_raw_20261013/rawcal_selected200_v1`, and existing source/token metadata. It has no model/PHY imports. The same original evidence is also SHA-verified on PFS in `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/rawcal_selected200_v1`. Repeating this audit is optional and does not justify repeating any scientific frame.

The source-copy steps are recorded in the development and rawcal hydration receipts. Final scientific call counts remain zero. Do not launch R1 or supplement the missing table until the bounded supplemental scope has been resolved by the root task.
