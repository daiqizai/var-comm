# Raw configuration mismatch: local engineering review

Date: 2026-10-10. This is metadata preparation, **not an executable scientific runner, deployment, registration, or completed experiment**. No PHY, model, scoring, bootstrap, new source selection, remote connection or existing result modification was performed. Historical reuse actually admitted at this stage: **0 frames**.

## Frozen scope and exact original identity

- N1024; raw whole and raw partial; original common500 sources at original indices 0 through 99, in the original order. This is a posthoc subset of the already evaluated holdout, not a new blind holdout.
- Actual SNR 4 dB uses configuration lookup 1/4/7; actual 7 uses 4/7/10; actual 10 uses 7/10/13. No policy optimization, new candidate, or extra SNR.
- Keep original noise seeds **6201, 6202, 6203**. Keep protocol `CONTENT-REAL-64QAM-H-20261006-V1`, stage `holdout`, canonical JSON and the original NumPy PCG64/float64 (1024,2) standard variate. No new experiment, method or lookup namespace enters its seed.
- Keep `public_counter=(index(actual_snr,[1,4,7,10,13,19])*500+original_source_index)*3+noise_index`. In particular the factor remains **500**, not 100. Source 0 at true 4/seed6201 has counter1500; source99 at true10/seed6203 has counter4799.
- Only the TX configuration lookup uses `config_snr_db`. The noise multiplier is `10**(-actual_snr_db/20)`. Both header and body decoders receive `actual_snr_db`. Preserve the original noise-variance convention; no extra factor1/2.

The source asset manifest is pinned to `b27128fb8eedee7f2cc74bed25c63e39e8add8c938c76d449f85c4b0d5a51e09`, original plan to `1ab63b0e0bdde865d677467232c956ebbae746239ac300dd7723e22dbe8c7bce`, original noise/counter source to `1acc294032f46166ff2b714b6d47de1ac7aa8bb37619c8b75729f801ac249fec`. The generator reads only the two JSON metadata files. Its input byte pins, original index order, all six original SNR policies and original noise specification are checked before output.

| Lookup dB | Whole TX profile, m, K | Partial TX profile, m, K |
|---|---|---|
| 1 | 183, 5, 0 | 141, 5, 18 |
| 4 | 152, 6, 0 | 181, 6, 16 |
| 7 | 264, 7, 0 | 264, 7, 0 |
| 10 | 274, 7, 0 | 231, 7, 75 |
| 13 | 387, 8, 0 | 243, 8, 9 |

Candidate IDs, complete wire keys and modulation are copied at original precision from the pinned plan, not reselected or transcribed from a report. Actual code-rate/layout remains the frozen public catalogue entry; the runner must bind its full metadata rather than infer rates from modulation.

## Cost and reuse ceiling

There are 18 logical cells ×100sources×3noise = **5400 logical frames**. At each true SNR, the two lookup7 cells share profile264: exact scheduled physical dedup gives **4500 physical frames**, including **1500 matched diagonal physical frames /1800 logical frames**, and **3000 off-diagonal physical frames /3600 logical frames**.

Before actual reuse admission the strict fresh packet ceiling is **9000**. If all 1500 matched physical frames pass the independent historical-reception gate, the new ceiling becomes **6000**. Each new frame pays one header and a body only when the received header is accepted; actual accounting is `1+int(header_ok)`. Reusing a paid RX is zero new packet calls. Equal logical arms must remain as separate result rows, including exact zeros, but cannot trigger duplicate physical work. Runtime dedup additionally verifies actual payload/waveform/received-array identities, not just the scheduled key.

The conservative image-render ceiling is one attempt per physical frame (4500 without reuse), reduced only by authenticated image reuse or exact received-state/model reuse; gray needs no VAR completion. This is not a model-callback ledger definition. The future runner must register actual model components and its stage limits explicitly. No qualification or probe calls are included or authorized by this metadata plan.

## Existing interfaces and minimum new owner

Read-only code reviewed:

1. `experiments/main-raw64-20261007/actualcal_r3/main_raw64_keep_receiver.py` in the local remote-source snapshot: `transmit_frame` (line124) already takes profile ID and public counter, without an SNR. `Receiver.receive` (line75) already takes one real SNR and sends it to **both** header and body demapping. Its full433 catalogue chooses body profile from the actually decoded header, including a legal incorrectly decoded ID. Reuse these lower interfaces unchanged, with a new independent charge callback/ledger and output directory.
2. The same directory's `main_raw64_packet_adapter.py`, `demap` and `receive_packet`, preserves q2/4/6 mapping. Do not introduce a new QAM mapping or substitute the lookup SNR as the demapper variance.
3. `.research/wcl_evidence_closure_20261009/remote_snapshot/main_raw_keep_receiver.py`, `validate_header`/`present_actual` (lines167/182), supplies the unchanged KEEP parser.
4. `.research/main_raw64_20261007/take_over_v1/unified500_same_rx_parallel_r2/raw64_unified500_same_rx.py`: `validate_policy` (line64), `SourcePairs.validate` and `SourcePairs.reconstruct` (line136) bind the **original matched schedule**. Do not pass new mismatched rows through this wrapper or forge its original fields. For matched historical admission its sealed source/point consumers remain useful; the new mismatch renderer should call the unchanged lower actual-state renderer with a new independent registration.

Minimal outstanding engineering is a new registration/ledger owner, a historical RX/image/metric admission adapter, a CPU transmit/noise/receive worker, a bounded actual-state GPU renderer, a source-paired four-metric consumer and closure/report stage. These have **not** been implemented or deployed here. Each physical frame must be durably reserved before its first paid callback; duplicate identity/request changes or unresolved reservations stop the run. Crash recovery must reconcile actual receipts rather than silently rerun uncertain calls. Actual parent wait/exit receipts and complete output seals are required before each stage claims completion.

Use explicit `actual_snr_db` and `config_snr_db` throughout. A compatibility field named just `snr_db` may represent actual SNR only, with this meaning declared. Event IDs must be new experiment IDs while the **scientific noise seed material and public counter remain the original ones**.

## Exact historical reuse gates

The matched schedule makes reuse a candidate; it does not prove reuse. The metadata checker reports `METADATA_IDENTITY_MATCH_ONLY` and always `scientific_reuse_admitted=false` even on equality. An independent admission owner must:

1. Authenticate original normal CPU completion, bound registration/config/final freeze, all actual worker waits/exit0 and the required output seals. The local CPU completion is `.../current/unified500_raw_parallel_execution_r2/completion.json`, SHA `b424925cfb4d6ea2295f2e3658cc870196e1a20419914a5b132d0181a32ba682`. Its `all_waited=true` is evidence to verify, not a replacement for bound actual receipts.
2. Authenticate the selected source archive/checkpoint, source ID/index, token hash and preprocessing hash. Locate matched point checkpoints through sealed metadata, not guessed filesystem search. Preserve the original 500-index correspondence.
3. Compare actual SNR/seed/counter/seed material and actual standard-noise, payload, TX waveform and observed-wave hashes. Compare intended profile/candidate/wire, full433 catalogue/aliases, exact old PHY/runtime/source bindings, original `raw_holdout` receiver phase, group0 and `actual-body` session. Verify original paid header/body event request/results; reparse stored actual hard bits/CRC with the old parser. **Do not invoke a decoder merely to validate reuse.**
4. Use accepted **RX** profile and actual hard tokens. No comparison may require RX profile to equal TX profile; a legal header misdecode is a result. Never replace actual body hard bits with source tokens or downgrade KEEP to DROP.
5. To reuse images additionally authenticate the normal visual closure and actual wait/exit, actual-state/receiver-view hashes, frozen model weights/flags/source identity, and actual FP32 image array hash. The local original visual completion SHA is `918766339ea91f1d53bfb0c1ee15fa33c284f4da32d381b94aba4b749fa2db8a`. It seals remote `outputs/MAIN-RAW64-20261007/unified500_same_rx_parallel_images_r2/{sources,source_checkpoints,images}/0000...0499` assets. Read the selected first100 exact paths from that seal. Historical archive or final PNG identity alone is insufficient.
6. For metric reuse compare exact source pixels, image hash, metric models/preprocessing/runtime/source-code identities and original per-frame values. Reuse four original metric values only after those conditions hold. Original500 aggregate summary/paired CSV cannot supply a100-source subset estimate. Any new statistics require their own separately frozen stage, never new policy selection.

At this review, the two completion metadata files were read, but selected remote point/image/ledger bytes and parent receipts were **not** newly verified; hence no reuse is claimed. The plan lists all required binding names per PHY/image/metric layer.

## Failure and result schema

Keep all5400 logical rows. Store source/index, seed, actual SNR, lookup SNR, family and physical alias; TX profile/candidate/m/K separately from actual RX profile/m/K; header CRC/legal/header_ok; body_attempted/body CRC; actual receiver state, image reference/hash, waveform/noise/observation hashes; E_frame/rho; paid event IDs and new-call/reuse evidence.

- Header reject: body is absent, one paid packet call, gray image; retain original `body_crc_accept=false` but set `body_attempted=false`. This is not a body-CRC trial.
- Header accept: two paid calls and parse the actual received profile. Body CRC failure still retains actual hard prefix and raster partial tokens (`KEEP_ACTUAL_HARD_TOKENS`), with no truth repair.
- Runtime/parser invariant failure: incomplete execution requiring diagnosis; never silently relabel it as a gray channel failure or drop its row.
- Metrics remain `psnr_db`, `lpips_alex`, `dinov2_vitl14_cosine`, `convnext_top1_source_prediction`. LPIPS sign is unchanged. ConvNeXt is prediction agreement, not class accuracy; plot differences in percentage points. Do not drop zero effects or failures. Whole/partial have identical modulation/coding permissions and partial may equal whole.

## Entropy-partial TX timing inherited from completed T4

The completed T4 convention is implemented in `experiments/wcl-evidence-closure-20261009/scripts/t4_timing_owner.py`, `t4_timing_endpoint.py` (`Meter`, `TimingReservations`, `fresh_case`) and `t4_raw_endpoint.py`. Do not modify these frozen files.

- Keep development fixed16 order `(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)`, SNR4/10/19, batch1 FP32, 3warmup plus3measured repetitions per image/workpoint. This timing population is distinct from mismatch100 or quality500; disclose it.
- Each new method costs144 measured+144 warmup = **288 fresh TX executions**, 48 measured observations per SNR. Explicitly register the number of new methods. A TX-only experiment has no packet-decoder calls and cannot report measured RX or software end-to-end time.
- Bind actual T4 environment/weights/flags/thread6/interop2/CPU affinity4..9/GPU0 and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; retain the actual venv executable path without resolving its symlink into system Python. Admit an exclusive GPU run and the exact bound waiting parent only. If the environment differs, report the difference rather than present pooled timing as one measurement.
- CUDA synchronize before and after each measured window; use `perf_counter`. Start TX with CPU uint8 CHW pixels and end after the complete1024-symbol IQ frame. Include fresh Encoder/VQ, conditional probability evaluation, entropy coding, every length/fallback attempt, header/FEC and modulation. Load weights beforehand; put logging/file I/O outside the window. Do not reuse cached tokens/CDF/streams/output as online work. Nested components cannot be summed twice.
- A new partial endpoint must use the newly frozen partial codec/profile/action and actual m/K. The old `EntropyEndpoint.choose` supports whole prefixes and its receiver renders K0; it cannot honestly measure the new partial mechanism unchanged. Freeze and qualify the partial endpoint before any timing.
- Register new durable TX reservations with explicit callback counts and no automatic retries. The original full T4 `TimingReservations` reserves two RX decoder slots and requires a positive packet cap; **do not use it unchanged for TX-only accounting**. Preserve its durable-before-work/no-reuse/unresolved-stop semantics in a separate TX owner.
- Bind per-case output/stream fingerprints across repeated uncached execution; retain failure or unfit attempts and their real cost. Report mean/median/p95/min/max, units, sample counts and boundaries. A partial-TX-only result leaves RX/E2E unmeasured. Existing original T4 totals are real historical comparators with their own population/environment receipt, not newly measured values.

## Local checks and reproduction

Twelve pure metadata tests passed, including all900 source×trueSNR×seed counter/seed-material identities against the exact original source, grid/dedup counts, illegal grid/source mutations, byte pins, each PHY/image/metric identity mutation, actual wrong-legal-header and bad-body-CRC KEEP cases, no-body header failure, duplicate/uncertain accounting, and rejection of a cache path as a reuse proof. No standard-noise array was generated and no PHY/model/scorer/statistics code was called.

From the repository root:

```text
python -B -m unittest discover -s experiments/entropy-partial-mismatch-20261010/scripts -p test_mismatch_plan.py -v
python -B experiments/entropy-partial-mismatch-20261010/scripts/mismatch_plan.py --original-plan .research/main_raw64_20261007/take_over_v1/current/unified500_raw_execution_v1/plan.json --source-manifest .research/main_raw64_20261007/take_over_v1/current/common500_source_assets_v1/manifest.json --out <fresh-path>/mismatch_metadata_plan_v1.json
```

Actual local metadata output: `.research/entropy_partial_mismatch_20261010/mismatch_metadata_plan_v1.json`, SHA `ca7e179bf5c9224a0301cea46f823bdf2b05114b5c89caf0df9edcbab3ed02ff`. The output is deliberately marked `METADATA_ONLY_NOT_SCIENTIFIC_REGISTRATION`; it refuses to overwrite an existing file.
