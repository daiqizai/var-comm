# T0 raw / candidate / data / energy audit

Read-only local audit, 2026-10-09. The user document was read in full. No SSH, model loading, inference, FEC decoding, channel simulation or bootstrap was executed by this audit. No old science file was changed. The server state and Git commit are for the root coordinator to verify independently; historical completion receipts do not certify that every remote cache is still present.

## Direct findings

| Asset | Status | Verified finding |
|---|---|---|
| Original N1024 common500 result tables | REUSE_EXACT | Local staging copy has all six SNRs and all four requested metrics for WHOLE, PARTIAL, PARTIAL Direct, P1024 and Swin. The 120 selected rows have no duplicate identities; 500 sources and three noises; mean lies inside each interval. |
| Original raw action catalogue | REUSE_EXACT | 433 profiles, 433 distinct wire keys; 106 K=0 actions and 327 K>0 actions. All actions remain in the derived coverage table. |
| Strongly protected whole actions | REUSE_EXACT | Seven K=0 full-budget profiles plus 99 nominal profiles. The full-budget configurations include m8/64QAM with k/n=3076/5736, 956 body symbols and no padding. |
| Original shortlist / proxy | REUSE_EXACT | All 433×6 proxy scores exist. Only three WHOLE actions per SNR entered the original final calibration. The old selection completion is normally closed and policies used all 1,000 calibration sources. |
| Expanded whole actual DINO pilot | NEED_NEW | No evidence here of all 106 actions having actual quality scores. The new pilot must retain all 106, with source/noise identity checks before old frame reuse. |
| Raw/P/Swin resources, energy, failure states | REUSE_EXACT | A4 v2 already includes 24 selected configurations, 36,000 frame memberships and 54,000 reconstruction-quality rows. Six original CSV hashes match its completion receipt. Actual energy was saved; no transmitter replay is needed to produce these summaries. |
| Calibration received-token evidence | REUSE_COMPONENT | 30 local source-0000 checkpoints inspected, each with the three original calibration noises. Receiver states, hard decoded payload bits, actual frame energy, waveform and received-observation hashes are retained. Full remote cache existence still needs confirmation. |
| Holdout raw and reconstructed images | REUSE_COMPONENT | Closed historical raw receipt has 16,500 physical frames / 18,000 WHOLE/PARTIAL memberships. Image receipt has 33,000 logical images represented by 9,282 stored images. Remote per-source NPZ payloads are referenced, not locally reopened by this audit. |
| Previously fixed 16 examples | REUSE_COMPONENT | They are development examples, not common500 holdout. Raw/P/Swin six-SNR PNGs exist locally. Preserve their source IDs and label their population accurately. |
| Adaptive BPG fixed16 target examples | BLOCKED_MISSING_ASSET | Existing A5 completion certifies 16 development images at 13 dB only. It does not certify a 10/19/low-SNR grid; remote matching records must be checked. |
| Train20k tokens for static frequencies | BLOCKED_MISSING_ASSET | The training population uses parquet-row identities, separate from ImageNet-val source IDs. The remote cache registration path is known, but actual token contents were not inspected locally. This is an asset audit gap, not a claim that training was never done. |

## Exact local paths

All paths below are relative to `C:/Users/11946/Documents/ChatGPT/comm`.

- Original public tables: `.research/main_raw64_20261007/take_over_v1/final_publication_r6/actual_staging_r6/results/main_raw64_20261007/final_common500_r6/`.
- Catalogue: `.research/main_raw64_20261007/assets/catalogue.json` (SHA `af772451d33a5750d77a57ba7f296b851cdff2baf098cd96896141818b3d2f03`).
- Proxy: `.research/main_raw64_20261007/current/prepare_contract/new/prescreen_r2/proxy_scores.json`.
- Final policies: `.research/main_raw64_20261007/take_over_v1/current/final_selection_v1/policies.json` (SHA `7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c`).
- Original frozen holdout: `.research/main_raw64_20261007/take_over_v1/final_freeze_v1/FINAL_FREEZE.json` (SHA `3f2b43a8a771be7f0c04d60bae7862b1f052c8c35f0236865ef904e6e388194a`).
- Calibration1000 source IDs / token archive references: `.research/main_raw64_20261007/current/prepare_contract/old/H/full1000_assets/manifest.json`. Its historical H-codec status was incomplete at that snapshot; token metadata and IDs still exist. Do not use that old status to negate later completion.
- Development100 source IDs: `.research/main_raw64_20261007/current/development_contract/CONTENT-REAL-64QAM-20261006/H/development100_source/manifest.json`.
- Holdout500 source IDs / source token+latent archive references: `.research/main_raw64_20261007/take_over_v1/current/common500_source_assets_v1/manifest.json`.
- Actual prior split: `.research/main_raw64_20261007/holdout_metadata/original_split_manifest.json` (SHA `576ffc8db4d9ced560af66e55c185fcd2d1a078b9ffa9315b372ea15f2532959`). The calibration manifest order matches this split exactly; calibration, development and holdout sets are pairwise disjoint.
- A4 actual tables: `results/paper_supplement_20261008/a4_resources/v2/`.
- Original fixed16: `.research/main_raw64_20261007/take_over_v1/current/development_fixed16_cached_display_v2/examples_manifest.json`.

`provenance.json` contains freshly computed byte hashes of all audited inputs, model file bindings, frozen state hashes, runtime metadata, and method/noise/source references. `data_source_lists.csv` exports all 1,600 verified source IDs; the first32/first100 calibration flags are stable-order proposals for this new protocol, not a claim of already launched new experiments.

## Original scientific facts relevant to T2/T4

- At 10 dB, selected WHOLE is m7/K0/64QAM, k/n=1876/5628, header/body/padding=68/938/18. PARTIAL is m7/K75/16QAM, 2776/3824, 68/956/0. There are 66 CRC-rejected KEEP frames for PARTIAL, and no gray frames; CRC rejection does not mean a gray output.
- At 19 dB, selected WHOLE is m8/K0/64QAM, 3076/3696, 68/616/340. PARTIAL is m8/K142/64QAM, 4780/5736, 68/956/0. The selected whole padding does not imply full-budget protection was prohibited; the catalogue proves the alternative exists.
- `existing_energy_summary.csv` directly transforms the saved E statistics to rho=E/(2N). It does not fabricate equal-energy quality curves or adjust old SNRs.
- Calibration noise labels are 4101/4102/4103; original raw holdout labels are 6201/6202/6203; baseline holdout labels are 2001/2002/2003. A source-level paired comparison does not assert identical continuous-waveform observations.

The public summary SHA is `5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859`; paired CSV SHA is `dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235`.

## Local deliverables

`candidate_coverage.csv` contains all 2,598 candidate×SNR rows with proxy ranks, shortlist membership, final winners, exact MCS/resources and full-budget permissions. Its new actual-quality fields are deliberately blank with `NOT_STARTED`; no pilot result is invented. `observed_rx_checkpoint_inventory.json` distinguishes observed local RX bytes from remote paths. `remote_asset_requests.json` provides bounded remote follow-ups. `audit_completion.json` records completed local validation only.
