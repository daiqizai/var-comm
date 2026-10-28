# N512 actual-frame evaluation and WCL delivery

Status: **scientific evaluation complete; manuscript compiled and visually reviewed**, 2026-10-28.
Base commit: `43ccd51473f978510f7310342b7b513f70c3da5d`. Historical N1024 code, results and figures remain unchanged.

## Completed scope

The N512 frame contains 68 protected header symbols and 444 LDPC–QAM body symbols, with nominal Es = 2. It uses actual terminated arithmetic streams, a paid predicted class for Ada, and an explicit paid NULL protocol. All 12 predeclared MCS layouts passed real construction and roundtrip checks. N512 was independently calibrated on cal1000; common500 quality was never used for selection. The pilot comprised 100 sources, one noise realization and 118,800 logical candidate outcomes. Final calibration comprised 1,000 sources, three noise realizations and 105,000 logical outcomes, within the 108,000 cap. Frozen policy SHA256: `dac20e371b1086be105d9c8234bcbc9a97416bc5d4cae954600c9efa310f7605`.

Formal evaluation completed **500 original sources × 3 methods × 3 SNRs × 3 noises = 13,500 logical frames**, with 1,500/1,500 outcomes per method and SNR. Original source indices 0–499 and noises 6201/6202/6203 are retained. Each source is averaged over three noise realizations before equal-source statistics. The existing 10,000 × 500 bootstrap bank (seed 2026100701) was reused; no new indices were generated. Outputs include 36 mean CIs, 24 direct paired CIs and 12 budget-interaction CIs. Every failed gray output remains weighted in the denominator.

## Main findings

- NeST−Ada PSNR is **+4.180, +4.055 and +1.740 dB** at 4/10/19 dB. LPIPS favors NeST at all three points.
- DreamSim favors NeST at 4/10 dB, but **Ada at 19 dB**: NeST−Ada = +0.013923 [+0.010342, +0.017337].
- **CLIP-I favors Ada at all three points**: differences are −0.023521, −0.010768 and −0.008283; all source-paired 95% CIs exclude zero. The results do not support universal superiority.
- NeST−WHOLE PSNR is +0.392170, +0.398573 and +0.792005 dB. All four metrics favor NeST at all three points. NeST/WHOLE utilization is 93.00/75.99%, 93.46/79.30% and 94.68/73.01%.
- Ada utilization is 99.02/99.54/99.52%. At 19 dB, 30 logical events (10 sources) send the full 256-token representation; the remaining sources use shorter legal prefixes. Spare capacity after a complete representation is transmitted is not a fitting failure.

At 4 dB, NeST/WHOLE/Ada body-CRC rejections are 4/7/5 out of 1,500 each; at 10 dB they are 1/0/0; at 19 dB all are zero. Header, parser/source-decoder failures, TX_NO_FIT and NULL counts are zero. These **17 logical failures** map to 13 distinct physical failures. The PHY/source audits found no unexplained discrepancy. The 12,588 new physical events support 13,500 logical outcomes, including 912 additional mappings through exact-state reuse. No N1024 physical event was substituted.

## New work and exact reuse

| Stage | Logical outcomes | Unique physical states | New physical frames | New NeST-family RX | New Ada RX |
|---|---:|---:|---:|---:|---:|
| Pilot | 118800 | 37875 | 37875 | 1508 | 1086 |
| Full calibration | 105000 | 67086 | 64845 | 6232 | 7978 |
| Formal | 13500 | 12588 | 12588 | 2690 | 1500 |

Full calibration reused 2,241 exact pilot physical states. Qualification additionally used 3,384 header-only decodes, 72 full frames and 2 end-to-end frames for numeric-route qualification. New full-frame PHY calls total 115,382; header/body decoder calls total 234,148, including header-only qualification. There were 1,400 new NeST source-probability passes and 100 cache reuses from the same PRO numeric route. All 1,500 Ada source caches were reused; there was no new tokenization. Numeric identity was qualified before source-route reuse, without changing model mathematics.

Formal scoring covered **4,209 unique native image pairs per metric: 4,199 new and 10 exact old Ada pairs**. DreamSim and CLIP each computed 4,187 new embeddings; 500 source features per metric were reused on the identical device/runtime. Formal CLIP uses the current Ada torch 2.10 route and does not mix the older 2.11 cache. Scoring did not trigger new reconstruction or PHY calls. Native arrays, frozen FP32 preprocessing and pixel hashes determine pair identity; scores from display PNGs are not substituted. See `actual_compute_and_reuse.json` for the full ledger and calibration DINO reuse.

Owned-worker wall accounting is 24.183753 GPU-device hours and 2.868865 CPU-process hours, each below its 96-hour cap. These are resource accounts, not isolated processing-latency measurements. The coordinator's elapsed wait is separate.

## Failures and lineage

Initial NeST qualification exposed a cross-device/CDF numerical-route mismatch; its 15 gray outputs and failure record remain preserved. Independent PRO source/RX qualification established the admitted route. The initial NeST/Ada qualification workers also encountered a request-file firewall error after their output loops; metadata auditing did not rerun successful calls. Four Ada pilot workers failed on PCI spelling before model loading and made zero scientific calls; separately registered `_pci_v1` workers performed their first calls. The original coordinator only waited, launched zero stages and was explicitly terminated (−15) for a static path error. The independent `science_chain_v2` completed normally.

Original sources, failed records and corrections are kept separately. All 68 successful worker roles exited with code 0; six historical failed roles exited with code 1. All processes are gone, and no unresolved ledger call remains. The chain and all 15 serial steps exited with code 0. Exact receipts are preserved in `../closed_execution_evidence/` and the post-output audits.

## Interpretation boundaries

The N512 systems independently selected policies from a common pool of 12 MCS layouts. Historical N1024 calibration used different candidate pools. Cross-budget differences therefore describe separately calibrated systems; they do not isolate a causal effect of N or next-scale versus raster ordering. The budget-interaction file reports all 12 prespecified comparisons. The PSNR gap change at 4 dB has a CI spanning zero; at 10/19 dB the gap increases. The manuscript does not claim uniform gap growth.

## Files and manuscript

- `frame_scores.csv`: all 13,500 logical frames, reception states, paid fields, actual endpoints and four metrics.
- `source_scores.csv`: 18,000 source/SNR/method/metric means with original source identities.
- `summary.csv`, `paired_deltas.csv`, `budget_interaction.csv`: full-precision means and CIs.
- `failure_summary.csv`, `capacity_summary.csv`, `endpoint_distribution.csv`: diagnostics retaining every evaluated outcome.
- `N512_Table.csv` and `N512_Table.tex`: three-row Table III, with NeST−Ada PSNR and DreamSim differences.
- `../protocol.json`, `../frozen_policies.json`, registrations, audits and `../delivery_review.json`: experiment provenance.
- `../../../paper/manuscripts/nestcom_wcl_n512_20261028_v1/`: a 5-page, 10 pt double-column PDF; a 14-page, double-spaced review PDF; an editable Word document; LaTeX sources; three unchanged figures; and Tables I–III.

All requested scientific points are complete. Large caches, native arrays, weights and raw logs remain on PFS; user-approved backups remain deferred. Repository checks are recorded separately in `../publication_validation/`.

## Absolute means and source-bootstrap intervals

| method | SNR (dB) | Metric | Mean | 95% CI |
|---|---:|---|---:|---|
| NEST | 4 | psnr_db | 15.498715 | [15.263388, 15.742034] |
| NEST | 4 | lpips_alex | 0.365175 | [0.357503, 0.372692] |
| NEST | 4 | dreamsim | 0.281865 | [0.272737, 0.291066] |
| NEST | 4 | clip_image_cosine | 0.765689 | [0.756483, 0.774891] |
| NEST | 10 | psnr_db | 17.219878 | [16.977684, 17.475980] |
| NEST | 10 | lpips_alex | 0.272596 | [0.266251, 0.278800] |
| NEST | 10 | dreamsim | 0.192165 | [0.184950, 0.199573] |
| NEST | 10 | clip_image_cosine | 0.814979 | [0.806229, 0.823410] |
| NEST | 19 | psnr_db | 19.429779 | [19.167060, 19.701499] |
| NEST | 19 | lpips_alex | 0.190053 | [0.184973, 0.195189] |
| NEST | 19 | dreamsim | 0.121676 | [0.116856, 0.126626] |
| NEST | 19 | clip_image_cosine | 0.864000 | [0.856777, 0.870999] |
| WHOLE | 4 | psnr_db | 15.106545 | [14.853202, 15.366067] |
| WHOLE | 4 | lpips_alex | 0.395124 | [0.385306, 0.404935] |
| WHOLE | 4 | dreamsim | 0.324663 | [0.313084, 0.336611] |
| WHOLE | 4 | clip_image_cosine | 0.746943 | [0.737012, 0.756805] |
| WHOLE | 10 | psnr_db | 16.821305 | [16.590066, 17.063879] |
| WHOLE | 10 | lpips_alex | 0.288079 | [0.282159, 0.293986] |
| WHOLE | 10 | dreamsim | 0.204857 | [0.197644, 0.212067] |
| WHOLE | 10 | clip_image_cosine | 0.807443 | [0.798681, 0.815856] |
| WHOLE | 19 | psnr_db | 18.637774 | [18.384511, 18.902154] |
| WHOLE | 19 | lpips_alex | 0.217542 | [0.212199, 0.222960] |
| WHOLE | 19 | dreamsim | 0.145797 | [0.140382, 0.151325] |
| WHOLE | 19 | clip_image_cosine | 0.848354 | [0.840399, 0.856020] |
| ADA | 4 | psnr_db | 11.318311 | [11.122966, 11.518543] |
| ADA | 4 | lpips_alex | 0.554156 | [0.546594, 0.561675] |
| ADA | 4 | dreamsim | 0.384728 | [0.372927, 0.396566] |
| ADA | 4 | clip_image_cosine | 0.789210 | [0.780741, 0.797634] |
| ADA | 10 | psnr_db | 13.164679 | [12.943322, 13.395177] |
| ADA | 10 | lpips_alex | 0.410977 | [0.403636, 0.418353] |
| ADA | 10 | dreamsim | 0.258171 | [0.248053, 0.268458] |
| ADA | 10 | clip_image_cosine | 0.825748 | [0.817850, 0.833499] |
| ADA | 19 | psnr_db | 17.690030 | [17.387370, 17.997444] |
| ADA | 19 | lpips_alex | 0.214568 | [0.207783, 0.221363] |
| ADA | 19 | dreamsim | 0.107753 | [0.102893, 0.112868] |
| ADA | 19 | clip_image_cosine | 0.872282 | [0.865442, 0.878992] |

## Direct source-paired differences

PositivePSNR/CLIP-I andnegativeLPIPS/DreamSim favorNeST.

| comparison | SNR (dB) | Metric | Mean | 95% CI |
|---|---:|---|---:|---|
| NEST_minus_ADA | 4 | psnr_db | 4.180404 | [3.990809, 4.377417] |
| NEST_minus_ADA | 4 | lpips_alex | -0.188981 | [-0.196626, -0.181392] |
| NEST_minus_ADA | 4 | dreamsim | -0.102862 | [-0.114222, -0.091749] |
| NEST_minus_ADA | 4 | clip_image_cosine | -0.023521 | [-0.031690, -0.015389] |
| NEST_minus_ADA | 10 | psnr_db | 4.055200 | [3.879932, 4.235463] |
| NEST_minus_ADA | 10 | lpips_alex | -0.138381 | [-0.144009, -0.132830] |
| NEST_minus_ADA | 10 | dreamsim | -0.066006 | [-0.074688, -0.057289] |
| NEST_minus_ADA | 10 | clip_image_cosine | -0.010768 | [-0.017624, -0.004078] |
| NEST_minus_ADA | 19 | psnr_db | 1.739748 | [1.577107, 1.908905] |
| NEST_minus_ADA | 19 | lpips_alex | -0.024515 | [-0.027959, -0.021242] |
| NEST_minus_ADA | 19 | dreamsim | 0.013923 | [0.010342, 0.017337] |
| NEST_minus_ADA | 19 | clip_image_cosine | -0.008283 | [-0.012729, -0.003769] |
| NEST_minus_WHOLE | 4 | psnr_db | 0.392170 | [0.340644, 0.444098] |
| NEST_minus_WHOLE | 4 | lpips_alex | -0.029950 | [-0.033659, -0.026404] |
| NEST_minus_WHOLE | 4 | dreamsim | -0.042797 | [-0.048305, -0.037342] |
| NEST_minus_WHOLE | 4 | clip_image_cosine | 0.018746 | [0.014518, 0.023103] |
| NEST_minus_WHOLE | 10 | psnr_db | 0.398573 | [0.355742, 0.444797] |
| NEST_minus_WHOLE | 10 | lpips_alex | -0.015483 | [-0.017026, -0.013743] |
| NEST_minus_WHOLE | 10 | dreamsim | -0.012692 | [-0.014959, -0.010407] |
| NEST_minus_WHOLE | 10 | clip_image_cosine | 0.007537 | [0.004280, 0.010793] |
| NEST_minus_WHOLE | 19 | psnr_db | 0.792005 | [0.740066, 0.844181] |
| NEST_minus_WHOLE | 19 | lpips_alex | -0.027488 | [-0.028826, -0.026162] |
| NEST_minus_WHOLE | 19 | dreamsim | -0.024121 | [-0.026389, -0.021963] |
| NEST_minus_WHOLE | 19 | clip_image_cosine | 0.015646 | [0.012348, 0.018995] |

## Paired cross-budget differences

(NEST−ADA)N512 minus(NEST−ADA)N1024. No new sampling indices.

| comparison | SNR (dB) | Metric | Mean | 95% CI |
|---|---:|---|---:|---|
| (NEST_minus_ADA)_N512_minus_N1024 | 4 | psnr_db | 0.064191 | [-0.121659, 0.248813] |
| (NEST_minus_ADA)_N512_minus_N1024 | 4 | lpips_alex | -0.057036 | [-0.063929, -0.050389] |
| (NEST_minus_ADA)_N512_minus_N1024 | 4 | dreamsim | -0.043633 | [-0.053599, -0.033917] |
| (NEST_minus_ADA)_N512_minus_N1024 | 4 | clip_image_cosine | -0.016147 | [-0.023638, -0.008760] |
| (NEST_minus_ADA)_N512_minus_N1024 | 10 | psnr_db | 3.600140 | [3.401734, 3.796943] |
| (NEST_minus_ADA)_N512_minus_N1024 | 10 | lpips_alex | -0.142495 | [-0.147945, -0.137014] |
| (NEST_minus_ADA)_N512_minus_N1024 | 10 | dreamsim | -0.086583 | [-0.095123, -0.077939] |
| (NEST_minus_ADA)_N512_minus_N1024 | 10 | clip_image_cosine | 0.000233 | [-0.006788, 0.007251] |
| (NEST_minus_ADA)_N512_minus_N1024 | 19 | psnr_db | 0.513466 | [0.333109, 0.700868] |
| (NEST_minus_ADA)_N512_minus_N1024 | 19 | lpips_alex | -0.005136 | [-0.008628, -0.001757] |
| (NEST_minus_ADA)_N512_minus_N1024 | 19 | dreamsim | 0.011706 | [0.008143, 0.014992] |
| (NEST_minus_ADA)_N512_minus_N1024 | 19 | clip_image_cosine | -0.016312 | [-0.020793, -0.011840] |
