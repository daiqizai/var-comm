# T2 finite whole-action calibration audit

The expanded finite calibration check retained both original whole-scale winners at 10 and 19 dB.
This is calibration evidence, not a new holdout result or a global-optimality claim.

## Actual completed scope

- Original catalogue: 433 unique wire actions, including 106 whole-scale actions and 7 whole-scale full-budget protection actions.
- Pilot: every one of the 106 whole actions, the first 100 original calibration sources, one original noise realization (4101), at 10 and 19 dB: 21,200 logical frames.
- Full recheck: the pre-registered union of pilot top5, both original winners and all seven full-budget whole actions. Each SNR has 10 eligible whole actions and one fixed partial reference; 1,000 original calibration sources and three noises (4101/4102/4103).
- Actual full closure: 66000 logical frames and 81200 newly charged packet decodes, within the 132000 independent cap.
- Pilot actual packet decodes: 41200; pilot plus full total: 122400. Exact received-state/image/score reuse did not invoke new packet decoding.
- Two GPU source workers passed the registered exact RGB and metric parity checks before full production; their qualification receipts remain in the scientific closure.
- Full rankings average the three noises within each source before averaging sources. Only DINOv2-L ranks eligible K=0 actions; candidate ID is the deterministic tie break.

## Retained winners

| SNR (dB) | Profile | Original proxy rank | Pilot rank | Full PSNR (dB) | Full LPIPS | Full DINOv2-L | Body CRC rejection | Header rejection | Changed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 10 | 274 | 1 | 1 | 18.40309 | 0.21868 | 0.65177 | 0.000% | 0.000% | False |
| 19 | 270 | 1 | 1 | 20.07435 | 0.16254 | 0.75775 | 0.000% | 0.000% | False |

CSV exports retain the input numeric strings at original precision; the table rounds for reading. No confidence intervals or bootstrap were newly computed.

## Interpretation and limits

The complete whole-action pilot makes the original proxy pre-screen visible. `candidate_coverage.csv` reports the original proxy rank, original shortlist membership, pilot rank, full eligibility and every actually measured quality/failure rate separately. A candidate missing full1000 measurements is left blank, never imputed from its pilot mean.
The 100-source, one-noise pilot and the 1,000-source, three-noise recheck have different populations and precision. Pilot quality levels should not be compared directly with full means as a performance change.
All seven full-budget whole controls were included in the full recheck. Padding in the retained 19 dB winner therefore describes the selected configuration; it does not establish that the whole family was denied full-budget protection.
The partial arm was a fixed reference and was never eligible to win the K=0 ranking. CRC rejection rates describe receiver events; under raw KEEP, a CRC-rejected packet may still supply a reconstruction and does not automatically mean a gray image.
This audit covers only 10 and 19 dB. It does not establish full1000 optimality across all106 whole actions, extend the check to the other four SNRs, or establish a global optimum over modulation, coding, packetization or source representations.
Both winners remained unchanged, so the registered policy requires no new500 holdout rerun. Published original holdout results and the original PARTIAL policy remain unchanged.

## Provenance

- Full request SHA256: `140ac78e9dcae9270c73524f0ef88b1569631f748a21075b52b3a293c122cc31`.
- Full completion SHA256: `39f76f1ae9b092c87c82ca3533711cbe1e2c631849b3e2bc4ac4da3ca5a1f748`.
- Pilot request SHA256: `67eaa4f68c26ed3a1b6f70e72cfb90201028864d223604c37d0aaf9cad17688e`.
- Full execution commit: `14b09ecd72984fb39c683d1a62bcd3221c69142f`.
- Original result files, policies and ledgers are unchanged. This export reads only JSON/CSV, writes a new directory and starts no scientific successor.
