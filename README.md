## Original100 pilot PHY complete; reconstruction running (2026-10-10)

The original-calibration pilot completed all 43,200 logical candidate frames: 100 fixed sources, 144 frozen candidates, SNRs 4/10/19 dB and noise seed 4101. Exactly 17,201 new physical observations used 34,402 actual LDPC decoder calls (17,201 header and 17,201 body), with no unresolved reservations. Another 102 logical rows reused individually matched closed48 observations and 25,897 reused an observation from this pilot. These reuse counts are not additional decoder calls. The CPU owner exited 0 after 856.43 seconds, with CUDA hidden and no neural inference or quality scoring.

All 60,418 exported archive members were verified by SHA256. Each logical row was checked against its actual physical receipt, including known closed48 references, and every new header/body outcome matched its durable decoder result. CRC failures remain present: 11,457/14,400 logical rows at 4 dB and 6,721/14,400 at 10 dB; all 14,400 rows at 19 dB parsed. These denominators include exact-event reuse and describe receive outcomes, not image quality.

At this publication snapshot, the separately registered reconstruction owner is RUNNING on shared GPU 2 with two CPU cores, a 16 GiB PyTorch allocator cap and a finite six-hour window. Its request SHA is `b46f78b01390516d8a8729bb6e7bac897ba24f290c246c456d0c0cd490176bb3`. It uses only actual received profiles and payloads, preserves fixed-gray failure outputs, and reuses an image only when the received source input, device, models, numerical settings and implementation agree. The frozen running source is not edited or restarted by this publication. This snapshot does not establish reconstruction completion, four-metric results or a ranking.

The first 100 source streams remain closed: prior32 are retained and new68 completed exactly one model load, 68 TX passes and 680 prior scales; their 1,632 new endpoint streams are byte-verified. Original source/PHY/image gates and all earlier failures remain preserved. Full1000 calibration, new100 confirmation and the mismatch evaluation remain NOT_RUN. The later pilot shortlist must retain the original full-calibration whole winner. Historical source-exposure auditing remains open and no new confirmation pixels have been selected.

[Closed pilot PHY receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_phy_v1.json) and [closed source100 receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_source100_v1.json).

## Completed WCL supplementary evidence (2026-10-10 publication)

[T0–T6 final report, results and figures](results/wcl_evidence_closure_20261009/README.md) are complete. The finite expanded calibration retains the original N1024 whole-scale winners. Same-prior VAR entropy coding improves N1024/19 dB quality at greater TX cost. The separate N2048 confirmation completed 2,700 method frames: partial-scale PSNR increments over raw whole-scale are +1.220451, +0.059652 and exactly zero dB at 4/10/19 dB. The report retains uncertainty, negative results and the different 500/100-source populations.

Original scientific results and frozen protocols are preserved. This publication follows the user's explicit push request; it runs no new scientific experiment. Restoration transfer status is tracked separately from Git publication, and historical monitors remain paused.

# VAR_COMM — single source worktree

## Completed Kodak24 generalization experiment (2026-10-09)

[Results and delivery index](results/generalization_kodak_20261009/README.md): all 24 predetermined 256x256 center crops, N1024, 4/10/19 dB, three noises, and four frozen methods completed 864 reconstruction frames. Unconditional VAR uses the null embedding; there was no training or Kodak calibration/policy selection. Four metrics, source-paired intervals, resource/configuration tables, and fixed-source visualizations are complete. The independent 1080-call PHY budget closed; the original root ledger remains 534231.

DINOv2-L similarity and source-prediction agreement favor the proposed method against every baseline at all three SNRs in pointwise paired intervals. The report retains the tradeoffs: PSNR is below adaptive BPG at all three SNRs, and 4 dB LPIPS is worse than latent continuous JSCC. Four 10 dB body-CRC failures remain in the prescribed KEEP population. Swin 19 dB is outside its training/calibration range. These 24 fixed crops provide limited cross-collection evidence, not full-resolution Kodak evaluation or proof of pretrained-data non-overlap. Earlier common500 results remain unchanged; historical monitors remain paused.


## Latest completed supplement (2026-10-09)

[Paper supplement and result index](results/paper_supplement_20261008/PUBLICATION.md): the A-class first round is complete, including the later-added adaptive BPG + LDPC baseline (500 sources x 6 SNRs x 3 noises), four metrics and source-paired comparisons, native Swin checks, five-method single-output timings, resource/failure tables and paper figures. The original frozen common500 results at commit `252176e` remain unchanged.

Adaptive BPG has higher PSNR at every tested SNR; the proposed method has better LPIPS, DINOv2-L and source-prediction agreement. All points and negative findings are retained. Swin at 19 dB is outside its training/calibration range; native256 BPG timings do not measure the adaptive source search. External native replication and B/C extensions remain conditional and were not started. No training or new scientific work is triggered by this publication; the monitor remains paused.


The actual project root is now the sole Git worktree for `daiqizai/var-comm`.
Edit, run, test, commit and push here. `publish/var-comm` is retained locally as an ignored historical backup; no source-copy or separate publication workflow is required.

## Latest completed study (2026-09-30)

[Frozen P4084 Step2 A conclusions](results/rx_posterior_step2_A_20260930_R1/report_reviewed.md) and
[complete results with exact table restoration](results/rx_posterior_step2_A_20260930_R1/README.md).
The run has zero training updates. Policies bypass at 1/4/7/13 dB; -5/-2 dB
outside the original training range exceed B2/A1/A2 with common-weight support,
at about 125 ms versus 13 ms receive time. B/P_low and Step3 require a separate
decision. [Execution and asset guide](experiments/rx-posterior-step2-A-20260930/README.md).

## Earlier study registration (2026-09-23 snapshot)

The following entries preserve earlier study registrations and runtime snapshots;
they do not restart the currently paused historical queues.

[Token/channel efficiency supplement](experiments/token_channel_efficiency_20260923/README.md)
adds source-codec accounting, P2048/P3060 and real16QAM. A/B now precede unstarted
short-prefix training; the active cache finishes unchanged. New holdout and
content selectors are deferred. Experiment A is complete; see [actual source ledger and representation report](reports/token_channel_efficiency_20260923_source_A.md).
P2048/P3060 real-device qualification passed and budget training is ongoing.
Runtime receipts distinguish implemented code from NOT_RUN real experiments.


[Short-prefix protocol and execution](experiments/var-short-prefix-hybrid-20260923/README.md): real preflight/qualification complete; full training and later evaluation stages pending.

## Completed review and bounded study

- [Phase1 repairs and affected real-weight results](reports/review_20260923_phase1.md).
- [Phase2 real experiments, numerical-protocol correction and measured tradeoffs](reports/review_20260923_phase2.md).
- [Phase2 evidence and run identities](results/review_20260923_phase2/README.md).
- [Actual execution commands and preserved launch history](experiments/var-latent-enhancement-20260917/research/EXECUTION.md).

These runs preserve original A/B and use the existing development population. They do not certify every historical issue, open a new holdout, or complete m10/new-Decoder digital adaptation.

## Check a checkout

```bash
python tools/verify_repository.py
python tools/run_cpu_checks.py
```

CPU checks require Python 3.10+, numpy, torch, PyYAML, LPIPS/torchvision and pytest (see requirements-cpu.txt). They use source from this checkout and include synthetic engineering tests and stored-result validation. They do not train models or recompute real image quality. External weights, datasets, caches and third-party implementations remain local and are not distributed. Historical GPU asset paths may need configuration on another host.

## Contents and provenance

- `src/`, `experiments/`, `scripts/`: actual execution code, configurations and tests.
- `src/cadsd_jscc/`: the small self-written historical runtime dependency closure, with source hashes in the migration records. External model libraries are not bundled.
- `tools/`, `tests/`: repository checks and lightweight result reproduction.
- `results/`: historical published data plus selected lightweight run artifacts, with run IDs and original hashes. `outputs/` remains in place and ignored.
- `reports/`, `docs/`: scientific evidence and scope limitations.
- [Migration record](reports/repository_migration_20260923.md).
- [Research status](RESEARCH_STATUS.md) and [historical result index](docs/RESULTS_INDEX.md).
- [Previous publication overview](docs/history/publication_README_before_20260923.md) and [previous worktree overview](docs/history/worktree_README_before_20260923.md).

Repository migration does not certify the previous repair task as complete. No original A/B training, new method, or GPU quality evaluation is part of this migration.
