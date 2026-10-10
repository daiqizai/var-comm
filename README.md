## Original100 pilot complete; full-calibration shortlist fixed (2026-10-10)

The original-calibration pilot has actually completed reconstruction and all four metrics for all 43,200 logical rows: 100 fixed sources, 144 frozen candidates, SNRs 4/10/19 dB and noise seed 4101. The reconstruction owner exited 0 after 1,019.01 seconds. Its separate ledger closes one model load, 1,408 source decodes, 1,408 VAR reconstructions, 1,408 frozen Dc calls and 25,142 prior scales. Exactly 1,409 new float32 image archives include the constant 0.5 failure image; 453 logical rows reuse closed48 images and 41,338 reuse an image from this pilot. All 103,369 archive members, every logical-to-physical-to-image reference and all 29,367 pairs of model-call receipts were independently verified. Failed receives remain in the population.

The four-metric owner exited 0 after 305.38 seconds. It constructed three metric models and prepared 100 references, then actually scored 1,530 unique source/reconstruction pairs; 41,670 logical rows reuse an identical pair under the same metric provider. Its 9,483 call reservations all completed, including 3,060 actual LPIPS backbone forwards. Every pair binds both source and reconstruction pixel hashes. PSNR, LPIPS-Alex, DINOv2 ViT-L14 cosine and ConvNeXt source-prediction agreement are retained for all 432 candidate-by-SNR means; agreement is not classification accuracy. No new bootstrap or holdout evaluation was performed by this pilot.

The complete-pilot DINOv2-L ranking admits the following top three candidates plus the original full-calibration whole winner at each SNR. Candidate-ID lexical order resolves exact ties. These are a shortlist for the original1000 full calibration, not its final winners:

| SNR (dB) | Complete-pilot top 3 | Retained original whole winner |
| --- | --- | --- |
| 4 | `m7_K25_q2_r2-3`, `m7_K50_q2_r2-3`, `m7_K75_q2_r2-3` | `m7_q2_r2-3` |
| 10 | `m8_K42_q4_r2-3`, `m8_K126_q4_r2-3`, `m8_K84_q4_r2-3` | `m9_q4_r2-3` |
| 19 | `m9_K192_q6_r5-6`, `m9_K128_q6_r5-6`, `m9_K64_q6_r5-6` | `m9_q6_r5-6` |

The source900 owner has now actually exited 0 after 543.45 seconds, closing one model construction, 900 fresh TX passes and 9,000 prior scales with no unresolved reservations. Combined with the previously closed source100, all original1000 source streams are available. This TX-only stage made no independent RX, reconstruction or quality calls, so it does not establish full-calibration quality. Its request SHA is `39bc1c524450ecd981a5c2dafdeb5c898bde06b2723e24f08da3efdf711354ca`.

At the checked publication snapshot, the separately registered original1000 PHY owner is actually RUNNING on CPU cores 0/1, request `81ae17e2dea3569424323d919cd84a089ed967a970a7a4cbc189bd8e48b5f2a5`. It evaluates the four admitted candidates per SNR, original1000 sources and seeds 4101/4102/4103, with at most 36,000 logical frames and 72,000 new decoder calls before exact-observation reuse. The one-step raw mismatch PHY owner is also actually RUNNING on CPU cores 2/3, request `9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf`. It retains the original common500 first100 sources, true-SNR noise and 500-stride counter, and admits 5,400 logical/4,500 physical frames with at most 9,000 new packet calls and no historical-event reuse. It uses the unchanged raw433 receiver domain; the entropy457 qualification is not presented as a raw433 qualification. Both workers keep CUDA hidden and have separate finite ledgers. These are running snapshots, not completed PHY, reconstruction or metric results.

The full1000 reconstruction and metric owners remain prepared but unexecuted at this snapshot. Final strategy freezing requires the complete original1000 population and all three registered noises. New100 confirmation remains pending: historical source-exposure/hash closure is still open, and no new confirmation pixels have been selected. No new bootstrap was performed.

The closed CPU pilot remains unchanged at 17,201 new physical observations and 34,402 actual LDPC decoder calls. Logical reuse never adds historical calls to the new-call ledger. Frozen protocol, models, actual received-input boundaries and whole-winner retention remain unchanged.

[Closed reconstruction receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_visual_v1.json), [closed four-metric receipt and candidate means](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_metrics_v1.json), [exact pilot finalists](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_finalists_v1.json), and [closed CPU pilot receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_phy_v1.json).

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
