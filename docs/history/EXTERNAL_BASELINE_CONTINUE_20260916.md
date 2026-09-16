# External baseline positioning: continuation

## Latest: non-diffusion work complete; HiFi resumed (2026-09-16 19:05 UTC+8)

The user requested completing other work first. Read `VAR_COMM/reports/external_baseline_non_diffusion_result_20260916.md` and the receipts before acting. The old "remaining" list below is a17:36 snapshot, **not a command to rerun completed digital calibration/evaluation/timing**.

- Swin4500/ADJSCC4500 and13500 protocol metric rows complete; old four-system6000 reference rows reused.
- `digital_calibration_001`:18000 transmissions,100 fixed `rint(linspace(0,999,100))` sources, two budgets4204/4498, raw/arithmetic, m7/8/9, original three calibration noises and five SNRs. `digital_policies_001.json` frozen before new-budget development. Both families/budgets choose m8 at1dB, m9 otherwise. Original LPIPS/PSNR-guard rule, DINO unused.
- `digital_development_001`:6000 selected-policy transmissions, all failures;757 new unique renders, other rows use exact observable-RX argument memoization. This is not GT repair or free online timing. Do not rerun the completed matrix.
- `digital_timing_001`:1920 complete CPU-to-CPU TX/RX pairs at the new budgets; every waveform and RGB exactly reproduces quality. No inference caches. VAE108948355+VAR310283520 parameters; single-system peak~1872MiB. Old R3/Deep single-model memory/full residency parameter scopes remain unmeasured; old co-resident memory cannot fill them.
- `non_diffusion_analysis_001`:25500 rows,119 summaries,1624 source-paired metric intervals; complete original/paid protocol tables, quality-resource and RX-cost plots,124 preselected PNG files including24 comparison boards. `audit.json`:all6000 new PHY replay,183 preselected pixel/PSNR/SSIM checks, row coverage and means PASS. LPIPS/DINO and all bootstrap intervals were not independently rerun.
- `completed_author_analysis_001` is the earlier pre-digital snapshot, retained. Use `non_diffusion_analysis_001` for current completed evidence.
- HiFi yielded after committed frame52 for the short queue; old queue interruption/failure record is an explicit user-priority scheduling event, not a method failure. `non_diffusion_priority_queue_001` contains the pause/resume provenance. All six author bindings unchanged; no images deleted, no sampling changed. It resumed19:05:20 with parent4160517/worker4160518; always recheck live `author_queue_001/status.json` rather than trusting these PIDs.

Next: let the original HiFi1500-frame queue and its scoring finish; then verify its standalone/companion ADJSCC pixels and transmitted/received observations agree, do not double-count that baseline, and extend the completed analysis in a **new output directory**. Existing summarizer intentionally excludes partial HiFi and is sealed by its completion receipt; make a new extension instead of rewriting completed evidence. Final `reports/external_baseline_positioning_result.md` remains absent/uncompleted. No new training, architecture, CSI study, final holdout or Git push is authorized by this continuation.

## Live state (2026-09-16 17:36 UTC+8)

- User uploaded all five official-link checkpoints. All supplied SHA256SUMS pass. ImageNet diffusion also matches the official Azure Content-MD5. Paths/hashes in `configs/protocol.json`.
- Private `.venv`: torch1.12.1+cu116, torchvision0.13.1+cu116, numpy1.23.5, Pillow9.5.0, PyYAML6.0.2, timm0.4.12. Shared environments untouched. Download/network failures retained in `downloads/` and tool logs.
- Source commits: Swin `a6d0e6da53548976acbe9317839a077ef31f190f`, DiffCom `a8cc4d63304f5bb514d69576e689ff2412510161` (pristine sparse checkout `vendor/diffcom_code`). Swin source archive under `vendor/SwinJSCC`.
- Output root: `../../outputs/EXTERNAL-BASELINE-POSITIONING-20260916/` relative to this directory's project context; absolute `/workspace/projects/VAR_COMM/outputs/EXTERNAL-BASELINE-POSITIONING-20260916/`.
- Author queue PID4031247, initial Swin worker4031248, tool session44962. Read `author_queue_001/status.json` for the actual current child, not these historic PIDs.
- Queue started17:34:31; at17:35:58 Swin2522/4500 frames. Queue sequence: Swin4500 → ADJSCC4500 → fast metrics → HiFi1500 → HiFi metrics.
- Full HiFi calibration at10dB:235 actual reverse steps,78.732s full RX,8.696GB peak allocated; source calibration only. Expect ~30–36 GPU-hours for its1500 default-setting frames. Do not silently reduce steps, images or noises to finish sooner.

## Completed qualifications

- `inputs_001`: exact original100 development inputs;8 fixed calibration inputs;6000 four-system reference rows with common SSIM. No old model reruns. PSNR check error3.81e-6; R3 SSIM CPU/GPU difference1.79e-7.
- `qualify_swin_001`: strict checkpoint load and original-path waveform/receive equivalence at three rates; max RGB error≤5.37e-7.
- `qualify_adjscc_001`: strict load and original-path equivalence, max RGB error2.38e-7. Its108-use metadata was a HiFi-side-info engineering fixture, not the standalone ADJSCC budget; formal standalone ADJSCC uses no header.
- `profile_hifi_001` failed because our adapter disabled parameter autograd required by the author's custom checkpoint backward. Failure retained. Fixed by preserving author autograd flags while using no optimizer; parameter version counters and lack of accumulated parameter gradients are checked. This is fixed-weight posterior sampling, not training.
- `profile_hifi_002` (3-step cost-only) and `qualify_hifi_full_001` (complete235-step author schedule) pass. Full output uses final `x_t`, as upstream does, not a selected intermediate.
- `calibration_adjscc_001` and `calibration_swin_001`:120 frames each,8 sources×5 SNR×one fixed calibration noise×3 rates. `calibration_metrics_001` complete. No choices fitted with DINO or development.
- Six CPU tests pass, including compact metadata roundtrip and exact originalN3060 digital waveforms.

## Frozen information/budget rules

- See `../../reports/external_baseline_protocol_freeze_20260916.md` from VAR_COMM context and `configs/protocol.json`.
- Author Swin needs actual normalization power and source-dependent active-channel mask. HiFi needs actual encoder power; bare ADJSCC does not. Never provide those fields silently for free in unified comparisons.
- Lossless float32 power plus combinatorial subset rank, original CRC16/tail6/K7/repeated-header PHY. Data first, metadata afterwards, preserving the entire same ADJSCC data observation for HiFi.
- Common budgets: ADJSCC C2=4096; HiFi C2=4204; Swin RA32=4498; ADJSCC C4=8192; Swin RA64=8754; ADJSCC C6=12288; Swin RA96=12952. Mean complex energy2. Author-assumed-info rows separate.
- Interleaver uses pre-shared numeric frame counter, not source filename/content. Sampling seed23 fixed. Legal failed-CRC metadata candidates retained; illegal Swin metadata→gray, illegal HiFi power→same-observation ADJSCC. Unexpected implementation/numerical errors halt for review, not silent frame removal.
- Bare C2 appears both standalone and as HiFi attribution output. Final analysis must verify equality and not double-count it. Prefer standalone timing/peak memory; paired output has diffusion weights resident and is explicitly marked.

## Do not edit running inputs

`author_queue_001/bindings.json` seals queue/evaluation/scoring drivers, `protocol.json`, `author_models.py`, `radio.py`. No editing these files while the queue is active. It stops on a detected other GPU process or an implementation failure. Preserve data and diagnose; do not restart duplicate workers. Per-frame commits allow same-version resume.

## Required work still remaining

1. Implement/complete the necessary digital equal-budget study atN4204 and4498, using existing raw/arithmetic codecs and candidate m7/m8/m9 only. `src/external_positioning/digital_budget.py` is implemented and its originalN3060/noiseless-budget CPU tests pass; **the common-budget evaluation driver and calibration are not implemented yet**.
2. Reuse old `outputs/COMMUNICATION-CONVERGENCE-20260915/{CALIBRATION,DEVELOPMENT}_001/images/{index:04d}/transmissions.npz`: it already stores exact arithmetic `payload_m7/m8/m9`. `source_tokens.npz` holds original tokens/label. Do not recompute TX entropy probabilities unnecessarily. Use100 fixed linspace sources from original1000 calibration, original4101/4102/4103 noises, five SNRs. Apply existing quality/BLER rule on calibration before new-budget development.
3. Old deterministic clean reconstructions/metrics can be reused only for identical receiver arguments; no truth-based recovery or output selection. Online timing must not receive cached TX source encodings or cached RX images for free.
4. Supplement necessary new-budget digital timings and individual peak memory/parameter scopes. ExistingN3060 timing CSV reports `GPU_all_models_peak_allocated_bytes`, not individual model peaks. Don't label that as per-model memory.
5. Aggregate author metrics with the four existingN3060 references and true equal-budget digital rows. Protocols/budgets/populations stay separate. Source-image paired intervals, primary1/4/7 and high13/19, all failures.
6. Preselected visual examples: indices0/25/50/75, seed2001. No choosing best-looking images or generation seeds.
7. Final report: `VAR_COMM/reports/external_baseline_positioning_result.md`. **Not yet produced; research positioning is not complete.** Author queue completion explicitly retains this remaining work rather than claiming full task success.

Do not rerun weight-closure, R3 completion, old digital full audits, train new methods, access new holdout, or start CSI research. New research files are in canonical VAR_COMM, not the publication clone; no new Git push was requested for this measurement turn.
