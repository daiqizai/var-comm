# Expanded whole-action calibration audit v1

Scientific base: 14b09ecd72984fb39c683d1a62bcd3221c69142f. The original raw policy and all old ledgers/results remain unchanged.

Pilot population is exactly the first100 entries in the original1,000-source calibration manifest, in stored order, without score selection. It uses all106 distinct K=0 wires from the original433-profile catalogue, including all seven full-budget protection layouts. SNRs are10 and19 dB, noise label4101. The original calibration `standard_noise` function and counter remain unchanged: full six-SNR order, original1,000-source dimension, and original three-noise stride. No new SNR-dependent noise namespace is silently introduced.

The pilot contains21,200 logical frames. Its independent SQLite ledger permits at most42,400 packet decode events, reserves durably before each call, caches exact completed results, and refuses unresolved-event retries. Header failure may result in fewer actual body calls. The original root ledger is never opened. Full1,000-source rechecks and holdout evaluations require separate frozen requests; the pilot does not automatically launch either.

CPU processes use the frozen mixed433 public catalogue, original raw12 payload serialization, real Sionna LDPC, original 68-symbol paid header, original QAM mapping, original KEEP receiver and known energized padding. The old `main_raw64_calibration_checkpoints.prepare_frame` constructs expected source/SNR/counter/energy/waveform/received hashes. An old selected frame is reused only after all expected fields match exactly and the actual hard-bit receiver state is rechecked under the effective mixed433 catalogue. Each source is checkpointed; independent worker locks prevent overlapping workers of the same index.

GPU execution uses the original `h_source_driver.render_received` and original `quality_driver.MetricScorer`, with unchanged model state, numerical flags and metric identity. Its receiver input contains only the actual received state. Image and quality reuse requires the identical state for the same source, frozen model and metric identities, and a hash-verified original reconstruction archive. It does not infer clean output from CRC success and does not substitute source tokens for corrupted KEEP output. Class labels are passed only to the historical metric scorer, never to VAR or the receiver.

Before new pilot rendering, one existing non-gray calibration state is rendered again and required to match the original float32 reconstruction exactly. One metric replay must match original PSNR/LPIPS/DINO-L within predeclared absolute tolerances1e-4/1e-5/1e-5. This finite qualification is one VAR call and one metric image, separate from the pilot accounting. The already frozen batch8 metric qualification is copied and verified; it is not rerun. This is not retraining or new model selection.

New render upper bound is21,200; realistic cost is determined after CPU execution from exact distinct received-state keys and existing scored-state matches. Correct same-prefix states across MCS can share an image. Error states can share only when all received tokens and recovery state match. No ETA based on optimistic BLER replacement is asserted.

The pilot ranking uses mean DINOv2 ViT-L/14 cosine across100 sources, ties by candidate ID. PSNR, LPIPS and failure rates are descriptive. Output rankings explicitly say pilot-only; no bootstrap or test-performance claim is produced. The next calibration union is pilot top5, original WHOLE and PARTIAL winners, and relevant full-budget WHOLE controls, deduplicated and separately registered before running. Negative or zero differences do not change the candidate set.

Execution stages:

1. `t2_pilot.py prepare --visual-request <old actualcal visual request> --out <new WCL output> --request <new request> --workers 8 --deadline-unix <explicit timestamp>`: metadata and hash registration only.
2. Original LDPC Python, CPU-only with OMP/MKL/OPENBLAS/NUMEXPR threads2: `t2_pilot.py cpu-worker --request <new request> --index <0..7>`. Source subsets are index modulo8; the global ledger cap is shared.
3. Original unified metrics Python, GPU0 with existing deterministic environment: `t2_pilot.py qualify-gpu --request <new request>`; then `gpu-worker` using the same request. GPU lock is shared with original visual jobs; no competing process is stopped.
4. `t2_pilot.py close --request <new request>` requires all100 CPU and GPU source checkpoints, all106×2 actions per source, and no unresolved packet reservations before emitting COMPLETE.

GPU resumes only completed state/source checkpoints; unresolved render reservations are preserved for explicit diagnosis, not repeated automatically. An incomplete CPU source ends GPU execution without silently dropping a candidate. No checkpoints or policy are written to historical result paths.
