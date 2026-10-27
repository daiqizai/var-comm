# Frozen 10 dB TX-policy SNR mismatch diagnostic

**Complete: 1800 logical receive evaluations; no further science is pending.** This uses the existing diagnostic100 population, not a new blind test. The main common500/Kodak curves and policies are unchanged. P2 was not started.

WHOLE targets (9,0), PARTIAL targets (8,126); both use q=4, rate=2/3, k=2549, 2520 entropy bits, N=1024 (68+956). Actual terminated-stream fitting uses the unchanged registered fallback. True SNR is 8/10/12 dB; RX LLR uses the true variance. Each source has three pre-generated, shared standardized complex AWGN vectors, scaled across the three SNRs; channel RNG is separate from generation/protocol RNG. `noise_identity.csv` identifies all 300 vectors and `source_list.json` all 100 original identities.

At 8 dB, both methods reject all 300/300 receptions (body CRC); all 600 gray outputs remain in the means. At 10 and 12 dB, both accept 300/300. Received prefixes and fixed reconstruction inputs at 12 dB exactly match 10 dB, so quality is unchanged. This exposes a downward operating boundary of the frozen policy, not robustness to arbitrary SNR error or fading.

## Four metrics and pointwise source intervals

Three noises are averaged within source, then 100 sources receive equal weight. All entries below include failures. Higher PSNR/CLIP-I and lower LPIPS-Alex/DreamSim are favorable. CSVs retain original precision.

| Family | True SNR | Metric | Mean [95% source CI] |
|---|---:|---|---|
| WHOLE | 8 | psnr_db | 11.454925 [11.050138, 11.881314] |
| WHOLE | 8 | lpips_alex | 0.951218 [0.934161, 0.967541] |
| WHOLE | 8 | dreamsim | 0.940848 [0.933318, 0.948181] |
| WHOLE | 8 | clip_image_cosine | 0.512354 [0.497318, 0.526763] |
| WHOLE | 10 | psnr_db | 18.951420 [18.367959, 19.538639] |
| WHOLE | 10 | lpips_alex | 0.201818 [0.187885, 0.215821] |
| WHOLE | 10 | dreamsim | 0.131544 [0.120236, 0.143151] |
| WHOLE | 10 | clip_image_cosine | 0.858719 [0.840972, 0.875278] |
| WHOLE | 12 | psnr_db | 18.951420 [18.367959, 19.538639] |
| WHOLE | 12 | lpips_alex | 0.201818 [0.187885, 0.215821] |
| WHOLE | 12 | dreamsim | 0.131544 [0.120236, 0.143151] |
| WHOLE | 12 | clip_image_cosine | 0.858719 [0.840972, 0.875278] |
| PARTIAL | 8 | psnr_db | 11.454925 [11.050138, 11.881314] |
| PARTIAL | 8 | lpips_alex | 0.951218 [0.934161, 0.967541] |
| PARTIAL | 8 | dreamsim | 0.940848 [0.933318, 0.948181] |
| PARTIAL | 8 | clip_image_cosine | 0.512354 [0.497318, 0.526763] |
| PARTIAL | 10 | psnr_db | 19.611725 [19.082782, 20.145736] |
| PARTIAL | 10 | lpips_alex | 0.175170 [0.165281, 0.185088] |
| PARTIAL | 10 | dreamsim | 0.107702 [0.100034, 0.115676] |
| PARTIAL | 10 | clip_image_cosine | 0.875937 [0.860688, 0.890518] |
| PARTIAL | 12 | psnr_db | 19.611725 [19.082782, 20.145736] |
| PARTIAL | 12 | lpips_alex | 0.175170 [0.165281, 0.185088] |
| PARTIAL | 12 | dreamsim | 0.107702 [0.100034, 0.115676] |
| PARTIAL | 12 | clip_image_cosine | 0.875937 [0.860688, 0.890518] |

## Direct source-paired PARTIAL minus WHOLE

| True SNR | Metric | Difference [95% CI] |
|---:|---|---|
| 8 | psnr_db | +0.000000 [+0.000000, +0.000000] |
| 8 | lpips_alex | +0.000000 [+0.000000, +0.000000] |
| 8 | dreamsim | +0.000000 [+0.000000, +0.000000] |
| 8 | clip_image_cosine | +0.000000 [+0.000000, +0.000000] |
| 10 | psnr_db | +0.660305 [+0.515228, +0.813956] |
| 10 | lpips_alex | -0.026648 [-0.032089, -0.021339] |
| 10 | dreamsim | -0.023841 [-0.029993, -0.017972] |
| 10 | clip_image_cosine | +0.017218 [+0.009911, +0.025193] |
| 12 | psnr_db | +0.660305 [+0.515228, +0.813956] |
| 12 | lpips_alex | -0.026648 [-0.032089, -0.021339] |
| 12 | dreamsim | -0.023841 [-0.029993, -0.017972] |
| 12 | clip_image_cosine | +0.017218 [+0.009911, +0.025193] |

The existing seed=2026100701 bank has shape 10000×100 and SHA256 `5e11cd2ee931a2de9f22f8e42034a71ef610eb83a48bac9e53224e666f5a447d`. It was used once for 24 mean intervals and 12 directly paired intervals. No new bootstrap indices were generated. These intervals describe sources, not 300 independent images.

## Actual computation and exact reuse

The 1800 logical evaluations comprise 1200 new physical frames (2400 header/body decodes) and 600 exactly verified old 10 dB events. The audit verifies input streams, paid profiles, waveforms, actual standardized noise and observations, plus numerical provenance. The 1200 accepted logical RX outputs reuse 161 exact existing native source/reconstruction pairs; 600 rejects map to 100 source/gray pairs. No new tokenizer, TX probability traversal, arithmetic source receiver or VAR reconstruction ran.

New scalar scores: 100 PSNR, 100 LPIPS-Alex, 100 DreamSim and 161 CLIP-I = 461. Existing unique scalar scores reused: 583. These map to all 7200 logical score positions. Metric weights, float preprocessing and registered FP32 device routes are inherited and pinned. Source embeddings are reused only under matching execution identity. `actual_compute_and_reuse.json` provides counts; `p1_metrics_registration.json` provides input/loader bindings.

## Preserved engineering failure and independent closure

The original link worker completed every packet call and logical loop, then exited 1 at a final assertion that CUDA must remain globally uninitialized. The qualified Sionna backend was explicitly CPU; its constructor initializes a global CUDA context with zero allocated CUDA bytes. The original failure and actual exit 1 remain untouched. This assertion is not a reception failure and did not trigger a scientific retry.

Independent `postprocess_link_v1.py` performed only saved-array/ledger metadata checks: all 2400 packet records were COMPLETE, 1200 saved physical results and all 600 old-event equivalence proofs were verified. Its completion is separately recorded; it does not relabel the original exit. Both metric processes and the single statistical job subsequently exited 0, with original PID/start identities checked gone. Exact receipts are retained in `../execution_evidence/`; large noise/observations and ledgers remain in PFS.

The 0.660305 dB contrast here is from the main frozen 10 dB image route. The earlier controlled-grid diagnostic reports 0.660149 dB from its separate registered route. They round alike but are not substituted for one another.
