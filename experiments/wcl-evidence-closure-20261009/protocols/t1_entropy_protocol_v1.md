# T1 source and finite-candidate protocol v1

Status: source/candidate implementation protocol, registered before new channel quality results. This is not a frozen winning policy or completed test report. The primary system and published common500 remain unchanged.

## Families and syntax

The two new families are EC_STATIC_WHOLE and EC_VAR_WHOLE. Both transmit a **pure arithmetic-coded whole prefix**. Unlike historical H-A, neither chooses raw bits when shorter. Both choose only a shorter whole prefix when the actual stream exceeds the paid capacity. There is no image-quality-dependent per-source decision.

Static CDFs use only the original20,000 training images, original views, the existing complete ten-scale full_tokens cache. The CPU fitting stage checks all200 shard SHA values against sealed training_statistics, source descriptors, exact registered source IDs/order and all680 token IDs. It uses full_tokens without splicing legacy base_tokens. Existing prefix differences are reported. Fixed per-scale shared counts use +0.5 pseudocounts, then the existing positive24bit CDF rule: floor(p*(2^24-4096))+1, remainder to first argmax. The CDF model is fixed and pre-shared; no per-image fitting, calibration frequency counts or labels.

VAR probabilities use the original frozen VAE/VAR/Dc identity and null class1000, FP32, full4096-codebook distribution, no CFG/truncation. Each RX context is independent and advances only with its decoded coarse scales. Existing H m6–m9 actual bitstreams and independent roundtrip receipts are reused only after source/pixel/token/model/source-SHA checks.

The integer arithmetic coder uses32-bit intervals and a24-bit CDF. Actual terminal bits are included. Streams are bit-addressed; there is no byte alignment on wire. The exactly30 deterministic zero-lookahead reads of the decoder are syntax, not transmitted source bits or source truth. Exact receiver re-encoding checks canonical termination; wrong streams that remain canonical are not rejected using truth.

## Paid N1024 framing

Header:12-bit public profile ID + CRC16 +6 tail bits, existing terminated convolutional code rate-matched to136 bits, QPSK68 complex symbols. Profiles identify family, actual whole m, q, k/n and decoder layout. No image class or transmitter-only CDF is sent or implicitly provided.

Body:956 complex symbols, q in{2,4,6}, n=956q. Nominal rates{1/2,2/3,3/4,5/6}; k=floor(n*r). Inside that fixed profile-known k are13-bit actual source length, arithmetic bitstream, deterministic known zero fill, and CRC16. Actual source capacity is k−29. Internal fill occupies FEC information positions. Frame-tail known-symbol padding is0 for this full-body layout. Constellation normalization is fixed, nominal average Es=2; actual frame energy is recorded, not forcibly normalized.

Target m belongs to{7,8,9}; fallback tries m=target,target−1,...,4, stopping at the first actual arithmetic stream that fits. The target is a calibration policy parameter; actual transmitted m is identified by the paid profile. The final144-profile codebook covers2 families×12 MCS×6 actual m values4…9. Actual Sionna layout and paid header qualification are separate required steps; a source-length fit is not an LDPC qualification or successful reception.

The first source32 CPU gate preceded the m4 extension and intentionally stops its source feasibility audit at m5. Its frozen files/results remain unchanged. The independent m4/m5 supplement resolves this exact low-SNR boundary without changing target candidates or consulting quality. m4's30 tokens bound its worst finite-CDF arithmetic length below the minimum927-bit source capacity; actual encoding still verifies that bound. A missing prefix is BLOCKED_MISSING_PREFIX, not a fabricated TX failure. Sources whose m6 already fits927 bits never use m4/m5 in any admitted bucket, so their unneeded VAR short streams are not generated.

Header rejection, body CRC rejection, impossible length/padding and InvalidSourceStream use the pre-registered fixed gray0.5 output. Model/resource/software errors stop execution and are not grey channel events. The raw main KEEP receiver is unchanged; the raw CRC-DROP same-observation diagnostic is reported separately.

## Fixed data and candidate selection

Source gate: first32 original calibration registration IDs in original order, m7/m8/m9. Pilot: first100 of that same original list, fixed calibration noise seed, SNR4/10/19. Each family has36 candidates (3targets×3modulations×4rates); no extra candidate search. Pilot upper bound is2×36×100×3=21,600 method-frames, before exact waveform/event reuse.

Retain the top3 legal candidates per family/SNR by actual all-frame DINOv2-L mean; fewer if fewer legal choices. Deterministic exact-score ties use candidate_id lexical order. Evaluate the retained candidates on the original1,000 calibration sources×3 original calibration noises; upper bound2×3×3×1000×3=54,000 method-frames. Logical head/body decode calls may be up to2×these frame counts, and are metered separately. Aliased actual transmissions can share actual receive evidence only after exact waveform/profile/source/noise matching, not merely same source ID or clean quality.

Per-family/SNR winning policy is frozen only after full calibration DINO-L. Holdout source streams are prepared only after policy freeze. Each new family formal test is500×3SNR×3noise=4,500 frames, marked post hoc same-source comparison. Existing raw common500 and its existing paired intervals are reused. New paired comparisons use the project-defined source-average3noise and10,000 resampling protocol only when those new comparisons actually exist.

No low-quality result triggers additional targets, data replacement, model training or extended searches. N2048 remains the separately planned next phase.

## Completed-source reuse and online cost

Source asset preparation is offline experimental acceleration and is not online timing. The unified per-source JSON contains exact stream identities and independently validated RX-token cache evidence. A RX cache hit requires family, m, frozen source-model identity, actual received bit length/hash and exact bit-array equality. It never matches by source ID/CRC success alone and never corrects received bits. Changed or syntactically valid wrong bits go through the actual source decoder.

Online TX/RX timing must disable these prepared source/reconstruction caches, include probability calculation and source coding/decoding, all attempted fallback work, FEC and waveform stages. The prepared clean-image arrays and construction timers cannot be reported as runtime latency.
