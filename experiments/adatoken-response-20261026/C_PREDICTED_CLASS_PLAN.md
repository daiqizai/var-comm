# Ada-TokenCom fixed-frame predicted-class adaptation — revision 1

2026-10-26 user steering supersedes only the unlaunched main C route in PLAN_zh.md. A endpoint granularity and this C external comparison have equal first priority. B is already scientifically closed and will be reported without new B inference. Registrations describe limits, not completion.

## Attribution and frozen choices

Name: **Ada-TokenCom (predicted-class, fixed-frame adaptation)**. This is not a reproduction of Ada's long-term average-budget controller, time-varying channel or ARQ. Preserve official LlamaGen-L, VQ-16, 256 RGB source preprocessing, raster token order and class-conditioned prefix entropy coding/tail generation. The null-class first8/24 qualification stays archived as **LlamaGen + LDPC (Ada-inspired)**; it never becomes the main Ada curve.

Primary evidence: Ada arXiv:2608.28086v1, sections II and IV; official LlamaGen commit ce98ec41803a74a90ce68c40ababa9eaeffeb4ec. The paper does not specify a complete sampling configuration. We therefore explicitly freeze the official sample_c2i defaults for tail sampling: temperature 1, top-k 2000, top-p 1, CFG 4, cfg_interval -1. FP32, TF32 off and math SDPA are this experiment's numerical adaptation, not claimed paper defaults. No temperature/CFG search or test-score tuning.

Source arithmetic probabilities use the conditional branch's full vocabulary, without top-k or CFG modification; integer CDF total 2^24 and canonical 32-bit arithmetic termination. TX and RX both compute the same two-branch conditional/null batch, so the conditional CDF uses the same numerical route. Tail sampling alone combines the two branches using the official CFG formula. Count the additional null-branch forward work explicitly.

## Class condition and paid frame

Predict one class from the sender's fixed 256x256 float RGB using the already frozen ConvNeXt-Tiny ImageNet1K_V1 weights (SHA256 983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d) and its original official tensor transform, FP32, batch one. Identical procedure on cal1000, common500 and Kodak24. Use zero-based ImageNet class ordering, never source directory labels or evaluation ground truth. Record classifier implementation, transform, weight and source pixel identities.

Actual body retains the common serializer order: **container length 13 bits + class ID 10 bits + terminated arithmetic stream + zero padding + CRC16**; the length counts the class and arithmetic stream together, and class IDs 1000–1023 reject. CRC protects the length/class/stream/padding together. Thus arithmetic capacity is min(k−39, 8191−10), not k−29. The existing 68-symbol protected profile header and 956-symbol body remain N=1024, Es=2. Shared paid profile identifies actual prefix/MCS. No free class side channel; the receiver uses only decoded class, actual stream and shared model. Header/body/length/class/arithmetic failure gives fixed gray and retains the denominator. Explicit null profile is used if no legal prefix fits. Do not amend the original PHY class in place.

## Population, candidates and budget

Keep cal1000, common500 and Kodak24 distinct. Retain the predeclared fine fixed-frame candidate set **all prefix lengths 1..256**, each independently terminated from one probability traversal. Report the paper's 32,64,...,256 subset as a deterministic capacity diagnostic from the same streams; no additional quality sweep or choice based on formal test results. Actual longest-fitting prefix follows paid capacity; no source bits are truncated.

Calibration uses the already declared six SNR/MCS combinations: 4 dB q2 r1/2 or 2/3; 10 dB q4 r2/3 or 3/4; 19 dB q4 r2/3 or q6 r5/6. DINO-L source-averaged calibration objective, three noises per source; freeze MCS before formal test. Formal PSNR, LPIPS-Alex, DreamSim and CLIP-I on common500 then Kodak24, with three noises and original source-level paired bootstrap banks. Keep ARPC; no other methods rerun.

Finite cumulative C caps, including preserved null qualification:

| Operation | Maximum |
|---|---:|
| Sender classifier forwards | 1,524 |
| Tokenizer calls | 1,524 total: 8 already done, at most 1,516 new |
| Source probability traversals | 1,532 total: 8 null archived + 1,524 conditional |
| Independently terminated streams | 392,192 total: 2,048 null + 390,144 conditional |
| Calibration logical frames | 18,000 |
| Formal logical frames | 4,716 (4,500 common500 + 216 Kodak) |
| Physical qualification events | 24 |
| Total new physical events | 22,740 |
| Header/body packet decodes | 45,480 |
| Conditional source-only RX qualification | 24 (8 sources × 32/128/256) |
| RX/render calls, conservative cumulative cap | 22,788: 22,740 physical + 24 conditional source-only + 24 archived null |
| Formal four-metric logical pairs, each metric | 4,716 |

The RX cap explicitly corrects the original plan's omission of source-only qualification from its conservative sum; no population or PHY expansion follows from this accounting correction. Exact received payload/class/prefix/model/generation/numerical identities permit reuse; all logical failed frames remain. Eight existing VQ token arrays are reused only after source/model/hash validation. Conditional recoding is a changed input, not a retry of the old null traversal. Unknown failures stop that stage; completed per-source commits are never silently repeated.

Keep the original total ceilings 96 GPU device-hours and 96 CPU process-hours, inclusive of archived qualification; 12-hour per-stage ceiling. No training, new seeds, MCS expansion, ARQ or resource-setting changes. Preflight qualification first: synthetic class framing/CDF/route tests, fixed first8 conditional source encodes, independent 24 source roundtrips and actual image review, then 24 PHY qualification events. Only their actual closure can admit calibration. Each later stage registers exact remaining caps and inputs before launch.

## Reporting

Report all predeclared SNRs and outcomes, actual prefix length, arithmetic bits including finish, class/length/CRC cost, utilization and acceptance. A identifies effects of endpoint granularity within fixed NeST models; C compares complete systems with different tokenizers/priors/decoders. Neither alone proves a representation-order causal effect. Update Fig.2 and claims only after qualified formal results; preserve ARPC and old results. Fig.3 is not expanded automatically.
