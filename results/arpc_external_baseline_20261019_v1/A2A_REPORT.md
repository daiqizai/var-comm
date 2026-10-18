# ARPC A1 and A2a: independently decoded, screened; actual link calibration next

Official ARPC weights, native RGB256 seven-scale schedule and paid caption/container
passed A1 independent decoding and actual visual/resource review. A2a then completed
92 missing caption/tokenizer/TX traversals and676 independent RX states, inheriting
8 source passes and24 exact prior states. The final cal100 contains all700 clean
prefix states. No ARPC wireless frame or evaluation-set score has run.

Three frozen metrics were actually computed on all700 native float outputs and100
constant0.5 gray references:800 pairs/2400 scalar scores,100 source DINO embeddings,
800 LPIPS pairs/1600 backbone forwards,800 reconstruction DINO embeddings,2 model
loads. DINO-L alone is the screening objective. Original FP32/TF32off preprocessing
and checkpoints are pinned. No ConvNeXt or other unneeded metric model was loaded.

## Source lengths and clean quality

These are clean-prefix calibration results, not wireless performance. Payload
counts include caption, compact mode/boundary metadata and termination/alignment;
the outer13-bit length and16-bit CRC are additionally paid within k. The complete
e7 representation cannot fit any allowed N1024 capacity in these100 sources.

| Prefix endpoint | Payload min / mean / max (bit) | PSNR (dB) | LPIPS | DINO-L |
|---|---:|---:|---:|---:|
| 1 | 144 / 293.28 / 456 | 10.2691 | 0.715762 | 0.126850 |
| 2 | 200 / 338.56 / 504 | 11.4428 | 0.658662 | 0.127517 |
| 3 | 488 / 624.16 / 792 | 13.1266 | 0.561377 | 0.168621 |
| 4 | 1064 / 1212.64 / 1392 | 14.5215 | 0.476154 | 0.217252 |
| 5 | 2088 / 2258.88 / 2464 | 16.0268 | 0.340014 | 0.340882 |
| 6 | 4416 / 4621.04 / 4824 | 17.6005 | 0.249277 | 0.554621 |
| 7 | 8576 / 8806.64 / 8992 | 20.7239 | 0.135421 | 0.860701 |

## Complete-strategy screening

All84 strategies are retained in actual_fallback_quality.csv (8400 source-strategy
records) and strategy_proxy.csv (252 SNR-strategy summaries). Actual per-source
length fallback supplies the quality term; no capacity-frequency ranking is used.
Historical3600 distinct packet events cover12MCS at4/10/19dB,100 sources percell.
They share the frozen physical backend but carry old NeST payloads/catalogue:
their success probabilities are screening proxies, not measured ARPC reliability.
Pointwise and36-cell Bonferroni95% Clopper–Pearson intervals retain uncertainty
even for100/100 successes. No new packet was generated for this extraction.

Each SNR fits the original two-candidate uncertainty gate. Proposed maximum target
is e7, with actual fallback:4dB QPSK2/3 andQPSK1/2;10dB16-QAM2/3 and16-QAM3/4;
19dB64-QAM5/6 and16-QAM2/3. These are nominations, not frozen final winners.
The next gate is a separately registered real header/body/source-parse qualification
within24 engineering frames, then at most18000 cal1000 logical frames. A3 remains
unrun until real calibration freezes each policy. No bootstrap has been performed.

## Execution and preserved failures

Clean TX/RX/audit actually exited0 (128.9032/587.0199/18.7810s), followed by
metric0/75.3699s and CPU screen0/.1698s. All known child/owner processes are gone.
The original A1 initialization failed before model load because this host exposes
different NVML/CUDA UUIDs; the separately registered PCI/index adapter fixed only
resource identification. A later metric launcher import omitted a wait validator,
stopping before preparation or any scoring. Its original bytes/error are preserved;
execute_checked.py supplies the validator,8 new CLI tests pass, and no successful
scientific call was repeated. Initial Windows clean-test import lacked fcntl;
the authoritative Linux qualification completed18 synthetic tests once.

Source/weight/runtime bindings, actual waits and failure evidence accompany this
milestone. Heavy arrays/models remain on shared PFS; restoration backup is deferred.
Historical raw NO_GO, LE, T1-Cm10, existing baselines and completed timing are unchanged.
