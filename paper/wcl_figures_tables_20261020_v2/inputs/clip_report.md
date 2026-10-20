# CLIP-I: current five-method N1024 common500 supplement

All22,500 logical pairs were scored; every method/SNR retains500 sources and
1500 frames. The byte/pixel inventory found no missing native array. NeST19dB
uses the T1-C m10-corrected current PARTIAL result. Final P100k/Swin200k,
adaptive BPG and all4500 ARPC receiver records are preserved.

CLIP-I is **CLIP image-image cosine similarity**, higher is better. It uses
OpenAI CLIP ViT-L/14,224 input, the original floating tensor preprocessing and
FP32 inference. Both inputs are images; values are raw cosines. This is distinct
from caption-oriented [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/).
No caption, softmax or percentage conversion is involved.

## Five-method means and pointwise95% source-bootstrap intervals

| Method | 4 dB | 10 dB | 19 dB |
|---|---:|---:|---:|
| NeST-Com | 0.8206 [0.8118, 0.8290] | 0.8731 [0.8663, 0.8799] | 0.9005 [0.8949, 0.9059] |
| Latent JSCC | 0.7408 [0.7330, 0.7485] | 0.7892 [0.7817, 0.7965] | 0.8041 [0.7965, 0.8116] |
| SwinJSCC | 0.6267 [0.6192, 0.6343] | 0.6691 [0.6613, 0.6768] | 0.6966 [0.6887, 0.7044] |
| BPG + LDPC | 0.5859 [0.5784, 0.5934] | 0.6350 [0.6268, 0.6431] | 0.6843 [0.6758, 0.6927] |
| ARPC + LDPC | 0.6442 [0.6359, 0.6522] | 0.7134 [0.7048, 0.7218] | 0.7880 [0.7799, 0.7962] |

## Direct source-paired NeST-minus-baseline differences

| Reference | SNR (dB) | NeST − reference | Direct 95% CI |
|---|---:|---:|---:|
| Latent JSCC | 4 | 0.07981 | [0.07326, 0.08635] |
| Latent JSCC | 10 | 0.08395 | [0.07873, 0.08923] |
| Latent JSCC | 19 | 0.09635 | [0.09110, 0.10180] |
| SwinJSCC | 4 | 0.19396 | [0.18618, 0.20162] |
| SwinJSCC | 10 | 0.20405 | [0.19718, 0.21103] |
| SwinJSCC | 19 | 0.20389 | [0.19721, 0.21057] |
| BPG + LDPC | 4 | 0.23478 | [0.22587, 0.24370] |
| BPG + LDPC | 10 | 0.23812 | [0.23034, 0.24575] |
| BPG + LDPC | 19 | 0.21618 | [0.20809, 0.22444] |
| ARPC + LDPC | 4 | 0.17647 | [0.16836, 0.18461] |
| ARPC + LDPC | 10 | 0.15977 | [0.15249, 0.16709] |
| ARPC + LDPC | 19 | 0.11250 | [0.10588, 0.11901] |

The original three noise realizations are averaged within each source before
equal-source aggregation. The existing seed2026100701,10000×500 index bank is
reused without drawing new indices. Differences are formed at source level
before resampling. These are pointwise intervals, not simultaneous or
multiple-comparison-adjusted claims. Historical baseline noise namespaces differ;
pairing is by source, not by an identical cross-method noise sample. All thirty
signed method-pair contrasts are reported in paired_differences.csv.

## Actual compute and reuse

13980 image embeddings =500 originals +
13480 distinct native reconstructed images.
13480 distinct source/reconstruction cosine scores serve22,500
logical positions; 9020 positions reuse an exact pair.
There were zero new training, channel, decoder, reconstruction, caption or
other-neural-metric calls. All original105 seven-metric summary cells and21
NeST-minus-ARPC paired cells are copied without numeric alteration into the
eight-metric combined tables. The existing main figures remain unchanged.

The actual inputs in this complete grid contain zero fixed-gray frames;
the inventory and statistics retain the original receive statuses and never
filter by success. Actual length fallback endpoints remain in frame_scores.csv.
The independent four selected/final two development examples are excluded.
N2048 has not been supplemented in this first-stage delivery.

Frozen weights and seven official source files were restored from the existing
static backup with every original SHA verified; the inference environment was
not upgraded. New feature caches, ledgers and native arrays remain outside Git.
Registration, input/file hashes, model/runtime identity and actual exit receipts
provide the lineage. Prior ARPC engineering telemetry limitations remain as
documented in its original delivery; this evaluation does not relabel them.
