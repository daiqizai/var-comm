# Adaptive BPG: verified resource and failure findings

The actual resource export is complete and internally consistent: all 13 listed
output hashes and all eight control-input hashes match their local mirrors.
There are exactly 9000 distinct source/SNR/noise rows and 3000 distinct
source/SNR choices from 500 unique sources, six SNRs and three noise seeds.
All 9000 budget/bit-count/energy identities and the six reported mean byte counts
and frame energies were checked against the exported rows. The six fixed
source-index-0 BPG examples match their recorded lengths and hashes.
No discrepancy was found. No original file was changed and no codec, PHY,
model, noise simulation or bootstrap was run for this verification.

## Actual findings

Every one of the 500 sources has a complete BPG file fitting the frozen capacity
at every SNR. Consequently there are zero SOURCE_UNFIT frames for this adaptive
baseline. The native256 BPG source-capacity failures cannot be generalized to
all traditional digital schemes: this separately named adaptive-downsampling
baseline pays for its actual complete file and may encode at 32, 64 or 128 pixels
before fixed bicubic restoration to the common 256-pixel output.

Only 7 dB has an observed receiver failure: 15 of 1500 frames (1.00%), from
15 distinct sources, are rejected by the body CRC and receive the frozen gray
fallback. The other 1485 frames at 7 dB decode, and each of the other five SNRs
has 1500 decoded frames. Header rejection, accepted wrong header, CRC-accepted
parser rejection, accepted differing payload, BPG decoder rejection and source
format rejection are all zero in this finite sample. Zero observed failures do
not establish zero population risk. Overlapping diagnostic rows are not added
to the mutually exclusive final-status counts.

At 1 dB, 412/500 sources use 32-pixel inputs (82.4%) and only one uses128 pixels;
none uses native256. At 19 dB, 226/500 use native256 (45.2%), while274/500 still
select a smaller input. The rate-fit result therefore comes with a source
resolution tradeoff. It is not evidence of pixel-detail or semantic superiority.
The official four-metric quality and paired comparisons are separate outputs;
this resource analysis makes no comparative quality claim ahead of them.

## Six-SNR accounting

Bytes include the complete BPG file and mandatory format header. QP ranges and
modes below come from the 500 source-side choices, not1500 repeated noise rows.

| SNR (dB) | Modulation | Actual k/n | Capacity (bytes) | BPG bytes mean [P05, P95] | Observed QP range; mode | CRC reject /1500 | Gray /1500 |
|---|---|---|---:|---:|---|---:|---:|
| 1 | QPSK | 637/1912 | 76 | 72.470 [66.0, 76.0] | 35–51; 46 (91/500) | 0 | 0 |
| 4 | QPSK | 1274/1912 | 155 | 147.982 [137.0, 155.0] | 28–51; 47 (77/500) | 0 | 0 |
| 7 | 16QAM | 1912/3824 | 235 | 223.240 [209.0, 235.0] | 33–51; 44 (54/500) | 15 | 15 |
| 10 | 16QAM | 2549/3824 | 315 | 298.336 [278.0, 314.0] | 30–51; 48 (56/500) | 0 | 0 |
| 13 | 16QAM | 3186/3824 | 394 | 373.506 [345.0, 392.0] | 28–51; 47 (74/500) | 0 | 0 |
| 19 | 64QAM | 4780/5736 | 593 | 559.792 [519.0, 591.0] | 32–51; 44 (58/500) | 0 | 0 |

| SNR (dB) | 32 px | 64 px | 128 px | 256 px | Sources |
|---|---:|---:|---:|---:|---:|
| 1 | 412 | 87 | 1 | 0 | 500 |
| 4 | 70 | 347 | 82 | 1 | 500 |
| 7 | 5 | 253 | 234 | 8 | 500 |
| 10 | 1 | 141 | 322 | 36 | 500 |
| 13 | 0 | 53 | 352 | 95 | 500 |
| 19 | 0 | 12 | 262 | 226 | 500 |

Every transmitted frame uses68 header +956 body +0 physical-padding complex
symbols, so frame effective use is1. The body information container has13 length
bits, the complete BPG file, known zero information padding, and16 CRC bits.
Its actual k/n includes these fields; a nominal rate label alone is insufficient.
Information padding is encoded within the body and is not free extra symbols.

The exact header waveform energy is136, derived from the frozen +/-1 real-axis
mapping. Body energy is the saved actual frame energy minus136; physical padding
energy is0. QPSK gives total energy2048 for every frame. At higher modulation
orders the saved energy varies, as expected from constellation-average Es=2
without per-frame normalization:

| SNR (dB) | Actual frame energy mean [P05, P95] |
|---|---:|
| 1 | 2048.000 [2048.000, 2048.000] |
| 4 | 2048.000 [2048.000, 2048.000] |
| 7 | 2047.886 [1987.200, 2104.000] |
| 10 | 2048.264 [1993.600, 2107.200] |
| 13 | 2047.742 [1990.400, 2108.800] |
| 19 | 2049.624 [1989.276, 2111.238] |

## Selection, samples and limits

The source rule was informed by the original32-source codec probe. MCS selection
used exactly100 original calibration sources with three noises per source,
not1000 and not holdout scores. Its candidate source resolution/QP rule and MCS
were frozen before this later-added500-source evaluation. The native256 baseline
and original raw results remain distinct and unchanged. This source-dependent
rate-fit rule uses source-side distortion and actual file size, never the current
noise or receiver outcome.

The completed holdout receipt records eight successful child exits and9000 rows.
The supplemental decoder ledger closes at31,916 of34,604 allowed calls:
qualification44, pre-screen3072, calibration10,800 and holdout18,000; unresolved
calls are0 and the original root budget is unchanged. This resource export adds
zero decoder or source-codec calls.

The stream examples are source index0 in the original order at all six SNRs,
chosen without looking at performance or noise outcomes. Their resolutions,
QPs and byte counts are:

| SNR | Resolution | QP | Complete bytes |
|---|---:|---:|---:|
| 1 | 32 | 47 | 72 |
| 4 | 64 | 47 | 154 |
| 7 | 64 | 45 | 222 |
| 10 | 128 | 50 | 287 |
| 13 | 128 | 49 | 335 |
| 19 | 128 | 46 | 556 |

The resource evidence supports feasibility and identifies the remaining digital
failure mode. It does not establish human semantic fidelity, global optimality,
training convergence, or the adaptive scheme's online timing. The existing A3
BPG timing table measures native256 BPG and must not be relabeled as timing for
this larger resolution/QP search.
