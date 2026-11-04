# RDVQ actual-bit prefix research (2026-11-04)

This independent project executes the user's PRO6000 Research Plan. Existing
NeST-Com manuscripts, measurements and stopped research branches stay frozen.
The first batch is P0/P1/P2, capped at 16/64/256 GPU device-hours, respectively.
P3 and P4 are not automatically authorized by this implementation.

Pinned upstream: CVL-UESTC/RDVQ `a2744eedd803d9e20b55b4b0c935ec211a07d184`.
The common public initialization is `lam_4.8_tau_0.1.pt`, HF revision
`d6223fb30f23f3805ec1d0a925c592fc43d223bd`, SHA-256
`827aaab038626bb306fc30fac37634eece418dd02bd4617f88e65d8d19e92294`.
It is a joint-RD checkpoint, not an assumed unpublished intermediate checkpoint.

## P0 preregistration

Before any model execution, use the first eight images from the fixed COCO
development split. Split seed 20261104 shuffles sorted official train2017 ZIP
filenames: first 20,000 train, next 1,000 development, next 1,000 calibration.
Test: first 500 after shuffling sorted val2017 filenames with seed 20261105.
Validate official ZIP CRC and JPEG bytes before computing native RGB identities.
These 256-pixel mechanism crops are not presented as the original paper's
full-resolution benchmark reproduction.

P0 sends all native slice endpoints 1..28 for each of eight sources (224 actual
source packets), plus eight explicit NULL packets. One image encoding and one
28-step causal probability pass per source generate independently terminated
prefix streams. An independent receiver process receives only packets and
shared configuration. It cannot read source pixels, TX indices, or TX states.
Qualification compares all received indices with TX references afterward and
retains every failed record. A failing scientific stage is never blindly retried.

The route is FP32, TF32 off, math SDPA, batch one, deterministic argmax suffix,
top-k/escape rANS K=1024 and CDF precision 16. This changes suffix generation
from the public default sampling by the research plan's preregistered choice;
entropy probabilities and native slice order remain the official implementation.

The task-specific dependency overlay does not modify shared environments.
Weights and training checkpoints live in the dedicated PFS checkpoint area;
logs, arrays, downloads and extension builds live in the task scratch area.
Only reviewed code, configuration and lightweight evidence enter Git.

## Source packet v1

| Field | Bits | Meaning |
|---|---:|---|
| Version and slice endpoint | 8 | version 1 (3 bits), 0..28 slice count (5 bits) |
| Top stream byte length | 16 | independently terminated rANS top/escape-rank stream |
| Escape stream byte length | 16 | independently terminated escape residual stream |
| Top and escape bytes | actual | includes each nonempty stream's 32-bit rANS state |
| Source CRC-32 | 32 | covers header and both entropy streams |

Total source wrapper = 72 bits; rANS termination is already inside the streams.
The actual source packet length, not estimated entropy or token count, is used
for the hard bit-budget check. Resolution, checkpoint, native slice ordering,
CDF and suffix settings are public fixed context. No source-specific free fields
are required. A NULL has endpoint zero and empty entropy streams (72 total bits),
and produces gray. Its overhead is still charged. This is not the future wireless
control header: dynamic MCS identification/protection must be specified and paid
separately before any wireless result. Old NeST 68/29-bit layouts are not assumed.

`packet.py` enumerates actual candidate lengths without a monotonicity assumption.
`check_packet.py` checks CRC rejection, canonical lengths and a deliberately
nonmonotone candidate set; it is not a neural-codec validation claim.

## Stage boundaries

P0 must close independent decoding and measured throughput before P1/P2.
P1 evaluates the frozen model using actual packet bits and complete denominator,
then tests a content-aware reference against a well-calibrated SNR-only control.
P2 uses common initialization, disjoint data, equal exposures and matched losses
for R1/R2/R3, and separately records estimated-only versus real-rate calibration.
Training update counts are resource-capped after measured throughput, not assumed
to reach convergence. Three independent key R2/R3 seeds are distinguished from
channel repetitions. Gate failures and exhausted budgets are reported as such.
