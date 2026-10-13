# EMAIN-v1: frozen entropy partial-scale extension on original common500

## Scope and population

The authorized new quality budget is exactly one method, `EC_VAR_PARTIAL`, on
the original 500 sources, N=1024 complex channel uses, SNRs 4/10/19 dB, and
three original T1 noise realizations: 500 × 3 × 3 = **4,500 logical method
frames**. The display name is **NeST-Com (entropy-coded)**. This population has
already been used in the study; these are frozen-policy extension results,
not a first blind test. Development, calibration, new100, Kodak and other
bandwidths do not enter the common500 statistical population.

The 256×256 RGB pixels, all 680 frozen source tokens, class labels used only
for evaluation, and original source order are bound in E0. All 500 archives
and token hashes were checked against the original source records. Existing
H800 calibration1000/new100 caches have no source-ID overlap with common500.
The verified existing tokens are reused; fresh H800 source probabilities and
terminated streams are produced without new tokenizer calls. Existing H800
end-to-end qualification is reused, so no additional 24-frame engineering
grid is run. Old-host arithmetic streams are not admitted as new RX inputs.

No training, model update, checkpoint choice, tokenizer change, entropy-model
change, class conditioning, candidate expansion, threshold adjustment or
common500-driven policy selection is authorized. E4 is disabled. The
manuscript's positioning of raw versus entropy-coded variants must reflect
the measured quality/compute tradeoff rather than determine it.

## Frozen source and link

The original tokenizer, unconditional VAR and frozen Dc remain fixed. Their
state hashes, native arithmetic implementation, original integer-CDF rules,
FP32/TF32-off settings, and actual receiver core are bound by registration.
H800 source/RX uses the already qualified runtime. Reusing historical quality
does not assert bitwise cross-host arithmetic or reconstruction equivalence.

| SNR (dB) | Frozen candidate | Target m,K | Modulation | Nominal rate | k / n | Actual arithmetic capacity |
|---|---|---|---|---|---|---|
| 4 | m7_K25_q2_r2-3 | 7,25 | QPSK | 2/3 | 1274 / 1912 | 1245 bits |
| 10 | m8_K126_q4_r2-3 | 8,126 | 16QAM | 2/3 | 2549 / 3824 | 2520 bits |
| 19 | m9_K192_q6_r5-6 | 9,192 | 64QAM | 5/6 | 4780 / 5736 | 4751 bits |

The target configurations are the already published full1000-calibration
winners. These targets are not guaranteed actual per-image endpoints. The
original deterministic fallback considers legal prefixes using each actual
terminated stream length. It may fall back to whole-scale transmission.
Selection never scores sender reconstructions. All 24 registered endpoints
are terminated independently during one source-probability progression; a
truncated longer arithmetic stream is not a substitute.

Every frame contains 68 paid header symbols and 956 body symbols. The body
capacity is k−29, including the 13-bit actual length field and 16-bit CRC.
Nominal rate is a label; actual k/n remains in the records. The receiver
decodes against the complete qualified 360-profile public catalogue. It does
not receive the selected TX endpoint as free side information. The reused
whole-scale reference targets are m7/q2/2-3, **m9/q4/2-3 at 10 dB**, and
m9/q6/5-6 at 19 dB. Raw433 and historical whole144 domains are not silently
replaced by the partial360 domain.

## Noise and counter identities

New entropy partial frames use `t1_phy.standard_noise(source_id, noise_seed)`
with seeds **6201,6202,6203**, the original T1 full1024 PCG64 standard-normal
variates, and the original noise amplitude `10**(-snr_db/20)`. The public
frame counter is:

```text
([1,4,7,10,13,19].index(snr_db) * 500 + source_index) * 3
    + (noise_seed - 6201)
```

The wire session remains `WCL_T1_ENTROPY_WHOLE_PHY_20261009_V1`. New frames
have their own actual waveforms and received observations. Numeric seed
equality does not mean identical observations across methods. Existing raw
results retain their original raw6201/6202/6203 namespace; P100k, Swin200k and
BPG retain their registered2001/2002/2003 rules. No old method is rerun merely
to replace its noise namespace.

## Finite calls, isolation and failure semantics

| Stage | Maximum / exact scope |
|---|---|
| Fresh source TX | 500 passes, all24 terminated endpoint arrays per source |
| Fresh tokenizer | 0; existing authenticated tokens reused |
| Actual PHY | 4,500 frames; at most9,000 header/body decoder calls |
| Source RX / VAR / Dc | At most4,500 each; exact received-input reuse allowed |
| Quality | 500 source references; at most4,500 unique pixel pairs /18,000 logical metric cells |
| New statistics | 12 marginal intervals plus60 direct source-paired intervals |
| New training / E4 | 0 / disabled |

The receiver's source input consists only of actual parsed RX family, m, K,
payload bits, and frozen public state. Independent RX CDF construction cannot
read TX CDFs or unseen source tokens. Original source tokens may be read
after decoding solely for the intact-stream consistency audit. If an intact
same-host registered stream cannot be independently decoded, that is an
engineering failure and stops the branch; it is not counted as an ordinary
channel loss or repaired with source truth.

Identical actual received inputs under the same frozen visual identity may
share a deterministic image. Metric reuse additionally requires identical
source and reconstruction pixels plus metric identity. Neither cache uses
source-truth tokens to stand in for received content. Every logical frame
retains its own header/body/parser status, received endpoint, and failure
classification even if its fixed-gray image is shared. LPIPS and all other
scores retain actual failed-frame outputs. The original header/CRC/length/
padding/arithmetic parse rejection rules remain fixed. Fallback to a shorter
source prefix is not itself a link failure. A source with no legal fitting
stream must retain the original explicit failure/stop semantics and cannot
be silently dropped or fabricated as a channel outcome.

## Existing references and E0 limitations

E0 exports the exact ordered source means and existing intervals for:

| Identity | Role | Existing evidence |
|---|---|---|
| EC_VAR_WHOLE | Direct matched-prior whole-scale reference | T1_entropy_whole/v3 |
| RAW_PARTIAL | Direct low-sender-compute reference | Original common500 source statistics |
| P100000 | Direct latent-space JSCC reference | baseline_strength common500 / paired_supplement_v1 |
| Swin200000 | Direct adapted Swin reference | baseline_strength common500 / paired_supplement_v1 |
| BPG_ADAPTIVE | Direct adaptive-resolution BPG+LDPC reference | T4_resources_cost/bpg_existing_v1 |
| RAW_WHOLE | Four-arm mechanism table | Original common500 source statistics |

The reuse bundle contains 36,000 source-metric means (6×3×4×500), 72 unchanged
published interval rows, the original summary rows and exact provenance.
Direct historical per-frame grids for entropy whole, P100k, Swin200k and BPG
are verified at the three selected SNRs. Raw's exact published source means
and original intervals are available; this export does not fabricate
unavailable historical raw per-frame metric rows. Its existing failure
breakdown remains linked to the T4 raw report.

The old Swin80k training-range limitation and the approved Swin200k weights-
initialized continuation with expanded1–19 dB range remain historical
disclosures. Swin200k reached its extra120k budget while still improving and
is not called converged. P is a self-built shared-representation control,
not a separately published external model. The whole-scale entropy control
is not relabeled ARPC.

## Statistics and E2 outputs

Use only PSNR (`psnr_db`), LPIPS-Alex (`lpips_alex`), DINOv2-L cosine
(`dinov2_vitl14_cosine`), and ConvNeXt source-prediction agreement
(`convnext_top1_source_prediction`). Agreement is not classification accuracy.
First average each source's three noise outcomes, then aggregate 500 source
means. Pair by authenticated source ID/order, not by unrelated population
means or assumptions of identical cross-method channel observations.

Reuse the original common500 source-level bootstrap implementation,10,000
replicates and seed2026100701. Only12 new method intervals and60 direct
new-minus-reference intervals are computed. Old intervals remain unchanged.
Paired intervals come from the source difference vector, not from subtraction
of two marginal interval endpoints. LPIPS deltas retain their natural sign;
agreement means/intervals are displayed as percentages, and differences as
percentage points. Retain negative, zero and unfavorable results. No result
feeds back into selection. New100 remains a separate confirmation table.

E2 uses the actual selected source stream once per source/SNR, not three
independent length samples for three noises. Preserve every fallback attempt
and its actual terminated length. Report b, 12T,1−b/(12T), k−29−b,
b/(k−29), target and actual m/K/T, fallback counts, and received endpoint
distributions. Undefined ratios remain explicitly missing for invalid/zero
denominators. The1,500 reused whole-scale selected length rows and3,000
available whole-scale endpoint rows retain their original identities. Missing
lengths are not reconstructed from −log probability.

The requested quality plot has five display methods: NeST-Com (entropy-coded),
NeST-Com (raw), Latent-space JSCC, SwinJSCC (adapted), and Adaptive-resolution
BPG + LDPC. The within-representation comparisons separately show partial
versus whole. Raw/entropy differ in packet format and failure handling, so the
four arms are not a pure orthogonal two-factor experiment. No unmeasured SNR
is interpolated into a new result. PDF/SVG/600 dpi PNG and the actual plot
data are delivered after statistical closure and visual inspection.

## E3 timing and explicit shared-resource amendment

Continue the existing six-method timing task: RAW_WHOLE, RAW_PARTIAL,
EC_VAR_WHOLE, EC_VAR_PARTIAL, P100k and Swin200k. The original development16
indices are `[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]` at N1024,
SNR4/10/19, batch1. Each method/source/SNR has three fresh warmups and three
fresh measurements:1,728 full calls,864 measured calls, at most2,304 digital
packet decodes. Timing noise keeps the registered P/Swin2001, raw development
6201/slot counter, and entropy T1 6201/SNR-index counter rules.

The user subsequently authorized shared resources on2026-10-13; the separate
timing wrapper v2 records this amendment without altering the v1 scientific
endpoint mathematics or budget. All six arms use one selected H800 and the
same runtime,6/2 threads,6 CPU slots,FP32 and TF32 off. The resulting table is
shared-environment latency, not uncontended latency or cross-hardware speedup.
E1 GPU work must be actually closed before the receipt-bound timing release;
no quality science starts until actual timing exit. E2 CPU statistics may
continue under this explicit shared-resource authorization, with concurrent
load recorded. No other user's work is stopped and no global GPU setting is
modified.

The registered timing gate requires at least18 GiB free and GPU utilization
at most25% at admission, retains a16 GiB private allocator cap and2 GiB
running free-memory floor, and logs other resident work outside measurement
windows. E1's separate quality gate retains18 GiB prelaunch/free-plus-own,
4 GiB absolute running free, and16 GiB own allocator cap. These scheduling
conditions do not change numerical algorithms or scientific scope.

TX includes fresh visual encoding/quantization, actual source probability and
arithmetic coding where applicable, all fallback length attempts, CRC/LDPC
and modulation. RX includes actual paid header/body decoding, protocol
validation, independent entropy recovery, VAR completion and Dc. The timing
wrapper moves reviewed static provenance verification outside the measured
case and records excluded pure-audit time; actual CRC, received fields,
arithmetic canonical checks, scrambling/seed hashes and fallback remain.
Raw wall, excluded audit, adjusted TX/RX/software total and environmental
records are preserved. No cross-frame token/CDF/stream/image cache is allowed.
Noise generation is recorded separately; metric networks, disk persistence,
model loading and static provenance are excluded. Full numerical qualification
is supplied by the existing fixed fresh warmups, not extra frames.

The once-only `after_emain.py` successor waits for the4,500-row scoring receipt
and all four actual exit-zero stage receipts, then checks release/resource
admission and exec-replaces itself with the v2 timing owner. It does not launch
a duplicate observer or retry failed science. Actual exit and complete case
coverage, not readiness or registration, are required for a timing result.
Report mean/median/P95 including failure paths; small-sample P95 and shared
contention remain explicit. No HiFi/BPG timing, old4090D/H800 speed ratio, or
air-interface propagation latency is introduced.

## State, failures and publication

`registration_v4.json` is the actual E1 identity; all prelaunch registrations
and their zero-science resource/storage/failure-label amendments are retained.
The active CPU analysis successor is `analysis_owner_v2.py`, bound by
`analysis_followthrough_v2/registration.json`. Its earlier wait-only owner was
retired only after verifying zero scientific calls and no child; the preserved
`retirement_zero_science_metadata_v2.json` is a retirement observation, not a
fabricated actual child exit receipt.
The finite owner waits for each child's real exit. All active bindings are
immutable. Durable ledgers reserve before scientific calls; unknown failures
stop without replay. Recovery must inspect failed/unresolved reservations,
preserve completed calls, and register only a necessary reviewed continuation
within the original remaining budget. No whole-grid retry is implied.

The first publication snapshot records verified E0, actual source and PHY
exit0, the full4,500-row physical grid, and E1 entering independent visual
reconstruction, with E2/E3 waiting. Parsed physical payloads are not a claim of
completed entropy recovery, image quality or timing. Only actual completion
receipts, full grids, provenance checks, output inspection and normal verified
repository push constitute delivery. Required repository checks remain
`tools/verify_repository.py` and `tools/run_cpu_checks.py`; explicit staging is
reviewed separately. Old results, exclusions and recovery dependencies remain
unchanged. E4, including additional entropy SNRs or Kodak evaluation, remains
disabled even if E1 is favorable.
