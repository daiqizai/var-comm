# Bounded extension protocol v1 — prepared, not executed

## Entropy source endpoints

Retain all 36 original whole target/MCS candidates (target m7/m8/m9;
q2/q4/q6; rates 1/2, 2/3, 3/4, 5/6). Add only three raster-prefix K values per
boundary, using floor of one quarter, one half and three quarters of the next
scale. The 108 additional target/MCS candidates give 144 total.

| Complete scales m | Next scale | K choices |
|---|---|---|
| 4 | 5×5 | 6, 12, 18 |
| 5 | 6×6 | 9, 18, 27 |
| 6 | 8×8 | 16, 32, 48 |
| 7 | 10×10 | 25, 50, 75 |
| 8 | 13×13 | 42, 84, 126 |
| 9 | 16×16 | 64, 128, 192 |

Original whole actions keep their exact whole-only fallback. Partial actions try
the registered partial and whole endpoints no longer than their target, in
descending token count, terminating at m4/K0. The first *actually encoded and
flushed* arithmetic stream that fits is selected. Missing streams stop execution;
they are not treated as poor quality, non-fit, or channel failure. No source
reconstruction score, token ordering search or entropy-model optimization enters
this decision. A new partial family is allowed to select a whole action.

CDFs come from the frozen null-class VAR and previously decoded complete scales.
TX and RX instantiate independent probability contexts. At the final partial
scale they use the first K rows of the next-scale CDF; they do not advance a
complete-scale provider using a partial list. Arithmetic termination and all
actual transmitted length bits are charged. The existing canonical re-encoding
and exactly 30 deterministic terminal zero-lookahead reads remain required.
Truncating an old flushed whole stream is not partial source coding.

## Paid frame and failure behavior

The 68-symbol header still transmits a 12-bit profile ID, CRC16 and six tail bits
through the existing convolutional encoder. Old IDs 0..143 retain their exact
semantics. IDs 144..359 append 216 partial profiles, identifying m, K and MCS.
All receiver interpretation uses the actual accepted received ID and the entire
public catalogue, including a legally decoded incorrect ID. The preserved
static-whole IDs require the frozen static source decoder if actually received.

The body remains 956 symbols with n=956q, k=floor(nr), and 13-bit actual length L,
L arithmetic bits, known zero information padding and CRC16. Source capacity is
k−29. All 1,024 uses and actual energy are recorded; no per-frame energy
normalization is introduced. Header rejection, CRC rejection, illegal framing
or noncanonical arithmetic uses the registered constant 0.5 RGB output.
Software/model faults stop; they are not channel failures. Raw KEEP is unchanged.

The extension has its own protocol identity, source bindings and ledger.
The old public scrambling/noise namespace and default wire session are retained
so unchanged whole transmissions can be compared exactly. The larger receive
catalogue can alter legal-ID outcomes; previous receipt reuse is permitted only
after checking the actual observations, decoded header ID, profile, body evidence,
source-model identity, numerical environment and output. No automatic reuse is
authorized merely by unchanged profile IDs, noise labels or CRC success.

## Finite stages and budgets

| Stage | Logical scope | Separate upper bound |
|---|---|---|
| Real source gate | first32 original calibration sources; 18 partial endpoints/source | 32 TX provider passes; 576 independent partial RX decodes; ≤64 missing short-whole RX checks |
| Real PHY gate | 12 original layouts, eight body cases each; all360 IDs plus unknown4095 | 96 body +361 header =457 decoder calls |
| Real image link gate | first4 original calibration sources ×3 SNR ×one noise ×four fixed endpoints | 48 frames; ≤96 decoder calls; ≤48 independent RX VAR and ≤48 reconstructions |
| Pilot | original calibration first100 ×3 SNR ×seed4101 ×144 candidates | 43,200 frames; 86,400 decoder calls before exact reuse |
| Full calibration | original1000 ×3 noises ×3 SNR ×at most4 finalists | ≤36,000 frames; ≤72,000 decoder calls before exact reuse |
| New confirmation | new100 ×3 SNR ×3 noises ×four methods | 3,600 frames; ≤7,200 decoder calls |
| One-step mismatch | old fixed100 ×3 actual SNR ×3 lookups ×3 noises ×two methods | 5,400 logical /4,500 unique physical frames; ≤9,000 decoder calls before verified matched reuse |

The full-calibration finalist set at each SNR is the top3 complete-pilot DINOv2-L
candidates union the original final entropy-whole winner. Exact score ties use
candidate-ID lexical order. Original whole winners cannot be dropped by pilot
ranking. Full1000 calibration uses original seeds4101/4102/4103 and its original
source ordering. Prepare actual source/render/model-call caps from the admitted
finite states and bind them before execution, rather than introducing unbounded
fallback work. Completion counters, actual model calls, reuse events and unresolved
reservations are reported separately from logical frame budgets.

The real image link gate follows both component gates, before pilot ranking.
At each SNR use the original entropy-whole winner's target m and MCS, with K=0
and the three registered fractional K targets. Use seed4101 and the same
calibration counter/noise/wire conventions as the following pilot. Transmit real
arithmetic streams through actual paid header/body processing, reconstruct from
the actual received tokens, and record fallback, CRC/parser outcomes and exact
stream identities. No quality ranking is performed by this gate. A channel
rejection retains its fixed output, while software defects stop execution; truth
comparisons are diagnostic and cannot alter RX acceptance. Pilot reuse requires
the same exact-event proof as any other reuse and never erases the gate's calls.

Freeze policies before accessing new100 content. New confirmation uses seeds
9301/9302/9303, with source IDs fixed from the audited unused pool by a deterministic
hash rank. This rank is independent of method performance and all four methods use
the same source manifest. Exhausted or incomplete deduplication is a gate failure,
not permission to replace images opportunistically.

Only four metrics are primary: PSNR, LPIPS-Alex, DINOv2 ViT-L14 cosine and
ConvNeXt source-prediction agreement. Source means average the three noises first.
Freeze pair definitions before new quality reads; within-representation contrasts
are raw partial−raw whole and entropy allowed-partial−entropy whole. New intervals
use the existing 10,000 source-resample protocol, seed2026100701, once per new
contrast. LPIPS keeps its original sign; agreement differences are percentage
points. All failures, zero increments and negative results remain in the population.

## Mismatch and timing

Mismatch retains common500 seeds6201/6202/6203 and the original counter:
`(actual_SNR_index_in_[1,4,7,10,13,19]*500 + original_source_index)*3 + noise_index`.
Never substitute the lookup SNR or a 100-source multiplier. The same true-SNR
matched condition anchors both quality and failure-rate differences. Original
matched evidence covers 1,800 logical/1,500 unique physical frames only after
individual identity verification; then at most6,000 new packet calls remain.
Coincident whole/partial profiles at lookup7 share one exact physical frame.

New entropy TX timing uses fresh tokenizer, VAR probability construction, actual
arithmetic coding and fallback work, paid framing and FEC. Prepared source caches
cannot masquerade as online cost. Reuse the existing exclusive timing protocol
(fixed16 development sources, three SNR, three repetitions, paired warmups,
same device/numerics/threads), documenting that this differs from quality100.
No concurrent GPU/PHY workload runs inside a timing window. Keep the original
raw low-transmitter-compute implementation and the published entropy-whole timing.

No result triggers new K values, sources, thresholds, training or automatic retries.
Engineering failures preserve their receipts and need a separately bound repair.
