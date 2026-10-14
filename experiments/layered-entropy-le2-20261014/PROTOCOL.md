# LE2 revision 1 — packet proxy only

Authorization: the user's 2026-10-14 instruction executes LE2 within 21,504
new physical packets, delivers results and LE3 candidates, then stops.
LE0/LE1, raw R1 NO_GO, model/source/token/clean-output bytes remain immutable.
No image wireless frame, VAR/tokenizer/reconstruction call, LE3, power expansion,
feedback, fading, KEEP, training, timing or common500 rerun is authorized.
Backup remains deferred at the user's request.

Pilot50 is the already hash-selected LE1 population. Objective is DINO-L:
J=.25 Q(true7)+.75 Q(true10), sender hat10. A body's source capacity is
floor(N*q*rate)-29 (length13+CRC16). Each arithmetic stream includes its own
finish; nothing is cut from a whole stream or derived by length subtraction.
Coarse N_B=256..768 step32, N_E=956-N_B. Each (j=6/7,total endpoint) additionally
admits its own even, nearest-ties-even proportional allocation
956*(median(b_B)+29)/(median(b_B)+median(b_E)+58), clamped2..954.
There is no capacity-occurrence weighting. No optional grid refinement is used.

Fallback is evaluated separately for every source: deepest complete base4..j
that fits, then longest registered endpoint above that actual base and no higher
than the target that fits enhancement; otherwise explicit L=0 enhancement.
No base4 fit means null and Q0, retaining all 50 sources in every mean.
Single956 also admits the same24 endpoints and longest fitting prefix below its
target. Every source has its actual cached Q_B/Q_F/Q0; no average-prefix image
or hypothetical successful target substitutes for fallback quality.

Gray is float32 RGB 0.5, 3x256x256. Admit existing exact Pilot50 gray/image/source/
metric pairs; score only missing fixed gray pairs with the identical frozen
DINO-L implementation/preprocessing/weights, no reconstruction models. This
necessary fixed-output evaluation is separate from packets and bounded33.

Stage A: all15 single956 MCS (q2/4/6 x rates1/3,1/2,2/3,3/4,5/6) at7/10 plus
original paid68 header at7/10, each128: maximum32 cells/4096 packets.
Stage B: nominate at most16 complete two-group physical strategies. Each nominee
requires BOTH groups at BOTH7 and10; deduplicate exact geometry/SNR cells. Six
EEP slots cover j6/7 x q2/4/6; six U slots cover j6/7 x base-q2/4/6; four open U
slots provide further distinct enhancement-MCS coverage. A source-resolved
quality transfer screen uses Stage-A single956 success frequencies for the same
MCS to *prioritize measurements*, including its Wilson lower endpoints. It is
explicitly a nomination heuristic, NOT a shorter-group BLER estimate, bound,
lookup entry or scientific quality result. Unknown group cells stay [0,1] in
the actual table. Equal screen scores prefer lower nominal-rate sums, then the
canonical policy key. Final ranking uses ONLY the actually measured exact
group cells. Maximum64 new cells x128, cumulative96 cells/12288 packets.
All policies with completely covered geometries are evaluated, including target
aliases and U=EEP overlaps. Uncovered layouts are reported NOT_MEASURED.

Stage C: rank completely measured policies and nominate ranking-influential
geometry bundles (both7/10), at most12 geometries/24 cells, each384 additional
packets. Prioritize header then top U/EEP/robust single/Sback3/Snom full bundles
and largest conservative interval width among remaining top policies. Never
top up one SNR alone. Total additional<=9216, total new<=21504, percell<=512.
Every call is reserved durably before encode/channel/decode, completed only
after actual hard RX and RX-only parse evidence is saved before truth audit.
There is no independent PHY benchmark/noiseless/high-SNR qualification outside
these caps. Correct observations in the paid probes are retained as roundtrip
evidence. Zero observed errors does not mean zero true BLER.

Independent backend accepts arbitrary paid group N using original Sionna2.2.0
RV0 constructors/encoder/20-iteration boxplus-phi decoder and original q2/4/q6
mapping. ep_phy.Backend.plan is untouched. FP32/TF32off, CPU2/inter1,
Es2 constellations, AWGN perrealcomponent sigma=10^(-SNR/20), no frame
normalization, no power allocation. Isolated codeblock trials use valid random
L13+CRC16+canonical zero padding; payload bits are iid and lengths cycle fixed
pilot fallback lengths, declared before each cell. This is a payload/proxy
approximation, not arithmetic parsing or wireless image evaluation.
Standard AWGN and public scrambling are SHA256/PCG64 with a new LE2 domain;
same trial/geometry draws across7/10, SNR affects sigma/RX likelihood only.
Header raw12+CRC+tail decoding is saved with all4096 field values legal; correct
header probability is source-independent. Wrong-profile acceptance under the
eventual LE3 public catalogue requires LE3 validation, is never claimed solved.

Existing raw cells are inspected for exact k/n/q/interleaver/backend match.
They remain auxiliary evidence if payload/parser protocol differs, never new
L13-parser qualification. No unmeasured reliability is assigned a success rate.

Correct-prefix proxy Q=Q0+pH*[pB*(QB-Q0)+pB*pE*(QF-QB)]. Null Q=Q0;
single Q=Q0+pH*pS*(QF-Q0). Empty enhancement gives QF=QB.
Report actual CRC rejection and undetected errors separately. The proxy excludes
wrong-header effects, undetected arithmetic corruption, conditional/source
dependence and receiver failures; no image-quality/GO guarantee follows.
Per-cell two-sided95% Wilson intervals and propagated interval sensitivity are
not simultaneous confidence or paired source-bootstrap intervals. In delta
comparisons the same header variable is shared. No bootstrap or threshold tuning.

Snom is the existing entropy anchor m8_K126_q4_r2-3 at hat10. Srobust maximizes
Pilot50 proxy J; Sback3 maximizes proxy true7. They may coincide. LE3 slots are
at most3 U,3 robust-single,2 EEP,1 old nominal,1 backoff, with physical/target
deduplication. Positive point estimates may be exploratory candidates only;
if interval ordering is unresolved report BLOCKED_TABLE_UNCERTAINTY. If no
measured positive estimate report NO_PROMISING_MEASURED_CANDIDATE, explicitly
not a global negative proof over the unmeasured grid. No automatic LE3 execution.

Only source/code/light results enter Git; arrays/ledger/logs stay on PFS runtime.
Once-only control records PID/start, argv, persistent stdout/stderr, actual wait,
source/runtime bindings and immutable failures. Unknown failure stops, no retry.
