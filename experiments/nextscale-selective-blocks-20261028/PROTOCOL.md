# Same-scale selective recovery: P0–P3

This is a separate, bounded study. Scientific baseline: a7eb2f3c; current
storage-only parent: e333620b. The completed WCL study and its negative raw R1
and LE2 findings are preserved. No old task is resumed. Main is the sole Git
worktree; the independent study has its own code, results and PFS run directory.

The frozen source set is the first 32 entries after sorting cal1000 by
SHA256(canonical JSON `["NESTCOM_SAME_SCALE_CAB_PILOT32_V1", source_id]`), then
source_id. Original source indices are retained. Selection uses no image or
quality measurements. The population is calibration, not common500 or a blind
test. Source streams and tokens use the qualified PRO6000 FP32/no_grad route.

TX has a fixed nominal 2 dB strategy, evaluated at 0/2/4 dB with equal weights
and three noises per source. J is the resulting DINOv2-L image cosine mean;
PSNR, LPIPS-Alex, DreamSim and CLIP-I are also reported. No objective switching,
bootstrap or significance claim is allowed in the 32-source nomination study.

N=1024, Es=2: 68 paid header symbols and 956 body symbols. The header is the
original 12-bit profile + CRC16 + terminated rate-1/2 convolutional encoder,
mapped to ±(1+j). The body is QPSK. C/A/B each carry L13, independently finished
arithmetic bits, known zero padding and CRC16 before their own LDPC encoding.
The three blocks incur 58 additional fixed syntax bits versus one block;
independent termination cost is measured, not inferred from the old two-stream
experiment. Actual k=floor(2*N_i*r_i) uses rational arithmetic.

C contains full scales through m. A and B partition the first K positions of
scale m+1 at floor(K/2). C alone determines one full boundary CDF; slicing its
rows for two newly started arithmetic streams does not depend on A. C is decoded
first in an independent RX context. Received positions overwrite boundary
predictions and update both residual latent and finer-scale generation.
No finer-scale conditional source stream is transmitted. Models, codebook,
decoder, class1000 greedy generation, integer24 CDF/arithmetic32, no_grad FP32
and TF32-off remain frozen. Model loaders verify their original state hashes.

Four targets: (m,K)=(4,12),(4,25),(5,18),(5,36). Allocations are
(478,239,239),(598,179,179),(718,119,119). Rates are 1/4,1/3,1/2,2/3,
with r_C<=r_E=r_A=r_B: 120 three-block candidates (48 uniform,72 strict).
The 25 original quarter/full endpoints, including m10, times four rates form
100 single-block candidates. Whole targets retain whole-only fallback; partial
targets permit shorter original quarter endpoints. There are 220 candidates.

Three-block fallback is full K → floor(L/2) → C-only, or half K → C-only;
failure to fit C gives paid NULL. Every decision uses independent finish lengths.
C-only keeps paid empty A/B packets and their allocated symbols. NULL uses
legal dummy symbols at the same energy. No source is dropped or allocation
silently changed. The public profile determines actual m/K, positions, geometry,
empty state and NULL; RX obtains its ID only through the actual header.

Header/profile/C rejection gives gray. A/B fail independently after C success.
Protocol errors alone may reject: CRC, length, known padding and invalid or
noncanonical AC. Software/OOM/numeric/shape failures stop with diagnostics.
Accepted wrong content/profile is preserved and reconstructed as received;
later TX truth audits neither repair nor override RX. Output is saved first.
Cache identity includes actual C tokens, positions, values, preprocessing,
model/numeric identities and generation configuration. A-only and B-only
cannot share an identity based merely on their token count.

P1: 32 new six-scale source traversals, zero tokenizer calls; 128 complete and
128 delete-A independent RX proofs plus missing-C checks. Normal clean-state
union has at most 928 source/state entries (25 continuous + four C+B per source).
Existing exact caches are preferred; actual reuse, proofs and calls are recorded
separately. Technical correctness and one positive mean DINO(C+B)−DINO(C) are
required before P2. All four configurations and negative effects are reported.

P2: at most 28 unique real noiseless LDPC qualifications, 10,752 noisy body
packets and 384 header trials. Impossible geometries are excluded explicitly.
J_proxy nominates at most three policies per source configuration: best uniform,
best strict, and best strict at 0 dB; duplicates are removed. The best single
policy per rate gives at most four. Ties use ascending candidate ID. Proxy
probabilities are only nomination evidence, not actual image-channel results.

P3: at most 4,608 complete physical frames, 11,520 bodies and 4,608 headers.
All actual outcomes remain in the denominator. The same decoded physical event
is replayed as ALL, SEQ and SEL. The measured SEL−SEQ mechanism contribution
is the C-success/A-failure/B-success indicator times the quality difference for
the actual received values. It must agree with the direct replay difference.
J_single_star is the best of the at most four actually measured single policies.
GO requires technical closure and one policy with J_SEL>J_single_star and
J_SEL>J_SEQ. Otherwise stop with the bounded negative result. No P4/P5 follows
automatically, and there is no training, new external baseline or paper rewrite.

Noise identities use namespace NESTCOM_CAB_20261028_V1, phase, actual SNR,
original source index, replicate and candidate ID; C/A/B enter the public
scrambler namespace. Seeds are derived from SHA256 canonical JSON, never Python
hash or run order. Each phase records exact derivation and counters before use.
Same-physical-event rule replay never creates another noise realization.

Conservative execution ceilings: 32 GPU-device-hours, 32 CPU-worker-hours,
eight hours per individual stage; each stage is independently registered with
stricter exact call caps and real owner wait receipts. P1 initially uses two
cards, later independent RX/metrics may use up to four. Admission requires
60,000 MiB free and <=25% utilization, and never changes another user's job or
device settings. No successful scientific call is repeated after a failure.
