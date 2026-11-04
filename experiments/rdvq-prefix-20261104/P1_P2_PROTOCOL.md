# First-batch protocol

Frozen before reading any P1 test reconstruction or quality result.

## Inputs and packet

RDVQ commit `a2744eedd803d9e20b55b4b0c935ec211a07d184`; all arms start from
`lam_4.8_tau_0.1.pt`, SHA-256
`827aaab038626bb306fc30fac37634eece418dd02bd4617f88e65d8d19e92294`.
The original COCO split is fixed at 20,000 train / 1,000 development / 1,000
calibration / 500 test. JPEG size and CRC come from the official archive;
SHA-256 and decoded/preprocessed pixel identities are retained. No missing
source is replaced, and subsets use manifest order rather than quality.

The mechanism operates on RGB 256x256. The native 28 causal slice endpoints
are used. Each candidate is independently terminated; actual bytes include
both rANS streams and states. Source wrapper v1 adds 72 bits: version+endpoint
8, two byte lengths 32, CRC32 32. A valid NULL is the 72-bit empty-stream
packet and yields gray 0.5. This is source framing; it is not free wireless
control signaling or the old NeST frame. All budgets include this wrapper.

## P1

Use the first 128 development sources, all 1,000 calibration sources and the
fixed 500 test sources. Preserve all 28 nonempty prefixes and NULL. The first
eight development cases exactly reuse the completed P0 native packets,
reconstructions and same-evaluator scores. TX and RX are separate processes;
RX cannot read source images, indices or TX state. Post-RX CPU audit checks
every received index. Metrics use native float reconstructions, never PNGs.

Report actual-bit curves at 384,512,768,1024,1536,2048,3072,4096 and unseen
640,1280,2560 bits, alongside the complete representation. Longest-feasible
selection scans actual packet lengths; it assumes neither monotone rANS
lengths nor monotone reconstruction quality. NULL and saturation stay in the
denominator. Main metrics: DISTS/DreamSim; auxiliary: PSNR/LPIPS-Alex.
Source-paired bootstrap uses 10,000 resamples, source order fixed, seed2026110417.

The P1 reference controller is a separately labeled source-informed upper
bound using known TX source and candidate successful reconstructions, never
future noise. A PHY-dependent benefit cannot be claimed until the new paid
dynamic-MCS header and reliability table are qualified. No learned selector
or P3/P4 extension starts automatically. Failure of that controller hypothesis
does not create a positive source-representation result.

## P2 matched screening

Predeclare 4,000 optimizer updates per arm, effective batch16, microbatch4,
four accumulation steps, FP32/TF32off, frozen codebook. P0 measured about4.8–4.9
seconds per warmed update and24.6GB peak memory for this route. These are
short throughput qualifications, not formal timing comparisons. First-batch
updates are a screening horizon, not convergence or the20,000-update ceiling.

R1: full-representation RD adaptation, seed2026110401.
R2: full plus two token-count prefixes, seeds2026110401/02/03.
R3: full plus two bit-budget prefixes with actual-rate calibration, same seeds.
R3-estimated: identical R3 budget/loss/data route without rate correction,
same three seeds. Thus ten matched training runs are specified.

E, D and P use Adam1e-5, betas(0.9,0.95), zero weight decay, no LR search.
The reconstruction loss is MSE + official RDVQ VGG-LPIPS; lambda4.8 multiplies
soft expected bits/(256x256). Commitment beta0.25 and the frozen-codebook
stability term are retained. Three-path losses have equal weights; R1 uses
its full path. See LITERATURE_SCOPE.md for the shared inactive-GAN interval.

R2 draws a token-count cutoff uniformly from1..168 and169..336, then rounds
down to the native slice grid. R3 draws one budget uniformly from the first
four training budgets and one from the last four. All arms use the same
seed-specific image order/crops. Endpoint RNG is separate from model/dropout
RNG. E runs once per image. Missing suffix indices come from deterministic
native causal argmax with only received inputs. Argmax is stopped; E/D use
received-token STE and P uses the soft entropy loss. There is no claim of
differentiable entropy coding or discrete rollout.

At initialization and each500 updates, audit actual terminated packets on
the same first128 development sources. Calibration uses the per-endpoint
median of actual packet bits minus estimated likelihood bits minus136 bits.
R3 updates this correction; R3-estimated retains zero correction and still
performs the same audit. All reporting/deployment uses actual hard lengths.
Development checks occur at2,000 and4,000 updates. Final endpoint choice and
metrics use independent actual RX outputs. Final test results are not used to
choose budgets, hyperparameters, checkpoints or examples.

Save complete model, optimizer, RNG, update count and data cursor every50
updates and on requested stop at an update boundary. Preserve milestone
checkpoints. Native bicubic backward is nondeterministic; P0 exact-state
restore followed by one update differed by at most1.19e-7, below the
preregistered1e-6 absolute tolerance. No bitwise training-resume claim.

P0/P1/P2 caps are16/64/256 GPU-device-hours, aggregate336. Reserve limits
before launch, count failed attempts, keep CPU/network resources separate,
and do not restart unknown failures. Use only currently idle cards. All arms
must be compared at their common completed update count if interrupted.
No P3/P4, old NeST task, or stopped training branch is automatically resumed.
