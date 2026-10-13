# R1 implementation gaps and frozen interfaces

## 2026-10-13 entry preparation closure

The implementation gaps described below are historical. The complete five-stage entry now exists under `r1_execution_v1/`, with independent CPU review and a strict zero-model import check. It is not registered or scientifically executed. Actual R0 clean closure and authenticated nominations remain prerequisites. See `results/layered_mismatch_raw_20261013/r1_execution_preparation_v1/`.


Status: read-only design review. No R1 registration, PHY call, reconstruction,
metric call, bootstrap, candidate selection, or R2 dispatch was performed by
this review. New R0 BLER/clean results were not opened. This document proposes
execution interfaces; it does not replace the frozen scientific protocol.

## What already exists

| Component | Available behavior | Remaining boundary |
| --- | --- | --- |
| `r0_screening.py` | Frozen full-domain DROP approximation and deterministic nominations | The caller must authenticate completed R0 evidence before calling it. |
| `r1_phy.py` | Registered finite actual TX, AWGN, RX, physical ledger, canonical logical views | It is a worker, not a complete owner/visual/score chain. Its completion is `COMPLETE_R1_PHY_ONLY`. |
| `grouped_wire.py` | Actual two-body packet path and mixed-public-domain DROP receiver | No real R1 frame has qualified the composed path merely because CPU tests passed. |
| `policy_selection.py` | Complete-grid validation, three-noise source means, frozen J selection and calibration gate | The caller must authenticate actual PHY/visual/score exits and identities. |
| `contract.py` | Source selection, noise/counters, finite budgets, selection objective and GO arithmetic | Metadata and arithmetic cannot substitute for execution evidence. |

The required additions are an authenticated nomination builder, a finite owner,
an RX-state projection and image worker, a calibration metric worker, and an
actual-closure/selection publisher. None should edit the already bound modules.

## 1. Authenticate R0, then freeze exact nominations

An independent CPU builder should open only completed R0 artifacts after both
new stages have actual successful child waits. It must verify:

- Original calibration200 source IDs **and order**, original1000 selection rule,
  pixel/token/checkpoint identities, and the frozen source/model/metric bindings.
- All 689 paid profiles and their canonical IDs; all 687 body layouts; exactly
  672 compatible archived selection points plus 1,389 new points at 4/7/10 dB.
  Every new point has exactly 64 closed trials and no unresolved reservation.
- The three archived header points by their frozen pins. No new header trials.
- Exactly 3,400 historical clean pairs plus 5,400 new pairs forming 44 x 200
  terminal states, plus the independently authenticated per-source gray values.
  The 200 source-DINO preparations are separate from the 5,400 reconstructions.
- The frozen `screening_contract.json`, approval addendum, exact missing keys,
  and `server_input_relocation.json`. Linux consumers must use the verified
  server mapping, not interpret Windows path strings as relative server paths.
- Every output pin and both actual stage closures. An `all_actual_exit_zero`
  boolean alone is insufficient without the referenced actual-wait and output
  records. Preserve historical values; do not rerun old intervals or inference.

Only then call `rank_and_nominate`. Its pure CPU API intentionally accepts small
synthetic fixtures, so the production caller must enforce the full 689/200
domain and precise 672/1,389 split itself. Do not change its formula, ties, trial
count, confidence interpretation, or family permissions after seeing new data.

Suggested nomination artifact: `FROZEN_R1_NOMINATIONS` with `rows`,
`selection_uses_holdout=false`, rule/input/output pins, source order, and ranking
provenance. Each row is exactly:

```text
snr_hat, arm, profile_id, candidate_id, physical_key, receiver_variants
```

Use catalogue canonical IDs even where an old frozen anchor names an alias.
`nomination_groups` merges aliases and preserves `arms[]` and
`nomination_indices[]`. The same physical two-group action may fill U and EEP
roles. Do not force distinct candidates or fill extra slots to reach a budget.
Keep both nominal and backoff anchors; their actual KEEP outputs remain eligible
for S_robust even when the proxy did not nominate them in the S pool.

## 2. Finite owner and immutable R1 registration

Create a new entry rather than adding implicit execution to the frozen PHY or
selection modules. Before dispatch, bind all code, nominated rows, both RX
catalogues, source arrays, models, metric implementations, noise namespace,
resource assignments, finite deadline and STOP path.

The PHY registration schema/required fields are already enforced by
`r1_phy.validate_registration`. Compute exact physical and logical grid sizes
from `nomination_groups`; retain the upper bounds:

- At most 36,000 distinct physical frames and 108,000 actual packet decodes.
- At most 46,800 unique candidate/receiver-rule logical VAR outputs. Actual
  identical-view reuse and deterministic gray output may require fewer model
  calls, but every registered logical scoring row remains in the denominator.
- No extra unregistered qualification frames, pilot noise realizations,
  tokenizer calls, training updates, or confidence-interval computation in R1.

The owner must acquire an exclusive execution lock, inspect prior launch,
completion and unresolved reservations, and save actual child PID/start ticks,
argv, pinned registration, actual wait code and timeout outcome. Unknown failure
preserves the scene and stops. The current PHY journal cannot reopen an old
attempt; an authorized recovery needs a separately reviewed complete-only resume
design, not a second launch of `run` into a new directory that repeats calls.

Use the original CPU LDPC environment for the PHY child. Keep new logs/journals,
images and caches under personal `modelTeam/liberai-tmp/liulu`; code and light
receipts remain under personal `code/liulu`. Do not add PFS symlinks.

Coordinate with timing through the **actual**
`results/baseline_strength_20261011/timing/six_method_v6/owner.lock`. The timing
owner takes an exclusive lock. New scientific owners should take a nonblocking
shared lock and pass its file descriptor to every child until actual child exit,
as reviewed for R0. Check real timing OUT/RUN paths and process identities too.
Do not modify the old timing owner or reintroduce its superseded exclusive-host
requirement. New resource windows must obey the currently authorized settings.

## 3. Keep the two actual receive domains separate

Original S_nom/S_back3/S candidates use the complete original raw433 receiver.
The raw receiver's `phase='raw_holdout'` is an unchanged internal API string;
the new source records, event prefix and noise namespace identify calibration200.
Do not switch to a holdout loader because of that string.

U/EEP use the complete 689-entry mixed catalogue. An erroneously received but
legal original single-group ID remains legal: parse and decode its actual body,
then apply the new grouped receiver's fixed DROP rule. Do not narrow the public
domain to the nominated IDs or correct the received ID from the transmitted ID.
The nominal group count selects the frozen method architecture/domain, never a
transmitted packet's true public configuration.

`receiver_views` already derives single KEEP and DROP from the **same** actual
RX event. No second header/body decode or fresh noise is allowed for DROP.
Grouped output is DROP only: G1 rejection means gray; G1 accepted/G2 rejected
means the received base prefix; both accepted means the received full prefix.
CRC acceptance is not token correctness. Wrong-but-accepted values and legal
wrong headers must remain in reconstruction and the logical denominator.

## 4. Project receiver states before the visual process

The raw physical frame file contains source provenance and post-RX truth audit.
Do not pass that whole object, its source archive, or clean source tokens into a
visual model wrapper. Add a pure CPU projection that authenticates the original
frame pin and emits an explicit allowlisted receive-state object:

```text
view_id, physical_frame_pin, actual_rx_digest, observation_digest,
receiver_rule, receiver_state_sha256,
state = {kind, m, K, order, prefix, partial_values, rgb_fill?},
model_and_numeric_bindings
```

The projection may keep opaque source/frame identifiers for joining results;
the render worker must not use them to open a source archive or obtain labels.
Exclude `post_receiver_truth`, clean token hashes/arrays, original pixels, source
image paths, transmitted ID/configuration and source-derived quality from its
model-facing input. Scoring and post-hoc failure audits may read source truth in
separate processes after actual RX state construction.

For `kind=tokens`, validate raw raster order, integer range [0,4095], exact full
scale lengths, legal m/K and total `OFFSETS[m]+K`. Flatten only the **received**
`prefix + partial_values` and call the original frozen
`render_received(native, received_tokens, m, K)`. Preserve its unconditional
class1000, deterministic completion, original Dc and numeric settings.

For `kind=gray`, create the registered constant float32 RGB output with fill0.5.
Do not run unconditional VAR to replace a failed frame, and do not fill missing
token positions with codebook index0. Gray remains a completed scientific failure
outcome, distinct from a software crash or invalid tensor.

The clean R0 wrapper is not an R1 receiver: its legitimate clean source prefixes
must never become a fallback input to this stage. Reuse its finite owner/ledger
patterns if useful, but add a received-state-only adapter rather than invoking
`render_one` with clean source tokens.

## 5. Reuse views without changing logical denominators

The current PHY output `logical_frames.json` has one canonical
candidate/receiver-rule row and lists arm aliases separately. Preserve that
structure. Do not emit one quality row per alias arm; the selector rejects those
duplicates and they would weight an observation repeatedly.

When KEEP and DROP yield byte-identical received model inputs for one actual
physical event, one image may be reused with two logical rule references. A cache
must bind the full source/pixel/protocol/configuration/noise/actual-RX/model
identity required by the user plan. Token count, m/K, a transmitted candidate ID,
or a shared gray image alone is not a sufficient cross-event reuse key.

Every logical row keeps its source, hat, true SNR, seed, canonical candidate,
rule, actual observation and actual-RX identity, image hash and reuse provenance.
Channel failure rows remain present. The ledger should distinguish requested
logical outputs, newly executed model calls, fixed-gray construction and exact
reuse, rather than equating every row with a new VAR invocation.

## 6. Calibration scoring and authenticated selector input

R1 needs PSNR and LPIPS-Alex as cost monitors and DINOv2-L as the selection metric.
`policy_selection.R1_METRICS` explicitly requires those three. ConvNeXt must not
enter nomination, strategy selection or GO. The plan requires all four metrics
for R2 reporting; do not silently make an extra classifier a prerequisite for
R1 selection merely because an earlier interface note mentioned four metrics.

The score worker may read the original source crop, but cannot supply it back to
the render/RX process. Authenticate equal 256x256 RGB preprocessing, frozen
metric weights/source code, source-reference features and actual reconstructed
image hash. Compute scores from the **actual** saved image, including gray and
accepted wrong-token reconstructions. Do not substitute R0 clean-Q estimates,
historical means, or a predicted reconstruction for actual R1 image scores.

Before `select_from_complete_rows`, produce exactly one row for each expected
canonical rule view with these required fields:

```text
source_id, snr_hat, snr_true, noise_seed, candidate_id, physical_key,
receiver_rule, execution_status='COMPLETE', source_truth_supplied_to_RX=false,
observation_key, actual_rx_key, reconstruction_sha256,
psnr_db, lpips_alex, dinov2_vitl14_cosine
```

`observation_key` should pin the paid waveform/noise/received-observation
identity; `actual_rx_key` should pin the one actual receive event. Both must be
identical across KEEP/DROP/aliases of that event. A Boolean declaration of
isolation does not prove it: bind and audit the projected renderer inputs and
code paths. Optional received-ID, CRC, accepted-prefix and energy columns belong
to the reporting join, not to source-conditioned reconstruction logic.

A separate closure verifier must check all physical/packet/model/score grids,
duplicates, failure states, reuse keys, source/model/metric identities, actual
child exits and output pins. Require all software stages to close; a single
saved `execution_status` string or a CPU-test report is not sufficient.

## 7. Freeze policies and separate GO from final claims

The existing reducer averages the three noises within each source, then computes
source J = 0.75 matched DINO + 0.25 true=hat-3 DINO, then averages the 200 sources.
True=hat+3 is reported pressure behavior and never enters selection. Preserve the
fixed candidate/anchor tie rules already in `policy_selection.py`.

S_robust includes actual S KEEP/DROP choices and existing anchor KEEP choices;
it may equal the nominal or backoff strategy. U and EEP may share a physical
winner. Save frozen winner identities and all negative/zero results before any
development access. No R0 proxy is allowed to decide GO.

The R1 engineering gate uses actual point estimates: mean U−S_robust and
U−S_back3 across the two hats are each at least0.01, and each corresponding
per-hat delta is strictly positive. No bootstrap is required to select R1.
The later R2 confidence-bound criteria are a separate result-claim test.
Report matched PSNR cost and high-estimate loss as separate engineering targets;
do not replace the DINO gate with them or adjust protection after a bad result.

On NO_GO/ONE_POINT_PROMISING, deliver and stop. Only a validated GO may authorize
the already specified finite R2 branch. R2 source binding, S_payload, two direct
Dc outputs, interaction statistics, plots and final delivery are separate work;
this document neither registers nor starts them.

## Minimal implementation order

1. Authenticate R0 and freeze nominations plus exact budgets.
2. Implement the finite owner and RX projection/visual/metric workers, with CPU
   shape/identity tests and pinned unmodified model/PHY math.
3. Register the complete R1 chain and its finite resource window; count any real
   engineering qualification inside the authorized physical/logical ledger.
4. Run actual PHY → received-state reconstruction → calibration scores, stopping
   on unknown failure and preserving all actual outcomes.
5. Verify actual closure, reduce source-first statistics, freeze policy and gate,
   publish the stage; dispatch R2 only if the fixed gate truly passes.

This order avoids calling the current PHY-only worker a completed experiment,
and avoids waiting until R0 finishes to define the missing execution boundaries.
