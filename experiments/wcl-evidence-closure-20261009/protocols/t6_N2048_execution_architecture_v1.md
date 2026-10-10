# T6 N2048 finite calibration and confirmation architecture

Implementation preparation only. No new N2048 channel, source-model, renderer or
metric calls are claimed by this document. The actual constructor-only catalogue
is sealed at SHA256 `cd2142263eef55f5212e4f5df4d0097479fd9e02ef2b53aefcf6f26caa370e01`:
521 unique raw wires, 131 whole-scale actions, 9 full-budget whole actions and
3 complete ten-scale actions. The earlier 7,372 integer queries are not candidates.

## Existing implementations and new qualification

`t6_raw_phy.py` loads the bound original Sionna classes, raw packet modulation /
demapping and scrambling, paid convolutional Header, raw serializer and KEEP
parser. Only a new independently identified N2048 catalogue and 2,048-symbol
frame shell are introduced. Header costs remain 68; the body budget is 1,980.
An actual accepted header, including a wrong accepted header, selects the body
layout and token endpoint. CRC rejection still retains actual raw hard tokens.
Rejected headers produce the original fixed gray receiver state.

The original planner was injected with the raw64 adapter identity for all
modulations. Therefore a q2/q4 physical key contains that adapter identity while
its internal layout preserves the original backend identity. Both layers are
checked against the actual backend; the metadata files are not rewritten.

`t6_qualify_raw_phy.py prepare` checks the actual constructors again without
decoder callbacks. `run` independently qualifies every body with four noiseless
and four 60 dB cases, each of the 521 paid IDs, and one unknown ID. The exact
budget is 521 × 8 + 522 = 4,690 packet decoder calls, in a new durable ledger.
No source images or neural models enter qualification. Formal runtime refuses
anything short of a completed, sealed qualification with matching identities.

## Finite calibration

The raw pilot uses every one of the 521 actual legal actions on the first100
original calibration sources, SNR 4/10/19 and seed4101: 156,300 logical frames,
312,600 maximum header/body calls. The two raw families share receptions; WHOLE
may rank only K=0, while PARTIAL ranks the same catalogue including all WHOLE.

The full raw recheck is registered only after the pilot: per SNR, union the five
best eligible WHOLE, five best PARTIAL, all nine full-budget WHOLE and all three
m10 actions. Deduplicate physical wires, then register the exact budget for the
original1,000 calibration sources × seeds4101/4102/4103. The conservative maximum
is 22 actions per SNR, 198,000 logical frames and 396,000 packet calls. These are
upper bounds, not receipts. Rank source means by DINOv2-L, break ties by candidate
ID and freeze both policies without reading confirmation images or quality.
This finite staged search does not establish a global optimum over all actions.

The single entropy family is inherited from the sealed N1024 family choice:
equal mean of its three independently calibrated SNR winners' DINOv2-L means,
with lexical family tie break. N2048 may not choose a family using new test data.
The 48 entropy resource queries extend the original modulation/rate grid to
target prefixes7/8/9/10. Prefix fallback remains actual-length based, down to4,
with no raw substitution. The paid uint13 length field limits source capacity to
`min(k - 29, 8191)`. Queries that real Sionna constructors reject must be retained
as unsupported metadata before reading calibration quality; they are not legal
candidate actions. In particular, q6 full-body k8910/k9900 require actual backend
admission and must not be made legal by silently truncating k.

The entropy pilot upper bound is 48 × first100 × three SNRs × one noise =14,400
frames /28,800 packet calls. The exact admitted candidate table determines the
real budget. Its full stage uses top3 per SNR × original1000 ×3 noises, at most
27,000 frames /54,000 calls. The independent m10 source codec keeps the old
probability, arithmetic and canonical rules. Unreachable short prefixes need not
be calculated solely to fill a table; actual wrong received prefixes must still
be independently decoded when an exact received-bit cache cannot be used.

`t6_calibration_plan.py preview` has been run locally against the actual
catalogue and confirmation registration. It emits complete metadata schedules
and budgets. `prepare-raw` additionally requires actual N2048 PHY qualification
and the sealed N1024 entropy-family decision. Preview is not execution admission.

## Reuse and GPU accounting

Reuse is by exact source preprocessing/token identity, actual received state,
renderer identity and numerical runtime. Equal source ID alone is insufficient.
An old state may represent erroneous tokens and is eligible only when all actual
tokens and positions match. Original packet receptions cannot be reused across
the new N2048 waveform/noise namespace.

The historical `m1-n2048-full-grid-20261004` calibration implementation saves
rows/timing but deletes its in-memory render cache; it saves float reconstruction
archives only for development. Historical completion counts therefore do not
establish calibration RGB availability. Fresh representative checkpoint and
archive metadata must confirm any exception. Existing original raw/H/T1/T2
calibration float images are the primary exact-state cache sources.

Before PHY, report the clean-state potential cache matches and conservative
per-stage upper bound separately. An exact new-render count for erroneous token
states is unknowable before actual reception. After the complete CPU receipts,
seal a per-source map of every unique actual state, eligible old image matches,
new source decodes and new renders, and its independent GPU upper bound.
Two disjoint source workers may share the GPU only after a separately budgeted
two-source exact RGB / metric parity qualification using the existing T2 cohort
ownership scheme. Candidate permissions and noise counts do not change for speed.

## Confirmation and consumer contract

The root-owned confirmation registration already fixed 100 source IDs outside
the supplied audited source-use manifests, in deterministic hash order, before
opening their pixels. It fixes seeds9201/9202/9203. The observed pool has50,000
files,46,500 eligible after known exclusions; it does not establish that other
projects have never accessed those images. Content-duplicate checking remains
a separate gate before confirmation execution.

New references use exact cached uint8 3×256×256 pixels and float32 `/255`;
`T6_CONFIRMATION100_SOURCE_MANIFEST_V1` has100 ordered records containing
source_index, source_id, evaluation_class_index, preprocessing_id, archive / SHA,
and checkpoint / SHA. Archives contain tokens:int64[680] and pixels:uint8.
Reference features from the former500 population must not be substituted.

The policy closure is `T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1`, with
source_count1000, noise_count3, selection_used_confirmation:false,
holdout_used_for_selection:false and policies RAW_WHOLE / RAW_PARTIAL /
ENTROPY_WHOLE at each of4/10/19. The duplicate gate is
`T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS`, with zero duplicate source
IDs and preprocessing hashes among checked sources, counts of available and
unavailable prior hashes, and scope `all_available_audited_source_content_hashes`.

The renderer closure is `T6_CONFIRMATION100_RENDER_COMPLETE`: N2048,100 ordered
sources, three noises, exactly2,700 method-frame rows, source_manifest,
calibration_freeze and entropy_family_selection descriptors, and an output seal
covering per_frame.csv plus all old/new referenced images. Method IDs are
`N2048_RAW_WHOLE_VAR_COMPLETION`, `N2048_RAW_PARTIAL_VAR_COMPLETION` and the one
selected `N2048_EC_STATIC_WHOLE_VAR_COMPLETION` or
`N2048_EC_VAR_WHOLE_VAR_COMPLETION`. Point IDs append `_SNR_<snr>`.
Each image reference carries image_path, image_key, image_slot,
image_sha256 (shared float32 dtype/shape/bytes convention), and image_file_sha256.
Source, channel, failure, fallback and measured resource fields stay per-frame.

The separate metric/statistics consumer supplies the original four metrics and
three source-paired comparisons: PARTIAL−WHOLE, EC−WHOLE and PARTIAL−EC. Exact
same-source RGB/model-identity cache reuse is allowed inside this new population.
Every worker must actually exit and be waited for by its supervisor before a
completion claims `actual_children_waited`; a child may not invent that receipt.
