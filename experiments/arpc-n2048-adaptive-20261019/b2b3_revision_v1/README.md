# N2048 B2/B3 explicit scope revision

This independent directory implements the user's 2026-10-19 manuscript request.
It preserves completed adaptive B0/B1, all seven bound original transport files,
old budget records, original RX exit1 and its metadata-only saved-call closure.
The new authorization is prospective: nine exact calibration cells, including
five NeST-Com 10dB policies. This directory does not train, tokenize, compute TX
probabilities, recover images, score, bootstrap, or automatically launch B3.

## Budgets and unchanged mathematics

B2 uses all original cal1000 sources and seeds4101/4102/4103:27000 logical
objects,54000 transmitted subframes,54000 header and at most54000 body decodes.
B3 uses the original external_common100_v1 and seeds6201/6202/6203 after freezing
four actual-DINO policies:1200 objects,2400 subframes,at most4800 decoder calls.
Both phases together are28200 objects,56400 subframes,and112800 maximum decodes.
Header rejection may skip its body call; this reduces the actual decoder count,
never the complete logical denominator. Rejection is an experimental result.

`plan.B2_CELLS` is the exact nine-cell user list. All actual endpoints follow
the original `adaptive_plan.choose` length-only fallback over the B0 catalogue.
No-fits transmit both actual null-profile subframes and retain their denominator.
Original `pair_protocol` and `pair_link.Runtime` implement every byte, actual
LDPC encode/decode,QAM,CRC,parse and energy computation. The noise namespace is
unchanged NEST_ARPC_N2048_EQUAL_PROTECTION_V1 with the original source/seed-based
2048-by-2 Gaussian draw. Every logical event actually encodes both subframes;
different MCS/SNR/profile events are never reused by predicted output equality.
The new disjoint paid pair-ID ranges start at1000000(B2) and2000000(B3).
Each body pays13 length+16 CRC+96 reassembly bits; each fragment also pays68
header symbols. Nominal Es2,totalN2048,CPU backend two intra/one interop thread,
FP32/TF32off and all original source/native models remain unchanged.

## Registration and controls

Only the actual remote repository is Git. New runtime roots are
`L/b2b3_revision_v1/{B2,B3}` and controls
`L/b2b3_revision_v1/controls/{B2,B3}_{tx,rx,audit}`. Lightweight registrations
and transport summaries are `O/b2b3_revision_v1/{B2,B3}`.

Root supplies an immutable copy of its original authorization JSON with schema
NESTCOM_MANUSCRIPT_COMPLETION_AUTHORIZATION_20261019_V1, nine B2_cells, explicit
B2/B3 caps and seeds, `new_budget_authorized=true`, `prospective_only=true`,
`historical_scope_records_unchanged=true`. `--request-path` supplies the remote
copy of the original request; its bytes/SHA must equal authorization.request.
The original authorization's Windows request path is retained without rewriting.

Metadata-only registration command (CPY is the original frozen LDPC Python):

```text
CPY -B E/b2b3_revision_v1/prepare.py B2 --authorization REMOTE_AUTH --authorization-sha256 SHA --authorization-bytes BYTES --request-path REMOTE_REQUEST
```

It verifies actual completed B0/B1 closure and opened native review without
rerunning them, the original source catalogue and implementation pins, and every
source's full endpoint availability. It creates an exclusive phase directory and
registration. Existing directories, unknown failures or changed assets stop;
there is no reset, resume or automatic retry. Registration source hashes include
all new files and inherited bound implementations. Root must independently
review and qualify the new code before registering or dispatching science.

Explicit one-stage control, called only once per role after checking prior state:

```text
CPY -B E/b2b3_revision_v1/control.py B2 tx --expected-registration SHA
CPY -B E/b2b3_revision_v1/control.py B2 rx --expected-registration SHA
CPY -B E/b2b3_revision_v1/control.py B2 audit --expected-registration SHA
```

The original adaptive execution.lock is held throughout the owned child wait.
Durable supervisor/launch/actual_wait/stdout/stderr records include PID,start
ticks,host,exactargv and bounded timeout. Engineering wall ceilings are7200s TX,
43200s RX,and1800s audit. They do not change any scientific call cap. Only the
new owned process session receives TERM/KILL on timeout. Unknown failures stop.
Root's persistent outer supervisor must preserve its own actual wait as well.

## Independent RX and outputs

TX output received/public.json contains only phase/caps,implementation pins,
backend catalogue,anonymous observation jobs and the registration digest.
Anonymous jobs contain key,SNR,two paid counters and observed array descriptor.
They contain no source ID,index,desired codec/endpoint,TX object or source truth.
RX receives only this public descriptor; it never reads the private registration.
Its PFS read boundary permits explicit implementation/environment roots and its
own received/output directories. The exact old protocol source directory is
allowed, including optional bytecode lookup, fixing the former qualification's
final metadata denial without changing or repeating that historical run.

Every header/body result is fsynced before its ledger becomes COMPLETE; every
logical assembly receipt is independently saved. Per-call RESERVED/FAILED rows,
physical checkpoints,logs and failed files are retained on any unknown error.
RX assembles only actual received fragments. A legal wrong object is passed on
as received and only labeled in the post-exit audit; truth cannot repair it.
Both channel rejection and source-parser rejection retain the logical frame.

After RX actualexit0 and PID/start gone, `audit` joins the actual received bytes
with TX records. O/transport_completion.json references the large runtime
`logical_frames.json` and `rx/source_jobs.json` by path/bytes/SHA. Logical rows
contain key,source_id,source_index,codec,policy_id,SNR,seed,target,endpoint,
layout_id,accepted,exact_post_exit,accepted_wrong,source_input_key,actual received
codec/endpoint,failure reason,energy and original receipt descriptors. Endpoint
and sent_endpoint are the actual TX fallback endpoint; actual_received_endpoint
is separately retained. Deduplicated source jobs contain key,codec,endpoint,
bit_length,input descriptor,generation_seed0. Native and metric stages must use
these actual bytes and exact cache identity, with their own call accounting.

## B3 frozen-policy JSON contract

Root's completed B2 analysis writes a new JSON, with this required schema:

```text
schema: N2048_B2_DINO_FOUR_POLICIES_V1
status: FOUR_POLICIES_FROZEN
metric: dinov2_vitl14_cosine
candidate_means: nine rows {codec,snr_db,policy_id,calibration_mean_DINO,
                          source_mean_count:1000,noise_per_source:3}
rows: plan.select_four(candidate_means)  # four exact dicts including tie_break
no_additional_candidates: true
calibration_bootstrap_calls: 0
B2_registration: descriptor
B2_transport_completion: descriptor
source_evidence: descriptor
metric_evidence: descriptor
selection_evidence: descriptor
score_completion: descriptor  # JSON status COMPLETE
score_actual_wait: descriptor # actual bounded wait0,gone process
```

The select_four rule is maximum actual source-mean DINO. Exact ties preserve
the registered lower q,lower nominal rate,higher target order,then lexical
policy ID as a deterministic final order. Overlapping intervals do not add
candidates. `prepare.py B3` accepts the above `--selection`, `--selection-sha256`
and `--selection-bytes`, verifies all evidence and recomputes the four winners
from the nine saved means without any model/PHY/bootstrap. Root must save the
actual selection calculation, not populate those means from historical proxies.
B3 stage commands use the resulting separate B3 registration digest.

## Engineering qualification

`test_revision.py` uses only synthetic metadata,byte transforms and a temporary
ledger. It prohibits numpy,torch,tensorflow,transformers,Sionna/PIL imports and
performs no Runtime constructor,transmit,channel,decode,model or score call.
It covers the amended budgets,complete source grids,fallback,null frames,
independent public schema,paid pair assembly,wrong/rejected inputs,exact DINO
selection ties,checkpoint-before-COMPLETE,failed call preservation and no retry.
The old B0/B1 qualification is never rerun by these tests.

No completed experiment or scientific source file outside this new directory
is edited. Source changes are prohibited once any new registered stage is active.
