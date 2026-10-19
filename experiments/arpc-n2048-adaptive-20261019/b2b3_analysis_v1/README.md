# B2/B3 actual native outputs and B2 DINO selection

This is an independent revision under the new 2026-10-19 manuscript request.
It does not modify the seven bound transport sources or any original model,
source codec, prior, decoder, score implementation, or successful result.
Preparation and synthetic qualification are not scientific completion.

## Contract and bounded work

Transport is read only after its real TX, RX and audit waits are zero and their
parents/children are gone. B2 contains exactly 27,000 logical frames; B3 exactly
1,200. The original nine B2 cells, two SNRs and three seeds remain unchanged.
Native work is bounded by the number of distinct **actual received** source jobs
that do not match a verified closed cache. No source TX, tokenizer, channel,
training or bootstrap is performed here.

The actual `(codec, endpoint, bit_length, object SHA)` selects cache candidates.
Input bytes, float pixels and frozen native model/numeric identity are verified
before reuse. A different source's exact object can be reused as a native result,
but its quality score is never borrowed for the current source. Conflicting
cache pixels or scores are fatal. The cache includes B0, the two qualification
native jobs, all completed cal100 native jobs, and closed B2 actual objects when
preparing B3. No successful native or score call is automatically retried.

Missing NeST jobs execute the original `nest_prefix_rx.py`: original independent
ReceiverCodec, unchanged VAR/Dc, FP32/TF32 off, six intra-op and two inter-op
threads, 12 GiB allocator, 15 GiB startup and 2 GiB running floor. Missing ARPC
jobs execute the original `arpc-external-baseline-20261019/a2b_rx_v1/native.py`
with the exact SHA-bound original A1 engine: four/two threads, preset 0.06M,
seed 0, CFG 3, tau .5, original scale/mask schedules and model weights. Its
existing 64 GiB startup / 8 GiB running floor and 70 GiB allocator are retained.
The current free GPU may be admitted on either shared-PFS host; its PCI/CUDA
UUID mapping is recorded before model work. No old CUDA index is assumed.

Each native worker is bounded to 43,200 seconds plus 300 seconds for outer
cleanup. This is a conservative engineering wall limit, not additional science
calls. Original qualified worker per-call ledgers and output semantics remain
unchanged. Only the owning session can be terminated on timeout. Any failure
preserves evidence and blocks automatic continuation.

Both native workers receive only copied actual RX bytes, public paid endpoint
metadata, shared static assets and models. No source ID, source image/token,
transmitter CDF, clean desired endpoint, or transmission truth is in their
public configuration or job list. Their original read firewalls remain active.
The post-exit join preserves legal wrong received inputs, parse failures,
NULL/source-no-fit frames and all failure denominators.

## Parent-controlled sequence

Let `A=R/experiments/arpc-n2048-adaptive-20261019/b2b3_analysis_v1`,
`L=.../arpc_n2048_adaptive_20261019_v1/b2b3_analysis_v1`, and
`O=R/results/arpc_n2048_adaptive_20261019_v1/b2b3_analysis_v1`.
Root supplies a persistent outer launch/wait for all mutating roles and reviews
the actual preparation receipts before launching models. Commands below are
templates, not claims that any step has run.

1. `CPY -B A/native.py prepare --phase B2 --expected-registration <transport SHA>`.
   This is a CPU metadata/closed-file audit. It writes `O/B2/registration.json`
   and exact cached/missing input lists. Inspect the new native counts.
2. On the intended native execution host, run
   `CPY -B A/native.py configure --phase B2 --name nest --gpu <index> --expected-registration <analysis SHA>`.
   ARPC uses the same command with `--name arpc`. These are device/asset
   metadata operations. If a codec has zero gaps, skip its configure.
3. On that host, run `CPY -B A/native.py run --phase B2 --name <nest|arpc> --expected-registration <analysis SHA>` once.
   The controller writes persistent stdout/stderr, exact child and parent
   PID/start ticks, argv, and actual wait. It holds the original shared adaptive
   execution lock. Native roles are serialized. Do not call the old worker
   directly or rerun a failed/partially successful role.
4. Once its controller has exited, invoke `native.py observe` with the same
   phase/name/SHA **on its execution host**. It checks both process generations
   are gone, the real wait is integer zero, timeout/termination flags are false,
   and there is no failure. It writes one immutable `closure_observation.json`.
   A shared filesystem does not make another host's `/proc` authoritative.
5. `CPY -B A/native.py seal --phase B2 --expected-registration <analysis SHA>`
   verifies both host observations and outputs; no native inference occurs.

The same sequence is used for B3 only after transport registration consumes
the closed four-policy B2 selection. No new calibration choices are introduced.

`native_completion.json` has `status=COMPLETE`, `logical_pixels`, `inputs`,
`outputs`, full manifest, `native_actual_waits`,
`process_closure_status=ALL_NATIVE_PROCESSES_GONE`, and
`process_closure_evidence` with host-specific observations. The logical rows at
`L/{B2,B3}/actual_logical_pixels.json` include key, codec, method, source ID/index,
SNR, noise seed, image descriptor (null means fixed RGB .5), array key, raw
float32 SHA, receiver status and `accepted_wrong_post_exit`. Method names are
`NEST_ADAPTIVE_N2048` and `ARPC_ADAPTIVE_N2048`.

## B2 scoring and freezing

After native closure:

1. `PY -B A/score.py prepare --expected-registration <B2 analysis SHA> --gpu <index>`
   verifies actual source/pixel pairs, the original DINO numeric metadata,
   existing cal100 and cal1000 DINO scores, and the 100 + 900 already saved source
   features. It writes `O/B2/score_registration.json`. Source forwards are zero.
2. `CPY -B A/score.py run --expected-registration <score SHA>` once. The bounded
   original DINO worker receives only the missing pair plan. Wall limit is
   21,600 seconds plus 300 seconds cleanup; exact reconstruction score cap is
   the registered missing count. If none are missing, its real child writes a
   zero-call receipt without importing or loading a model.
3. After controller and child exit, `CPY -B A/score.py select --expected-registration <score SHA>`.
   All 27,000 logical rows, including failures and reused pairs, are joined.
   Three noise values are averaged within each source, then 1,000 sources are
   equally averaged. No bootstrap is called. All nine candidate means are
   saved. Maximum DINO selects one policy per method/SNR. Exact ties use the
   original lower q, lower nominal rate, higher target order, then lexical ID.

`dino_worker.py` pins original `cal100_dino.py` SHA
`dd90bd108e3c36c7aff78eb4219a76a616e06c9e7e5c68ec67d85bee54e2beb0`.
Its four reversible substitutions change only the registration/output paths,
the source-feature population label, and fixed-gray loading. Metric evaluator,
preprocessing, source uint8/255 conversion, float32 native images, source feature
reuse, requested DINO-L, numeric flags and score-call AST are unchanged.
All opposite signs and failures are retained. B2 never scores test sources.

`O/B2/selection.json` uses `N2048_B2_DINO_FOUR_POLICIES_V1` and exactly the
transport `plan.frozen_cells` schema. It binds transport registration/completion,
source references, metric identity, candidate means, score completion and real
score wait. B3's seven-metric scoring and one paired bootstrap are a separate
root-owned stage, consuming the closed B3 logical pixels. This code never
auto-starts B3 or B3 scoring.

## Qualification

`test_analysis.py` is synthetic CPU only. It checks exact-input identity,
public job isolation, corruption detection, actual image validation, all nine
cells and three-noise source weighting, failure-denominator handling, exact
tie rules, reversible DINO adaptation, unchanged metric call AST, cross-host
closure constraints, no retry paths and zero-bootstrap selection. It does not
load models, invoke packet backends, consume real scientific assets or call
the production preparation/worker functions.

## Review fixes: closed native and metric identity

Before accepting a cached input, preparation now binds the original closed
registration, public native configuration, actual runtime and completion/state
membership for every ARPC origin and both full N2048 receivers. The A2 clean
cache's 24 inherited A1 outputs also bind the saved A1 closure and runtime.
Saved qualification/cal100 NeST outputs compare both model routes and static
CDF assets, actual model state hashes and complete numeric runtime/flags.
Original entropy outputs retain their existing six-field native identity check.
B3 inherited B2 outputs must also match B2's model routes, static CDF and ARPC
weight descriptors. Unknown origins, mismatched evidence and conflicting native
pixels stop preparation; no cache miss is silently converted into a retry.

ARPC configuration uses a bounded 120-second **metadata-only** child under the
actual `/usr/bin/python3` and original independent overlay. It imports only the
standard library, reads `torch/version.py` as AST, and records the resolved
interpreter binary SHA, full Python version, Torch metadata/build, and each
overlay package's version and METADATA/RECORD SHA. These must match the original
environment policy and closed Python/Torch runtime. The binary and metadata
files are rechecked before the actual native child and at sealing. Historical
interpreter binary equality is not claimed: its original SHA was not recorded.
No new runtime installation, model import or native qualification is involved.

Sealing compares actual ARPC runtime to the closed reference, excluding only
PID, GPU name/UUID, timestamp and per-run cache directory. Sampling, precision,
native arguments, schedules and Python/Torch remain exact. New NeST native
identity is also checked after real exit. Main H800 GPU0 for NeST and a qualified
free leo-tactile GPU for ARPC are suitable execution routes; admission is always
rechecked on the actual host. No cross-hardware timing equality is claimed.

DINO cache reuse binds the original cal100 score registration's `all_states`,
the verified-cache completion's exact cache and input descriptors, and the
original worker's scores plus metric identity. The historical cal100 registration
has zero cross-endpoint logical pixel reuse; this fact is checked explicitly.
Nonempty states must match their saved source, endpoint, pixel and scalar.
New old-worker scalars must also equal their exact saved per-pair JSON values.
The T1-B four-metric metadata with identity `cea0a051...` supplies the original
pilot/T1-B suite provenance; its complete metadata hash is verified and its DINO
subidentity, weights, preprocessing, package versions, flags and runtime must
match the original ARPC DINO metadata. Different global suite identities are
retained. This is a metadata compatibility check, never a new score computation.

The original 35 synthetic tests and their prior receipt remain intact. The new
`TargetedIdentityTests` class covers these review fixes and can be selected alone
with `python -B test_analysis.py TargetedIdentityTests`; root performs one final
original-environment qualification of the final source set after review.
