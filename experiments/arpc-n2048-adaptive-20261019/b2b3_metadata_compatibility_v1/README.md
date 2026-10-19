# Exact entropy-cache metadata compatibility and preparation continuation

This is an independent metadata adapter. The six `b2b3_analysis_v1` files, seven
transport files, old registrations, failed preparation, reconstructed pixels,
source streams, scientific functions, numerical settings and budgets remain
unchanged. This directory must not be used to run a model, channel or metric.

## Diagnosis and admitted schemas

The failed B2 preparation assumed every `ORIGINAL_ENTROPY_RX` image record had
an inline `original_visual_identity`. The closed catalogue actually contains:

* 6,540 original inline records, exact members of the old saved image index;
* 332 T1-B `receiver_recovery_v1` ACTUAL_NEW records with separate image evidence;
* 75 T1-C ACTUAL_NEW records with separate evidence plus image_key/image_slot.

There are no other admitted schemas. B2's 7,787 received source jobs include
175 candidates from the T1-B schema; they are verified cached outputs, not gaps.
The first former KeyError was at zero-based received job 2,363. No native
registration or model call occurred during the failed preparation.

The metadata diagnosis verifies all 407 newer records against their closed
images maps, actual received header/profile/payload, exact bit length, raw-bit
SHA, packed-byte SHA, source catalogue endpoint, per-image decode/render proof,
and false receiver-truth/TX-CDF flags. It binds both registrations and shared
model/static assets, actual native model states/runtime/numerical flags,
registered source/CDF/render code, successful receiver wait, launch and gone
processes. T1-B's actual native identity was not in its historical publication
copy list: it is explicitly pinned now and matched to the closed registered
implementation and reference identity; no historical publication claim is made
for that file. T1-C's identity is additionally bound by its published copy list.
Historical T1-B/T1-C PID records had no hostname; their original schema is kept,
and they are checked on the original main host. New records require host identity.
Both B2 and B3 preparation explicitly require the hostname recorded by the
SHA-bound full metadata diagnosis before checking any hostless historical PID;
passing a remote B2 closure on leo does not authorize preparation there.

`lineage.build()` reads only JSON and source files. It does not open image or
token arrays. Subsequent cache wire/pixel checks are the original frozen
`verified_candidate` body. New proof is attached beside the original cache as
`original_entropy_metadata_revision`; the old image object is unchanged and no
inline identity is manufactured from `model_routes`.

## Exact source adaptation

The original native.py bytes must match 34279 bytes / SHA-256
`d02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889`.
Three exact text replacements are made in memory, with an exact inverse and
same-interpreter AST comparison of every unaffected function:

1. The two-line inline-identity check calls the explicit three-schema validator.
2. The preparation directory creation calls a narrowly scoped directory guard.
3. The lineage write calls a preservation guard.

Only `verified_candidate` and `prepare` are recompiled into the original module.
No global filesystem monkeypatch, broad exception suppression or scientific
function replacement is used. Original code pins stay in the unchanged native
registration. An independent revision receipt binds the extra adapter, root
authorization, failure audit, actual command and new metadata proof.

## Preserving the failed B2 preparation

The failure audit is fixed at
`results/arpc_n2048_adaptive_20261019_v1/b2b3_analysis_metadata_failure_v1/failure_audit.json`,
6556 bytes, SHA-256
`d2b8a044152987359dca74e2489e52c62c332a38f0a761b91301dadfeb1575c8`.
Before continuation it must still describe a genuine returncode 1, both original
processes gone, no native registration and zero new model/metric/channel calls.
Every preserved original/copy descriptor and original six-source pin is checked.

For B2, the existing runtime root must contain exactly the original
`failure.json` (999 bytes, `5bb9c1ab...`) and
`native_identity_compatibility.json` (14906 bytes, `671ed778...`), with exact SHA.
The existing output directory must be empty. The newly computed lineage must
equal the original saved JSON; its original bytes are kept. The failure file is
never deleted, renamed or overwritten. For B3 both phase directories must be
absent and are created exclusively as originally intended.

A new exclusive directory and O_EXCL claim at
`L/b2b3_metadata_compatibility_v1/<phase>` permit one metadata attempt. A failure
is saved there and there is no retry path. Do not remove a failed claim or call
the original direct preparation again.

## Root authorization and execution interface

Root supplies a separate immutable JSON with these exact fields (extra evidence
is permitted):

```
schema: B2B3_METADATA_COMPATIBILITY_AUTHORIZATION_V1
status: AUTHORIZED_METADATA_ONLY_REVISION
allowed_prepare_phases: [B2, B3]
scientific_math_changed: false
new_scientific_calls_authorized_by_revision: 0
automatic_retry: false
original_code: metadata.original_pins()  # all six exact remote descriptors
adapter_code: exact descriptors of these five files
failure_audit: metadata.FAILURE_AUDIT
diagnosis: exact descriptor of the persisted full metadata diagnosis
independent_review: exact descriptor
qualification: exact descriptor of the new original-environment qualification
B2_transport_registration: exact descriptor
```

The B3 authorization is a new file, additionally containing `B2_closed_revision`
as a direct `{path, bytes, sha256}` descriptor. Do not modify the B2 authorization
or wrap this descriptor inside another file. The original B3 transport
registration must already exist and satisfy the original frozen-policy gates.

The only preparation command (outer control owns true Popen.wait and logs) is:

```
CPY -B E/b2b3_metadata_compatibility_v1/runner.py prepare \
  --phase B2 --expected-registration TRANSPORT_SHA \
  --authorization AUTH_PATH --authorization-sha256 AUTH_SHA --authorization-bytes AUTH_BYTES \
  --actual-control L/b2b3_analysis_root_v1/B2_prepare_metadata_compatibility_v1 \
  --expected-runner RUNNER_SHA
```

Use B3 consistently for its later separate preparation. The outer control path
is exactly `<phase>_prepare_metadata_compatibility_v1`; the old
`B2_prepare_none` control is never reused. The actual preparation walltime remains
the root's existing 1800-second metadata bound; it authorizes no scientific calls.

After actual wait 0 and both wrapper child and outer owner have exited, run on
that same main host, using a new bounded root control:

```
CPY -B E/b2b3_metadata_compatibility_v1/runner.py close \
  --phase B2 --expected-registration NEW_NATIVE_REGISTRATION_SHA \
  --authorization AUTH_PATH --authorization-sha256 AUTH_SHA --authorization-bytes AUTH_BYTES \
  --expected-runner RUNNER_SHA
```

This writes `O/b2b3_metadata_compatibility_v1/<phase>/closure_observation.json`
and `closed_revision.json` exactly once. It checks precise wrapper argv,
PID/start/host, returncode strict integer zero, timeout/termination flags false,
metadata_error null, source pins, preserved files and registration. It binds
outer stdout/stderr without rewriting them. The close command must itself be
wrapped by root's real wait and log recording; it never claims its own future
exit as part of its outputs.

## Downstream gates and roles

Only native **prepare** requires this wrapper. Frozen native configure/run/
observe/seal and score prepare/run/select retain their original commands and
registration schema. None of those entry points reads the phase-root historical
failure as an active failure, nor do they require an old unwrapped prepare argv.
Before every root action, validate the extra closed revision:

```
CPY -B E/b2b3_metadata_compatibility_v1/runner.py guard \
  --phase B2 --expected-registration NEW_NATIVE_REGISTRATION_SHA \
  --closed-revision CLOSED_PATH --closed-revision-sha256 CLOSED_SHA --closed-revision-bytes CLOSED_BYTES \
  --expected-runner RUNNER_SHA
```

Python API: `runner.guard_closed(exact_closed_descriptor, expected_native_sha)`.
The root helper's previous `assert not(AL/phase/'failure.json').exists()` must
be replaced only at this exact boundary with the guard result. This retires
exactly the one SHA-bound historical metadata failure. It does not hide a new
failure, an altered old failure, failed worker/control, nonzero actual exit, or
incomplete stage. Root must retain its separate original stage wait/failure gates.

The close observation is created on the preparation host. Main-host guards also
recheck real /proc identities. Guards on leo consume the exact SHA-bound closed
main-host observation; they do not treat a remote PID missing from local /proc
as gone. Both root invocation and guard output should remain in the actual
per-stage control evidence. Include the revision receipt in downstream root
publication/report manifests; no immutable native registration is edited.

## Qualification

`test_metadata.py` runs only new synthetic metadata cases: exact known schema,
actual payload and endpoint, native numerical/model identity, no truth/CDF,
explicit provenance, immutable residue, exclusive creation, source inverse/AST,
and real/cross-host closure semantics. It does not import NumPy/Torch/models,
read production data, invoke production prepare, run the old 56 tests, launch
subprocesses, repeat a scientific call, or execute the live compatibility build.
Fixture files are the only PFS files allowed by the synthetic audit guard apart
from explicitly allowed source text. No whole-environment upgrades are involved.
