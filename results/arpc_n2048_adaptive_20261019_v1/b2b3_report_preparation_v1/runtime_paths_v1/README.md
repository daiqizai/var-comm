# Exact runtime source-path compatibility for the deterministic report

Prepared locally only. The original report failed before creating its output
directory because its input-root check omitted five already registered source
files in the shared runtime. The original report source, three source pins,
authorization, statistics, CSV values, rendering functions and failed receipts
remain byte-for-byte unchanged. This revision does not authorize a scientific
call, rescoring, bootstrap, or training.

`deployment_plan.json` lists the exact source, revision, literal proof and original
read-only diagnosis bytes. Deploy these once without replacing an existing file.
The wrapper belongs beside the original report at
`experiments/arpc-n2048-adaptive-20261019/b2b3_report_v1/runtime_paths_v1.py`.
The other three files belong under
`results/arpc_n2048_adaptive_20261019_v1/b2b3_report_preparation_v1/runtime_paths_v1/`.
The raw diagnosis stdout becomes `diagnosis.json` with its original bytes.

The wrapper allows only the five exact `(path, bytes, SHA256)` triples which occur
in both transport registrations, both native registrations and the B3 metrics
registration. It rechecks all five registrations and their source descriptors.
It changes one literal predicate in `checked`, evaluates the original report from
these checked source bytes without using a pycache, and calls its unchanged
`run(auth)`. Original stat/SHA/symlink checks remain effective for every input.
The byte inverse and all other top-level function ASTs must remain exact.

The original control `.../b2b3_report_root_v1/report/` is preserved. Its wait1,
launch, supervisor, stdout and stderr are pinned by this revision. At entry the
three old main-host PID/start generations must be gone and the report output
directory must still be absent. An unknown second failure is not retried.

The root owns a new exclusive control directory:
`.../arpc_n2048_adaptive_20261019_v1/b2b3_report_root_v1/report_runtime_paths_v1`.
Use the exact argv from `deployment_plan.json`, original CPY and original
`shared.wait_child` with a 600-second bound. Preserve actual launch argv,
parent/child host/PID/start, stdout, stderr and actual wait. The wrapper writes an
additional exclusive `runtime_path_transform.json` there before calling run.
This package does not launch that command.

## Actual success route, to be written only by root after real completion

Fixed path:
`O/b2b3_report_preparation_v1/runtime_paths_v1/success_route.json`.
Required fields:

- `schema`: `B2B3_REPORT_RUNTIME_PATHS_ROUTE_V1`;
- `status`: `CLOSED`;
- `wrapper`, `source`, `authorization`, `revision`: exact triples;
- `original_failure_receipts`: original five-name descriptor map from revision;
- `actual_wait`, `launch`, `host_observation`: exact triples for the new control;
- `report_completion`, `report_manifest`: exact actual original output triples;
- `runtime_path_transform`: exact new proof receipt triple;
- `scientific_calls`: integer zero.

The host observation must bind this actual wait, launch, exact argv and new
child/controller PID/start identities and show both gone on their real host.
The wait must be strict integer returncode0, flags false and metadata_error null.
The original shared waiter has no `actual_wait` boolean; do not fabricate one.
Its producer identity and original contract remain bound by the root controller.
The original pending-visual-review completion remains unchanged. The separate
actual visual review and final delivery review manifest close that requirement.

The final compiler can bind this route and the new actual wait. For the already
reviewed publisher/assembler, root replaces exactly the `B3_report` wait spec in
the final draft with this new control and retains the old wait1 in preserved
history. The corresponding one closed-copy plan item must use the new control;
the old five receipts remain explicit failure evidence. Do not edit publisher,
assembler, original authorization, original report or original completion.

## Local proof boundary

`proof.json` records the literal inverse and metadata path rejection cases. It
did not execute the original report module, old tests, plotting, model code,
packet code, scores or bootstrap. The fixture substitutes PurePosixPath for Path
only in its local stdlib module namespace, because it runs on Windows; the
production wrapper bytes retain Path. One initial local packaging attempt stopped
at Windows path spelling before writing any package file or executing a report;
`local_packaging_failure_note.json` preserves that fact. The corrected package
builder ran once. No remote qualification or actual report execution was done.
