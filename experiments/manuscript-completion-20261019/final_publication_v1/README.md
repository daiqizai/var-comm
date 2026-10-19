# Final manuscript publication — root preparation only

These five files are local preparation. They have not been deployed or invoked.
They do not launch a model, a channel, scoring, bootstrap, plotting, a TeX build,
or a scientific audit. Root must first complete the actual authorized work and
fill the descriptor with real receipts. No illustrative result is executable.

Fresh read-only repository observation during preparation: `main` at
`e6ef4dcd0daa2f20ee9be27036b6c36aa7e64798`, tracked tree clean, normal origin
`https://github.com/daiqizai/var-comm.git`. This is a historical observation, not
a hardcoded future parent. Root reads HEAD again before assembling final input.
The current remote manuscript directory did not yet exist. Scientific B2 native
work was still running, and B3/manuscript integration was not final. No complete
descriptor has been written by this preparation task.

## Fixed locations and phases

- Sole Git root R: `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM`.
- Remote helper deployment S: `R/experiments/manuscript-completion-20261019/final_publication_v1`.
- Root descriptor P: `R/results/manuscript_completion_20261019_v1/final_publication_preparation_v1/authorization.json`.
- New controls A: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/manuscript_completion_20261019/final_publication_v1`.
- Existing Python for the original tools: `.../var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python`.
- CPY is used only to run the stdlib publication controller.

Root reviews and deploys these exact five source bytes, includes them in the
eventual exact publication manifest, and creates P exclusively. Root updates
the four documents with the actual final scientific results before checks; the
publisher never writes scientific prose or replaces historical document bytes.
Local `descriptor_pin.json` is just the final absolute `{path,bytes,sha256}`.

Run only the intended phase, after inspection of the preceding real output:

```text
python -B invoke.py checks --descriptor-pin descriptor_pin.json
python -B invoke.py commit --descriptor-pin descriptor_pin.json
python -B push.py --descriptor-pin descriptor_pin.json
python -B sync.py --push-receipt local_push_verified.json \
  --expected-local-before exact_local_before.json --attempt ABS_HELPER_DIR/sync_attempt_01
```

Root may run the remote `publication.py checks|commit` under its existing bounded
supervisor instead. Its cwd must be R. Arguments are `--descriptor P --sha256 SHA
--bytes N`. Never pass credentials on argv. `push.py` is the local credential
client and is the only supported push invocation.

`checks` starts once in a new A, stages positive exact paths in batches using
`git add --sparse --`, runs the original manifest updater and stages the resulting
`release_manifest.json`. It then runs the default cached diff check, original
repository verifier, and original 264-test CPU entry in that order. The three
tool bytes must match both descriptor SHA and current parent commit. CUDA is
hidden, Python bytecode disabled, and original CPU thread settings retained.
Actual stdout/stderr, exact argv, worker/controller PID/start/host and actual
wait are persisted per command. Limits are 300 s for manifest update, 120 s for
diff check, 1200 s each for verifier and CPU checks, and 180 s for commit/push.
Only each newly launched own session can be terminated after its deadline.

The staged tree, file bytes, source and closure inputs are checked again after
checks and before a normal commit. The commit author identity is inherited via
per-command `-c` from the current parent, without persistent configuration edits.
Normal push checks that remote main is still the parent or the already-created
head, never force-pushes, and requires a fresh `ls-remote` matching the exact new
head. A successful command is not substituted for remote SHA verification.

Any existing attempt directory, source/hash drift, unknown failure, nonzero exit,
timeout, unreviewed changed path or default whitespace failure stops and preserves
the evidence. There is no automatic retry, rollback, cleanup, formatting change,
CRLF exemption, filter change, global Git setting, merge or rebase. Diagnose a
real failure before a separately reviewed exact continuation. Do not rerun
successful scientific work or requalify old tests because a publication failed.

## Descriptor schema

Schema is `FINAL_PUBLICATION_AUTHORIZATION_V1`, status must be
`ROOT_REVIEWED_COMPLETE`; `repository`, `branch` and `publication` match above.
`parent` is the freshly observed actual 40-character HEAD. `scientific_calls` is
0, `large_backup_deferred` is true, and `commit_message` is factual final text.

Every descriptor is exactly `{path,bytes,sha256}`, with an absolute normalized
PFS path beneath exactly R or BASE. Arbitrary system paths and RT environment or
weight paths are not publication evidence. Repository `files` use normalized
repository-relative paths instead.
Only regular files without symlinks are accepted. Bytes and SHA are checked
before and after stat. Each light file is at most the existing 10,000,000 bytes.

Required fields:

1. `files`: exact final new/modified files, including four root documents and
   publisher sources. P itself and the newly generated release manifest are
   added explicitly by the controller; do not put either into this list. Do not
   include unchanged historical dependencies here: bind them as evidence.
2. `documents`: keys `README.md`, `RESEARCH_STATUS.md`, `PROGRESS.md`,
   `EXPERIMENTS.md`. Each has `before:{bytes,sha256}` matching parent Git bytes,
   and `after` equal to its exact row in `files`.
3. `publication_sources`: keys exactly `publication.py`, `invoke.py`, `push.py`,
   `sync.py`, `README.md`, each pointing to its actual deployed S file.
4. `repository_tools`: original absolute pins for `update_repository_manifest.py`,
   `verify_repository.py`, `run_cpu_checks.py` beneath R/tools.
5. `source_packages`: exact relative-path lists with these keys:
   `N1024_audit`, `N1024_repair`, `N1024_report`, `B2B3_transport`, `B2B3_analysis`,
   `analysis_continuation_v1`, `analysis_continuation_v2`, `metadata_compatibility`,
   `B3_metrics`, `B2B3_report`, `manuscript`, `publisher`. Every member must be in
   `files`; identities therefore cannot be replaced by directory names alone.
6. `closures`: keys exactly `N1024_audit`, `N1024_repair_delivery`, `N1024_report`,
   `B2_transport`, `B2_native`, `B2_DINO`, `B2_selection`, `B3_transport`,
   `B3_native`, `B3_metrics`, `B3_statistics`, `B3_report`, `manuscript_compile`,
   `manuscript_delivery`. Each value has `record` (closed receipt pin), nonempty
   `assertions` (`{pointer,equals}` exact JSON-pointer values), `wait_ids`
   (nonempty references below), `light_members` (exact pins), and `manifests`
   (see below). Use real producer status/count fields; no synthetic normalization
   that relabels a failed or intermediate receipt as scientific success.
7. `actual_waits`: the actual finite successful stages with exact launch/process
   provenance, schema below. Reused completed stages may be referenced by more
   than one closure, with no re-execution. Distinguish model, PHY, DINO, metric,
   selection, statistics, deterministic report and actual compile waits.
8. `visual_reviews`: keys `N1024_utilization`, `N1024_quality`, `B3_figures`,
   `manuscript_PDF`. Each supplies `record`, nonempty exact `assertions`, optional
   `status_pointer` (default `/status`), `final_not_intermediate:true` and nonempty
   `opened_bindings`. Actual status must begin PASS/COMPLETE without PENDING,
   INTERMEDIATE or NOT_RUN. Each opened binding contains an `opened_pointer`
   whose saved value is true and an artifact binding below. Bind actual final
   PDF/page images and their real all-pages inspection; opening an earlier
   intermediate PDF cannot close the final manuscript.
9. `history`: `qualification_attempts` and `preserved_failures`, each a nonempty
   list of `{description,records:[exact pins]}`. Record actual attempts separately.
   Analysis qualification is 19 PASS, then 0 additional PASS, then 37 PASS across
   three attempts, not one 56-test pass. Report qualification retained 15 passes
   before interruption and then 11 remaining passes, not one 26-test attempt;
   bind its actual combined receipt. Preserve the metadata-only cache-schema
   failure, the source-order/dispatch syntax observations and report-qualification
   failures with their actual continuation evidence; do not describe engineering
   qualification as real-image science.
10. `failure_locations`: explicit exact current phase/worker/control failure
    paths. If any exists, it must be an exact unchanged pin in the preserved
    failure records. Root must list all real relevant source-defined locations;
    an unknown current scientific failure is fatal. A historical exception does
    not waive an adjacent worker failure or a new SHA.
11. `additional_evidence`: optional exact light pins (root authorizations,
    request/scope revisions, source/root reviews, closure observations, etc.).
12. `excluded_untracked`: exact repository-relative path to nonempty explanation
    for deliberate omitted light artifacts in the positive roots. This is never
    a Git ignore/filter edit. All untracked self-written `.py/.cpp/.h/.sh` must
    still be positively included, as required by the original verifier.

An artifact binding is `{pointer,descriptor,relative_base?,path_mapping?}`. `pointer` selects
the producer's saved `{path,bytes,sha256}` within the pinned record; SHA and size
must equal `descriptor`. An absolute saved path must match directly. A relative
saved path must resolve by the explicit absolute `relative_base` to that same
descriptor. `manifests` entries are `{record,member_bindings:[bindings]}`. Only
light output members are rehashed here; arrays, feature archives and checkpoint
weights remain bound by the original actually closed scientific audit. Do not
pass them to this publication controller or re-audit active data.

An original Windows review is kept unchanged. Its artifact binding may set
`path_mapping:"WINDOWS_WORKSPACE_TO_REPOSITORY_V1"` only when the saved absolute
Windows path is beneath exactly `C:/Users/11946/Documents/ChatGPT/comm`. The same
relative path must map to R and match the actual bytes/SHA. This mode rejects
other drives/roots, traversal, alternate stream names and mixed relative-base
rules; it does not silently treat arbitrary Windows paths as repository files.

An actual-wait entry has this shape (explanatory strings below are not valid pins):

```json
{
  "id": "B3_score_outer",
  "receipt": "actual wait descriptor",
  "object_pointer": "",
  "actual_wait_key": "actual_wait",
  "returncode_key": "returncode",
  "launch": "actual launch descriptor",
  "argv_pointer": "/argv",
  "argv": ["exact", "already", "executed", "command"],
  "execution_host": "actual uname hostname",
  "identity_records": [],
  "processes": [
    {"record": "launch descriptor", "pointer": "/process", "host_key": "host"},
    {"record": "launch descriptor", "pointer": "/parent_process", "host_key": "host"}
  ],
  "host_observation": "only required for a different host"
}
```

Use real key aliases if a frozen receipt uses `exit_code`, `child_process`, etc.
Each process spec can set `pid_key`/`start_ticks_key`. Every existing `host` and
`hostname` field is checked automatically, whether `host_key` is supplied or not;
an explicit custom `host_key` is also checked. At least two distinct actual child
and controller `(PID,start_ticks,host)` generations are required. Repeating a
single generation twice is rejected.
The wait's actual child must equal the launch's child, and the bound generations
must contain that child and that launch's different real controller. Optional
`wait_child_identity`, `launch_child_identity` and `launch_parent_identity` use
the same `{pointer,pid_key?,start_ticks_key?,host_key?}` shape for real historical
aliases. Defaults select `/process`, `/process` and `/parent_process`; the wait
pointer is relative to `object_pointer` when a receipt contains multiple passes.
Additional identity records must be explicitly pinned, never guessed PIDs.
Compile receipts with `passes` use `object_pointer:/passes/0` and `/passes/1` to
bind each real wait rather than inventing a flattened exit. Current true wait
must be strict integer0. Existing timeout/terminate/kill flags must be false;
existing metadata_error must be null.

A genuinely hostless historical launch requires a wait-level `host_provenance`:
`{record,pointer,anchor_record,anchor_pointer,host_key?}`. The record must also be
in `identity_records`; it selects an actual host-aware process. `anchor_record`
must be this wait's launch, and `anchor_pointer` must be `/parent_process`. The
host-aware generation must equal that exact launch controller's PID/start. This
can reuse an existing outer launch/actual process receipt and does not require
an extra observer for a hostless child whose launch controller is already bound.
A caller-supplied hostname alone is insufficient.

Most new B2/B3 controls use the original `shared.wait_child`, whose saved receipt
does not contain an `actual_wait` boolean. Select the explicit
`receipt_schema:"FROZEN_SHARED_WAIT_CHILD_V1"` and add `producer` pin for
`R/experiments/arpc-n2048-adaptive-20261019/shared.py`, SHA
`beab2a0b405b2d04cd5a0dcf4019743204e5daef5965b63ddaac506ac615f67e`.
The actual bytes must also equal the parent Git blob; the frozen function must
contain its bounded `p.wait(timeout=timeout)` contract. The exact original saved
keys, strict zero exit, false interruption flags, null metadata error, launch
argv, identical receipt/launch child and distinct launch child/controller are
required. Original receipts are not rewritten to add fields. Other receipt
schemas retain the explicit actual-wait-true requirement.

Same-host generations are checked fresh using `/proc/PID/stat` start ticks.
For another host, root obtains the following metadata-only observation **on that
actual execution host after wait**, without launching science:

```json
{
  "schema":"FINAL_PUBLICATION_PROCESS_OBSERVATION_V1",
  "status":"CLOSED_ON_EXECUTION_HOST",
  "observer_host":"actual host",
  "execution_host":"same actual host",
  "observed_unix":0,
  "wait":"exact receipt descriptor",
  "launch":"exact launch descriptor",
  "argv":["exact existing command"],
  "processes":[{"pid":1,"start_ticks":1,"host":"actual host"}],
  "all_process_generations_gone":true,
  "observer_source":"exact metadata-only observer source descriptor"
}
```

The process list must equal the actual wait specification, and observation time
must follow the real receipt mtime. `OTHER_HOST`, a missing PID on a different
machine, launch receipt alone, or a completion marker without actual exit is
never treated as gone. Observation is additional evidence, not an exit rewrite.

Already-closed native controls on leo may reuse their frozen actual observation
without a duplicate publication observer. Set
`host_observation_schema:"FROZEN_NATIVE_CLOSURE_OBSERVATION_V1"`, `host_observation`
to the original `closure_observation.json`, and `observer_source` to the exact
`b2b3_analysis_v1/native.py` (SHA
`d02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889`). Its child/parent,
host, launch/wait descriptors, actual gone flags and post-wait time must all
match. Same-host checks still read `/proc` freshly. No new scientific work is
performed to obtain or validate these metadata observations.

At the checks→commit and commit→push boundaries, both the preceding commands'
children and their old controllers must be gone. A role does not require its
own currently running controller to disappear before writing its result.

## Positive scope and omissions

Only this round's N1024 experiment/result tree; the explicit B2B3 transport,
analysis, continuation, metadata, report and B3 metric source/result subtrees;
new manuscript v6; publication preparation; and four root docs are eligible.
Old R0/R1/LE/common500 baselines and original source/model packages are unchanged
dependencies, not a new scope. Precise retained old qualification/failure copies
belong beneath the current result preparation directories. If additional final
receipt copies are needed, use `results/arpc_n2048_adaptive_20261019_v1/b2b3_execution_receipts_v1`
or `results/manuscript_completion_20261019_v1` and bind their copy/source manifest.

Existing source/weights/data/environment/vendor exclusions and the 10 MB cap
remain. Arrays, databases, archives, fonts, big runtime logs, lock files and model
weights cannot be staged by this entry. The final manuscript PDF and native
vector figures are legitimate current products; external paper PDFs and fonts
are not bundled. `source_packages` is a provenance list for code, not model/data
source packaging. Heavy backup remains deferred.

The final controller intentionally does not recursively walk runtime trees.
Root supplies closed exact small receipts and output manifests. Original
scientific closure/audits retain responsibility for full arrays and budgets.

## Local sync after remote-SHA-confirmed normal push

`push.py` follows the existing local GCM memory-only client and the already
authorized reverse proxy `127.0.0.1:23098 -> local127.0.0.1:7897`. It never writes
or prints a credential or mutates persistent Git settings. Uncertain outcomes
retain the attempt and must be inspected before any retry.
Local GCM, SSH push transport and remote push drop inherited `GIT_TRACE*`,
`GIT_CURL_VERBOSE`, `GCM_TRACE*` and `GCM_DEBUG` only from their process environments,
so an external trace setting cannot persist the in-memory authorization header.

`sync.py` reads only the pushed commit's final exact `staged_files.json`, verifies
its SHA against `push_verified.json`, and requests each file with `git show
HEAD:path`. It does not tar the repository, read runtime arrays or build a backup.
It checks final HEAD before and after transfer. Each blob is rehashed remotely
and locally. Windows extended-length spelling is used, with path traversal,
symlink/junction/reparse point checks. Existing same-SHA files are reused.

`exact_local_before.json` maps only explicitly approved differing local paths
to `{bytes,sha256}`. Use `{}` when all local files should match or be absent.
Any other difference stops **before the first replacement**. An approved old
local source/doc is kept under the small local sync attempt's `replaced_local/`
before atomic replacement by its exact committed bytes. This is only an audit of
the explicitly replaced light file, not a scientific/data archive. A changing
local file is rechecked before replacement. The final `local_verification.json`
lists every actual file/hash and same-SHA reuse. Failure preserves verified
partial transfers and never silently retries or overwrites an unfamiliar edit.

No credential, array, scientific count or completed result is fabricated by
these helpers. All final facts, package completeness, actual visual reviews,
remaining author-identity gaps and precise failure/compute reporting belong in
the root-reviewed descriptor and completed manuscript.
# Exact historical fixed-SSH exception (one N1024 audit)

The original `N1024_audit` wait and launch contain no host field. Keep those
bytes. `EXACT_HISTORICAL_FIXED_SSH_V1` accepts only the original wait
`db9e4d805626f45804d8c62a15651c84ab95a72922adb0d4d6ffa47115c6b300`,
launch `3ff48698558cbdb7f57e96d017fea624d7a8bfaf03628b17b76ed3e712761752`,
child `3316552/208506583`, and controller `3316543/208506575`. This is an
explicitly accepted historical reconstruction, not an original inline host
record. All other hostless cases retain the reviewed host-aware anchor rule.

Root first copies the exact original small evidence bytes into
`results/n1024_consistency_20261019_v1/audit_host_provenance_v1/`:
`audit_host_provenance_reconstruction_v1.json`, `launch_v2.py`,
`launch_remote_audit_v2.py`, `audit_v2_launch_stdout.txt`,
`audit_v2_launch_stderr.txt`, `inner_launch.json`, `control_stdout.txt`, and
`control_stderr.txt`. The route hard-pins every size and SHA, the original
remote audit source, original runtime inner launch and logs, actual command,
and the two original generations. It runs only on the exact currently mapped
main host `aibox-rfbd2ce574fd-5df9766c45-vz6d9` and freshly checks both
generations gone. No audit, model, decoder, or test is rerun.

Root separately writes `root_acceptance.json` there with the following actual
fields. Do not replace placeholders without reading the evidence. All
descriptors use exact `{path,bytes,sha256}` triples:

```json
{
  "schema": "N1024_FIXED_SSH_HOST_MAPPING_ACCEPTANCE_V1",
  "status": "ROOT_ACCEPTED_HISTORICAL_RECONSTRUCTION",
  "execution_host": "aibox-rfbd2ce574fd-5df9766c45-vz6d9",
  "ssh_alias": "DevMachineModel-H100-4C-P1660-01",
  "reconstruction": "EXACT COPIED ORIGINAL RECONSTRUCTION DESCRIPTOR",
  "actual_wait": "EXACT ORIGINAL RUNTIME WAIT DESCRIPTOR",
  "launch": "EXACT ORIGINAL RUNTIME LAUNCH DESCRIPTOR",
  "process": {"pid":3316552,"start_ticks":208506583},
  "parent_process": {"pid":3316543,"start_ticks":208506575},
  "original_inline_host_present": false,
  "host_aware_same_generation_anchor_found": false,
  "current_alias_and_proc_are_historical_host_records": false,
  "basis": "Root's actual explanation of the accepted fixed-SSH chain and its limitations",
  "reviewer": "Actual root reviewer",
  "audit_source": "EXACT ORIGINAL REPOSITORY AUDIT SOURCE DESCRIPTOR",
  "copied_evidence": {
    "launch_v2.py": "DESCRIPTOR",
    "launch_remote_audit_v2.py": "DESCRIPTOR",
    "audit_v2_launch_stdout.txt": "DESCRIPTOR",
    "audit_v2_launch_stderr.txt": "DESCRIPTOR",
    "inner_launch.json": "DESCRIPTOR",
    "control_stdout.txt": "DESCRIPTOR",
    "control_stderr.txt": "DESCRIPTOR"
  }
}
```

The N1024 wait specification keeps the original wait/launch triples, argv and
process pointers. Set `execution_host` to that exact main host, include the
acceptance descriptor in `identity_records`, and set:

```json
{"host_provenance":{"schema":"EXACT_HISTORICAL_FIXED_SSH_V1","record":"EXACT ROOT ACCEPTANCE DESCRIPTOR"}}
```

The publisher does not infer an execution host from a caller string or from
an arbitrary alias. Its one-case pins, root acceptance, original saved SSH
output, and reconstruction remain visible in the published evidence.

