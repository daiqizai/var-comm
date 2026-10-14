# Final recovered R1 science increment — prepared entry

This entry packages an **actually closed calibration200 experiment**. Its local
synthetic tests do not establish scientific completion. At preparation time the
recovered scientific chain is active; no real final backup or final figure
review has been produced by this entry. Do not run it from the older
`r1_completed_backup_v1` draft: that draft assumes an all-success original chain
and is incompatible with the preserved original initialization failure.

There are no model, channel, metric, selection, bootstrap or R2 calls here.
Imports are inert. Build is for the original Linux server; fetch is for the
authorized Windows machine. Both need future explicit actual-result hashes.

## Required closure, in order

1. Authenticate the fixed recovery master `ad068190...`, its reviewed nine
   scientific source bindings and the `f37df9e9...` CPU delivery entry. Reuse its
   read-only `preflight`, not `deep_audit` or any scientific computation. This
   verifies original PHY/projection exit zero, original visual/owner exit one,
   precisely one failed model-load reservation and zero completed loads,
   forwards or images, the original failure audit and its separately pinned
   `resource_history_at_failure.json` (SHA
   `7094c96a7bf5e1e71d4e12cb9601b5075880d106c583c4401c08dacf866a13c0`,
   2,466 bytes), and all three new stages/new owner exit zero. Every recorded
   PID/start generation must be gone, including zombies; locks must be free.
2. Use the fixed `b5e5d8f4...` figure entry's metadata-only `guard_delivery` to
   authenticate the real `38a7a17d...` CPU dispatcher wait zero, no failure,
   exact command/master, completion same-file marker, full delivery manifest
   and every light/heavy artifact SHA. Compare its saved scientific audit and
   current process closure. No quality CSV is parsed or policy reselected.
3. Require a future root-pinned descriptor for the actual final figure launch,
   wait zero, process generations, stdout/stderr, final manifest/completion,
   and the two-PNG human visual review. Verify all eleven final artifacts,
   vector PDF/editable SVG/600-dpi PNG scope, same-file completion marker,
   correct CPU delivery input manifest and no plot failure. Both reviewed PNG
   pins must exactly match the final manifest. A synthetic preview is invalid.
4. Verify normal publication on `main`, original required repository checks,
   current clean tracked tree, and Git byte identity of all final light results,
   figures, entries, master and visual review. Hold the five original/recovery/
   CPU dispatch locks while inventorying and packing. Require closed model and
   metric ledger summaries and no nonempty SQLite sidecars.

Any unmet prerequisite stops before creating the backup directory. An existing
backup directory is never overwritten or automatically rebuilt. Failures after
creation preserve the original attempt and `build_failure.json` for diagnosis.

## Included scope and size accounting

The package is
`layered_raw_R1_final_science_20261014_v1`, under the personal
`baseline_strength_20261011` backup area. The exact allowlisted roots are:

- Original `r1_execution_v1`: complete 32,400 PHY physical frames and packet
  journal, 43,200 logical RX projection, the failed visual initialization,
  its empty image directory, ledger, resource-history attachment and all actual
  launch/wait/log records. Successful channel failures/CRC rejections remain.
- Original dispatch and initial detached-launch metadata.
- Recovery `r1_cache_recovery_v1`: complete visual pixels and logical mappings,
  metric rows and ledgers, source preparation metadata, frozen selection and
  source means; new owner/dispatch/initial-launch receipts and logs.
- Completed CPU delivery: all heavy per-frame tables and full large source
  means, light report/CSV/TeX/audit/manifest, dispatcher receipts and logs.
- The two final figures, exact plot-data copies and captions, input audit,
  final manifest/completion, real plot receipts and the human visual review.
- Exact bound R1 code, lightweight final-chain/failure metadata, authorization,
  four publication receipts and restore dependencies. Registered master/model
  bundle inputs already covered by previous verified packages remain dependencies.

Only these three exact reproducible cache roots are excluded:

```
L/r1_execution_v1/cache
L/r1_cache_recovery_v1/cache
L/r1_cache_recovery_v1/visual/runtime_cache
```

The last contains the private `tmp`, `inductor` and `triton` subdirectories.
No generic “cache” name filter is used: scientific reuse arrays and metric
identity/cache records are retained. Omitted cache roots and any explicitly
referenced cache pins are listed in the manifest. A missing scientific reference
is fatal. Duplicate publication staging and external Matplotlib caches are not
included. No model weights, original source images, older R0 arrays or complete
runtime environments are copied.

No numerical byte estimate is claimed before actual closure. After closure,
the manifest records canonical file count/bytes, same-inode alias count/bytes,
and logical restored size **before compression**. No data is truncated to meet
a target size. There can be at most 43,200 unique float32 `3×256×256` image
arrays (32,400 MiB of raw pixel values, plus NPZ headers); the exact recorded
unique-image count is typically smaller because strict actual-input reuse is
preserved. PHY packet/metadata tables add their actual inventory size. Tar uses
gzip level 1, with no assumed compression ratio. A finite scope does not promise
a small archive or a fixed compression runtime.

Every payload's path, size, SHA, device/inode and modification/change timestamp
are recorded. Exact same-inode aliases are stored once; equal bytes in different
inodes are not silently deduplicated. After tar creation, every member and every
original source/stat is reverified. Aliases restore as ordinary verified files,
not links; empty scientific evidence directories are explicitly restored. All
symlink/junction targets and traversal/case-collision names are refused.

## Future root descriptor (template only)

The template below intentionally contains **no real completion values**. It is
not an authorization file and cannot pass validation. The root must populate it
only after the actual figure process exits zero and both final PNGs have been
opened and reviewed. Every `PIN` means exactly
`{"path":"<absolute actual path>","sha256":"<actual 64-hex SHA>","bytes":<actual integer bytes>}`.
The completed descriptor itself is passed by path/SHA/bytes on the build CLI.

```json
{
  "schema": "R1_FINAL_SCIENCE_BACKUP_AUTHORIZATION_V1",
  "status": "AUTHORIZED_AFTER_ACTUAL_FINAL_CLOSURE",
  "scientific_calls": 0,
  "registration": "PIN for O/r1_cache_recovery_v1/registered/registration.json",
  "expected_commit": "<actual published main SHA>",
  "publication_directory": "<actual PUB directory under personal BASE>",
  "push_verification_sha256": "<actual SHA of PUB/push_verified.json>",
  "delivery_actual_wait": "PIN for L/r1_cache_recovery_delivery_dispatch_v1/actual_wait.json",
  "delivery_completion": "PIN for O/r1_cache_recovery_delivery_v1/completion.json",
  "delivery_audit": "PIN for O/r1_cache_recovery_delivery_v1/closure_audit.json",
  "figure_entry": "PIN for E/r1_cache_recovery_figures_v1/figures.py",
  "figure_manifest": "PIN for O/r1_cache_recovery_figures_v1/figure_manifest.json",
  "figure_completion": "PIN for O/r1_cache_recovery_figures_v1/completion.json",
  "visual_review": "PIN for the actual published review JSON under O",
  "figure_execution": {
    "actual_wait": "PIN for a real completed plot parent wait under L or BASE",
    "launch": "PIN for the corresponding actual launch under L or BASE",
    "stdout": "PIN for the JSON successful plot stdout under L or BASE",
    "stderr": "PIN for the actual plot stderr under L or BASE",
    "process": {"pid": "<actual positive integer>", "start_ticks": "<actual positive integer>"},
    "parent_process": {"pid": "<actual positive integer>", "start_ticks": "<actual positive integer>"},
    "argv": ["<registered runtime.python_cpu>", "-B", "<E>/r1_cache_recovery_figures_v1/figures.py", "--expected-registration", "ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c"]
  }
}
```

The actual plot `launch` must record `process`, `parent_process` and exact
`argv`. Its `actual_wait` must have `actual_wait: true`, strict integer
`returncode: 0`, matching `child` PID/start (or strict matching `pid` and
optional `start_ticks`), `terminated: false`, `killed: false`,
`timed_out: false`, and `metadata_error: null`. Existing optional timeout/stop
flags must be false. The descriptor pins the real schema without fabricating
an exit based on a completion marker. Both process generations must be gone.

The separate actual visual-review JSON must contain `status: "PASS"`,
`actual_opened_both_pngs: true`, exact `registration` and `figure_manifest`
pins, a nonempty `reviewer` description, and `figures`, a list of exactly two
full path/SHA/bytes PNG pins matching the final figure manifest. Keep this file
under O and publish it normally before packing. The figure program's original
`VISUAL_REVIEW_PENDING` receipts remain unchanged; the separate review resolves
that requirement without rewriting provenance. Never use a synthetic review.

## Future invocation and restore

Only after original-environment synthetic qualification and independent review,
the root may deploy these entry files, then use the original CPU environment:

```text
python -B build.py --authorization-path <ACTUAL_DESCRIPTOR> --authorization-sha256 <ACTUAL_SHA> --authorization-bytes <ACTUAL_BYTES>
python -B fetch.py --receipt-sha256 <ACTUAL_SERVER_RECEIPT_SHA> --sync-light
```

The first command produces an exclusive personal backup directory, archive,
manifest and receipt. The second retrieves only that exact fixed archive to
`D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/layered_raw_R1_final_science_20261014_v1`,
verifies archive/manifest/every extracted file and alias, and optionally syncs
light published files into W without changing the existing 10,000,000-byte
publication filter. Windows filesystem access uses extended-length paths.
Existing matching downloads are verified and reused, not downloaded again;
conflicting/partial artifacts require inspection and are never silently removed.

Retain the five explicitly hash-pinned R0 BLER, R0 clean, R1 registered,
cache-recovery-entry and cache-delivery-entry packages, plus their transitive
`required_static/source_restore`, original models/data/environments, baseline
training/formal/diagnostic/input, EMAIN and timing recovery dependencies.
This is **not an independent full restore package**. Previous dependency
receipts and archive presence are checked; large prior archives are not rehashed.

## Local qualification and remaining boundary

```text
python -B -m unittest discover -s .research/layered_mismatch_raw_20261013/r1_final_science_backup_v1 -p "test_*.py" -v
```

Tests create only small synthetic temporary files. The CLI fixture mocks the
production closure/payload list and exercises actual packaging helpers on a
single non-scientific text; the independent regression restores the old early
main ordering and confirms it fails. No model, live quality, network, remote
archive or real figure is used. Local PASS is not Linux/PFS qualification or a
real scientific completion claim. The future genuine descriptor, Linux check,
normal publication, archive build and Windows verification remain root actions.
