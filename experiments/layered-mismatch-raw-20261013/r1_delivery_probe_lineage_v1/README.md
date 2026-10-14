# Explicit CPU preparation-probe lineage revision

This revision fixes one diagnosed CPU metadata failure: the frozen recovered
delivery guard classified three preserved preparation-probe failure receipts as
unreviewed scientific failures. The published original compatibility report,
its frozen review bundle and root review already bind all six zero-model probes.
Only the exact original v1, v2 and v5 failure paths are additionally passed to the
original guard, after SHA/bytes, ancestry and strict zero-model checks. No probe,
model, channel, metric, selector or bootstrap is run by this adapter.

Original delivery, dispatcher, figure, final-backup and fetch files remain
unchanged and are loaded only after matching their frozen SHA. No startup hook,
environment import injection, executable replacement or global scientific
module patch is used. The wrapper changes only explicitly loaded CPU metadata
module functions. The original source pins retain their actual meaning;
an additional `metadata_revision` records this revision and its root descriptor.

## Fixed routing and truth of receipts

The `runner.py` roles are `dispatcher`, `delivery`, `figures`, `backup`, and local
Windows-only `fetch`. All remote roles use original CPY, `-B`, hidden CUDA and
the fixed master SHA. Dispatcher launch/prepared/wait records contain the actual
wrapper child argv, including role, runner/adapter SHAs and root authorization
triple. Figure guards expect that exact argv and exact metadata revision. Final
backup expects the actual figure wrapper argv and the same revision in both
root execution descriptor and actual plot launch. No record claims that the
original unwrapped program was the OS command.

The explicit route is:

```
CUDA_VISIBLE_DEVICES='' CPY -B E/r1_delivery_probe_lineage_v1/runner.py ROLE \
  --expected-adapter ADAPTER_SHA --expected-runner RUNNER_SHA \
  --expected-registration ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c \
  --revision-authorization O/r1_delivery_probe_lineage_preparation_v1/authorization.json \
  --revision-authorization-sha256 AUTH_SHA --revision-authorization-bytes AUTH_BYTES
```

`dispatcher` is the only delivery launch, and launches `delivery` itself once.
Do not run both manually. All original no-attempt, once-only claim, finite actual
wait, PID/generation, closure and publication guards remain. Figures wait for
actual CPU completion. Backup adds its original `--authorization-path`,
`--authorization-sha256`, `--authorization-bytes` flags, after actual final plot
review and verified normal publication. Local `fetch` uses this same local
runner with expected adapter/runner/master and `--receipt-sha256`, optionally
`--sync-light`; it does not need remote revision-authorization flags. It verifies
the frozen local source bytes and exact historical metadata rows in the archive.

## Root metadata authorization (not an additional science authorization)

After independent review and original-environment bounded synthetic tests, root
creates the exact O path above, schema `R1_CPU_PROBE_LINEAGE_AUTHORIZATION_V1`,
status `AUTHORIZED_METADATA_ONLY`, strict integer `scientific_calls: 0`, and
`registration_sha256` equal to the fixed master. `files` contains exact path/SHA/
integer bytes for these five revision files (`adapter.py`, `runner.py`,
`runtime_inventory.json`, `README.md`, `test_adapter.py`), the five
original final-backup files (`build.py`, `fetch.py`, `README.md`, `test_backup.py`,
`test_cli_review.py`), and actual root/reviewer/qualification/failure-observation
evidence copied to O/r1_delivery_probe_lineage_preparation_v1. No self pin is
included inside that descriptor. The descriptor's SHA/bytes are passed by the
actual command, then bound in all four remote roles' closure receipts.

## Exact historical backup scope

`runtime_inventory.json` is the reviewed runtime projection of the independent read-only inventory of
16 exact historical preparation files and six exact empty cache directories,
with published report/bundle/root pins. It discloses which four resource-guard
receipts are newly byte-pinned instead of falsely calling them historical
SHA-bound. There are no standalone per-probe actual waits; none is fabricated.
The original local inventory remains untouched (SHA
`8d3850bfa6e56ada3f9efd93f95b43bec09c196d4fea6548305660f6b12efdfd`).
The projection renames only its two local D-drive evidence pin fields `path` to
`local_path` and adds the parent SHA and explicit projection explanation. Those
local dependency receipts are informational, not server file inputs. All remote
paths, historical values and byte pins remain unchanged. The projection SHA is
`fd7e626188e95e18b361fe4ff788f607a13b4a8e471065253705ba2a33fd102f`.
The final archive preserves all these exact files and empty directories, rejects
added files under the nine exact historical roots, and retains the original
failed-science evidence. The server and Windows fetch share the exact inventory.
Fetch also requires every historical row's source path, byte count and SHA to
match the frozen inventory. There are no general preparation-prefix exemptions.

The final backup includes the root authorization, all its exact published code
and evidence files, and all seven original published preparation receipts.
Original source, model, R0 and environment archives remain dependencies.

## Narrow change proof and qualification

Delivery replaces only `no_failure_files` (three authenticated historical paths
on the exact old-run base) and `preflight` (adds explicit provenance). Original
guard runs unchanged on every call, including other roots. Dispatcher `run`
changes only child argv and its metadata_revision field. Figure guard changes
only expected argv and required revision descriptors. Backup changes only
metadata loader/figure argv/provenance/payload-name/payload-evidence routing;
archive encoding, physical/scientific checks, statistics, metric computation,
selection, original CSV functions and render functions are untouched.

Each source-derived metadata function uses exact substitutions and checks
inverse AST equality before compile. Tests verify other functions retain the
identical function code object. `test_adapter.py` is bounded synthetic metadata
only. For original Linux qualification provide the four fixed original sources
as `delivery.py`, `dispatcher.py`, `figures.py`, `backup.py` in an isolated source
fixture directory via `R1_PROBE_LINEAGE_SOURCE_FIXTURES`; fixture source SHAs are
checked. No production preflight, deep audit, model, runtime array, metric or
original completed test suite is invoked by these tests. The root qualification
wrapper must restrict opens to its fixture, immutable source snapshot and
interpreter/dependency imports; no real scientific R/L data is needed.
