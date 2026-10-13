# Read-only R1 calibration200 delivery

This is a separately reviewed CPU delivery entry for the already registered R1.
Preparing, testing or deploying it does not establish scientific completion.
At this entry's release, no actual R1 delivery result has been produced by it;
synthetic CPU tests do not prove that the active R1 run has closed. The root
coordinator must verify actual closure before invoking it. It does not launch
or implement R2.

The fixed master SHA is
`70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47`.
The authoritative run is `L/r1_execution_v1`, where
`L=/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013`.
The repository is
`R=/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM`.

## Gate and audit

Before opening quality outputs or making output directories, standard-library
checks require the repository COMPLETE_R1_CHAIN, all five actual successful
child waits, the resource dispatcher's real owner wait, exact registration/argv
bindings, original PID/start identities gone (zombies are rejected), unheld owner
and dispatch locks, and no CPU failure/partial markers. The initial detached
launcher has only a launch receipt, not a child wait; this is explicitly reported.

After that gate, the entry installs the already reviewed checkpoint metadata
adapter, verifies all original master code bindings, and calls only the existing
CPU audits. These import NumPy to validate saved arrays, but never Torch, Sionna,
TensorFlow, LPIPS models, reconstruction, PHY or metric execution. Model imports,
subprocesses, network connections and file writes are denied while auditing.
The original received-state access audit, exact image/cache hashes, source pixels,
full 32,400 physical and 43,200 logical grids, 200 sources, three noises, packet and
model ledgers, metric identity and per-view records are checked.

The original selected source means and tables are checked against arithmetic
reductions of the completed rows. Saved candidate ranks, tie order, saved winners,
J values and the already stored gate must agree. The entry never calls
`select_from_complete_rows`, `score.select`, bootstrap or any model method. It
retains the existing three metrics and all negative/zero/channel-failure results.

## Outputs

`R/results/layered_mismatch_raw_20261013/r1_delivery_v1/` receives:

- unchanged selected CSV/JSON copies when below the original 10,000,000-byte limit;
- candidate and frozen-method receive-state tables, including zero-count states;
- the frozen method parameter table with actual information/transmitted bits,
  rates, modulation, header/body/idle symbols and the complete group metadata;
- `REPORT.md`, an English `manuscript_results.tex` fragment, closure audit, manifest
  and final completion receipt.

`L/r1_delivery_v1/` receives complete physical-event, logical-frame and
frozen-method frame CSVs, plus any original selected output too large for Git.
No truncation, splitting to bypass the filter, arrays, model weights, new metric
values or new confidence intervals are introduced. The publication manifest pins
every heavy artifact. Channel CRC rejection remains a valid measured outcome;
an unresolved CPU/model call prevents all delivery. Unattempted CRCs are blank,
not counted as rejection. Legal wrong IDs stay distinct from full/base success.

No output is created until all closure and derived-table checks pass. Then one
exclusive permanent `O_EXCL` publication claim prevents concurrent or automatic
replay. Files are built in new staging directories. Each final directory is
reserved by exclusive `mkdir`; every destination file is created with
`O_EXCL`/`O_NOFOLLOW`, copied, fsynced and checked by bytes and SHA. Existing files,
directories and dangling symbolic links are rejected. The heavy payload is
copied first, then the light files except `completion.json`. All final artifact
hashes and the manifest are verified again. Only then is the already completed
receipt exclusively copied to `completion.pending.json` inside the final light
directory, checked against its original bytes/SHA, and fsynced. That local
pending receipt is published as `completion.json` by one same-directory,
non-overwriting `os.link`, followed by directory fsync. Both completion metadata
names remain available; neither participates in a circular self-hash. There is
no fallback to overwriting rename or copy and no cross-area hard link.

Directories can become visible before delivery finishes. A directory is never
completion evidence: acceptance requires a valid final completion marker, its
manifest and all artifact hashes, an actual successful delivery-process exit,
and no publication failure receipt. In particular, a directory-fsync failure
after linking the marker still exits with an error and must be diagnosed.
Staging, the permanent claim, partial final directories and detailed failure
receipts (phase/type/errno) are retained. Successful staging is also retained;
the entry deletes no old or newly reserved evidence and never automatically
retries a partial publication.

This explicit PFS publication revision replaces the original unexecuted entry's
`renameat2(RENAME_NOREPLACE)` requirement. The original Linux synthetic test
encountered PFS `EINVAL` (errno 22); its failure receipts and original source
snapshot remain preserved separately. Root's zero-science PFS capability probes
verified same-area completion-file hard links, rejection of an existing link
with errno 17 without changing its bytes/inode, and directory fsync. A separate
actual tmp-to-code link probe returned `EXDEV` (errno 18), which is why the final
pending receipt is first exclusively copied to the code-area result directory.
This revision concerns
file publication and completion-marker semantics only; science and selection
bindings are unchanged. Windows development cannot exercise Linux directory
fsync; the original Linux CPU environment must also run these synthetic tests
before entry release. The dangling-link fixture simulates `lstat` metadata and
forbids following the destination; it creates no PFS symbolic link.

## Future invocation

Only after the actual owner and all children have exited, the root coordinator
may run the frozen deployed script using the original CPU environment:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_delivery_v1/deliver.py --expected-registration 70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47
```

Deployed synthetic checks, still without accessing the scientific run:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B -m unittest discover -s /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_delivery_v1 -p 'test_*.py' -v
```

Local development uses the same test discovery under
`.research/layered_mismatch_raw_20261013/r1_delivery_v1`.

## Limits and remaining risks

- This entry is source-bound and fixed to this one run; any future scientific
  recovery requires an explicitly reviewed new delivery binding.
- The full audit reads all saved frames, arrays and ledgers. It can be I/O-heavy,
  but does not spend scientific calls or repeat model execution.
- Final directory contents do not form one filesystem transaction. The
  completion marker is atomically linked only after both sets of files pass
  verification; actual process exit and failure evidence must also be checked.
- New derived output is a dependent increment, not a self-contained recovery
  package. Original R0/R1 inputs, models, environments and run records remain needed.
- These are calibration200 results without confidence intervals or ConvNeXt.
  A stored GO permits consideration of the already authorized R2, but this script
  does not register, execute or claim that R2 is complete. No GO does not justify
  changing p, SNR, samples, configurations or models.
- No plotting is included in this bounded delivery entry.
