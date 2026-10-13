# R1 recovered calibration: CPU delivery entry

This entry is for delivery **after** the registered recovery has actually finished.
Synthetic CPU tests are not evidence that R1 has finished. This entry neither
launches an experiment nor resumes a failed scientific process.

## Exact scope and lineage

- Original master: `70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47`.
- Recovery master: `ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c`.
- Recovery visual child: `c1eeef69de031fd9e370690268c14d06c63810a94ff323b950db0f0ad14f6255`.
- Reused CPU delivery source: `94b6986e1db8a6082f0a02a6a91855fd79b926a390bb1551a1bc785d06bfd1b3`.
- Preserved failure audit: `daad4d721815607b9b9186ed33ded6b778c9a91b91d169afbfb5000f15add7d4`.

The original PHY and RX projection retain their original registration, files and
successful actual waits. The original visual process and owner exited with code
1. Their one failed model-load reservation, zero completed model loads, zero
VAR/prior/decoder forwards, empty partial frame list and empty image directory
must still agree with the reviewed original failure evidence. Every file in that
evidence inventory is SHA checked; unknown original failure artifacts are rejected.
No old failure is deleted, relabelled as success, or treated as a completed frame.
The original audit did not inventory `visual/resource_history_at_failure.json`.
This closed historical auxiliary attachment is separately pinned at 2,466 bytes,
SHA `7094c96a7bf5e1e71d4e12cb9601b5075880d106c583c4401c08dacf866a13c0`;
it is admitted only with that exact path, byte count and SHA. The original audit
is unchanged.

The new owner and its visual, score and selection children must each have an
authenticated actual exit of zero. Their exact registered command lines, child
registrations, CPU affinity, deadline, duplicate waits, completion pins and
mixed-stage receipt are checked. All original and new process generations must
be gone, including zombies; four control locks must be unheld. Live identities,
missing receipts, interruptions, any new failure, or altered original evidence
stop the command before parsing quality data or making delivery output.

The nine frozen recovery files are loaded only after their registered hashes are
verified. The reviewed `adapter.install(registration, 'selection')` provides the
original scorer and receiver metadata adapters. Three exact substitutions in
the original CPU delivery audit change only its module loader, new stage root
and inherited projection path. An inverse AST check restores the original audit.
All metric arithmetic, source/noise aggregation, cache identity checks, fixed
selection verification and CSV functions remain the original implementations.

The complete fixed grid remains 32,400 physical events and 43,200 logical views,
with 200 calibration sources and three predetermined noises. The entry reuses
the stored winner/ranking/GO results and checks their arithmetic; it does not
select a new policy, generate a confidence interval, bootstrap, reconstruct,
score a model, call a channel, train, or start R2. PSNR, LPIPS-Alex and DINOv2-L
remain the three R1 metrics. Negative values, zero-count receive states, gray
failures, legal wrong IDs and CRC failures retain their original denominators.
This is calibration200 reporting, not blind confirmation.

## Future invocation

Only after the full recovery and its outer dispatcher have closed, use the
original CPU environment with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/deliver.py --expected-registration ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c
```

This command requires a persistent external dispatch with its own exact source
binding, command, PID/start, stdout/stderr and real `Popen.wait` receipt. No
dispatcher is created here. A marker or command-start receipt alone cannot
establish successful CPU delivery. A failed attempt is never automatically
retried, even when the scientific inputs remain unchanged.

## Outputs and publication semantics

Lightweight output is a fresh
`results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1` directory.
Heavy complete CSV/JSON goes to the personal temporary root's fresh
`r1_cache_recovery_delivery_v1` directory. The permanent claim, staging directory
and any publication failure receipt use the same new name. Original
`r1_delivery_v1` paths are not reused. The original 10,000,000-byte publication
limit is unchanged; large tables are neither truncated nor split to bypass it.

Outputs include complete physical event and logical frame tables, frozen-method
frames and parameters, candidate and frozen-method receive-state summaries,
unaltered saved source means/quality summaries/selection, a closure audit, an
English report and a LaTeX results fragment. The report explicitly discloses the
old failed initialization and the inherited versus new successful segments.

The original reviewed PFS publication protocol is reused: permanent exclusive
claim; exclusive final directories; `O_EXCL`/`O_NOFOLLOW` file copies with SHA
verification; heavy files before light files; complete manifest verification;
and a fully written, fsynced same-directory `completion.pending.json` hardlinked
without overwrite to the final `completion.json`. There is no cross-area link,
directory-rename fallback, overwrite or cleanup of another attempt. Directories
can be visible before completion. Claim and staging evidence stay in place.

Successful delivery requires **all** of: actual CPU delivery exit zero, no
publication failure receipt (including in the preserved staging directory), a
valid completion marker, its exact manifest SHA, and every manifest file's full
SHA/size. A link followed by an fsync failure still produces a failed CPU exit
and is not a completed delivery.

These are incremental results. Original model/environment/source restoration
packages, the complete inherited PHY/projection and failure evidence, and the
full new recovery run remain required. Arrays/checkpoints are not duplicated.

## Synthetic CPU tests and limitations

After deployment the isolated command is:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B -m unittest discover -s /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1 -p test_deliver.py -v
```

Local development uses the same discovery command with the new directory under
`.research/layered_mismatch_raw_20261013`. Tests construct temporary synthetic
metadata and a one-row failed-call ledger, never real scientific inputs. They
check mixed actual exits, blocked active/failed paths, PID reuse, old failure
immutability, command/deadline/child binding, code identity, and inverse AST
preservation. Linux advisory locks are mocked only in synthetic fixtures;
production uses the original lock checks. The original published CPU delivery
tests remain the evidence for its unchanged arithmetic and PFS publication core.

This adapter has not yet produced real R1 delivery results. Future genuine
closure can still expose a missing or inconsistent receipt; fail-closed behavior
must be diagnosed from preserved evidence, not bypassed. Python read-only audit
hooks prevent model imports and writes during auditing but are not an OS sandbox.
