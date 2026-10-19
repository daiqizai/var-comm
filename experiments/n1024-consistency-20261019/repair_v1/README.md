# N1024 minimal source-version repair v1

Scope: common500 only, WHOLE at the original 4 and 10 dB operating points. Exactly 2,421 logical / physical events are changed (1,491 at 4 dB and 930 at 10 dB); maximum 4,842 header/body decoder calls. No new source-probability pass or tokenizer call. The other 6,579 logical events, including 579 exact cross-family events and every T1-C 19 dB event, are excluded from new PHY execution. The original 9,000-frame result remains immutable.

This is an independent entry. It does not modify the original T1-C or legacy experiment code. `source_transformations.json` records the exact original hashes and the limited text transformations for PHY, RX, metrics and control. Independent review must inspect these changes, not infer qualification from the old tests alone.

## CPU preparation and qualification

`prepare.py` is metadata-only. It pins the existing 500 source/token archives, current terminated streams, old PHY receipts, fixed images, model/author assets and original 10,000-by-500 bootstrap index bank. It verifies 579 cross-family noise identities. It builds a closed RX-to-pixel cache from current source-version outputs; a conflicting pixel for one actual RX key blocks preparation. No stream transmission, decoding, model loading, scoring, or bootstrap draw occurs.

`test_repair.py <exclusive-output-directory>` tests only synthetic scope, duplicate/counter/budget rejection, RX cache isolation, real-wait requirements, full-denominator source means and direct paired intervals. It must run once with CPY and CUDA hidden, with persistent logs and actual process wait metadata. No production stage is invoked by the tests.

## Root registration

Only after the CPU preparation and qualification have closed, root reviews the exact deployed source set and creates a review JSON with `status: PASS`, `scope: N1024_COMMON500_WHOLE_4_10_SOURCE_VERSION_REPAIR_2421_ONLY`, `scientific_launch_authorized: true`, and `code` equal to the sorted descriptors of every file in this directory. The GPU is chosen from the actual current inventory. Registration requires at least 15 GiB free and utilization at most 25%; the original 12 GiB allocator and runtime free floor 2 GiB remain. A UUID is used for both CUDA visibility and NVML checks, avoiding ordinal remapping.

```
CPY -B register.py --gpu-uuid <GPU-uuid> --expected-head <current-main-SHA> --review-path <root-review.json> --review-sha256 <SHA> --review-bytes <bytes>
```

This writes `results/n1024_consistency_20261019_v1/repair_v1/registration.json` with schema `N1024_SOURCE_VERSION_REPAIR_REGISTRATION_V1`, and a limited `receiver_registration.json`. No role runs at registration. The RX context contains only shared model/static assets, GPU/runtime limits and scope; it excludes source archives, TX streams, old images and truth. The root master binds its hash.

## Actual stages, once each

Use CPY to run the control entry (it chooses PY only for receiver/metrics). Root must wait for each control and child to exit before the next role. Every role has a permanent exclusive output directory; failures must be preserved and diagnosed, never blindly retried.

```
CPY -B nc_control.py phy <registration-SHA>
CPY -B nc_control.py receiver <registration-SHA>
CPY -B nc_control.py metrics <registration-SHA>
CPY -B nc_control.py delivery <registration-SHA>
```

The runtime root is `liberai-tmp/liulu/liulu-var-comm/n1024_consistency_20261019_v1/repair_v1`. Scientific controls use the existing shared T1-B lock. All actual argv, parent/child PID and start ticks, stdout/stderr, timeout decisions and `child.wait` results are persisted. Each stage is limited to 14,400 seconds plus 60 seconds for control shutdown; this is a finite repair scope, not a new search.

PHY uses the unchanged paid 68-symbol header and 956-symbol body, Es=2, original session, counter and noise namespace, 13-bit length and 16-bit CRC. Frozen policies are m7 whole at 4 dB and m9 whole at 10 dB, with their original complete-scale fallback. Only changed actual events enter `rt.receive`; public profiles and real observations enter RX.

The new actual RX key is checked against the closed current-version cache. An exact hit reuses its immutable saved output; otherwise independent arithmetic decoding and the original frozen greedy VAR/Dc mathematics run once per unique actual RX input. Rejected headers, CRC or source parsing retain fixed gray outputs and all denominators. No TX CDF, original image, untransmitted tokens or correction oracle enters RX. Source truth is consulted only after the output has been fixed, for an explicit audit; any exact-stream/CDF mismatch blocks the run.

Metrics reuse source-plus-pixel pairs only with identical saved seven-metric values; ambiguous legacy values are excluded from reuse. Missing fixed-output pairs use the original four-metric implementation plus the frozen DISTS/DreamSim/MS-SSIM implementations and preprocessing. No model or checkpoint is selected. Registration pins model, loader, overlay and source-code identities.

Delivery joins 2,421 repaired logical rows with 6,579 saved rows, validates the complete common500 × two families × three SNRs × three noises grid, averages the three noises per source, and uses the **existing** common500 bootstrap index array SHA `5ca12fe75af6a09e1d3c18e205d7394bf4cdca53c25f7ca21005e5de30117634`. No random draws are made. Intervals use direct paired per-source differences. New100, other methods, timing, training, model/strategy selection and any new operating point remain out of scope.

After actual success, publish the code, registration, original source-transform proof, closed controls and complete light delivery. Publication is root-owned. This bundle does not commit, push, schedule successors or construct backups.
