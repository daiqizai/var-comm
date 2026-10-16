# Actual closure: T1-B completed, stop before T1-C

The successful chain is the original PHY, receiver_recovery_v1,
metrics_recovery_v2 and selection. All four actual stage waits are zero;
the final owner and CPU delivery audit exited zero and all processes are gone.
The two earlier owners exited one, and their evidence is preserved: the first
RX initialization was blocked before Torch import; the first metric
initialization rejected CUDA UUID spelling before any metric model construction
or score. Only exact read-only runtime library roots and optional GPU- UUID
prefix normalization were corrected. No completed PHY/RX/model score was rerun.
Original registered source bytes remain immutable.

The final CPU delivery entry was deliver_recovery_v2.py. Its closed output is
results/t1b_m10_calibration_20261016_v1/delivery, including all four metrics,
all 36000 logical scores, source means and failure denominators. 4/10 dB
winners are unchanged. At19 dB, whole and partial winners are the new m10
fallback strategies, with calibration DINO-L deltas +0.0003139766454696158
and +0.0001045141816139461. These are calibration differences, not test gains.
There is no pending scientific launch. Do not rerun any registration, owner,
receiver, scorer, selector or delivery. T1-C/LE3/training remain NOT_RUN;
backups are deferred. The following is the original pre-execution description,
retained for provenance; its launch instructions have already been consumed.

# T1-B: symmetric m10 calibration, then stop

User authorized T1-B only. Basis main is 3818fe6b05121347dbfa96f5d708818d8c0d21c8. The already delivered first batch, raw R1 NO_GO, LE0/LE1/LE2, fixed16 results and six-method timing are immutable.

Both whole and partial strategy families gain maximum target (10,0), using the original 12 MCS. Whole fallback is m10 then m9..m4. Partial fallback is m10 then all 24 original complete/quarter endpoints in descending token count. RX receives actual paid profile and actual bits, never the intended TX target or strategy family. The original 360 paid profiles remain byte exact; append 12 whole m10 profiles shared by the two families. Header 68 symbols + single-codeword body 956 symbols = N1024; the body pays 13 length + 16 CRC bits; Es=2. No grouped LDPC, power change, KEEP or feedback.

The published m10 source streams are reused; zero new source TX probability passes or tokenizer calls. Every source's stream identity/length is checked against the published cal1000 length table. All old CRC-passing headers selected retained legal IDs, so expanding the public catalogue does not change their decisions. Reuse of a physical event additionally requires identical source/noise/counter/profile/payload, recomputed wave/noise/observations, and unchanged original decoding mathematics. Image reuse requires identical actual RX input, model/source mathematics and numerical/device identity; metric reuse requires identical source and reconstruction pixels and metric identity. New source decoding cannot read TX arrays/CDFs or source truth; a separate audit reads truth only after images are fixed. An unrecognized model/CDF failure stops; it is not converted into a channel gray frame or automatically retried.

Before any new quality scoring, 24 symmetric strategies are reduced by source lengths: 20 have exactly the same calibration prefixes as retained legacy maximum-target strategies. Four differ: whole/partial at q6,r3/4 and q6,r5/6. They are all evaluated on the complete original cal1000 at 4/10/19 dB and seeds4101/4102/4103, independently of pilot rank. This finite refinement avoids discarding rare m10-fit sources through a small pilot. It retains all 144 old pilot candidates and all 12 old full finalists; zero-change aliases are documented, not assigned invented full-grid scores.

The new grid has 36,000 logical frames: 6,884 existing exact logical events, 6,977 same-batch duplicates, and 22,139 new physical events. The new packet ceiling is 44,278 (one header and at most one body per new physical event), below the previously reported 72,000 conservative ceiling. Independent source RX/VAR renders each cap at22,139; metric pairs cap at36,000. Failures and logical denominators remain. No standalone scientific qualification images are added.

Selection uses the original four metrics and DINO-L as the sole objective: average three noises within each source, then all1000 sources. Whole candidates include the old whole winner and the two new whole strategies. Partial candidates include all retained full finalists and all four new strategies, because its admissible endpoint set contains whole endpoints. Exact ties prefer legacy candidates, then lexical IDs. No test-set pixels, scores or m10 source boundary is read; no bootstrap or training. This is a calibration correction, not evidence of novelty or test performance.

Only `tb_control.py REGISTRATION_SHA` runs the four sequential finite stages: PHY, receiver, metrics, selection. `register.py` is metadata-only. Each stage has durable PID/start identity, stdout/stderr and actual parent wait. Existing output directories block a second launch. The controller does not launch T1-C, LE3, tests, backups or training. Unknown failure preserves the ledger and stops.

GPU2 is the same physical H800 used by the old scored caches. NVIDIA NVML and CUDA report different UUID namespaces; both are separately bound, with physical PCI checked. Start free15GiB/util<=25, private allocator12GiB, runtime free2GiB; FP32, TF32off, math6/interop2, CPU[0,1]. These resource limits are disclosed separately from original runs. CPU PHY uses its original2/1 thread settings. Per-stage finite safety limits are 21600/21600/14400/1800 seconds plus120 seconds parent cleanup; they do not increase call caps.

After all actual exits are zero and all stage processes are gone, deliver the closure/call audit, complete calibration table, source means, frozen policy comparison and limits of inference. Publish normal reviewed source/light results to main with original repository checks. Defer backup at the user's request. Stop after T1-B delivery; T1-C and any further experiment require a new user request.
