# Supplemental v6 startup resource revision

The user explicitly said “计时放开一下门槛”. This revision lowers only the
startup free-memory admission from 18432 MiB to 8192 MiB on the same H800 GPU0.
Startup utilization remains at most 25%; the original private PyTorch allocator
cap stays 16 GiB and the running free-memory floor stays 2048 MiB. No GPU global
settings or other jobs are changed. The GPU remains shared.

The existing v6 registration is immutable historical metadata. The supplemental
registration truthfully supersedes only its startup admission. It does not claim
that all resource conditions were identical across the two execution segments.
The original 576 complete calls and their full journal remain unchanged; only the
original 1152 missing complete calls run. The cumulative remaining time is 13069
seconds. GPU identity, 6/2 threads, six CPU slots, FP32/TF32-off, models, TX/RX,
source/noise identities, numerical-runtime equality and packet budgets remain.

The v5 resource history has 578 boundary samples, 577 containing its actual worker
PID 1720426: observed own occupancy was 2644–3128 MiB. Its last failure was the
documented `IndependentProvider` KeyError, not an OOM. These are boundary samples,
not continuous GPU peaks or allocator high-water marks. The remaining four
methods have no complete peak measurement in that record. The 8 GiB admission
therefore provides headroom over observed occupancy but does not establish a
guaranteed peak bound. No new model probe is added. Unknown failure stops once.

## Exact scope

`resource_adapter.py` extracts the original resource function from the SHA-pinned
v2 source. It substitutes only `free >= 18432` with `free >= 8192` and verifies an
exact inverse text and AST proof. Both original snapshot references receive that
same function. The run-time 2048 floor, utilization, coordination and conflicting
GPU process guards are preserved. The original six predecessor locks remain.

The wrapper sets only `old.__file__` to its own entry so the original owner's
actual `_worker` command enters the same explicitly bound resource wrapper.
Original source files are never rewritten. Original v6 verification, source SHA
checks, worker scientific functions, numerical-runtime checks and journal resume
execute unmodified. The change to the compiled traceback filename is execution
provenance, not a numerical change. Both actual process argv and supplemental
source/registration hashes are separately recorded under personal temporary
storage, including original registration and adapter proof.

The supplemental registration lives at
`results/baseline_strength_20261011/timing/six_method_v6_resource_v1/registration.json`.
Actual execution receipts live at the personal temporary
`baseline_strength_20261011/six_method_timing_v6_resource_v1` directory. Original
v6 science output and runtime paths remain unchanged.

## Root sequence; no agent may launch science

1. Review these four files and run the new inert CPU tests once in the original
   environment, with CUDA hidden and no model/Torch import. `python -B
   test_resource_adapter.py` uses embedded copies of the original two functions,
   fake device responses and synthetic receipts. It never reads a real ledger,
   calls an original readiness entry, registers a run, loads models or opens CUDA.
2. Deploy the four exact reviewed source files to a new independent experiment
   directory. Root creates an authorization JSON with schema
   `TIMING_V6_STARTUP_RESOURCE_AUTHORIZATION_V1`, status `AUTHORIZED_18_TO_8_GIB`,
   user_instruction exactly `计时放开一下门槛`, startup_free_MiB 8192,
   previous_startup_free_MiB 18432, scientific_calls 0, nonempty review_evidence
   exact `{path,sha256,bytes}` JSON pins, and `expected_source_sha256` mapping all
   four filenames to the reviewed hashes.
3. Root runs `timing_resource_owner.py register --authorization ABS
   --authorization-sha256 SHA --authorization-bytes N` with CUDA hidden. This
   creates only the supplemental registration, after authenticating the original
   v6 SHA, historical evidence, all source pins, absent original attempts and six
   available locks. Save the returned supplemental registration SHA.
4. Root rechecks live GPU0 and coordination, any R1/backup coordination locks,
   absence of all original and supplemental attempts, then dispatches exactly
   once using the original visual Python: `timing_resource_owner.py run
   --expected-resource-registration SHA`. The original owner owns and really
   waits for its child. Keep a separate outer supervisor, stdout/stderr and
   actual wait record. A directory or returned launch PID is not completion.
5. If a failure occurs, retain it and diagnose; no blind retry or new scientific
   attempt. The wrapper's owner/worker receipts are exclusive and cannot be
   overwritten. Root must verify actual wait0, no failure, both process generations
   gone, original completion and all new execution bindings before delivery.

## Postprocessing in this same revision

After actual success only, run the wrapper's `postprocess --output ABS
--expected-resource-registration SHA`. It verifies the exact root-bound
supplemental registration, true owner/worker argv, source hashes, original
wait0/completion, returned receipts and both dead process generations. The original
postprocess function is source-derived with only its 18432 admission comparator
changed to 8192. A routing guard enables that comparator **only** for the exact
authenticated v6 registration, actual saved admission and actual worker PID.
The inherited v5 segment uses the untouched original validator with 18432 MiB;
unknown registrations cannot borrow the exception. All statistics functions,
rows, inherited validation, clocks and numeric/resource checks stay original.

The extra registration, authorization, sources, actual execution/control receipts
and postprocess binding must be included in the final timing publication and
backup. Preserve old historical registration and failures. These source files
are preparation only and do not constitute a timing result.
