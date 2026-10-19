# Analysis original-environment qualification continuation

This independent engineering wrapper preserves the frozen six analysis sources.
The first original CPY attempt returned 1 after 19 tests passed. Test 20,
`IdentityTests.test_image_hash_and_shape`, reached `np.savez` and its lazy standard
library `zipfile` import was rejected by the unchanged fixture-only PFS guard.
Error formatting then hit the same guard when opening NumPy source. The original
wait, launch, supervisor, stdout and stderr remain immutable and SHA-bound in
`continuation_plan.json`; there were no model, channel or scoring calls.

The plan statically derives the original unittest alphabetical order from the
exact test source: 56 unique cases, 19 retained passed cases, and 37 cases starting
at the interrupted test. `continue_tests.py` preloads only standard library
`runpy`, `unittest` and `zipfile`, verifies the original source and failure chain,
then executes the exact original test source with `runpy.run_path`. The original
model import blocker, subprocess prohibition and fixture-only PFS audit hook
remain active. It selects only those 37 cases, so the prior 19 are not repeated.

Root owns static deployment and a single finite 180-second CPY `wait_child` run.
Deploy `continue_tests.py`, `combined_review.py` and `continuation_plan.json`
together in a new independent light control directory under the existing L tree.
Use a new, exclusive control directory for the continuation's stdout, stderr,
supervisor, launch and actual wait. Do not reuse the first qualification directory.
Set `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONNOUSERSITE=1`, clear PYTHONPATH, and set TMPDIR to the new continuation
control directory, matching the original fixture restriction.

Actual child argv is exactly:

```
CPY -B /absolute/deployed/continue_tests.py --plan /absolute/deployed/continuation_plan.json
```

After the continuation child and its parent/supervisor have really exited,
root loads `combined_review.py` as a standard-library-only metadata helper and
calls:

```
proof = validate_combined(plan_path, continuation_control_dir, wrapper_files)
```

`wrapper_files` contains exact `{path, bytes, sha256}` descriptors for the three
deployed files. The helper verifies both attempts, original source bytes, the
first failure boundary, exact real child argv/wait, host-local PID/start closure,
all 37 individual pass lines, and the disjoint 19+37 union. It returns a combined
qualification proof for root to save with its own source and review evidence.
It writes nothing itself and cannot start tests or science.

The root preparation gate must consume that saved proof and revalidate its
bound evidence; it must not require or replace the preserved first wait1 with
wait0. Report **19 retained PASS plus 37 continuation PASS across two attempts**.
Do not claim a successful single 56-test attempt. A new unexpected failure is
preserved for diagnosis; there is no automatic retry or guard relaxation.

This package has only been statically checked and independently reviewed before
root execution. Its existence is not an original-environment qualification PASS
or any scientific result.
