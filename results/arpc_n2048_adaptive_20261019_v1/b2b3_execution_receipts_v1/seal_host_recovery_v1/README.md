# B2 seal: exact execution-host correction

This is an engineering diagnosis and root control proposal. No seal retry,
native inference, scoring, packet experiment, registration or production
adapter has been run by this task. The six registered analysis sources remain
unchanged. Root must independently review this helper and its failure guard
before deployment or invocation.

## Actual failure and preserved evidence

The original MAIN control `b2b3_analysis_root_v1/B2_seal_none` actually waited
for child 3940558/start 209507359 and returned integer 1 after
82.10256171226501 seconds. Controller 3940443/start 209506951 and its parent
3940441/start 209506941 have also exited. The original five control files and
the earlier 999-byte phase metadata failure are preserved byte-for-byte here.
The new read-only `main_failure_observation.json` binds all three generations,
the actual failed wait and the original launch/supervisor descriptors.

The frozen `native.py` (34,279 bytes, SHA256
`d02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889`)
calls `validate_runtime_metadata()` at line 449. It checks the ARPC execution
metadata against paths on the **seal host**. MAIN's `/usr/bin/python3.12`,
`torch/version.py` and `torch/__init__.py` differ from the ARPC configuration
recorded on leo. On leo, every recorded runtime file and every package
METADATA/RECORD descriptor matches. The original shared CPY successfully ran
the read-only metadata observer on both hosts; no Torch/model import occurred.

The ARPC runtime and reference runtime are equal under the original frozen
canonical comparison. NeST completed 2,896 native jobs, with unresolved 0;
ARPC completed 377 native jobs, with blocked reads empty. Both workers retain
their successful original waits and host-specific closure observations.

No partial seal output exists: `post_exit_native_audit.json`,
`actual_native_objects.json`, `actual_logical_pixels.json`,
`native_manifest.json` and the output `native_completion.json` are absent.
The failed attempt only read closed metadata/pixels before reaching its first
write. It did not invoke a receiver, generator, channel or metric. The earlier
phase failure is still 999 bytes with its original SHA; the frozen exception
handler did not overwrite it.

## Minimal route

Use the existing frozen `native.py seal` on the ARPC configuration host,
leo-tactile (`aibox-rb80bf7d8e38-dc54b8fc7-zrgl7`). No production-source
adaptation is needed. The other seal operations use the shared PFS or the
already saved cross-host native observations. There is no MAIN-only process
check inside seal's call path. The original `check_observation` verifies the
actual execution-host observation; it does not inspect another host's `/proc`.

Root helper: `H/seal_execution_host_v1.py`.

1. Root deploys only the files in `deployment_plan.json` under
   `O/b2b3_execution_receipts_v1/seal_host_recovery_v1/`, preserving bytes.
   This task has not performed that deployment.
2. Root integrates `failure_guard_code()` into the existing
   `analysis_execute.metadata_revision_guard` **only** for phase B2 and suffix
   `seal_none`, then continues that exact branch. The function verifies the
   diagnosis, original five receipts, source, registration, failed returncode
   1, exact PID generations, execution-host failure observation and unchanged
   999-byte phase failure. Unknown failures still fail. It never converts a
   failure to wait0. No string-rewrite or broad skip is used in the helper.
3. Root explicitly invokes `run leo-tactile B2 <analysis registration SHA>`.
   Preflight requires absent new control and all five absent seal outputs,
   both native closures, the actual phase configuration host, matching local
   interpreter/Torch/package files, and unchanged registered sources. It
   exclusively creates `ROOT/B2_seal_execution_host_v1`, records actual host,
   registration, original failure, runtime identity, source and exact argv,
   and uses the original `shared.wait_child` with CPY, hidden CUDA and 1800s.
   Persistent stdout/stderr and true child/controller identities are saved.
4. After that command has really exited, root explicitly invokes
   `observe leo-tactile B2 <analysis registration SHA>`. This checks strict
   actual wait0, original argv, same-host child/controller/outer-parent gone,
   no new failure, and the completed seal manifest. It creates the host
   observation and success route only then.

The route is `O/b2b3_execution_receipts_v1/seal_host_recovery_v1/B2/route.json`.
Its schema is `B2B3_SEAL_EXECUTION_HOST_ROUTE_V1`, with exact `phase`,
`registration`, `native_source`, `actual_wait`, `launch`, `host_observation`,
`native_completion`, `manifest`, `original_failure_audit`, and full log/helper
descriptors. The successful observation uses schema
`FINAL_PUBLICATION_PROCESS_OBSERVATION_V1`, status `CLOSED_ON_EXECUTION_HOST`,
host/observer_host/execution_host, full child/parent identities, canonical
processes, actual_wait/wait, launch/argv, all-gone flags, observation time and
the exact root-helper source. It can be consumed by report and publisher.

After success the helper stores the returned actual route descriptor/content
as `H/seal_route_B2.json`. A visible directory or dispatch return does not count
as success. A new failure must be preserved and diagnosed, with no automatic
retry. No scientific calls are repeated by this root control.

For B3, the same explicit helper accepts only B3's own registration,
configuration, native closures and fresh B3 outputs/control. It must not reuse
B2's configuration or interpret the B2 failed predecessor as a B3 failure.
The current helper intentionally requires a nonzero new ARPC job count and
leo configuration; another situation requires a separate explicit root
decision. It does not auto-start B3.

## Reporting changes needed after actual success

The report root authorizer must use the exact new success route instead of the
failed `B2_seal_none/actual_wait.json`, include its cross-host observation, and
disclose the failed MAIN wait1 as an engineering exception. The old wait1
remains wait1. A B3 route, if later actually completed, uses its own descriptors.
The report's frozen production code and all numerical/statistical functions
stay unchanged. Publication must not relabel this as a fresh native/quality
measurement or a successful initial MAIN seal.
