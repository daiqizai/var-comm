# Six-method timing wrapper revision v2

This timing-only wrapper preserves the v1 scientific endpoint functions and
registration. No v1 model/case/packet launch had occurred at preparation.
The user subsequently authorized shared resources rather than waiting for all
GPUs to become empty. The selected H800 remains fixed across the six arms.
These measurements describe a shared environment, not uncontended latency.

The fixed budget is unchanged: development16, N1024, SNR4/10/19, batch1,
three fresh warmups and three fresh measurements per condition, 1728 total
calls / 864 measured calls, and at most 2304 digital packet decodes. No training,
extra qualification frames, source/output caching or algorithm changes occur.
The original six CPU slots, 6/2 threads, FP32 and TF32-off settings remain.

The static raw433 catalogue hash is checked before and after every full case.
The original per-frame calls to that static verification return only after the
outside check. Actual received-profile, CRC, length/padding, arithmetic
canonical termination, input-shape and fallback checks remain on the timed path.
Only the individually reviewed pure provenance helpers below are measured in
explicit excluded sections; they still execute and return their actual hashes:

* raw receiver `array_sha`: waveform / observation provenance;
* raw receiver's imported `identity`: received-state metadata digest;
* raw planner `identity`: request/result/physical-description provenance;
* raw planner `new_identity`: static backend descriptor construction;
* original whole and partial entropy PHY `array_sha`: packet array provenance.

Hashing for scrambling/noise seeds, protocol CRC, arithmetic CDF validation,
canonical re-encoding and scientific cache keys is unchanged and included.
There is no global clock/hashlib monkey patch. Pending CUDA science synchronizes
before an excluded CPU audit. Raw wall time, excluded audit time, adjusted
TX/RX/software end-to-end time and per-function audit counts are all retained.
Exclusion wrappers do not change returned metadata or scientific outputs.

Start admission requires the chosen GPU to have at least 18 GiB free (16 GiB
private PyTorch allocator cap plus 2 GiB safety) and reported utilization <=25%.
During execution the 2 GiB free safety floor remains; other resident processes,
GPU load and timestamps are recorded outside each measurement window. No other
process is stopped and no GPU global setting is changed. Six methods execute
serially and E1 quality science must remain stopped for this timing window.
E2 CPU statistics may run concurrently under the user's shared-resource
authorization; their presence is recorded, and timings are not uncontended CPU
or GPU latency estimates.

The root coordinator must create `coordination_release.json` in the v2 result
directory, binding E1 stage/closure receipts and explicitly declaring
`quality_jobs_stopped=true` and
`no_quality_science_until_timing_actual_exit=true`. Missing release returns WAIT.
The wrapper holds the original v1 owner lock through actual child exit, rejects
any pre-existing v1/v2 launch, and never retries a real attempt automatically.

The separately registered `after_emain.py` waits once for E1's complete 4500-row
scoring receipt and all four actual exit-zero stage receipts. At the root
coordinator's explicit authorization, it writes the evidence-pinned release,
checks admission, and exec-replaces itself with the v2 timing owner. It does not
leave a second observer, restart E1, or retry a failed timing attempt. Polling is
once per 60 seconds, with a 24-hour finite waiting deadline.

From the repository root, after checking real receipts and process identities:

```sh
python experiments/baseline-delivery-20261012/timing_revision_v2/timing_owner_v2.py readiness
python experiments/baseline-delivery-20261012/timing_revision_v2/timing_owner_v2.py run
```

`prepare` is once-only and does not qualify numerical endpoints. Only complete
actual warmups and measurements with actual exit0 permit timing delivery. The
CPU tests are synthetic engineering checks, not scientific calls or acceptance.
