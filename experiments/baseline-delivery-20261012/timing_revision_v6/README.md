# Six-method timing v6: reviewed missing-work recovery

This independent wrapper is prepared after the v5 `IndependentProvider` closure
failure. It does not modify or retry any older owner. Registration and the
single launch remain separate, explicitly coordinated actions.

The v5 journal contains 576 completed calls (both raw methods, all fixed16
sources and three SNRs, three fresh warmups and three measurements) and the
reserved first entropy-whole warmup. That failed call completed fresh visual
encoding but produced no entropy stream, waveform, channel observation or
packet decode. Its temporary tokens were not persisted.

The new journal inherits the exact 576 completed rows and 1152 packet calls.
Only the remaining 1152 complete calls enter the endpoints. The failed partial
warmup is archived separately and is never counted as a completed warmup.
Success means 1728 complete calls and 864 measurements; it also means 1729
known fresh visual encoder/endpoint entries across the failed and recovered
attempts. One incomplete encoder/VQ computation is explicitly additional.
The global digital packet cap stays 2304. No completed call is rerun.

The four-hour execution envelope is conservatively cumulative: enclosing
launch-to-actual-wait intervals from v2, v3 and v5 are each rounded up with
five seconds of margin and deducted from 14400 seconds. Those prior attempts
charge 1331 seconds, leaving 13069 seconds for v6. v4 was never launched.
The inherited 30-minute spec window applies only to initial model-loader
validation; fresh timing uses the remaining cumulative owner boundary.
The inherited 1800-second child-wait watchdog grace is not scientific budget.

The user's experiment document defines 1728 as complete end-to-end calls and
allows recovery of genuinely missing or invalid work while preserving
completed rows and failures. The coordinator's separately pinned authorization
must record this interpretation and the additional failed subcomputation;
this wrapper never infers approval from elapsed time or a failed receipt.

Changes are restricted to: capturing the original `IndependentProvider` class
in a lambda default before `ns` becomes the P namespace; retaining the already
reviewed Swin attribute compatibility and live globals; seeding the old owner
with complete576 data and slicing its loop before any endpoint invocation.
Exact substitutions have inverse AST proofs. The shared-H800 gate, 16 GiB
private allocator cap, 6/2 threads, six-CPU affinity, FP32/TF32-off math,
model/config/source/noise identities, receiver isolation, audit exclusion and
constructor-only byte-equivalent module aliases are inherited unchanged.
The already-qualified `h800_ep48_link_core_v1.bind_static` also loads and
validates the original frozen train20k CDF into the entropy codec, preserving
legal received `EC_STATIC_WHOLE` profiles in the 144/360 public receiver domains.
It is not a new transmitted method or a new static fit. Its original relocated
completion/table hashes are added to the input bindings. New and inherited
runtime/device identities are checked before the first remaining endpoint.

`test_recovery.py` contains CPU-only inert tests. It imports no Torch models
and performs no real encoding, channel, packet, reconstruction or scoring.
No v6 registration, launch or model call is performed by preparing these files.

After separate review and coordinator authorization, the existing explicit
`register-recovery`, `verify`, and once-only `run` commands are available.
All predecessor locks are held for the attempt; any new failure stops without
automatic recovery. v2/v3 really constructed models with zero cases/packets;
v4 was never launched; v5's 576 complete plus one incomplete call are retained.
