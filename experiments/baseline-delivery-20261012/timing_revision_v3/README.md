# Six-method timing: explicit v3 constructor recovery

The first real v2 attempt ended with `actual_wait.exit_code=1` before any timing case was reserved. Its case ledger contains 0 cases and 0 packet decodes. VAE, VAR, and Dc construction had occurred; the failed attempt must not be described as zero-model execution. No timing warmup or measured frame completed.

The failure is an import-identity conflict. The raw433 factory first loads two preserved, relocated modules into their historical names. The T1 whole-entropy constructor then requests the canonical repository paths under the same names. Both module pairs are byte-identical, but the unmodified strict loader rejects a different `__file__` path.

The v3 wrapper admits only these two exact registered path pairs during construction:

- `main_raw64_plan_only`, SHA `6005bf8b5668034b63cd22082a21dfacfe02c9c06f812172746bc6f6f901ddd0`
- `main_raw64_packet_adapter`, SHA `8414b29d63cbe340a164708da9278769479d5559915b97de64e4383d4caeb928`

The original raw module objects and `__file__` values remain intact. The private T1/EP loaders verify both byte hashes and return the same objects. Every other name/path retains the original strict loader. The temporary constructor hooks are restored before the first timing case. Full catalogue/constructor identity, receive domains, mathematical functions, profiles, models, 6/2 threads, FP32/TF32 settings and timing/audit accounting remain inherited unchanged.

All v1/v2 source, registration, launch, failure and actual-wait evidence remains untouched. The v3 registration binds the failed ledger and receipts, rejects an existing v3 attempt, and shares the previous owner locks. Its run command is separate from recovery registration; there is no retry loop or new waiter.

The existing budget remains 1,728 total cases (864 measured) and at most 2,304 digital packet decodes. All of that case/packet budget is unused. The existing fixed16 sources, three SNRs, three fresh warmups and three measurements apply. Numerical qualification remains NOT_RUN until actual fresh warmups pass.

CPU tests extract only the original two small strict-loader functions and use synthetic temporary Python modules. They reproduce the original path failure, test the four admitted loader/path combinations, byte/path/planner-object rejection, original-loader restoration, and unaffected imports. They do not import Torch, construct models, generate a channel or decode a packet.

`register-recovery` records the reviewed recovery without running scientific work. `verify` checks its current bindings only. An actual `run` requires separate coordinator dispatch after review. Shared-H800 resource admission and E1 completion evidence remain inherited from v2; no exclusive or uncontended timing claim is made.
