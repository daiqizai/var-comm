# Six-method timing v5: preserve runtime wrapper ordering

v4 was reviewed and registered but never launched. Its one-line constructor correction compiled a copy of the original build function using a copied globals dictionary. The subsequent v2 worker replaces `runtime.visual_backend` with the shared-resource adapter; the copied globals would still resolve the previous function and lose the updated resource metadata. The v4 files and registration are preserved unchanged as prelaunch evidence.

v5 keeps the exact corrected code object, defaults and closure, but constructs the function with `runtime.__dict__` as its globals using `types.FunctionType`. Therefore later changes to `runtime.visual_backend` remain visible. No scientific expression, model, policy, timing boundary, thread/precision choice or budget changes.

The effective chain is:

1. v5 provides the live-globals adapter to the inherited v4 worker.
2. v4 installs the single-line Swin constructor correction with the original runtime globals.
3. v3 wraps that constructor with the two exact-byte private module aliases.
4. v2 installs the shared visual-resource wrapper, memory cap, pure-audit timing adapter and parameter inventory.
5. The original once-only worker constructs the system and invokes the unchanged registered fresh cases.

CPU tests reproduce the v4 lost-hook behavior and exercise the actual v4/v3/v2 worker functions in their true nested order using inert stand-ins. They verify that the two private module aliases are actually used and restored, the shared visual metadata is true, the audit hook is installed before a stub first-call sentinel, and the corrected SNR list remains intact. The actual original build definition is also compiled with inverse-AST proof and its globals identity checked without invoking it. Tests do not import Torch, run a model, generate a real channel or consume a timing case.

v2 and v3 failures remain immutable. Those attempts did construct real models, but each reserved 0 timing cases and decoded 0 packets. v4 has no runtime directory or launch. v5 retains the full existing 1,728/864 case and 2,304 packet budget, shares predecessor locks, and has no retry scheduler. `register-recovery`/`verify` perform metadata checks only. Actual v5 launch remains a separate coordinator action; actual numerical qualification is still NOT_RUN.
