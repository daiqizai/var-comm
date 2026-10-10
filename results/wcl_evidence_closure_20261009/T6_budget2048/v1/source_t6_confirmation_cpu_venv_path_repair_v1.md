# T6 confirmation CPU launcher path repair

The first confirmation CPU cohort exited before ledger initialization and before any scientific worker launch. The registered virtual-environment interpreter was incorrectly passed through `Path.resolve()`, which dereferenced its symlink to `/usr/bin/python3.11`. That system interpreter failed while importing NumPy. The original registered virtual-environment path imports its installed NumPy successfully.

The actual failed parent and initialization child were waited. The output namespace has no packet ledger, physical frames, raw source checkpoints, entropy source checkpoints, or scientific worker reservations. Thus the failed attempt consumed zero packet decodes and zero model calls. The original source preparation completed separately and remains valid: 100 Encoder calls, 100 VAR source encodes, and 292 VAR source decodes.

The root operator reviews this zero-call failure explicitly. The repair uses a separately versioned CPU cohort owner preserving the registered virtual-environment interpreter path. All original failure receipts and scripts remain unchanged. A fresh `T6_confirmation100_render_v2` namespace reuses the exact completed source manifest and nine-policy freeze. It keeps the original render deadline, four workers per branch, shared 5,400-packet cap, 2,700 method-frame conditions, scientific scripts, source IDs, seeds, policies, and numerical settings. No automatic retry is enabled.

The source preparation is not repeated. The failed render v1 is never described as a successful scientific result. Each subsequent stage still requires actual completion and parent wait before the next stage is launched. This operational repair changes neither the scientific comparison nor the frozen configuration selection.

Remote zero-call audit: `outputs/WCL-EVIDENCE-CLOSURE-20261009/T6_cpu_venv_path_repair_v1/zero_call_audit.json`, SHA256 `9e0d5367a82fc9418e34424b2448272c5c88738c26cbc64c42e4078125a4d932`.
