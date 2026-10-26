# Exact device mapping and output-lineage revision

The initial source process failed at `torch.cuda.set_device(0)` before model loading or any scientific forward. Its sole model-load reservation is FAILED. The NVML UUID differs from CUDA Driver UUID on this host; both map to physical PCI 61:00.0. The original checked mapping implementation identifies CUDA PCI_BUS_ID ordinal 0.

This revision uses numeric CUDA_VISIBLE_DEVICES=0 and CUDA_DEVICE_ORDER=PCI_BUS_ID. NVML resource admission still checks the original physical UUID and PCI. Original model loader independently checks the visible device PCI. Precision, model weights, source encoding, PHY, receive generation, policies, sampling and budgets remain unchanged.

All original lane files and original registration remain immutable. The old failed output and wait logs remain in place. The revision imports original scientific modules after setting only their execution output root to nest_v1/recovery_v1; final light delivery remains the original, previously absent delivery directory. Each role binds the revision and parent registration. Every child uses a real bounded wait, persistent stdout/stderr, PID/start ticks and an exclusive stage directory. No retry or next-stage launch occurs automatically.

The failed model initialization is disclosed separately from the one permitted successful source model load. It did not execute tokenizer, TX probability, prior or decoder calls. No successful scientific call is rerun.

Use `execution.py dispatch STAGE EXPECTED_REVISION_SHA` once per explicitly verified stage. Read both owner and supervisor actual waits, identities and failure files before moving to the next stage. PHY and delivery keep CUDA hidden. All execution on the same PRO host avoids cross-host PID interpretation.
