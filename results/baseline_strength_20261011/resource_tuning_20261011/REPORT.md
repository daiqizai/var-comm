# Live resource tuning — 2026-10-11

The user authorized higher utilization where spare resources permit. Only operating-system scheduling and P GPU placement changed. No bound training source or protocol was edited. Original optimizer, source order, RNG, loss, batch, precision, stopping and calibration rules remain in force.

P saved and SHA-verified its complete step65356 state, exited with the existing safe-pause receipt, and resumed on GPU1. The previous GPU0 process is absent. Swin continued on GPU2. Each branch now has six logical CPU cores available on NUMA0, retaining six compute threads, niceness10 and the16GiB allocator cap. Only two training workers remain. No other users’ processes or shared GPU settings were changed.

| Branch | Before (s/update, last200) | After (s/update,100 updates) | Update-time reduction |
|---|---:|---:|---:|
| P | 1.3239 | 0.9193 | 30.6% |
| Swin | 0.6169 | 0.4658 | 24.5% |

These are short observational windows on a shared host; occupancy and image mix can change. They are not the paper’s unified timing measurement. No cross-device bitwise-equivalence claim is made. All original source-binding hashes and contiguous training-update sequences were verified after migration.

Current P owner: jobs/P_train_gpu1_v2 (worker383981). Original P_train_v1 is safely retired; do not restart it. Swin_train_v1 (worker92548) remains active. CPU affinity is an external live adjustment, recorded in receipt.json; the unchanged launcher still carries its original two-core default on any future restart. A future authorized restart must inspect this receipt and the live device load before choosing resources.

Checkpoints remain under personal liberai-checkpoint/liulu, operational logs under personal liberai-tmp/liulu, and this lightweight evidence under code/liulu. Final training, diagnostics and frozen holdout evaluation remain pending.
