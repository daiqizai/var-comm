# DINO-only C calibration entry

Prepared independently while calibration PHY runs; no deployment, registration,
model invocation or strategy freeze follows merely from the presence of files.
Do not execute worker.py/freeze.py without exact registered closed inputs and
the original durable control path. All RX shard wait0/gone/conformance audits
must pass before scoring. Never read active reconstruction arrays for admission.
The pair-plan preparation must observe worker, owner and audit PID/start identities
on PRO and save a pinned closure observation for each new RX shard. The H800
metric worker verifies that evidence; it never treats absent PIDs in MAIN's
different process namespace as evidence of PRO process exit.

Reuse the original H800 GPU0 DINO-L model, metric implementation, FP32/TF32off,
batch1, threads6/2 and feature preprocessing. Source features can reuse the
verified old100+900 calibration caches only after exact metadata/runtime/source
pixel/feature hashes match. A metric-selection attribution string is the sole
permitted metadata difference. No current source feature inference is requested.
No LPIPS, ConvNeXt, CLIP or DreamSim model is loaded during calibration.

The immutable pair plan must provide every original source with archive,
source_float_sha256, reference descriptor, reference_feature_sha256 and pairs.
Each pair binds native image descriptor (or None for constant0.5 gray), pixel
hash, optional exact prior score JSON descriptor, and all logical frame records.
Repeated identical source-image pairs reuse current scalar scores; rejected
received outputs stay in their original18 logical positions per source. An
existing gray score may reuse only under the full unchanged donor identity.
No quantization, clipping, imputation or source-truth RX correction occurs.

Scoring writes full precision source-pair and18000 logical scores with actual
new/reuse counts. All1000 original indices are required. The subsequent separate
CPU freeze uses the SHA-pinned original ARPC aggregate() AST and only the DINO-L
metric tuple. It first validates a complete six-candidate/three-noise grid.
Three noises are averaged within source; all1000 source means have equal weight.
Exact maximum ties choose lexicographically smaller candidate ID. No bootstrap,
new MCS, generation changes or formal samples participate in calibration.

Frozen policy explicitly describes common500+Kodak24 formal scope,4716 logical
frames, fine1..256 finished-prefix fallback and paid10-bit class. It does not
run that successor or import the historical ARPC external_common100 main().

Before registration, complete the closed-output pair-plan/dependency/resource
preparer and review source pins. Those preparations are not in this directory.
The worker's8GiB metric memory assertion does not bypass outer admission: use
the frozen project resource policy and verify actual original H800 device.
