# PRO6000 same-device DINO calibration migration

User instruction (2026-10-27): 换到有空的pro6000上继续，别一直等。
The original H800 metric registration never dispatched; its files remain intact.
This new entry uses PRO GPU6, PCI A3, with unchanged frozen Python, original
DINO-L weights/evaluator/preprocessing, FP32/TF32off, batch1 and threads6/2.
The same original 1000 sources, 18000 logical calibration frames, received pixels,
source-pair plan, six MCS candidates and source-mean DINO selection are retained.
No PHY, reconstruction, training, formal evaluation or bootstrap runs here.

Hardware is part of runtime identity. Thus all1000 source DINO references and
at most5061 distinct source/pixel pairs are evaluated on this PRO device.
The 828 historical H800 scalar reuse candidates and1000 H800 reference features
are explicitly NOT reused. Compared with the unstarted H800 plan, this adds
1000 reference forwards and at most828 pair forwards, not new logical events.
Identical current-run source/pixel pairs may still reuse scores with a key
including complete runtime identity. Gray/failure frames retain every weight.

Limits stay14400seconds and cumulative96GPU-hours. Start admission24GiB/util<=80,
worker initial8GiB, allocator6GiB, running free2GiB. No foreign workloads changed.
Original worker/contract/registration and pair plan remain byte-identical.
New worker only changes runtime validation, same-device fresh reference handling,
cache policy, explicit operation caps and provenance. Original source image
validation, metric.score call, frame expansion and grid validation are retained.
Unknown failure is saved without automatic retry. Original frozen CPU policy
aggregate may run only after actual score exit0 and process disappearance.
