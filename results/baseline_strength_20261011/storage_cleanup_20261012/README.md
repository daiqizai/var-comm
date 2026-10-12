# User-requested redundant storage cleanup, 2026-10-12

Actual deletion completed:437 files,10,526,892,735 logical file bytes (9.80GiB).
Only VAR_COMM-owned paths under liberai-checkpoint/liulu/var_comm and
liberai-tmp/liulu/liulu-var-comm were touched. This is a logical file-byte count,
not a claim about immediate filesystem quota or snapshot reclamation.

-116 superseded full recovery snapshots:3,145,659,106 bytes. These intermediate
  optimizer/RNG states are deliberately retired and are not individually backed
  up by this cleanup. Latest full P100k/Swin200k recovery states and their verified
  Lu_Data copies, original parent states, migration anchors and every calibration
  model-weight snapshot remain.
-9 redundant transfer/publication archives:7,372,973,061 bytes. Each server file
  was matched by full SHA256 and byte count to its preserved Lu_Data copy before
  removal. Extracted original training/calibration inputs remain in place.
-312 completed synthetic CPU-test temporary files:8,260,568 bytes.

Original scientific results, training histories, qualification/failure records,
source images, frozen environments and pending timing dependencies remain.
The timing owner lock was held through deletion; no scientific job was running,
no timing attempt had started, and every registered timing input/code binding
plus both final full checkpoint hashes passed verification after deletion.
Other projects and users were untouched. No inference, training or resampling ran.

plan.json records every deleted path, SHA, size and reason; deletion_journal.jsonl
records actual removals; completion.json records post-cleanup checks. Historical
checkpoint references describe prior states and must not be mistaken for current
file availability. Retired intermediate recovery snapshots cannot be restored
from the final-state backup. No completed scientific stage should be rerun.
