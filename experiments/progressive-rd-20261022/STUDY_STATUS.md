# Progressive RD follow-up (2026-10-22)

Independent research from frozen WCL publication. Read user_request.txt, then
results/progressive_rd_20261022_v1/stage0_delivery and seed_delivery_v1.
Stage0 and seedv2 are actually closed. The original failed seed import remains
preserved with zero completed model/source/RX/metric calls.

Current decoder registration: decoder_pilot_registration.json. Three arms,
5000 updates x4 each, common Dc initialization and matched576-parameter FiLM.
One final calibration100 x8 per arm. No checkpoint selection or held-out claims.
Only decoder weights update; old recipe historical Fb_TX is training-only.

Expansion: expansion_registration.json is a13.7MB runtime metadata file, omitted
from Git's10MB/file limit. The deduplicated expansion_registration_compact.json
and restore_expansion_registration.py recover its exact bytes and original SHA.
Do not modify the active runtime registration. CPU synthetic accounting checks
and original seed runtime are reused. Successful source calls are not repeated.
Workers3-7 start on free GPUs;0-2 depend on the real decoder actual_wait0 and
gone process. All share the original absolute six-hour-window deadline.

Never hot-edit bound code. Inspect PID/start_ticks, actual_wait, failure and
ledger before action. No automatic science retry. All data/checkpoints remain
outside Git on PFS; normal exact-scope repository checks/publication required.
No new environment, training data download, tokenizer, PHY, selector or encoder
fine-tuning is authorized by an old record. See the current user request.
