# Phase2 historical metric recovery R1

This is a separate, versioned continuation of the selected R3 queue. It keeps the original 12 jobs, coverage, contrasts, model weights, policies, sources and noise seeds. It reads the seven complete R2 studies and four complete R3 studies at their original paths.

Only OPTIONAL_PHASE2_MAIN changes its replay adapter to historical_phase2_alias. The first 81 source checkpoints and float caches are admitted read-only with their original registration and payload bindings. Only sources 81 through 99 (855 rows) receive new metric inference. No original R3 file is written.

The old Phase2 pure-continuous implementation and continuous-grid implementation have different floating-point normalization/noise evaluation paths. Their complete 1,500-row historical difference is disclosed. Actual new images and metric values are not shifted. New replay must pass the original continuous-grid tolerances and waveform/observation checks. The source-81 historical PSNR difference is not relabelled as direct Phase2 scalar parity.

Execution order: audit_selected_runtime.py --root ROOT --output OUT/selected_constructor_audit.json; register_selected_queue.py --root ROOT --audit OUT/selected_constructor_audit.json; history_controller.py --root ROOT --queue-registration OUT/queue_registration.json. OUT is ROOT/outputs/HISTORICAL-PHASE2-RECOVERY-20261004-R1. Use the already qualified native metric environment and the registered decoder gate. Upload all flat .py and .md files in this directory. Never use the old R3 output as the new OUT.

The controller runs CPU tests, checks the previous numeric delivery and the exact R3 controller/bootstrap/child process identities have exited, publishes this source normally, reuses completed studies, scores 19 sources, summarizes all 12 studies, and publishes results normally. Completion does not claim the old R3 run itself finished. The external training controller additionally waits for this new controller and all its children to exit.
