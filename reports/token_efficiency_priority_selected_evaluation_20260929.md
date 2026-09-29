# Completed-model C evaluation scheduled ahead of remaining training

The user requested evaluation of already available models, then asked whether it could run in parallel with training. GPU0 was at 100% utilization, approximately 10/24 GiB used, and 80 C. The unchanged runtime rejects every other GPU process. Full online timing would also be confounded by concurrent training. This interlude therefore keeps GPU work serial; independent CPU checks remain concurrent.

The separately registered development population contains 14 frozen selected models: six original N4084 arms, two N3060 arms, four second-seed arms, and the third-seed H6-V/H6-P pair. Every training group has a calibration stop decision. The active third-seed H8-V and unstarted third-seed P4084 are excluded. The 100 original development sources, five SNRs, and three noise seeds imply 21,000 primary frames, plus original B_RX/C_RX diagnostics, 1,400 full online timing calls and 420 warmups. The original C_evaluate module performs real selected cache/online/replay qualification before population evaluation. No new holdout, selection rule, model, loss, FEC code, runtime precision, hardware setting or active source was changed.

The original outer delivery chain was verified by PID, start ticks and command before SIGTERM. Its nested training saved a safe checkpoint, both scheduling layers exited with the requested-stop path, and the independent worker acquired the original main, C-followup and predecessor locks. Unknown failures stop without automatically resuming training. Successful complete evaluation releases the locks and starts the original unchanged delivery entry, preserving old receipts and registering the new real process identity. Two historical workers remain waiting. The author and N3060 timing waiters treat upstream STOPPED_BY_REQUEST as a protected failure and exited without GPU execution. Their original failure/status/launch/registration/console evidence is archived. A separate CPU-only recovery waiter verifies the successful priority completion and resumed original main identity before restoring those two original wait commands in order; their registrations and all GPU gates remain unchanged.

The early grid has its own run/context and output directory. It does not claim to be the final 16-model grid or to have completed the overall experiment. The original final-grid schedule remains intact; future reuse, if implemented, requires explicit compatible-identity verification rather than relabeling this population.

## Recorded safe handoff

- Output: outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/C_priority_selected_v1
- Original chain outcome: STOPPED_BY_REQUEST, returncode 75.
- Third-seed m8 checkpoint step: 17642.
- Checkpoint SHA256: fe2589b31c0aa057da0d4b2b33ec7d6a7672d4a83c65b5f4ff7f09381fc037c7.
- Priority registration SHA256: c6b75bd9c66c6b2a15118713a2ee23612413a94f1c218c8622f39c0aa1885309.
- Immutable priority bindings: 198.
- CPU checks: 259 passed; repository source accounting passed after explicit staging.
- Real selected GPU qualification: 56/56 cache/online/replay checks passed (REAL_C_SELECTED_CACHE_ONLINE_REPLAY_PASS). Full 1,400-cell population evaluation has started and remains incomplete. The acceptance receipt does not certify full-population quality.

## Independent tools

The new tools/priority_selected_evaluation.py and shell entry are outside every prior active binding. Regression checks reject recycled/changed/zombie process identities, unfinished calibration decisions, incomplete or synthetic evaluation completions, and accidental inclusion of unfinished third-seed controls. Existing experimental sources and receipts were preserved.

## Correction after live handoff inspection

The initial scheduling report incorrectly generalized the first two waiters' behavior to all four. Live checks found the author and N3060 guard exits at the requested-stop boundary. This is an identified scheduling side effect, not a measured quality failure. The priority evaluator is unaffected and its bound source is unchanged. tools/restore_priority_reference_waiters.py only accepts the two exact archived upstream-stop errors after this pause, rejects live or duplicate workers and unrelated failures, waits for the original main to resume, then restores author first and N3060 only after author has a fresh waiting status. The recovery process itself is CUDA-masked and makes no GPU calls; child workers regain their original environment and remain subject to all predecessor/idle/thermal gates. No claim is made that they have resumed before the recorded recovery completion exists.
