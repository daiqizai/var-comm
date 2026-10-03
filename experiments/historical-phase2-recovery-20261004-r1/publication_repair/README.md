# Final publication audit repair

The R1 scientific computation completed all 4,500 Phase2 rows, including 81 inherited sources and 19 newly measured sources. Final publication then stopped with `KeyError: 'dino_mismatched'`.

The legacy numerical-alias proof contains exactly three metrics: PSNR, LPIPS and matched DINO. Its tolerance map also includes an optional mismatched-DINO field used by other historical tables. The final verifier incorrectly enumerated all tolerance keys. This cold repair enumerates the three registered legacy metrics and retains their original tolerances. It also checks exact metric-key coverage and the recorded difference identities.

The separate independent mismatched-image metrics remain in the completed evaluation rows. No legacy mismatch difference is invented. No image, waveform, model, metric, scientific source, queue or stopping rule is changed. No reconstruction or metric calculation is repeated.

The new publication-only process verifies the original frozen queue and all completed study evidence, installs the repaired final audit callable into its own publisher process, runs the original repository checks and performs a normal push. Its sources, tests, registration and real process identity are included in the publication. The original failed controller records are preserved. The completion receipt identifies this new process, not the exited original controller.
