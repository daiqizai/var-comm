# Swin implementation and support check

Actual execution on 2026-10-09: **20/20 original calibration sources PASS** at 13 dB and C6 with the original 80k checkpoint. The author native forward was run; its actual noisy decoder input was projected into the paid adaptation's IQ representation. The two decoders used the same mask, float32 power and actual observation. The maximum absolute RGB difference was 5.364418029785156e-7, and the maximum display uint8 difference was one level. This was a free-context implementation diagnostic, not a paid-information performance baseline.

`native_v1/native_parity.csv` contains all 20 cases. `native_v1/completion.json` binds 44 output files, including the original float images and received IQ; all 44 were downloaded and hash-verified locally. No training, paid PHY decode, policy update or holdout evaluation was performed.

The checkpoint trained C6 and C13 at integer SNRs 1 through 13 and was calibrated at 1/4/7/10/13 dB. At total N1024, C13 requires 1664 body uses and is infeasible; C7 is accepted by the code but was not trained. The supported frozen adaptation therefore remains C6, 768 body uses plus 256 protected header uses. Merely shortening that header without a supported additional body channel creates padding rather than extra source capacity. No new header/body policy was adopted and the original 500-source results are reused.

This check does **not** establish global optimality of the 256-use header, full convergence of the 80k checkpoint, or support for untrained channels. The 19 dB points remain outside both training and calibration ranges. Paper labels should remain “SwinJSCC-80k (adapted)”; do not call this an official optimal or fully converged benchmark.

Reproduce on the original host using the two commands in `experiments/paper_supplement_20261008/a2_swin/README.md`. A completed unchanged request is reused, not re-executed.
