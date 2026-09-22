# Isolated HiFi execution profiling

- This is the explicitly authorized separate experimental branch, not a replacement of the frozen1500-frame implementation.
- Do not edit vendor source, existing author adapters, checkpoints, frozen configs or old results. The direct-attention variant is process-local and limited to bypassing AttentionBlock's forced checkpoint wrapper.
- Preserve precision, actual observation, RNG seeds, reverse steps and model autograd flags. No full-path no_grad/inference_mode, no optimizer/training or shared environment changes.
- Validate complete RGB, matched-state input gradients, parameter immutability, peak memory and complete RX time. Keep instrumented timing separate from uninstrumented timing.
- Three fixed calibration cases only. No automatic sweep or switching of the original queue.
- The new20-source stage remains original author execution; all systems use the same100 transmissions, failures included. Retain the original1500-frame plan and all completed data without claiming it is complete.
