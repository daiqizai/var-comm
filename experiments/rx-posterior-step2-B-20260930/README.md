# Step2 B: single P_low continuation and receiver evaluation

The user explicitly authorized B after the completed Step2 A result. This run continues the C priority first seed2026092304 P4084 selected27500 and keeps N4084/E8168, PureContinuous, the visual Encoder/VQ, Dc and VAR unchanged in structure and identity. Only the PureContinuous parameters receive updates.

The actual original AdamW settings match the proposed B settings: lr2e-4, weight decay1e-4, betas(.9,.999), eps1e-8, effective batch16, microbatch4, clip1. The full populated selected27500 optimizer, source order and CPU/CUDA/channel RNG states are resumed. There is no fresh optimizer or learning-rate schedule. The intentional change is uniform training SNR sampling from -5/-2/1/4dB.

## Frozen rules

See [protocol.json](protocol.json) and the original [execution plan](EXECUTION_PLAN.md). The original plan's default A-only instruction is superseded by the user's subsequent explicit request to execute B.

Original20000 training sources remain disjoint from the1000 calibration sources. The same original image/latent loss is used. Full1000-source calibration at all four training SNRs and three original calibration noises selects among the initial parent and every2500-update checkpoint. Minimum utility wins, with ties choosing the earlier checkpoint. The first milestone is10000 added updates. Each later10000 extension requires at least0.2% improvement in both latest2500 intervals, measured from current full-calibration utilities. Maximum added updates:30000.

After selection, the fixed200-source receiver calibration estimates new error/fusion/lambda/BYPASS parameters for P_low. The reused100-source development set is opened only after these policies freeze. G2 is reported pointwise at -5/-2/1dB,4dB is supporting, and13dB is an extra out-of-training-range side-effect diagnostic. Each P_low method shares the same P_low waveform and observation. Old P4084 comparisons share source/noise seed/N/E and retain separate waveforms and observations.

## Execution

```bash
python3 experiments/rx-posterior-step2-B-20260930/test_protocol.py
python3 experiments/rx-posterior-step2-B-20260930/train.py --qualification-only
python3 -u experiments/rx-posterior-step2-B-20260930/supervisor.py
```

The detached supervisor runs resumed-parent GPU qualification, training, frozen evaluation, then repository checks and publication. It retries only safe resource pauses (exit75); unknown failures stop with logs preserved. Candidates, failed checkpoints and receipts remain in the new output directory. The original P4084/A assets and paused queues are preserved.

Outputs: `outputs/RX-POSTERIOR-STEP2-B-20260930-R1/`. Results: `results/rx_posterior_step2_B_20260930_R1/`. Large model/tensor artifacts stay outside Git. The publication stage follows the user's request to submit results to the repository and verifies required CPU checks plus the remote commit SHA.

Scientific completion requires both the training and evaluation completion receipts; a running supervisor is not a completed result. Final publication additionally requires `publication.json` with status `PUSHED`. No P_low_perc or subsequent training study is authorized by this pipeline.
