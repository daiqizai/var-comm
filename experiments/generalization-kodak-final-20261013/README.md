# Frozen Kodak24 final-baseline supplement

This independent supplement follows `results/generalization_kodak_20261009/protocol.json`: all24 original RGB256×256 center crops, N1024, SNR4/10/19dB, three original noise labels2001/2002/2003 and the original method-specific noise generators. Shared labels do not imply identical frame-level observations. Checkpoints and policies remain frozen from ImageNet; Kodak is never used for selection, calibration or training.

Only P100k and Swin200k produce new transmissions, channel observations and receiver reconstructions:216 frames each,432 total, batch1. P uses the frozen source VAE and the original continuous decoder. Swin uses the final200k encoder/decoder with its paid256-use auxiliary header and768-use body. Transmitter auxiliary context is compared with received context only after reconstruction; it is never supplied to the receiver. Header rejection and wrong accepted headers remain in the evaluation.

NeST-Com raw and adaptive-resolution BPG+LDPC are copied from their verified historical scores. Their source pixels, frozen models/configurations, requested metric mathematics and weights must match before reuse. The original raw BODY_CRC_REJECT_KEEP outcomes are retained. Historical results, files and intervals are not changed.

The four scores are PSNR, LPIPS-Alex, DINOv2-L(ViT-L/14) cosine and ConvNeXt source-prediction agreement. Agreement is not classification accuracy. The H800 scorer has its own recorded runtime identity; matching frozen mathematics and weights is not a claim of bitwise equivalence to the original host.

`run_final.py prepare` registers the finite experiment without model calls. A unique `owner.py inference` owns one actual child and records its exit. `owner.py scoring` can run only after the actual inference exit0 and complete432-frame grid. No attempted scientific stage automatically retries. Code/model bindings are rechecked; do not edit bound code during an active run.

`postprocess.py` reuses the original Kodak source-level interval function, three-noise source means,10000 bootstrap indices and seed2026100901. It computes24 new summary intervals and24 new NeST-minus-final-baseline intervals. Original raw/BPG summary/source/paired row bytes are preserved in the combined exports. `statistics_completion.json` is written before plotting; `--render-only` can recover layout without recomputing statistics. See [README_STATS.md](README_STATS.md) for exact output schemas and reproduction.

All light outputs are under `results/generalization_kodak_final_20261013/`; fresh RGB, IQ and noise arrays/logs remain in the registered personal temporary runtime directory. No entropy coding, HiFi, extra bandwidth, extra crop, training or tuning is included. Execution is an ordinary bounded evaluation on a shared GPU, not a timing experiment or a timing-resource qualification.
