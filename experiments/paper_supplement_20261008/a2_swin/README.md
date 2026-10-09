# A2: frozen Swin implementation and support checks

This independent diagnostic uses the original registered first 20 calibration
sources, the exact 80,000-step checkpoint, C=6 and 13 dB. It does not train,
change policies, read development/holdout images, invoke header/PHY decoding,
or repeat the published 500-source evaluation.

The author `SwinJSCC.forward` and `Channel.forward` are called without replacing
either implementation. Hooks record the actual encoder output/mask and the
actual noisy masked decoder input. The latter is projected to the wrapper's
active IQ coordinates, with the same mask and float32 power, and passed to
the existing `SwinCodec.receive_data`. Both outputs are compared before display
quantization and after rounding to uint8. All 20 cases are retained.

Native normalization uses complex energy 1 and per-axis noise standard
deviation `1/sqrt(2*gamma)`, then restores `sqrt(2*P)`. The paid adaptation uses
complex energy 2 and standard deviation `1/sqrt(gamma)`, then restores
`sqrt(P)`. Projection of the actual noisy native decoder coordinates compares
the same observation; separate RNG draws are not used for the two arms.

The exact mask and power are deliberately free context **only for this
implementation diagnostic**. It is not a paid performance baseline, a test of
header reliability, or evidence that 256 header symbols are globally optimal.

The frozen checkpoint was trained only at C=6/C=13 and SNR 1–13 dB. C=7 is
code-executable but untrained and is recorded as UNSUPPORTED for the proposed
trained-channel comparison. C=13 requires 1664 body symbols and cannot fit
N1024. C=6 uses 768 body symbols; reducing its header does not increase source
capacity without an unsupported change of channel count. No weaker-header
policy was adopted, so no 200-source head/body search or holdout repeat is
performed. The original paid results are reused and the 19 dB training- and
calibration-range warning remains.

Run on the original server in its registered Torch 1.12.1 environment, after
the coordinator confirms the GPU is idle:

```bash
PY=experiments/external-baseline-positioning-20260916/.venv/bin/python
$PY experiments/paper_supplement_20261008/a2_swin/native_parity.py prepare --root /home/liulu/projects/VAR_COMM --output /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a2_swin_native_v1
$PY experiments/paper_supplement_20261008/a2_swin/native_parity.py run --request /home/liulu/projects/VAR_COMM/outputs/PAPER-SUPPLEMENT-20261008/a2_swin_native_v1/request.json
```

`prepare` freezes identities, hashes and tolerances before inference. It does
not initialize CUDA. `run` has a process lock and verifies completed outputs
before reusing them. A failed run is retained and requires diagnosis rather
than silently restarting. Preparation alone does not constitute completion.

Deliverables: `native_parity.csv`, `native_parity_summary.json`,
`support_scope.json`, `environment.json`, 20 case receipts and pixel/IQ
archives, and a completion file binding their hashes. Paid PHY decode count
is zero; 20 native AWGN forwards are explicitly recorded.
