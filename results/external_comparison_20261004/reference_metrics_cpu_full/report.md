# Metric reference levels

100 fixed development originals, six conditions, one paired observation per source. No channel or noise triplication.
Same-class donors come only from the original training/calibration populations. All donor identities were selected before scoring.
These values illustrate the scale of each metric; they are not tuned thresholds or communication results.
Execution device: cpu. Weights and preprocessing match the frozen suite; backend identity is recorded separately.

| Condition | PSNR | LPIPS | DINOv2-L | Semantic error | Confidently wrong |
|---|---:|---:|---:|---:|---:|
| identity | +infinity (identical pixels) | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| unrelated | 8.2373 [7.9039, 8.5785] | 0.7878 [0.7740, 0.8017] | 0.0155 [0.0099, 0.0210] | 1.0000 [1.0000, 1.0000] | 0.2600 [0.1800, 0.3500] |
| same_class | 8.6865 [8.2893, 9.0882] | 0.7509 [0.7359, 0.7661] | 0.4728 [0.4158, 0.5277] | 0.2300 [0.1500, 0.3100] | 0.0600 [0.0200, 0.1100] |
| gaussian_blur_sigma1 | 28.5590 [27.8314, 29.2931] | 0.2378 [0.2202, 0.2557] | 0.9923 [0.9906, 0.9938] | 0.0800 [0.0300, 0.1400] | 0.0000 [0.0000, 0.0000] |
| gaussian_noise_sigma2_255 | 42.1767 [42.1510, 42.2083] | 0.0046 [0.0035, 0.0059] | 0.9978 [0.9973, 0.9983] | 0.0200 [0.0000, 0.0500] | 0.0000 [0.0000, 0.0000] |
| jpeg_quality90 | 37.6901 [37.1098, 38.2806] | 0.0039 [0.0035, 0.0044] | 0.9965 [0.9958, 0.9971] | 0.0600 [0.0200, 0.1100] | 0.0000 [0.0000, 0.0000] |
