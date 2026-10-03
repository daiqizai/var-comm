# Historical classification confidence

These are the same 6,600 measured frames selected in Step 0. Only frozen ResNet-50 ImageNet1K V2 softmax inference was added, on CPU with six threads.
Confident disagreement means reconstruction top-1 probability >= 0.5 and reconstruction prediction differs from the classifier prediction on the original image. Softmax confidence is not a calibrated probability of correctness.
Every original and reconstruction argmax exactly matched its historical measurement. Any mismatch stops the run; no image, preprocessing, batch or backend substitution is allowed.
First average the three registered noise repetitions for each source. Confidence intervals and paired differences use the same 10,000 bootstrap resamples of 100 sources (seed 20261002).
Paid-class digital rows are descriptive: the generation pipeline receives the true class. D0 and Dc remain separate protocols. These comparisons do not isolate a causal effect of bandwidth.
Source confidence is reported for each exact historical float reference. Original uint8 images match; legacy normalization may retain its registered 1-ULP float difference.

| Method | N | SNR | Confident disagreement [95% CI] | Original top-1 confidence |
|---|---:|---:|---:|---:|
| Digital arithmetic / Dc | 2048 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital arithmetic / Dc | 2048 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital arithmetic / Dc | 3060 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital arithmetic / Dc | 3060 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital arithmetic / Dc | 4084 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital arithmetic / Dc | 4084 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 2048 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 2048 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 3060 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 3060 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 4084 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Digital raw / Dc | 4084 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Legacy arithmetic / D0 | 3060 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| Legacy arithmetic / D0 | 3060 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| Legacy raw / D0 | 3060 | 7 | 0.3% [0.0, 1.0] | 0.4048 |
| Legacy raw / D0 | 3060 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| P | 2048 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| P | 2048 | 13 | 0.3% [0.0, 1.0] | 0.4048 |
| P | 3060 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| P | 3060 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
| P | 4084 | 7 | 0.0% [0.0, 0.0] | 0.4048 |
| P | 4084 | 13 | 0.0% [0.0, 0.0] | 0.4048 |
