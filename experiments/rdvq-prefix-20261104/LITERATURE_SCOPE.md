# Mechanism and prior work

This is a new study; the existing NeST-Com WCL artifacts remain frozen.

| Work | Existing mechanism relevant here | What this study must test separately |
|---|---|---|
| [RDVQ](https://arxiv.org/html/2604.10546v2) | Differentiable soft-index rate loss jointly trains image transforms and an autoregressive entropy predictor; the predictor also supports limited prefix completion. | Matched token-count versus actual-source-packet-budget prefix training, using the same initialization, data, losses and training exposure. VQ, rate loss and suffix generation are inherited. |
| [GIST](https://arxiv.org/html/2610.15515v1) | Adds a DINOv3 semantic-error control to KARL's reconstruction-error conditioning and trains adaptive halting with two forward passes. | GIST is directly relevant to adaptive token allocation. Our encoder instead produces one budget-independent sequence; the controlled quantity is the charged, independently decodable source packet. A general adaptive-tokenization claim would be too broad. |
| [AdapToC](https://arxiv.org/html/2610.03822v1) | Selects complete tile-level drop/protection policies using content, CSI and an exact digital ledger; a contextual receiver uses reliability evidence. Its evaluation shares a frozen VQGAN tokenizer and decoder. | Content/CSI adaptation and exact accounting are established ideas. Our first test concerns training the codec for actual-bit prefixes. A later controller must be compared with the same strong SNR-only controller for both R2 and R3. |

The full GIST and AdapToC HTML methods were available on 2026-11-04 and were
read for this revision; the attached plan's earlier abstract-only limitation
does not describe the present review. No external model replication is claimed.

The pinned RDVQ implementation uses tensor-rANS, not NeST's arithmetic coder.
No arithmetic termination identity or monotonicity theorem is transferred.
The public checkpoint contains model state only; missing discriminator and
optimizer state are recorded rather than invented. The matched first-stage
fine-tuning starts fresh optimizers and uses the common pre-adversarial window
(upstream discriminator start 25,000, above this study's maximum 20,000 updates).
All arms therefore use the same MSE, official VGG-LPIPS, fixed-codebook
commitment/stability terms, and differentiable rate loss, with no active GAN
term in this first batch. This is a declared adaptation, not a replay of the
authors' complete training trajectory.
