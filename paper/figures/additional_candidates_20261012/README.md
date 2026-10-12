# Additional reconstruction candidates 09–16

All eight sources were recovered from the original server caches and verified by the existing collector. No inference, simulation, scoring, bootstrap or training was rerun. N1024,13 dB; historical development only, latent40k/Swin80k, fixed first noise2001 for latent/Swin/HiFi and6201 for digital. This is not final100k/200k holdout evaluation. All missing and unfavorable outcomes are retained.

Candidate09–12 (group03): source33 hippopotamus,41 child,52 respirator,60 singer.
Candidate13–16 (group04): source64 fabric,87 volleyball,92 broccoli,95 pastries.
Candidate06/source21 remains available in the previous group02; no final user selection has been applied.

Both five-column and four-column panels have PDF,SVG,600dpi PNG. Native individual PNGs and verified display NPZ bundles are in the local/Lu_Data delivery, outside Git. Source provenance and received-token pairing checks are recorded per group. All four PNG panels were actually opened for visual review.

Reproduction on the original cache host uses the existing script, for groups3 and4:

```sh
python3 scripts/plot_paper_hifi_development_examples.py --group 3 --collect --render
python3 scripts/plot_paper_hifi_development_examples.py --group 4 --collect --render
```

Collection reads existing caches only. The original server outputs were exported to a separate directory and transferred with archive SHA451feb0ae80067a2f56c59395a829584f34ecd293a89239e2ba9594792b0d16f. All82 transferred files match their manifests on both local disk and Lu_Data.
