# Fixed scope and stop decision

Base: `2f13240b9338c8e79f14d6087e221714d3259c6b`. Authoritative original registration: `results/shared_codeword_fit_20261028_v1/scope.json`, SHA256 `8868ea94ba3f9f595e0fc239edf24e3f1cc4694326856d1dadec9e730556536a`.

Reuse the original 32 calibration/development sources without renumbering. TX design SNR is 2 dB; evaluation uses 0, 2 and 4 dB, three fixed noises each. N=1024, header68+body956, Es=2. Five new arms each288 actual frames, plus two exact historical arms each288: 1440 new,2016 total. ALL/SEQ/SEL replay the same reception and add no channel events. Source and image caches require exact received content, frozen model and numeric identity.

Controls: original strong single; original best three short blocks; token-wise S_FILL; fixed-source minimum-fitting-k S_PROTECT; unchanged three information blocks shared in one long codeword; compact fixed K12; compact token-wise K. No training. Caps:16 GPU-device hours,16 CPU-worker hours; per-stage registrations impose tighter limits.

Qualification precedes image/metric review, which precedes actual frames. Main selection criterion is equal-SNR DINOv2 cosine J with failures retained; PSNR, LPIPS-Alex, DreamSim and CLIP-I are all reported. First average three noises per source, then sources and the three SNRs. Historical arms have different noise namespaces and are source-paired only. No new bootstrap for this developmental screening.

Result gate: no shared candidate exceeds the stronger single control S_FILL. Stop A before the128-source confirmation, UEP, calibration or formal evaluation. The finer single-stream control remains a useful reference for route B. Any UEP or formal study would require a new justified gate; better utilization alone is insufficient.
