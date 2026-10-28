# N512 frozen actual-frame experiment (2026-10-28)

This independent implementation uses 68+444 complex symbols, nominal Es=2, actual Sionna LDPC/QAM, paid NULL frames, fitted terminated source streams and independent RX. Historical N1024 files remain unchanged. The protocol, candidate pool, native inputs, code hashes and resource limits are recorded in `results/nestcom_n512_20261028_v1/protocol.json` and the per-role registrations.

**Scientific work is complete. Do not rerun successful launchers.** The original `science_chain` waited without launching stages and was retired. The completed route uses `science_chain_v2`, `stage_operations_v5` and `pipeline_lineage_v3`. Earlier source versions are immutable evidence. Qualification, PCI and path failures remain preserved; amendments identify first calls and exact reuse.

The stages were source/numeric-route qualification, 12-layout qualification, a 100-source pilot, cal1000 three-noise reevaluation, policy freezing, 13,500 formal common500 outcomes, four metrics and source-paired statistics. There was no training. DINO selected policies on calibration data only. All formal PSNR, LPIPS, DreamSim and CLIP-I points are reported, including failures and adverse findings.

Read the [complete report](../../results/nestcom_n512_20261028_v1/delivery/REPORT.md), [execution evidence](../../results/nestcom_n512_20261028_v1/closed_execution_evidence/) and [review](../../results/nestcom_n512_20261028_v1/delivery_review.json) before any future work. Closed registrations use exclusive output paths and must not be launched again. Caches, raw logs and native arrays remain outside Git on PFS. Publication checks are engineering checks, separate from the recorded experiment.
