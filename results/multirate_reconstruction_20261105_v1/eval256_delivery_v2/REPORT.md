# Early-stopped source-codec comparison

User-authorized early stop. The original 150000-update schedule is incomplete. The last safely saved checkpoint is evaluated; no test-based checkpoint or hyperparameter selection.

The 256 test uses the same 500 held-out source IDs and fixed center crops. The 512 test is NOT RUN under the latest user instruction. All on-wire wrapper and entropy bytes count. New codec has a 136-bit wrapper; original RDVQ has a 72-bit wrapper with a fixed shared resolution. These are system-level actual source-packet comparisons, not equal-overhead ablations or radio-PHY results.

The original 256 codec uses identity-verified existing packets, arrays and metrics; one fixed cached pair per shard revalidates the metric route. The new codec uses its whole-image representation. No 512 quality claim is made. No favorable-result selection or parameter scan.

| Resolution | Budget | Metric | New minus original [95% source-paired CI] |
|---|---:|---|---|
| 256 | 512 | psnr_db | +2.119081 [+2.018027, +2.218280] |
| 256 | 512 | lpips_alex | -0.106018 [-0.112292, -0.099680] |
| 256 | 512 | dists | -0.087555 [-0.092251, -0.082664] |
| 256 | 512 | dreamsim | -0.096092 [-0.105314, -0.086707] |
| 256 | 768 | psnr_db | +2.259899 [+2.164916, +2.355921] |
| 256 | 768 | lpips_alex | -0.077829 [-0.083427, -0.072107] |
| 256 | 768 | dists | -0.053637 [-0.057475, -0.049777] |
| 256 | 768 | dreamsim | -0.086712 [-0.095325, -0.078126] |
| 256 | 1024 | psnr_db | +2.144185 [+2.042107, +2.244096] |
| 256 | 1024 | lpips_alex | -0.055645 [-0.060602, -0.050618] |
| 256 | 1024 | dists | -0.029545 [-0.032611, -0.026448] |
| 256 | 1024 | dreamsim | -0.063695 [-0.071371, -0.056027] |
| 256 | 1536 | psnr_db | +1.389806 [+1.274141, +1.500284] |
| 256 | 1536 | lpips_alex | +0.004002 [-0.000434, +0.008422] |
| 256 | 1536 | dists | +0.004995 [+0.002471, +0.007532] |
| 256 | 1536 | dreamsim | +0.008877 [+0.002517, +0.015253] |
| 256 | 2048 | psnr_db | +0.129893 [+0.035633, +0.215436] |
| 256 | 2048 | lpips_alex | +0.068921 [+0.066286, +0.071585] |
| 256 | 2048 | dists | +0.038726 [+0.037116, +0.040364] |
| 256 | 2048 | dreamsim | +0.078624 [+0.074794, +0.082541] |
| 256 | 3072 | psnr_db | -0.389527 [-0.469617, -0.316841] |
| 256 | 3072 | lpips_alex | +0.089150 [+0.086362, +0.092006] |
| 256 | 3072 | dists | +0.048707 [+0.047137, +0.050279] |
| 256 | 3072 | dreamsim | +0.098866 [+0.094922, +0.102835] |
| 256 | 4096 | psnr_db | -0.383567 [-0.463844, -0.310759] |
| 256 | 4096 | lpips_alex | +0.089803 [+0.086982, +0.092662] |
| 256 | 4096 | dists | +0.049567 [+0.047985, +0.051162] |
| 256 | 4096 | dreamsim | +0.099622 [+0.095667, +0.103587] |
| 256 | 8192 | psnr_db | -0.388869 [-0.469231, -0.315573] |
| 256 | 8192 | lpips_alex | +0.089195 [+0.086390, +0.092036] |
| 256 | 8192 | dists | +0.049017 [+0.047457, +0.050589] |
| 256 | 8192 | dreamsim | +0.098962 [+0.095085, +0.102849] |

Positive PSNR favors new; negative LPIPS/DISTS/DreamSim favors new. All mixed and unfavorable values are retained. NULL remains gray and in the denominator. Unknown execution failures block that resolution and are not converted into successes or gray outputs.

{
  "256": {
    "status": "done",
    "logical_positions": 8000,
    "sources": 500
  }
}
