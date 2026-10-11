# Independent interim baseline diagnostic

The user requested evaluation without pausing either training branch. This separate process pins P40k/P80k and Swin80k/Swin150k, original calibration first100, N1024, all six registered SNRs and three original noises. It performs7200 method frames and four frozen metrics with no optimizer, no holdout images, no bootstrap and no use for selection/stopping. It is not the complete1000-source milestone or final500-source evaluation.

The completed result is results/baseline_strength_20261011/interim_cal100_v1/REPORT.md. Original bound training files were read and remain unchanged. Swin paid metadata and all failures are retained. The report validates4200 overlapping original-calibration outcomes before making comparisons. Swin80k at19dB remains out of range; the new arm includes the disclosed optimizer reset, longer training and wider SNR coverage.

run_interim_evaluation.py requires the existing frozen_visual_py310 environment, fixed asset roots/checkpoint hashes and an explicitly selected GPU0 with adequate free memory. The actual detached owner used CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0, CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONDONTWRITEBYTECODE=1, PYTHONNOUSERSITE=1, six CPU threads, niceness10, a16GiB allocator cap, and personal temporary/cache paths under liberai-tmp/liulu. Its actual launch and waited exit receipts are retained with the results. No repeated scheduler was created.

Do not rerun the completed registered output. A deliberate reproduction needs a new versioned output and temporary directory, a new registration and a fresh device check; use the existing result for ordinary inspection. The independent CPU report entry can regenerate the report/PNG/SVG from the completed CSVs and actual exit receipt:

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python experiments/baseline-diagnostics-20261011/report_interim_evaluation.py

Use the frozen environment Python and a personal MPLCONFIGDIR when regenerating. Model weights, environments and pixels remain outside Git. The plotting entry performs no inference or bootstrap.
