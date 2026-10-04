# Qualification evidence and registration correction

The frozen execution README incorrectly says calibration rows contain
reconstructed floating-point image hashes. They do not. The actual check
compares every saved field of all 12,420 rows per benchmark mode, including
PSNR, LPIPS, DINO, latent error, link decisions, and waveform/observation hashes.
This establishes exact recorded-row parity. It does not establish pixelwise
image parity.

The registered README remains an immutable execution input while the current
workers run. This correction and the published benchmark report describe the
actual verification scope; no scientific computation or acceptance condition
has been changed.

Four workers split six benchmark sources as 2/2/1/1. The measured time includes
the uneven tail. The small benchmark chooses an execution setting; its speedup
is not a guarantee for every subsequent image or study. Estimated remaining
time in the decision receipt covers calibration compute and one startup only.

See [measured results](../../reports/m1_gpu_workers_20261004.md).
