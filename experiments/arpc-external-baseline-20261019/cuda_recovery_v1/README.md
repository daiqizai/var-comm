# A1 CUDA admission recovery

The first A1 TX stopped at CUDA initialization before creating its call ledger,
loading any model, or accessing a source image. Its registration, original frozen
sources, caches, errors and all actual exit-1 receipts remain unchanged.

On this host NVML identifies the selected device as
`GPU-535fda57-9774-5902-7375-bcab46fb9895`, while CUDA reports
`e3133767-ae42-afba-3515-0873255753fc`. A zero-model diagnostic proved that selecting
the NVML UUID fails, but CUDA index 4 initializes. An independent CUDA-driver query
confirmed PCI `0000:A1:00.0`, matching the originally selected NVML GPU 4.

This separate registration only changes CUDA admission to index 4, with a fail-closed
PCI and CUDA UUID check in each TX/RX process before the original native entry runs.
It does not alter the original native source, weights, numerics, source selection,
wire format, calls, limits, audit, RX data boundary, or reference comparisons.
The original owner implementation retains registration checks, locks, serial stages,
waits and failure records. New control paths retain the failed attempt separately.
The original absolute A1 deadline is 1792361947.8323371; it is never reset.

The CUDA runtime wrapper reads only the image-independent public configuration.
It supplies no source/CDF/token data to RX. Recovery is allowed only with verified
original failed waits, no original call ledger, exact diagnostic/source pins and
the coordinator's resource-only authorization. No successful scientific call is
replayed; all original scientific counts were zero. A further failure is preserved
and stops successors. Actual A1 passes and visual/resource review still gate A2.
