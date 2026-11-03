# Timing execution history

These historical execution notes were moved from the supplement without altering measurements or archived failures. The supplement retains the final resource protocol, independent decoding validation, measured costs and paired quality results.

## Preserved operational notes

The previous affinity [38,39] selected two SMT siblings of one physical core, although the earlier manuscript called them two CPU cores; its intra-op pool had six threads. This was the same allocation for all methods. The corrected run replaces the available main timing cells; Ada's cells remain unreported rather than presenting its serial adaptation as an author-aligned implementation. The initial GPU0 session was interrupted by a foreign GPU diagnostic process; all GPU0 timings are excluded. GPU6 supplies every final GPU cell, and only the completed, exactly matching CPU-only BPG cells are reused. Prior resource use and interrupted-call ledgers are retained. No foreign process or GPU clock setting was changed.

Three unsuccessful compilation attempts (missing development headers, parent and compiler-child module-name collisions) precede the formal run. They produced no timed TX/RX results and are retained with their actual nonzero exits and resource use. Private matching Python headers and the existing child-namespace binding repair the environment without changing the qualified model calculation. These failures are not included in per-image processing times. All study attempts remain charged against the original absolute deadline and the separately authorized resource extension.

## Serial implementation audit row

```latex
Ada (audit only) & 7653.77 & 7888.77 & 15542.54 [15525.31, 15560.32]\\
```

This is an excluded serial-implementation audit, not Ada's matched implementation cost. Original logs, failed exits and ledgers remain in the timing and unified-calibration result directories.

## Earlier serial implementation and two-source audit diagnostics

```latex
The frozen Ada adaptation evaluates probabilities serially for all 256 source positions at TX. It executes conditional and unconditional branches, although only the conditional branch supplies entropy CDFs, and retains its FP32 math-attention route. RX decodes the prefix and continues generation along the same context. NeST predicts maps scale by scale. These model and implementation choices remain part of the measured costs.

Ada-TokenCom's published Section IV-E instead describes a single encoder prefill and \texttt{torch.compile}. Its Table IV reports 19 ms encoding and 1012 ms decoding on an RTX 5090 and Xeon Platinum 8260. The reported boundary comprises tokenizer, AR probability estimation, softmax and arithmetic coding, whereas our spans additionally include class prediction and digital-link processing. The paper also describes repeated full-prefix evaluation at RX to preserve entropy-model numerical consistency. Thus, neither its published numbers nor our serial-adaptation numbers establish a matched end-to-end speed comparison. In particular, the serial TX loop and unused TX branch are implementation overheads, not requirements of Ada's method. The published latency section does not specify enough numerical detail to assume that a different precision is responsible for the discrepancy.

An auxiliary diagnostic uses only the first two registered sources. It defers Ada's redundant per-token finite-logit audit until after timing and removes unused CDF audit hashes; for NeST it removes an unused completed-token hash. CRC, length/parser checks, arithmetic canonical re-encoding, actual generation and final output identity remain. The NeST TX path is unchanged and serves as a negative control. The following means compare this diagnostic with the original-audit measurements on the same two sources. Their session variation and tiny sample preclude replacing the 32-source results or subtracting these differences from them.

\begin{longtable}{@{}llrrr@{}}\toprule
System & Span & Original audits & Diagnostic & Difference (ms)\\\midrule\endhead
NeST & TX & 246.78 & 248.51 & +1.72\\
NeST & RX & 406.41 & 404.05 & -2.37\\
Ada (audit only) & TX & 7637.67 & 7664.81 & +27.14\\
Ada (audit only) & RX & 7872.74 & 7688.25 & -184.49\\
\bottomrule\end{longtable}
The measurements characterize the frozen implementations on a common resource allocation. They do not measure equally optimized kernels or an architecture-independent speed limit. Ada's raw measurements remain audit evidence only, with no NeST/Ada speedup claim. All diagnostic outcomes, including the unchanged-path control, are retained.
```
