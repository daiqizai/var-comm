# Child-only x265 CPU-count adaptation

The unchanged bpgenc binary's first 256px/QP0 call timed out at 120 seconds.
No BPG timing case completed, and the output container remained empty. The
reserved first warmup, input image, empty output, traceback and real exit are
preserved. Previously completed NeST, P100k and Swin200k calls are reused.

A separate synthetic 256px test reproduced the stall: the old x265 created
64+64+52 worker pools from the host's 180 CPUs. It printed an encoded-frame
summary, but all remaining 117 threads waited on futexes and the process failed
to exit. The 32px synthetic input, which disables WPP/pools, exited normally.
The archived original `threadpool.cpp` shows that CPU count is obtained through
`sysconf(_SC_NPROCESSORS_ONLN)` without checking the inherited affinity.

The explicit LD_PRELOAD resource adapter affects only bpgenc children. That one
query returns the actual allocated CPU affinity count, with a strict six-CPU
check. Other sysconf queries delegate to libc. The existing encoder/decoder
binaries, presets, chroma, QP grid, source search, PHY and numerical math retain
their original bytes. Python, neural methods and bpgdec have no preload.
No system library, process outside this task, or shared machine setting changes.

Synthetic 32px and 256px encodes and decodes now exit0. The 32px stream is exactly
identical to the successful original. Original 256px byte parity cannot be
claimed because its hung process did not publish a BPG container. The adapted
256px log records a six-worker pool and two frame threads; small images retain
the encoder's no-pool behavior. This resource difference is explicitly reported.

One failed incomplete real encoder attempt precedes the measured grid and is
retained; no completed case is rerun. The new owner performs only BPG TX/PHY/RX
and the unstarted ARPC TX/PHY/RX. Counts remain 1440 completed full calls,
720 formal measurements, with this failed attempt disclosed separately.
It stops on an unknown failure and cannot automatically retry.
