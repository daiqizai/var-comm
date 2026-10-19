# Adaptive N2048 B3: frozen seven metrics and source-paired inference

This independent entry consumes only completed B3 actual native pixels. It does
not send packets, choose policies, run tokenizers, recover images or train.
It evaluates the original external_common100_v1, two frozen adaptive methods,
4 and10dB and seeds6201/6202/6203:1200logicalframes. All link and source-parser
failures retain their source/noise denominator as the fixed receiver gray image.
The existing19dB fixed-full reference remains a separate experiment.

Paths are adaptive O/b3_metrics_v1 and L/b3_metrics_v1. `prepare.py` verifies the
B3 transport registration, closed native outputs and host-specific closure
evidence, and deduplicates by exact source archive and actual float32pixel hash.
Cache reuse comes only from the completed original ARPC A3 and fixed-full N2048
seven-metric runs, after checking numerical metric identity and array equality.
No marginal means or endpoint labels are used as a pair-cache identity.
Native completion must expose inputs,outputs,native_actual_waits,logical_pixels,
process_closure_status=ALL_NATIVE_PROCESSES_GONE and process_closure_evidence.
The actual native array key is used; no quantization or color conversion is added.

`adapter.py` pins the original N2048 metrics.py SHA and extracts only its four
existing functions csvwrite,load_image,ScoreLedger,score. Exactly two literal
logical denominators1800 become1200 and the3500s resource guard becomes14340s
inside a14400s finite engineering wait. An inverse AST proof establishes that
all other metric,preprocessing,model and cache mathematics are unchanged.
The original four and supplemental metric implementations,weights,metadata,
package versions,FP32/TF32 settings and DreamSim preprocessing remain frozen.
Runtime uses the previously verified H800 GPU0 CUDA mapping and a12GiB allocator
cap,8GiB starting free,2GiB running free,6intra/2interop threads. These resource
ceilings are documented, not claimed to reproduce inference latency experiments.
Recheck the actual mapping and free memory before dispatch. A zero-missing-pair
run performs only CPU reuse and does not construct models.

`statistical.py` requires exactly three noise values per method/source/SNR.
`worker.py statistics` generates one10000x100 draw-index array withseed2026101902,
after source-wise noise averaging. It saves the indices and direct paired
NeST-minus-ARPC percentile95intervals. No interval is produced by subtracting
marginal endpoints. Negative differences and intervals crossing/touching zero
are retained. No further candidate,noise,source or bootstrap round is triggered.

Root first reviews and qualifies new code with synthetic tests; do not repeat
old scientific qualification or actual scores. Then, after upstream actualexit0
and process closure, run metadata preparation once:

```text
CPY -B E/b3_metrics_v1/prepare.py --expected-native SHA
```

The resulting registration explicitly records unique,reused and new pairs.
For the two finite actual stages, root preserves a durable outer supervisor:

```text
CPY -B E/b3_metrics_v1/worker.py score --expected REGISTRATION_SHA
CPY -B E/b3_metrics_v1/worker.py statistics --expected REGISTRATION_SHA
```

The worker itself runs the registered original visualPython child, records
exactargv,PID/start_ticks,stdout/stderr and actual bounded wait, and owns the
adaptive execution.lock. Run only the next explicit stage after actualwait0
and process gone. Existing directories and any unknown failure are retained
and diagnosed; there is no automatic retry or silent reset. Successful output
files alone do not establish process closure. Root performs final full-manifest
verification and manuscript/report delivery after the statistics child exits.
