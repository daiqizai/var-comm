# Closed-result report and checkpoint inventory

`report_closed_results.py` is local preparation only until the correct existing summary has actually finished. It imports only the stdlib input guards from `plot_closed_results.py`; importing that file does not import Matplotlib or perform any work.

After six scientific jobs and both controller chains have closed as registered, and after the one lineage-aware CPU summary has exit0/gone, deploy the report source beside the unchanged plotting source. Run with CUDA hidden, the existing CPY interpreter and `--results O --run L`, recording actual process/argv/source hashes/stdout/stderr/wait in a new exclusive `L/report_dispatch_v1`. Output `O/report_v1` is exclusive. Existing output or failure requires diagnosis; never repeat a completed summary.

The report verifies the sealed summary manifest, copies saved point estimates and paired intervals into readable tables, and creates the six-job compute/recovery inventory. It reads closed completion/wait metadata and rechecks checkpoint existence/size; checkpoint SHA validation is explicitly attributed to the completed CPU audit. It does not load checkpoints, rehash every cache, perform inference, recompute scores, or generate another bootstrap bank.

The report lists all endpoints and all contrasts for both individual seeds and their mean. It keeps frozen-Dc and equal-update-control comparisons separate. The root must still read the actual full results and write a balanced short interpretation, inspect the actual figures, perform repository checks, and publish normally. A generated report is not a visual review or a publication receipt. No resumed update or cross-device checkpoint test is claimed.
