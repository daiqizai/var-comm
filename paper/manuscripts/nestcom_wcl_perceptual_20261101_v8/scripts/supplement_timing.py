"""Insert only actually closed E5 records; keep incomplete drafts explicit."""
import csv,json,hashlib

def enrich_timing(text,root,table):
    p=root/'inputs/e5'
    if not (p/'execution_closure.json').exists():return text
    c=json.loads((p/'execution_closure.json').read_text())
    assert c['actual_wait_returncode']==0 and c['owner_gone'] and c['child_gone']
    assert c['summary_sha256']==hashlib.sha256((p/'timing_summary.csv').read_bytes()).hexdigest()
    with (p/'timing_summary.csv').open(newline='')as f:rows=list(csv.DictReader(f))
    assert len(rows)==48
    short={'WHOLE':'WHOLE','Token_NeST':'NeST','Ada':'Ada','ARPC':'ARPC','BPG_LDPC':'BPG','SwinJSCC':'Swin','Latent_JSCC':'Latent'}
    section=r'''All 16 groups and 32 TX/RX stages completed with actual zero exits and no live owner or child. There are 5120 fresh timing records: 2048 warmups and 3072 measured boundary spans. The 14 previously completed groups were retained; only ARPC and BPG required the explicitly authorized final continuation. Failed setup/measurement attempts remain in the execution/resource record and are excluded from completed-trial latency statistics, rather than converted to zero latency.

ARPC retains its original model runtime and invokes the unchanged qualified CPU PHY through an in-memory socket. Per-frame IPC is included in the measured boundaries; both processes share the same two-CPU affinity. A Python module-name collision was repaired by prebinding the exact standard-library module in the model process and its compiler subprocesses. BPG retains the original codec and search; its child-process CPU-count adapter now matches the registered two-core affinity. Neither repair changes model weights, numerical precision, codec options or transmitted information.

\subsection{All measured groups}
Values below are milliseconds. Means average the three measurements within each of 32 sources; median and p95 are calculated from the 96 measured calls. Total pairs the TX and RX spans for the same source and repetition. The success count refers to measured RX calls; conditional RX means exclude rejected receptions and are not substituted for overall RX means.
'''
    body=[]
    for r in rows:
        label=short[r['method']]
        if r['group'].startswith('granularity_'):label='Grid '+r['grid'].title()
        body.append([label,r['N'],r['snr_db'],r['direction']]+[f"{1000*float(r[k]):.2f}"for k in ['equal_source_mean_seconds','call_median_seconds','call_p95_seconds']])
    section+=table(['System','$N$','SNR','Span','Mean (ms)','Median','p95'],body,'lrrlrrr')
    body=[]
    for r in rows:
        if r['direction']!='RX':continue
        label=short[r['method']]
        if r['group'].startswith('granularity_'):label='Grid '+r['grid'].title()
        body.append([label,r['N'],r['snr_db'],r['successful_RX_calls'],f"{1000*float(r['success_conditional_RX_mean_seconds']):.2f}"if r['success_conditional_RX_mean_seconds']else'--'])
    section+=table(['System','$N$','SNR','Success / 96','Conditional RX (ms)'],body,'lrrrr')
    section+=r'''\subsection{Computation and interpretation}
The matched grid comparison uses ceiling $(8,126)$; its token arm reuses the identical NeST timing group at $N=1024$, 10 dB. The main WHOLE timing group has its own family target and is not the matched-grid WHOLE control. The timed implementation constructs all candidate streams and scans the complete list for feasibility before selecting the longest one. Its recorded legacy field \texttt{length\_checks} is the selected endpoint's descending rank, not the actual number of tests. The corrected accounting records 5/20/39/352 candidate tests for matched Whole/Quarter/Eighth/Token, separately from the early-stopping comparisons used by the cached quality control in S3. Both rules select the same endpoint; the timing includes this implementation's full scan.

Fresh source-probability traversals, CDF construction steps, independent finalizations, actual tests and endpoint ranks are in \texttt{source\_fitting\_accounting.csv}, derived from the bound code and per-trial witnesses. Stage-level instrumented model/PHY calls are in \texttt{model\_call\_accounting.csv}; they are not exhaustive operator counts. Counters include warmups and are not interchangeable with latency. No prior source-output cache replaces timed model work.

This software implementation includes actual CDF construction, independent arithmetic termination and real digital decoding. The result is a deployment-cost comparison on this host and these original runtimes, not an optimized-hardware throughput bound. Zero observed failures in a timing cell is not a reliability guarantee. No confidence interval is fabricated from the timing repetitions.
'''
    old=r'\textbf{Draft: unified timing remains incomplete. No historical timings or partial measurements are substituted for the required 16 groups and 32 TX/RX stages.}'
    assert text.count(old)==1;text=text.replace(old,section)
    old=r'\textbf{Working draft.} E1--E4 and E6 contain complete results. Unified timing and final document acceptance remain pending.'
    assert text.count(old)==1
    return text.replace(old,'Complete experimental results and implementation details. Numerical tables use the same frozen CSV sources as the main manuscript.')
