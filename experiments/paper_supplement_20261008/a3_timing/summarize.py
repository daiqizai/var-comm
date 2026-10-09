"""Read-only audit and publication tables from completed A3 timing receipts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS={
    'RAW64_PARTIAL':'Partial-scale digital + VAR',
    'RAW64_WHOLE':'Full-scale digital + VAR',
    'P1024':'Latent continuous JSCC',
    'SwinJSCC80k':'SwinJSCC-80k (adapted)',
    'BPG_LDPC_native256':'BPG + LDPC (native 256)',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('x',encoding='utf8',newline='')as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def save(path,value):
    with Path(path).open('x',encoding='utf8')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def optional(value):return None if value is None or value==''else float(value)


def summarize(source,out):
    source,out=Path(source),Path(out)
    require(not out.exists(),'Use a new report output directory; preserve any existing snapshot')
    all_rows=[];timing_rows=[];storage_rows=[];audit=[];missing=[]
    for method,label in METHODS.items():
        directory=source/method;completion=directory/'completion.json'
        if not completion.exists():
            missing.append(dict(method=method,status='MISSING_OR_NOT_COMPLETE',expected=str(completion)));continue
        done=read(completion)
        require(done['status']=='COMPLETE'and done['source_count']==16 and done['SNRs']==[7,13,19], 'Unexpected completed A3 scope')
        require(done['method']==method,'Method pairing mismatch')
        for remote_path,expected in done['outputs'].items():
            local=directory/Path(remote_path).name
            require(local.exists()and sha(local)==expected,'Missing/changed completed output: '+str(local))
        rows=table(directory/'timings.csv');summary=table(directory/'summary.csv');storage=read(directory/'model_storage.json')
        require(len(rows)==192 and sum(row['phase']=='measured'for row in rows)==144,'Incomplete source/repetition grid')
        seen=set()
        for row in rows:
            key=(int(row['source_index']),int(row['snr_db']),int(row['repetition']))
            require(key not in seen,'Duplicate timing case');seen.add(key)
            if method=='BPG_LDPC_native256'and row['source_unfit']=='True':
                require(row['TX_seconds']==row['RX_seconds']==row['software_e2e_excluding_channel_seconds']=='',
                        'Source-unfit case must not have complete-frame/link timing')
                continue
            outer,channel,excluded,tx,rx,residual=[float(row[k])for k in [
                'software_e2e_including_simulated_channel_seconds','channel_elapsed_seconds',
                'software_e2e_excluding_channel_seconds','TX_seconds','RX_seconds','uncategorized_overhead_seconds']]
            require(all(math.isfinite(x)for x in [outer,channel,excluded,tx,rx,residual]),'Nonfinite timing')
            require(abs(outer-channel-excluded)<1e-8 and abs(outer-channel-tx-rx-residual)<1e-8,
                    'Actual outer-window decomposition mismatch')
        by_key={(int(row['snr_db']),row['metric']):row for row in summary}
        require(len(by_key)==len(summary),'Duplicate summary metric')
        storage_rows.append(dict(method=method,label=label,unique_loaded_parameters=storage['unique_loaded_parameters'],
            unique_parameter_bytes=storage['unique_parameter_bytes'],unique_buffer_bytes=storage['unique_buffer_bytes'],
            total_checkpoint_file_bytes=storage['total_checkpoint_file_bytes'],
            codec_library_bytes=storage.get('codec_library_bytes',''),
            parameter_scope_note=storage['parameter_count_note'],
            offloaded_unused_modules=';'.join(storage['offloaded_unused_modules']),minimal_deployment_size_claimed=False,
            evidence=str(directory/'model_storage.json')))
        for snr in [7,13,19]:
            group=[row for row in rows if row['phase']=='measured'and int(row['snr_db'])==snr]
            require(len(group)==48 and len({row['source_id']for row in group})==16,'Measured source denominator')
            unfit=sum(row.get('source_unfit')=='True'for row in group)
            gray=sum(row.get('header_reject_gray',row.get('gray'))=='True'for row in group)
            result=dict(method=method,label=label,snr_db=snr,source_count=16,measured_attempts=48,
                warmups_excluded=16,source_unfit_attempts=unfit,actual_link_attempts=48-unfit,gray_attempts=gray,
                parameters_million=storage['unique_loaded_parameters']/1e6,
                checkpoint_file_GiB=storage['total_checkpoint_file_bytes']/1024**3)
            for metric,label_metric in [('TX_seconds','tx'),('RX_seconds','rx'),
                ('software_e2e_excluding_channel_seconds','software_e2e_excluding_channel'),
                ('software_e2e_including_simulated_channel_seconds','software_e2e_including_channel'),
                ('channel_elapsed_seconds','simulated_channel'),('uncategorized_overhead_seconds','uncategorized_overhead')]:
                values=by_key[snr,metric]
                result[label_metric+'_available_samples']=int(values.get('metric_available_samples',values.get('samples')))
                for stat in ['mean','median','p95']:
                    value=optional(values[stat]);result[label_metric+'_'+stat+'_ms']=None if value is None else value*1000
            result['peak_allocated_MiB']=max(float(row['peak_allocated_bytes'])for row in group)/1024**2
            result['peak_reserved_MiB']=max(float(row['peak_reserved_bytes'])for row in group)/1024**2
            if method=='BPG_LDPC_native256':
                for metric in ['source_encoding_attempt_seconds','failed_source_attempt_elapsed_seconds']:
                    values=by_key[snr,metric]
                    result[metric.removesuffix('_seconds')+'_available_samples']=int(values['metric_available_samples'])
                    for stat in ['mean','median','p95']:
                        value=optional(values[stat]);result[metric.removesuffix('_seconds')+'_'+stat+'_ms']=None if value is None else value*1000
            timing_rows.append(result)
        all_rows.extend(dict(row,method_label=label)for row in summary)
        audit.append(dict(method=method,status='COMPLETE_HASH_VERIFIED',completion=str(completion),
            completion_sha256=sha(completion),verified_output_files=len(done['outputs']),
            source_attempts=192,packet_attempts=done['packet_attempts'],packet_cap=done['independent_packet_cap'],
            runtime=done['runtime']))
    require(timing_rows,'No complete timing methods available')
    require(all(row['runtime']==audit[0]['runtime']for row in audit),'Methods have different runtime flags')
    out.mkdir(parents=True)
    write(out/'timing_table.csv',timing_rows);write(out/'timing_single_output.csv',timing_rows)
    write(out/'model_storage.csv',storage_rows);write(out/'all_summary_rows.csv',all_rows)
    save(out/'audit.json',dict(status='COMPLETE'if not missing else 'PARTIAL_COMPLETE_METHODS_ONLY',
        completed_methods=audit,missing_methods=missing,read_only=True,training=False,inference=False,
        new_PHY_packets=0,bootstrap=False,original_scientific_results_modified=False))
    bpg_rows=[row for row in timing_rows if row['method']=='BPG_LDPC_native256']
    bpg_note=('BPG TX/RX/link-E2E averages are conditional on an actual transmitted frame. At 7/13/19 dB these comprise only '+
        '/'.join(str(row['actual_link_attempts'])for row in bpg_rows)+' measured samples ('+
        '/'.join(str(row['actual_link_attempts']//3)for row in bpg_rows)+
        ' distinct sources, each repeated three times). They are not full-population deployment averages; source-unfit encoding-attempt costs are separately retained in the CSV.'
        if bpg_rows else 'BPG timing is not yet present in this snapshot; no BPG timing is inferred from old quality results.')
    lines=['# A3 unified single-output timing','',
        ('All five requested methods have actual completed receipts.'if not missing else
         'This snapshot contains only methods with actual completed receipts. Missing methods: '+', '.join(x['method']for x in missing)+'.'),
        '', 'Each method uses the same 16 pre-existing development sources at 7/13/19 dB, with one warmup and three measured repetitions per source/work point. Summaries therefore contain 48 measured attempts per SNR. Mean, median and empirical P95 describe these repetitions, not independent noise trials or a population tail guarantee.',
        '', 'The software E2E values come from an actual synchronized outer window. The separately measured software channel window is reported both included and subtracted. TX+RX is not substituted for E2E; unclassified overhead remains explicit. This is one pass, not a separately repeated E2E run, and is not air-interface latency. Nested components are inclusive and cannot be summed with their parent totals.',
        '', 'All neural/source-codec work is fresh. Model loading, filesystem I/O, quality scoring and verification are outside timing. Required device copies are inside the endpoint totals. Header/decoder failures are retained. BPG SOURCE_UNFIT attempts have a real source-coding cost, but no transmitted frame, RX or link E2E: these fields are unavailable, not zero. Tables give each metric its available-sample count.',
        '', bpg_note,
        '', 'Memory is the maximum measured PyTorch peak allocated/reserved memory per SNR, including active loaded model baselines. Metric networks are on CPU; P also offloads its unused VAR. Full VAE modules remain loaded and include unused decoder parameters. Parameter counts are unique objects; checkpoint file size can include training metadata and is separate from tensor bytes. This is not a minimal deployment-size or energy measurement. Classical BPG has no learned weights; codec library bytes are separate.',
        '', 'SwinJSCC-80k is the frozen adapted model; 19 dB is outside its training and calibration range. RAW uses frozen development noise seed 6201; P/Swin use their original deterministic noise seed 2001. BPG uses its documented A3 development namespace and seed 2001. No method chooses a favorable noise realization.',
        '', 'All receipt-listed output hashes and case grids were checked before export. Published holdout quality results were neither recomputed nor modified. `timing_table.csv` uses milliseconds and MiB/GiB as labelled. `all_summary_rows.csv` preserves original seconds/bytes values and summary precision; `model_storage.csv` retains byte counts and scope notes.',
        '', '`timing_single_output.csv` is an identical content alias of `timing_table.csv`, matching the experiment-plan filename. It adds no measurements or statistics.',
        '', '| Method | SNR (dB) | TX mean (ms) | RX mean (ms) | Software E2E excl. channel mean (ms) | Gray / 48 | Source unfit / 48 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'NA'if value is None else f'{value:.2f}'
    for row in timing_rows:
        lines.append(f"| {row['label']} | {row['snr_db']} | {fmt(row['tx_mean_ms'])} | {fmt(row['rx_mean_ms'])} | {fmt(row['software_e2e_excluding_channel_mean_ms'])} | {row['gray_attempts']} | {row['source_unfit_attempts']} |")
    lines+=['','Runtime: '+audit[0]['runtime']['device']+', Torch '+audit[0]['runtime']['torch']+', CPU threads '+str(audit[0]['runtime']['threads'])+', interop threads '+str(audit[0]['runtime']['interop_threads'])+'.','',
        'Reproduce this read-only report with:', '', '```text',f'python experiments/paper_supplement_20261008/a3_timing/summarize.py --source "{source.resolve()}" --out NEW_EMPTY_OUTPUT_DIRECTORY','```','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf8')
    return dict(status='COMPLETE'if not missing else 'PARTIAL_COMPLETE_METHODS_ONLY',completed_methods=len(audit),
        missing=[x['method']for x in missing],output=str(out.resolve()),verified_files=sum(x['verified_output_files']for x in audit))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(summarize(a.source,a.out),indent=2))
