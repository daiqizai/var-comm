"""Read-only export of the completed historical external timing measurements."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path

EXTERNAL=('P1024','SwinJSCC80k','BPG_LDPC_native256')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csvout(p,table):
    with Path(p).open('x',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in table for k in r)));w.writeheader();w.writerows(table)

def run(root,out):
    require(not out.exists(),'Use a fresh directory; preserve all old and new timing snapshots')
    base=root/'results/paper_supplement_20261008/a3_timing';actual=base/'actual_v1';final=base/'final_v1'
    audit=read(final/'audit.json');require(audit['status']=='COMPLETE'and audit['missing_methods']==[],'Completed historical table required')
    protected=root/'results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2'
    protected_hashes={str(p):sha(p)for p in protected.iterdir()if p.is_file()}
    require(protected_hashes,'Current T4 summary must be present and remain unchanged')
    full=rows(final/'timing_single_output.csv');require(len(full)==15 and sha(final/'timing_table.csv')==sha(final/'timing_single_output.csv'),'Exact published table alias')
    selected=[r for r in full if r['method']in EXTERNAL];require(len(selected)==9,'All external3 x3 SNR points required')
    inputs={str(p):sha(p)for p in [final/'audit.json',final/'timing_single_output.csv',final/'README.md']}
    verified=[];metadata=[];copy_plan=[]
    request_paths={'P1024':actual/'materials/request.json','SwinJSCC80k':actual/'materials/request.json','BPG_LDPC_native256':actual/'materials_bpg/request.json'}
    for method in EXTERNAL:
        d=actual/'results'/method;cp=d/'completion.json';done=read(cp)
        previous=next(x for x in audit['completed_methods']if x['method']==method)
        require(sha(cp)==previous['completion_sha256']and done['status']=='COMPLETE'and done['method']==method
                and done['source_count']==16 and done['SNRs']==[7,13,19],'Historical completion differs')
        for remote,h in done['outputs'].items():
            p=d/Path(remote).name;require(p.is_file()and sha(p)==h,'Changed sealed historical output: '+str(p))
            verified.append(dict(method=method,remote_path=remote,local_path=str(p),sha256=h))
        rp=request_paths[method];request=read(rp);require(sha(rp)==done['request_sha256'],'Historical request binding differs')
        require(request['SNRs']==[7,13,19]and request['batch_size']==1 and request['warmup_per_source_snr']==1
                and request['measured_per_source_snr']==3 and request['holdout_used']is False,'Historical timing population differs')
        calls=rows(d/'timings.csv');summary=rows(d/'summary.csv')
        require(len(calls)==192 and sum(r['phase']=='measured'for r in calls)==144,'Original192 calls/144 measured required')
        keys={(r['source_index'],r['snr_db'],r['repetition'])for r in calls};require(len(keys)==192,'Duplicate historical timing calls')
        for snr in [7,13,19]:
            display=next(r for r in selected if r['method']==method and int(r['snr_db'])==snr)
            calls_at=[r for r in calls if int(r['snr_db'])==snr and r['phase']=='measured']
            require(len(calls_at)==48 and {r['source_id']for r in calls_at}==set(request['source_ids']),'Fixed16 actual source grid differs')
            for metric,prefix in [('TX_seconds','tx'),('RX_seconds','rx'),('software_e2e_excluding_channel_seconds','software_e2e_excluding_channel')]:
                sr=next(r for r in summary if int(r['snr_db'])==snr and r['metric']==metric)
                require(int(display[prefix+'_available_samples'])==int(sr.get('metric_available_samples',sr.get('samples'))),'Available count differs')
                for stat in ['mean','median','p95']:
                    require(math.isclose(float(display[prefix+'_'+stat+'_ms']),float(sr[stat])*1000,rel_tol=1e-15,abs_tol=1e-12),'Published seconds-to-ms value differs')
            if method=='BPG_LDPC_native256':
                unfit=[r for r in calls_at if r['source_unfit']=='True']
                require(len(unfit)=={7:45,13:36,19:18}[snr]and int(display['source_unfit_attempts'])==len(unfit),'BPG source-unfit counts differ')
                require(all(r['TX_seconds']==r['RX_seconds']==r['software_e2e_excluding_channel_seconds']==''for r in unfit),'SOURCE_UNFIT link timing is NA, not zero')
        metadata.append(dict(method=method,completion_path=str(cp),completion_sha256=sha(cp),request_path=str(rp),request_sha256=sha(rp),
            N=1024,SNRs=[7,13,19],source_indices=request['source_indices'],source_ids=request['source_ids'],population='fixed16 development',
            warmups_per_source_snr=1,measured_repetitions_per_source_snr=3,timing_protocol=request['timing_protocol'],
            runtime=done['runtime'],affinity=request['affinity'],nice=request['nice'],source_cache_read=done.get('source_cache_read'),
            reconstruction_cache_read=done.get('reconstruction_cache_read'),summary_available_counts_retained=True,
            BPG_version='native256; not adaptive downsampling'if method=='BPG_LDPC_native256'else None))
        for name in ('completion.json','summary.csv','timings.csv','components.csv','model_storage.json'):
            copy_plan.append((d/name,Path('original_methods')/method/name));inputs[str(d/name)]=sha(d/name)
        inputs[str(rp)]=sha(rp)
    series=read(actual/'series_completion.json')
    require(series['all_children_waited']is True and all(x['exit_code']==0 for x in series['methods']),'Historical neural series actual wait differs')
    out.mkdir(parents=True)
    for name in ('README.md','audit.json','timing_single_output.csv','timing_table.csv','all_summary_rows.csv','model_storage.csv'):
        copy_plan.append((final/name,Path('original_final')/name));inputs[str(final/name)]=sha(final/name)
    copy_plan.extend([(actual/'series_completion.json',Path('original_receipts/series_completion.json')),
                      (request_paths['P1024'],Path('original_receipts/neural_request.json')),
                      (request_paths['BPG_LDPC_native256'],Path('original_receipts/bpg_native_request.json'))])
    for src,rel in copy_plan:
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        require(sha(src)==sha(dst),'Read-only byte copy differs')
    csvout(out/'external_timing_single_output.csv',selected)
    csvout(out/'external_summary_original_precision.csv',[dict(r,source_file=str(actual/'results'/m/'summary.csv'))for m in EXTERNAL for r in rows(actual/'results'/m/'summary.csv')])
    csvout(out/'verified_original_outputs.csv',verified);save(out/'population_environment_and_boundaries.json',metadata)
    (out/'README.md').write_text('''# Reused external single-output timing

This directory restores a direct delivery link to the already completed P1024 (public label: Latent continuous JSCC), SwinJSCC-80k adapted, and BPG+LDPC native256 timing measurements. It performs no new timing, inference, channel simulation or bootstrap.

`external_timing_single_output.csv` contains the nine external rows from the published A3 table without changing their numeric strings. `external_summary_original_precision.csv` preserves the original seconds/bytes summaries and available-sample counts. Byte-identical original full tables, per-method completions, raw per-call timings, components, model storage and requests are supplied. Every output listed by the three method completions was checked against its SHA; the entire local verification list is included.

All historical points use N1024 and SNR7/13/19dB, the same fixed16 development sources, batch1, one warm-up and three measured repetitions per source/workpoint (48 measured attempts per point). These repeats are not three independent noise trials. P/Swin use seed2001; BPG uses its A3 development namespace with seed2001. Hardware is RTX4090D, Torch2.11.0+cu128, six CPU threads/two interop threads, affinity4–9, nice15. Full frozen flags and endpoint boundaries are in population_environment_and_boundaries.json and original requests.

The original boundary is CPU CHW uint8 input -> complete CPU1024x2 I/Q -> received CPU I/Q -> one CPU float32 RGB. Actual synchronized outer E2E is measured once; simulated channel time is separately included and subtracted. TX+RX is not substituted for E2E, and inclusive nested component times must not be added to their parent totals. Model loading, disk I/O and quality scoring are excluded. This is software timing, not over-the-air latency or a universal deployment-tail guarantee.

BPG here is native256, not the adaptive-downsampling quality baseline. At7/13/19dB, only1/4/10 of the16 sources can fit, hence only3/12/30 measured link samples. Its TX/RX/link-E2E means are conditional on a transmitted frame. The45/36/18 SOURCE_UNFIT attempts retain real encoding-attempt costs, while full-frame TX/RX/link-E2E remain unavailable. These are not zero latency and cannot be used as full-population deployment means. Swin19dB remains outside training and calibration range.

The historical neural series receipt records actual waits/exit0 for P/Swin and old raw anchors. BPG is authenticated here by its published completion, request, sealed outputs and completed publication audit; a separate BPG process-exit receipt is not present in this local synchronized A3 folder. This export does not invent a new wait observation.

The newer WCL four-arm timing remains independently sealed at `../summary_completed_v2/`, with actual timing source at `.research/wcl_evidence_closure_20261009/t4_actual_timing_v3/`. It uses4/10/19dB and three warm-ups plus three measured repeats. Even where hardware/frozen numerical settings coincide, the historical raw values are retained as historical anchors, not overwritten by or pooled with the new raw measurements. No4/10dB external timings are imputed from7/13dB. Adaptive-BPG and HiFi do not acquire new unified timing rows from this export.

Original published entry: `results/paper_supplement_20261008/a3_timing/final_v1/timing_single_output.csv`, described in `reports/paper_supplement_20261009.md`. The copied original README retains all existing limitations and memory/weight-size scope.

Reproduce to a fresh directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/t4_reuse_external_timing.py --root . --out results/wcl_evidence_closure_20261009/T4_resources_cost/external_timing_reuse_NEW
```
''',encoding='utf-8')
    require(all(sha(p)==h for p,h in protected_hashes.items()),'Already sealed summary_completed_v2 changed')
    save(out/'completion.json',dict(status='T4_EXTERNAL_TIMING_REUSE_COMPLETE',methods=list(EXTERNAL),N=1024,SNRs=[7,13,19],
        source_count=16,measured_repeats_per_source=3,external_table_rows=9,verified_original_output_count=len(verified),
        input_bindings=inputs,code_sha256=sha(__file__),new_model_calls=0,new_timing_calls=0,new_packet_calls=0,new_bootstrap_calls=0,
        historical_neural_actual_wait_receipt=str(actual/'series_completion.json'),BPG_standalone_exit_receipt_locally_present=False,
        summary_completed_v2_unchanged=protected_hashes,not_pooled_with_new_timing=True,
        outputs={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()}))
    save(out/'DELIVERY_MANIFEST.json',dict(status='COMPLETE_VERIFIED',files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.rglob('*'))if p.is_file()}))
    return dict(status='T4_EXTERNAL_TIMING_REUSE_COMPLETE',external_rows=9,verified_original_outputs=len(verified),completion_sha256=sha(out/'completion.json'),delivery_sha256=sha(out/'DELIVERY_MANIFEST.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root.resolve(),a.out.resolve())))
