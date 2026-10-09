"""Independent A3 native256 BPG timing; no changes to the four neural runs.

SOURCE_UNFIT records the actual source-encoding attempt but has no transmitted
frame, RX, or software link E2E timing. A gray display fallback is saved outside
the timing window and must not be called an actual received reconstruction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(path,name,expected):
    import importlib.util
    if sha(path) != expected:
        raise RuntimeError('Changed executable: '+path)
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(request_path,deadline_unix):
    r = read(request_path)
    d = r['a3_sources']['benchmark.py']
    base = load(d['path'],'_a3_unchanged_neural_measurement_helpers',d['sha256'])
    require,save = base.require,base.save
    require(sha(__file__)==r['bpg_runner']['sha256'], 'BPG runner/request mismatch')
    require(r['schema']=='A3_FIXED16_BPG_TIMING_V1' and r['SNRs']==base.SNRS
            and r['source_indices']==base.FIXED_SOURCES and r['warmup_per_source_snr']==1
            and r['measured_per_source_snr']==3 and r['batch_size']==1
            and r['bpg_packet_cap']==384 and r['training'] is r['policy_selection'] is r['original_ledger_mutated'] is False,
            'Separate fixed16 BPG scope required')
    require(sys.platform.startswith('linux') and os.path.abspath(sys.executable)==r['python']
            and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
            and sorted(os.sched_getaffinity(0))==r['affinity']
            and os.getpriority(os.PRIO_PROCESS,0)==r['nice'], 'Same fixed UM resources')
    require(time.time()<deadline_unix, 'Explicit future deadline required')
    h = load(r['helper_module'],'_a3_bpg_qualified_endpoint_helpers',base.HELPER_SHA)
    sys.path[:0] = r['pythonpath']
    prior = h.descriptor(r['prior_completed_timing']['P1024'])
    require(prior['status']=='INDEPENDENT_UNIFIED_UM_ENDPOINT_TIMING_COMPLETE_V1', 'Prior shared-environment receipt required')
    for path,expected in r['executable_bindings'].items():
        require(r['bindings'].get(path)==expected==sha(path),'Frozen dependency changed: '+path)
    out = Path(r['bpg_out'])
    require(not out.exists(),'Fresh BPG output; preserve failed attempts')
    out.mkdir(parents=True)
    method = 'BPG_LDPC_native256'
    calls = h.Calls(r['bpg_packet_cap']);rows = [];started = time.monotonic()
    def guard():
        require(not h.STOP and time.time()<deadline_unix and time.monotonic()-started<r['max_seconds_per_method']
                and not any(Path(p).exists() for p in r['stop_files']) and not (out/'STOP').exists(), 'STOP/deadline')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,h.stop)
    try:
        with h.exclusive(r['visual_lock']):
            import numpy as np
            guard();sources = base.preload(r,np)
            d = r['bpg_endpoint'];module = load(d['path'],'_a3_native256_bpg_endpoint',d['sha256'])
            endpoint = module.BPGEndpoint(r,calls,h)
            t = endpoint.t;runtime = h.runtime_flags(t)
            require(runtime==prior['runtime'] and str(t.__version__).startswith('2.11.0')
                    and not t.is_autocast_enabled(), 'Same actual FP32 runtime')
            save(out/'attempt.json',dict(method=method,status='STARTED_NOT_COMPLETE',runtime=runtime,
                source_ids=r['source_ids'],request_sha256=sha(request_path),script_sha256=sha(__file__),
                deadline_unix=deadline_unix,parity_completion=r['bpg']['parity_completion'],
                new_qualification_packets=0,neural_measurement_files_changed=False))
            save(out/'model_storage.json',endpoint.storage)
            with t.no_grad():
                for snr in base.SNRS:
                    for item in sources:
                        for repetition in range(4):
                            guard();endpoint.snr=snr
                            phase='warmup' if repetition==0 else 'measured'
                            index=item['record']['source_index'];calls.case=f'{phase}/snr{snr}/source{index}/repeat{repetition}'
                            tx,rx=h.Meter(t),h.Meter(t);calls.components=None
                            t.cuda.synchronize();t.cuda.reset_peak_memory_stats()
                            memory_start=dict(allocated=t.cuda.memory_allocated(),reserved=t.cuda.memory_reserved())
                            outer_start=time.perf_counter()
                            wave=tx.call('TX_total',lambda:endpoint.encode(item,tx))
                            if wave is None:
                                require(endpoint.source_unfit,'Only a true source-capacity failure omits the frame')
                                t.cuda.synchronize();failed_elapsed=time.perf_counter()-outer_start
                                # No channel/RX is invented. The display gray image is made later.
                                timing=dict(TX_seconds=None,RX_seconds=None,channel_elapsed_seconds=None,
                                    software_e2e_including_simulated_channel_seconds=None,
                                    software_e2e_excluding_channel_seconds=None,uncategorized_overhead_seconds=None,
                                    source_encoding_attempt_seconds=tx.seconds['TX_total'],
                                    failed_source_attempt_elapsed_seconds=failed_elapsed)
                                observed=image=None
                            else:
                                require(not endpoint.source_unfit,'Conflicting source fit status')
                                channel_start=time.perf_counter()
                                observed=endpoint.noise(wave,item,snr)
                                channel_elapsed=time.perf_counter()-channel_start
                                calls.components=rx
                                image=rx.call('RX_total',lambda:endpoint.receive(observed,snr,rx))
                                t.cuda.synchronize();outer_elapsed=time.perf_counter()-outer_start
                                timing=base.partition_window(outer_elapsed,channel_elapsed,tx.seconds['TX_total'],rx.seconds['RX_total'])
                                timing.update(source_encoding_attempt_seconds=None,failed_source_attempt_elapsed_seconds=None)
                            calls.components=None
                            memory_peak=dict(allocated=t.cuda.max_memory_allocated(),reserved=t.cuda.max_memory_reserved())
                            if image is None:
                                image=np.full((3,256,256),.5,np.float32)
                            else:
                                require(wave.shape==(1024,2) and np.isfinite(wave).all(),'Complete transmitted IQ frame')
                            require(image.dtype==np.float32 and image.shape==(3,256,256)
                                    and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,'Valid final display RGB')
                            outcome=endpoint.last_outcome
                            row=dict(method=method,source_index=index,source_id=item['record']['source_id'],snr_db=snr,
                                phase=phase,repetition=repetition,source_unfit=endpoint.source_unfit,
                                actual_link_executed=wave is not None,gray=endpoint.gray,
                                link_status='SOURCE_UNFIT' if outcome is None else outcome['status'],
                                selected_qp=endpoint.last_fit['selected_qp'],capacity_bytes=endpoint.last_fit['capacity_bytes'],
                                actual_encode_calls=endpoint.last_fit['actual_encode_calls'],source_bytes=endpoint.last_source_bytes,
                                **timing,peak_allocated_bytes=memory_peak['allocated'],peak_reserved_bytes=memory_peak['reserved'],
                                baseline_allocated_bytes=memory_start['allocated'],baseline_reserved_bytes=memory_start['reserved'],
                                incremental_peak_allocated_bytes=memory_peak['allocated']-memory_start['allocated'],
                                output_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                waveform_sha256=None if wave is None else hashlib.sha256(wave.tobytes()).hexdigest(),
                                observation_sha256=None if observed is None else hashlib.sha256(observed.tobytes()).hexdigest(),
                                TX_components=tx.seconds,RX_components=rx.seconds,source_search=endpoint.last_fit)
                            if repetition==1:
                                path=out/f'rgb_source{index:04d}_snr{snr}_rep1.npz'
                                with path.open('xb') as f:
                                    np.savez_compressed(f,rgb=image)
                                row['display_archive']=str(path);row['display_archive_sha256']=sha(path)
                                row['display_role']='source_unfit_gray_no_received_frame' if endpoint.source_unfit else 'actual_first_measured_received_output'
                            rows.append(row);save(out/f'case_{len(rows):04d}.json',row)
                            del image,wave,observed
                            guard()
            endpoint.finish();guard()
            require(len(rows)==192 and all(x['status']=='COMPLETE' for x in calls.rows),'All bounded attempts complete')
            flat=[{k:v for k,v in row.items() if k not in ('TX_components','RX_components','source_search')}for row in rows]
            base.write_csv(out/'timings.csv',flat)
            component_rows=[dict(method=method,source_index=row['source_index'],snr_db=row['snr_db'],phase=row['phase'],
                repetition=row['repetition'],source_unfit=row['source_unfit'],direction=direction,component=name,seconds=value)
                for row in rows for direction in ('TX','RX') for name,value in row[direction+'_components'].items()]
            base.write_csv(out/'components.csv',component_rows)
            metrics=[k for k in flat[0] if k.endswith('_seconds')or k.endswith('_bytes')]
            summary=[]
            for snr in base.SNRS:
                group=[row for row in flat if row['snr_db']==snr and row['phase']=='measured']
                require(len(group)==48 and len({row['source_id']for row in group})==16,'16 sources x3 measured attempts')
                for metric in metrics:
                    available=[row for row in group if row.get(metric)is not None]
                    values=np.array([row[metric]for row in available],dtype=np.float64)
                    summary.append(dict(method=method,snr_db=snr,metric=metric,
                        mean=None if not len(values)else float(values.mean()),median=None if not len(values)else float(np.median(values)),
                        p95=None if not len(values)else float(np.percentile(values,95)),
                        minimum=None if not len(values)else float(values.min()),maximum=None if not len(values)else float(values.max()),
                        source_count=16,attempted_samples=48,metric_available_samples=len(values),
                        metric_available_source_count=len({row['source_id']for row in available}),
                        source_unfit_samples=sum(row['source_unfit']for row in group),gray_samples=sum(row['gray']for row in group)))
            base.write_csv(out/'summary.csv',summary);save(out/'packet_events.json',calls.rows)
            outputs={str(p):sha(p)for p in sorted(out.iterdir())if p.is_file()}
            save(out/'completion.json',dict(status='COMPLETE',schema=r['schema'],method=method,source_count=16,SNRs=base.SNRS,
                fresh_source_attempts=192,actual_link_frames=sum(row['actual_link_executed']for row in rows),
                source_unfit_attempts=sum(row['source_unfit']for row in rows),warmup_attempts=48,measured_attempts=144,
                packet_attempts=len(calls.rows),independent_packet_cap=calls.cap,display_archives=48,
                outputs=outputs,runtime=runtime,request_sha256=sha(request_path),script_sha256=sha(__file__),
                source_unfit_RX_and_E2E_link_are_NA=True,source_search_fresh_each_call=True,
                source_cache_read=False,reconstruction_cache_read=False,disk_IO_in_timed_windows=False,
                e2e_is_actual_outer_window=True,independently_repeated_e2e_pass=False,over_the_air_latency_measured=False,
                nested_components_addable=False,noise_namespace='A3_DEVELOPMENT_TIMING',noise_seed=2001,
                new_qualification_packets=0,new_quality_calls=0,training_updates=0,policy_selection=False,
                original_ledger_mutated=False,neural_measurement_files_changed=False))
            return dict(status='COMPLETE',completion=str(out/'completion.json'),sha256=sha(out/'completion.json'))
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),
            method=method,completed_attempts=len(rows),packet_attempts=len(calls.rows),packet_events=calls.rows,
            original_ledger_mutated=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();print(json.dumps(run(a.request,a.deadline_unix),allow_nan=False))
