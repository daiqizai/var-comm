"""Fresh bounded A3 timing. Model work runs only when --request is supplied.

One genuine outer elapsed window encloses fresh TX, deterministic channel
simulation, and RX. TX/RX are nested inclusive windows; their sum is never
reported as an end-to-end measurement. No scientific results are overwritten.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

METHODS = ('RAW64_PARTIAL', 'RAW64_WHOLE', 'P1024', 'SwinJSCC80k')
SNRS = [7, 13, 19]
FIXED_SOURCES = [0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95]
HELPER_SHA = '9ee58a09810fb92b04c14db77abcad03c81e6dd2e22c8672bcb99ebc045bddee'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def import_pinned(path, name, expected):
    require(sha(path) == expected, 'Changed executable: ' + path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preload(r, np):
    records = r['records']
    require([x['source_index'] for x in records] == FIXED_SOURCES
            and [x['source_id'] for x in records] == r['source_ids'], 'Original fixed16 ordering')
    result = []
    for record in records:
        require('/development100_assets/' in record['archive'], 'Development source only')
        require(sha(record['archive']) == record['archive_sha256'], 'Source archive changed')
        with np.load(record['archive'], allow_pickle=False) as z:
            require(set(z.files) == {'pixels', 'tokens'}, 'Two-key frozen source archive')
            pixels, tokens = z['pixels'].copy(), z['tokens'].copy()
        require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256)
                and tokens.dtype == np.int64 and tokens.shape == (680,), 'Source tensor domain')
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record['preprocessing_id']
                and hashlib.sha256(b'int64:680\0' + tokens.astype('<i8').tobytes()).hexdigest() == record['tokens_sha256'],
                'Original source preprocessing/token identity')
        result.append(dict(record=record, pixels=pixels, tokens=tokens))
    return result


def partition_window(outer, channel, tx, rx):
    """Preserve real timing and explicit uncategorized overhead, without clipping."""
    require(all(x >= 0 for x in (outer, channel, tx, rx)), 'Negative elapsed duration')
    require(outer + 1e-6 >= channel + tx + rx, 'Nested windows exceed outer wall clock')
    return dict(software_e2e_including_simulated_channel_seconds=outer,
                channel_elapsed_seconds=channel,
                software_e2e_excluding_channel_seconds=outer-channel,
                TX_seconds=tx, RX_seconds=rx,
                uncategorized_overhead_seconds=outer-channel-tx-rx)


def parameter_storage(endpoint, method, r):
    if method == 'SwinJSCC80k':
        modules = {'SwinJSCC': endpoint.model}; offloaded = []
    else:
        loaded = endpoint.native.loaded
        offloaded = ['lpips', 'dino'] + (['var'] if method == 'P1024' else [])
        for name in offloaded:
            loaded[name].cpu()
        modules = {'VAE_full_loaded_module': loaded['vae'], 'Dc': loaded['decoder']}
        modules['P1024' if method == 'P1024' else 'VAR'] = endpoint.model if method == 'P1024' else loaded['var']
    unique_parameters, unique_buffers, components = {}, {}, {}
    for name, module in modules.items():
        parameters = list(module.parameters()); buffers = list(module.buffers())
        components[name] = dict(parameters=sum(p.numel() for p in parameters),
            parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
            buffer_bytes=sum(p.numel()*p.element_size() for p in buffers))
        unique_parameters.update({id(p): p for p in parameters})
        unique_buffers.update({id(p): p for p in buffers})
    files = []
    for entry in r['weight_files'][method]:
        require(sha(entry['path']) == entry['sha256'], 'Checkpoint changed')
        files.append(dict(entry, bytes=Path(entry['path']).stat().st_size))
    t = endpoint.t; t.cuda.synchronize(); t.cuda.empty_cache()
    return dict(components=components, unique_loaded_parameters=sum(p.numel() for p in unique_parameters.values()),
        unique_parameter_bytes=sum(p.numel()*p.element_size() for p in unique_parameters.values()),
        unique_buffer_bytes=sum(p.numel()*p.element_size() for p in unique_buffers.values()),
        weight_files=files, total_checkpoint_file_bytes=sum(x['bytes'] for x in files),
        offloaded_unused_modules=offloaded, disk_checkpoint_may_include_training_metadata=True,
        full_vae_loaded=True if method != 'SwinJSCC80k' else False,
        minimal_deployment_size_claimed=False,
        parameter_count_note='Unique parameter objects across loaded method modules; full VAE includes unused decoder. Shared modules deduplicated by object identity.',
        baseline_allocated_bytes=t.cuda.memory_allocated(), baseline_reserved_bytes=t.cuda.memory_reserved())


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def run(request_path, method, deadline_unix):
    r = read(request_path)
    require(r['schema'] == 'A3_FIXED16_UNIFIED_TIMING_V1' and method in METHODS
            and r['SNRs'] == SNRS and r['source_indices'] == FIXED_SOURCES
            and r['warmup_per_source_snr'] == 1 and r['measured_per_source_snr'] == 3
            and r['batch_size'] == 1 and r['training'] is r['policy_selection'] is r['original_ledger_mutated'] is False,
            'A3 fixed protocol required')
    require(sys.platform.startswith('linux') and os.path.abspath(sys.executable) == r['python']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Shared UM single-GPU environment')
    require(sorted(os.sched_getaffinity(0)) == r['affinity'] and os.getpriority(os.PRIO_PROCESS, 0) == r['nice'],
            'Pinned CPU resource budget')
    require(time.time() < deadline_unix, 'Explicit current deadline required')
    require(sha(__file__) == r['a3_sources']['benchmark.py']['sha256'], 'Request/script mismatch')
    h = import_pinned(r['helper_module'], '_a3_qualified_endpoint_helpers', HELPER_SHA)
    sys.path[:0] = r['pythonpath']
    baseline = h.descriptor(r['baseline_request'])
    prior_method = 'RAW64_PARTIAL' if method.startswith('RAW64_') else method
    prior = h.descriptor(r['prior_completed_timing'][prior_method])
    require(prior['status'] == 'INDEPENDENT_UNIFIED_UM_ENDPOINT_TIMING_COMPLETE_V1'
            and prior['method'] == prior_method and prior['migration_validation_cases'] == 48
            and prior['source_count'] == 8, 'Actual prior backend validation must exist')
    for path, expected in r['executable_bindings'].items():
        require(r['bindings'].get(path) == expected == sha(path), 'Frozen dependency changed: ' + path)
    out = Path(r['out']) / method
    require(not out.exists(), 'Fresh output required; never overwrite or silently retry')
    out.mkdir(parents=True)
    started = time.monotonic(); calls = h.Calls(r['packet_caps'][method]); rows = []
    def guard():
        require(not h.STOP and time.time() < deadline_unix and time.monotonic()-started < r['max_seconds_per_method']
                and not any(Path(p).exists() for p in r['stop_files']) and not (out/'STOP').exists(), 'STOP/deadline')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, h.stop)
    try:
        with h.exclusive(r['visual_lock']):
            import numpy as np
            guard(); sources = preload(r, np)
            if method.startswith('RAW64_'):
                d = r['a3_sources']['raw_endpoint.py']
                module = import_pinned(d['path'], '_a3_raw_primitives', d['sha256'])
                endpoint = module.RawEndpoint(r, baseline, r['bindings'], calls, h, method)
            else:
                cls = h.PEndpoint if method == 'P1024' else h.SwinEndpoint
                endpoint = cls(r, baseline, r['bindings'], calls)
            t = endpoint.t; runtime = h.runtime_flags(t)
            require(str(t.__version__).startswith('2.11.0') and runtime['threads'] == 6
                    and runtime['interop_threads'] == 2 and not t.is_autocast_enabled(), 'Shared FP32/B1 backend')
            require(runtime == prior['runtime'], 'Actual environment must equal completed endpoint validation')
            storage = parameter_storage(endpoint, method, r)
            save(out/'attempt.json', dict(method=method, request_sha256=sha(request_path), script_sha256=sha(__file__),
                runtime=runtime, source_ids=r['source_ids'], deadline_unix=deadline_unix,
                prior_backend_validation=r['prior_completed_timing'][prior_method],
                new_qualification_packets=0, new_migration_validation_cases=0,
                historical_request_equivalence_claimed=False, output_status='STARTED_NOT_COMPLETE'))
            save(out/'model_storage.json', storage)
            with t.no_grad():
                for snr in SNRS:
                    for item in sources:
                        for repetition in range(4):
                            guard(); endpoint.snr = snr
                            phase = 'warmup' if repetition == 0 else 'measured'
                            index = item['record']['source_index']
                            calls.case = f'{phase}/snr{snr}/source{index}/repeat{repetition}'
                            tx, rx = h.Meter(t), h.Meter(t)
                            calls.components = None
                            t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
                            memory_start = dict(allocated=t.cuda.memory_allocated(), reserved=t.cuda.memory_reserved())
                            outer_start = time.perf_counter()
                            wave = tx.call('TX_total', lambda: endpoint.encode(item, tx))
                            channel_start = time.perf_counter()
                            observed = endpoint.noise(wave, item, snr)
                            channel_elapsed = time.perf_counter() - channel_start
                            calls.components = rx
                            image = rx.call('RX_total', lambda: endpoint.receive(observed, snr, rx))
                            t.cuda.synchronize()
                            outer_elapsed = time.perf_counter() - outer_start
                            calls.components = None
                            memory_peak = dict(allocated=t.cuda.max_memory_allocated(), reserved=t.cuda.max_memory_reserved())
                            require(wave.shape == (1024, 2) and np.isfinite(wave).all(), 'Full CPU1024 waveform')
                            require(image.dtype == np.float32 and image.shape == (3,256,256)
                                    and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1, 'Full CPU RGB')
                            if method.startswith('RAW64_'):
                                require(hashlib.sha256(b'int64:680\0' + endpoint.last_tokens.astype('<i8').tobytes()).hexdigest()
                                        == item['record']['tokens_sha256'], 'Fresh RAW Encoder token identity')
                            gray = not endpoint.context.accepted if method == 'SwinJSCC80k' else bool(getattr(endpoint, 'gray', False))
                            row = dict(method=method, source_index=index, source_id=item['record']['source_id'],
                                snr_db=snr, phase=phase, repetition=repetition, header_reject_gray=gray,
                                **partition_window(outer_elapsed, channel_elapsed, tx.seconds['TX_total'], rx.seconds['RX_total']),
                                peak_allocated_bytes=memory_peak['allocated'], peak_reserved_bytes=memory_peak['reserved'],
                                baseline_allocated_bytes=memory_start['allocated'], baseline_reserved_bytes=memory_start['reserved'],
                                incremental_peak_allocated_bytes=memory_peak['allocated']-memory_start['allocated'],
                                output_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                                observation_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                                TX_components=tx.seconds, RX_components=rx.seconds)
                            rows.append(row)
                            # Audit writes and hashes are outside every measured window.
                            save(out/f'case_{len(rows):04d}.json', row)
                            if hasattr(endpoint, 'last_f'):
                                del endpoint.last_f
                            del wave, observed, image
                            guard()
            endpoint.finish(); guard()
            require(len(rows) == 192 and all(x['status'] == 'COMPLETE' for x in calls.rows), 'Complete bounded run')
            flat = [{k:v for k,v in row.items() if k not in ('TX_components','RX_components')} for row in rows]
            write_csv(out/'timings.csv', flat)
            components = [dict(method=method, source_index=row['source_index'], snr_db=row['snr_db'], phase=row['phase'],
                repetition=row['repetition'], direction=direction, component=name, seconds=value)
                for row in rows for direction in ('TX','RX') for name,value in row[direction+'_components'].items()]
            write_csv(out/'components.csv', components)
            summary = []
            numeric = [k for k in flat[0] if k.endswith('_seconds') or k.endswith('_bytes')]
            for snr in SNRS:
                group = [row for row in flat if row['phase'] == 'measured' and row['snr_db'] == snr]
                require(len(group) == 48 and len({row['source_id'] for row in group}) == 16, '16 sources x3 measured')
                for metric in numeric:
                    values = np.array([row[metric] for row in group], dtype=np.float64)
                    summary.append(dict(method=method, snr_db=snr, metric=metric, mean=float(values.mean()),
                        median=float(np.median(values)), p95=float(np.percentile(values,95)), minimum=float(values.min()),
                        maximum=float(values.max()), source_count=16, samples=48,
                        gray_samples=sum(row['header_reject_gray'] for row in group)))
            write_csv(out/'summary.csv', summary)
            save(out/'packet_events.json', calls.rows)
            outputs = {str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()}
            done = dict(status='COMPLETE', schema=r['schema'], method=method, source_count=16, SNRs=SNRS,
                fresh_frames=192, warmup_frames=48, measured_frames=144, batch_size=1,
                packet_attempts=len(calls.rows), independent_packet_cap=calls.cap,
                runtime=runtime, request_sha256=sha(request_path), script_sha256=sha(__file__), outputs=outputs,
                new_quality_calls=0, training_updates=0, policy_selection=False, original_ledger_mutated=False,
                same_case_outer_window=True, independently_repeated_e2e_pass=False,
                e2e_excluding_channel='actual outer elapsed minus explicitly timed channel simulation window',
                over_the_air_latency_measured=False, components_inclusive_not_addable=True,
                disk_IO_in_timed_windows=False, model_loading_in_timed_windows=False,
                noise_seed_RAW=6201, noise_seed_baselines=2001,
                prior_8_source_timings_reused_as_new_measurements=False,
                peak_memory_note='PyTorch allocated/reserved GPU memory, including active model baseline; not device-wide nvidia-smi VRAM.')
            save(out/'completion.json', done)
            return dict(status='COMPLETE', method=method, completion=str(out/'completion.json'), sha256=sha(out/'completion.json'))
    except BaseException:
        save(out/'failure.json', dict(status='FAILED_PRESERVE_NO_RETRY', method=method,
            traceback=traceback.format_exc(), completed_frames=len(rows), packet_events=calls.rows,
            packet_attempts=len(calls.rows), original_ledger_mutated=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--deadline-unix', required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.request,args.method,args.deadline_unix), allow_nan=False))
