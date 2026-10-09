"""Bounded Kodak24 inference using the frozen A3 neural endpoint implementations.

Preparation is metadata-only. Execution is one explicit method per process,
with no training, calibration, warmup, timing repetitions or automatic retry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = 'KODAK24_FROZEN_NEURAL_REQUEST_V1'
METHODS = ('VAR_UNCONDITIONAL', 'P1024', 'SWIN80K')
OLD_NAMES = dict(VAR_UNCONDITIONAL='RAW64_PARTIAL', P1024='P1024', SWIN80K='SwinJSCC80k')
SNRS, SEEDS = [4, 10, 19], [2001, 2002, 2003]
CAPS = dict(VAR_UNCONDITIONAL=432, P1024=0, SWIN80K=216)
PROFILES = {4:181, 10:231, 19:378}
PREPROCESSING = 'kodak_rgb_center_crop_256_v1'
NAMESPACE = 'KODAK24_GENERALIZATION_20261009_V1'
HELPER_SHA = '9ee58a09810fb92b04c14db77abcad03c81e6dd2e22c8672bcb99ebc045bddee'
POLICY_SHA = '7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c'
STOP = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def descriptor(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def pin(d):
    require(sha(d['path']) == d['sha256'], 'Pinned input changed: ' + d['path'])
    return read(d['path'])


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def load(path, name, expected):
    require(sha(path) == expected, 'Executable changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_records(manifest):
    records = manifest['records']
    require(manifest['schema'] == 'KODAK24_CENTER256_DATASET_V1'
            and manifest['source_count'] == len(records) == 24, 'Fixed Kodak24 population')
    require([x['source_index'] for x in records] == list(range(24))
            and [x['source_id'] for x in records] == [f'kodak/kodim{i:02d}' for i in range(1,25)],
            'All 24 Kodak sources in fixed public order')
    require(all(x['preprocessing_id'] == PREPROCESSING for x in records), 'Fixed center crop preprocessing')
    require(all(not {'label','class','class_id','true_label','target_label'}.intersection(x) for x in records),
            'No class-label metadata admitted to neural inference')
    return records


def frozen_points(a3):
    cfg = pin(a3['raw_config'])
    policy_path = str(Path(a3['root'])/'outputs/MAIN-RAW64-20261007/final_selection_v1/policies.json')
    require(a3['bindings'][policy_path] == POLICY_SHA == sha(policy_path), 'Original frozen RAW policy')
    policy = read(policy_path)
    require(policy['status'] == 'POLICIES_FROZEN_ON_CALIBRATION1000'
            and policy['objective'] == 'dinov2_vitl14_cosine', 'Original selection identity')
    catalogue_path = cfg['adapter_config']['catalogue']
    require(a3['bindings'][catalogue_path] == sha(catalogue_path), 'Frozen full RAW catalogue')
    catalogue = read(catalogue_path)['profiles']; points = {}
    for snr in SNRS:
        wins = [x for x in policy['winners'] if x['family'] == 'PARTIAL' and x['snr_db'] == snr]
        require(len(wins) == 1, 'One frozen PARTIAL winner per SNR')
        p = catalogue[PROFILES[snr]]
        require(wins[0]['candidate_id'] in p['alias_candidate_ids'] and p['N'] == 1024,
                'Kodak may not reselect transmitter policy')
        points[str(snr)] = dict(snr_db=snr, profile_id=p['profile_id'], candidate_id=wins[0]['candidate_id'],
            wire_key=p['wire_key'], m=p['m'], K=p['K'], modulation=p['groups'][0]['modulation'])
    return points, descriptor(policy_path), descriptor(catalogue_path)


def source_extension(a3):
    """Freeze newly published source inventory while preserving every old hash.

    The legacy Native inventory includes all Git-tracked Python/C++ files, even
    unrelated new experiments. New files therefore need explicit registration;
    replacing or removing any previously admitted file remains forbidden.
    """
    baseline = pin(a3['baseline_request'])
    old = baseline['P_native_source_bindings']
    for path, expected in old.items():
        require(a3['bindings'].get(path) == expected == sha(path), 'Original Native source changed: '+path)
    tracked = subprocess.check_output(['git','ls-files','*.py','*.cpp','configs/*.yaml','configs/*.json'],
                                     cwd=a3['root'],text=True).splitlines()
    paths = [str(Path(a3['root'])/relative) for relative in tracked]
    additions = {path:sha(path) for path in sorted(paths) if path not in old}
    return additions, len(old)


def build_native_with_registered_additions(a3, baseline, bindings, helper, additions):
    """Same original loader/models; exact old-plus-registered-new source proof."""
    quality = helper.load(baseline['quality_driver'],'_kodak_original_quality',bindings)
    native = quality.build_native(Path(a3['root']),baseline['native_runtime'],helper.stop)
    require(native.loaded['identity'] == baseline['P_native_runtime']['frozen_visual_identity']
            and native.flags == baseline['P_native_runtime']['numerical_runtime'],
            'Actual original Native weights/numerical runtime changed')
    old = baseline['P_native_source_bindings']
    require(not set(old).intersection(additions), 'Additional source paths may not replace originals')
    expected = dict(old,**additions)
    require(native.driver_bindings == expected, 'Actual Native source inventory differs from registered old-plus-new inventory')
    for path, expected_sha in old.items():
        require(bindings.get(path) == expected_sha == sha(path), 'Original Native source not pinned: '+path)
    for path, expected_sha in additions.items():
        require(sha(path) == expected_sha, 'New registered source changed: '+path)
    return native


def make_request(original_a3_request, source_manifest, out, request_path, deadline_unix, max_seconds=7200):
    """Freeze host paths and unchanged policies; no neural or PHY work occurs."""
    a3_ref, data_ref = descriptor(original_a3_request), descriptor(source_manifest)
    a3, manifest = pin(a3_ref), pin(data_ref)
    require(a3['schema'] == 'A3_FIXED16_UNIFIED_TIMING_V1', 'Completed A3 endpoint request required')
    records = manifest_records(manifest)
    for record in records:
        require(sha(record['png']['path']) == record['png']['sha256'], 'Frozen Kodak crop bytes')
    points, policy, catalogue = frozen_points(a3)
    additions, old_count = source_extension(a3)
    require(sha(a3['helper_module']) == HELPER_SHA, 'Qualified original helper')
    request = dict(schema=SCHEMA, original_a3_request=a3_ref, source_manifest=data_ref,
        out=str(Path(out).resolve()), script=descriptor(__file__), methods=list(METHODS),
        SNRs=SNRS, noise_seeds=SEEDS, source_count=24, frames_per_method=216,
        N=1024, batch_size=1, packet_caps=CAPS, deadline_unix=float(deadline_unix),
        max_seconds_per_method=int(max_seconds), affinity=a3['affinity'], nice=a3['nice'],
        python=a3['python'], visual_lock=a3['visual_lock'], raw_points=points,
        frozen_raw_policy=policy, frozen_raw_catalogue=catalogue,
        original_native_source_binding_count=old_count,native_source_bindings_additions=additions,
        source_extension_rule='Every original path/hash unchanged; only explicit newly tracked paths added to inventory',
        stop_files=[str(Path(a3['root'])/'STOP'), str(Path(out).resolve()/'STOP')],
        raw_noise=dict(namespace=NAMESPACE, distribution='PCG64 standard_normal float64 IQ',
            seed_key='canonical JSON [namespace, RAW, source_id, snr_db, noise_seed]; SHA256 first16 little endian',
            real_coordinate_noise_scale='10**(-snr_db/20); unchanged Es2 convention',
            public_counter='((SNR-index * 24 + source_index) * 3 + noise-index)'),
        baseline_noise='Original method-specific deterministic channel functions with seeds 2001,2002,2003',
        shared_noise_labels_do_not_mean_shared_waveform=True, VAR_condition='null embedding index 1000',
        true_labels_used=False, training_updates=0, policy_selection=False,
        warmup_frames=0, timing_repetitions=0, original_ledger_mutated=False, automatic_successor=False)
    save(request_path, request)
    return descriptor(request_path)


def raw_noise(source_id, snr, seed):
    import numpy as np
    require(snr in SNRS and seed in SEEDS and source_id.startswith('kodak/kodim'), 'Kodak noise identity')
    key = json.dumps([NAMESPACE,'RAW',source_id,snr,seed], separators=(',',':'), ensure_ascii=True)
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(value)).standard_normal((1024,2))


def frame_counter(snr, source_index, seed):
    require(snr in SNRS and type(source_index) is int and 0 <= source_index < 24 and seed in SEEDS,
            'Kodak public frame counter domain')
    return (SNRS.index(snr)*24+source_index)*3+SEEDS.index(seed)


class PassThroughMeter:
    """Endpoint signature compatibility only; these runs are not timing samples."""
    @staticmethod
    def call(_name, function):
        return function()


class Calls:
    """Separate durable pre-reservation; old experiment ledgers are never opened."""
    def __init__(self, out, cap, guard):
        self.out, self.cap, self.guard = Path(out), cap, guard
        self.rows, self.case, self.components = [], 'initialization', None

    def invoke(self, kind, function):
        self.guard()
        require(len(self.rows) < self.cap, 'Kodak independent packet cap')
        row = dict(index=len(self.rows), case=self.case, kind=kind, status='RESERVED')
        self.rows.append(row)
        save(self.out/'packet_events'/f'{row["index"]:04d}_reserved.json', row)
        try:
            value = function(); row['status'] = 'COMPLETE'
            save(self.out/'packet_events'/f'{row["index"]:04d}_completed.json', row)
            return value
        except BaseException:
            row['status'] = 'FAILED'
            save(self.out/'packet_events'/f'{row["index"]:04d}_failed.json', row)
            raise


class NeuralAdapter:
    def __init__(self, method, request, a3, helper, calls):
        self.method, self.request, self.calls = method, request, calls
        self.h, self.meter, self.null_calls = helper, PassThroughMeter(), 0
        baseline = helper.descriptor(a3['baseline_request'])
        # Scope the admission extension to construction in this new process.
        # The original helper file, numerical routines and RX methods are intact.
        original_builder = helper.build_native
        helper.build_native = lambda rr,bb,bindings: build_native_with_registered_additions(
            rr,bb,bindings,helper,request['native_source_bindings_additions'])
        try:
            if method == 'VAR_UNCONDITIONAL':
                d = a3['a3_sources']['raw_endpoint.py']
                raw = load(d['path'], '_kodak_original_raw_endpoint', d['sha256'])
                self.endpoint = raw.RawEndpoint(a3, baseline, a3['bindings'], calls, helper, 'RAW64_PARTIAL')
            else:
                cls = helper.PEndpoint if method == 'P1024' else helper.SwinEndpoint
                self.endpoint = cls(a3, baseline, a3['bindings'], calls)
        finally:
            helper.build_native = original_builder
        if method == 'VAR_UNCONDITIONAL':
            var = self.endpoint.native.loaded['var']
            require(var.class_emb.num_embeddings == 1001 and not var.training
                    and float(var.cond_drop_rate) == 0, 'Frozen VAR null embedding domain')
            def check_null(_module, args):
                ids = args[0]
                require(tuple(ids.shape) == (1,) and bool((ids == 1000).all().item()),
                        'Only unconditional VAR null embedding may enter Kodak inference')
                self.null_calls += 1
            self.null_hook = var.class_emb.register_forward_pre_hook(check_null)
        self.t = self.endpoint.t
        old = OLD_NAMES[method]
        d = a3['a3_sources']['benchmark.py']
        benchmark = load(d['path'], '_kodak_original_storage_inventory', d['sha256'])
        self.storage = benchmark.parameter_storage(self.endpoint, old, a3)

    def encode(self, item, snr, seed):
        import numpy as np
        e, m = self.endpoint, self.meter
        e.snr = snr
        self.tx_detail = {}
        if self.method == 'VAR_UNCONDITIONAL':
            tokens = e.timing.native_encoder(e.native, item['pixels'].copy())
            require(tokens.shape == (680,) and tokens.dtype == np.int64, 'Fresh frozen VQ token domain')
            e.last_tokens = tokens
            point = self.request['raw_points'][str(snr)]
            profile = e.catalogue.entry(point['profile_id'])
            require(point['wire_key'] == profile['wire_key'], 'Frozen RAW wire')
            payload = e.original.serialize_raw(e.source.split_tokens(tokens), profile)
            e.counter = frame_counter(snr, item['record']['source_index'], seed)
            wave, detail = e.rxmod.transmit_frame(e.catalogue,e.backend,e.legacy,e.header,
                profile['profile_id'],payload,e.counter)
            self.tx_detail = dict(point=point, profile=profile, waveform=detail)
            return wave
        if self.method == 'SWIN80K':
            image = e.t.from_numpy(item['pixels'].copy()).to('cuda:0').float().div(255)[None]
            data, power, indices = e.model.encode_data(image,snr,e.protocol.layout(1024)['channels'])
            self.tx_detail = dict(power=float(np.float32(float(power[0]))),
                indices=indices[0].cpu().tolist(), layout=e.protocol.layout(1024))
            return e.protocol.transmit_frame(data[0].cpu().numpy(),float(power[0]),tuple(self.tx_detail['indices']),1024)
        return e.encode(item,m)

    def observe(self, wave, item, snr, seed):
        e, sid = self.endpoint, item['record']['source_id']
        if self.method == 'VAR_UNCONDITIONAL':
            return wave + raw_noise(sid,snr,seed)*10**(-snr/20)
        if self.method == 'P1024':
            return e.plugin.apply_channel(wave,snr,sid,seed,e.plugin.CELL)
        return wave + e.protocol.standard_noise(sid,seed,1024,snr)/10**(snr/20)

    def receive(self, observed, snr):
        before = self.null_calls
        result = self.endpoint.receive(observed,snr,self.meter)
        if self.method == 'VAR_UNCONDITIONAL':
            expected = 0 if self.endpoint.gray else 1
            require(self.null_calls-before == expected, 'Each non-gray frame uses one audited null embedding')
        return result

    def evidence(self, wave, observed):
        import numpy as np
        e = self.endpoint
        arrays = dict(transmitted=np.asarray(wave),observed=np.asarray(observed))
        if self.method == 'VAR_UNCONDITIONAL':
            actual = e.last_actual
            state = e.pair.received_view(actual)
            received, mask = e.pair.known_information(state)
            arrays.update(source_tokens=e.last_tokens,received_tokens=received,known_mask=mask)
            profile = self.tx_detail['profile']; g = profile['groups'][0]
            rx_correct = bool(np.array_equal(received,e.last_tokens[:len(received)])) if len(received) else False
            header_ok = bool(actual['header']['header_ok'])
            header_correct = bool(header_ok and actual['header']['profile_id'] == profile['profile_id'])
            crc = actual['body_crc_accept']
            status = ('HEADER_REJECT_GRAY' if actual['gray'] else
                      'BODY_CRC_REJECT_KEEP' if not crc else
                      'CRC_ACCEPTED_INCORRECT_INFORMATION' if not (header_correct and rx_correct) else 'RECEIVED_OK')
            receipt = dict(actual_reception=actual, receiver_state=state, transmitter=self.tx_detail,
                header_correct=header_correct, received_tokens_equal_source_prefix=rx_correct,
                crc_accepted_incorrect_information=bool(crc and not (header_correct and rx_correct)),
                null_embedding_index=1000, true_class_supplied=False,
                audit_comparison_is_after_reconstruction=True,
                actual_decode_authority='this independent Kodak packet_events directory')
            header, body, idle = 68, g['symbols'], profile['idle_symbols']
            resource = dict(N=1024,header_complex_uses=header,body_complex_uses=body,idle_complex_uses=idle,
                m=profile['m'],K=profile['K'],source_token_count=profile['token_count'],
                modulation=g['modulation'],nominal_rate=g['nominal_rate'],k=g['layout']['k'],n=g['layout']['n'],
                effective_rate=g['layout']['k']/g['layout']['n'],received_token_count=len(received),
                header_energy=float(np.square(wave[:header]).sum()),
                body_energy=float(np.square(wave[header:header+body]).sum()),
                padding_energy=float(np.square(wave[header+body:]).sum()))
        elif self.method == 'SWIN80K':
            context = asdict(e.context)
            correct = bool(context['accepted'] and context['power'] == self.tx_detail['power']
                and list(context['indices']) == self.tx_detail['indices'])
            status = 'HEADER_REJECT_GRAY' if not context['accepted'] else ('RECEIVED_OK' if correct else 'HEADER_WRONG_ACCEPT')
            receipt = dict(actual_context=context,transmitter=self.tx_detail,
                received_context_matches_transmitted=correct, same_context_to_decoder=True,
                outside_training_and_calibration_range=(e.snr==19))
            resource = dict(N=1024,header_complex_uses=256,body_complex_uses=768,idle_complex_uses=0,body_channels=6,
                header_energy=float(np.square(wave[768:]).sum()),body_energy=float(np.square(wave[:768]).sum()),padding_energy=0.)
        else:
            status, receipt = 'RECEIVED_OK', dict(status='CONTINUOUS_NO_DISCRETE_CRC',class_conditioning=False)
            resource = dict(N=1024,header_complex_uses=0,body_complex_uses=1024,idle_complex_uses=0,
                header_energy=0.,body_energy=float(np.square(wave.astype(np.float64)).sum()),padding_energy=0.)
        resource['actual_frame_energy'] = float(np.square(wave.astype(np.float64)).sum())
        return status,receipt,resource,arrays

    def finish(self):
        if self.method == 'VAR_UNCONDITIONAL':
            self.null_hook.remove()
        self.endpoint.finish()


def preload(records):
    import numpy as np
    from PIL import Image
    items = []
    for record in records:
        require(sha(record['png']['path']) == record['png']['sha256'], 'Source PNG changed')
        with Image.open(record['png']['path']) as image:
            require(image.mode == 'RGB' and image.size == (256,256), 'Already frozen RGB center crop required')
            pixels = np.asarray(image,dtype=np.uint8).transpose(2,0,1).copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record['pixels_chw_uint8_sha256'], 'Kodak pixel identity')
        items.append(dict(record=record,pixels=pixels))
    return items


def stop(*_):
    global STOP
    STOP = True


def run(request_path, method):
    import numpy as np
    r = read(request_path)
    require(r['schema'] == SCHEMA and method in METHODS and r['methods'] == list(METHODS)
            and r['SNRs'] == SNRS and r['noise_seeds'] == SEEDS and r['packet_caps'] == CAPS
            and r['frames_per_method'] == 216 and r['N'] == 1024 and r['batch_size'] == 1
            and r['training_updates'] == 0 and r['policy_selection'] is False
            and r['original_ledger_mutated'] is False and r['true_labels_used'] is False,
            'Explicit fixed Kodak protocol required')
    require(sha(__file__) == r['script']['sha256'], 'Kodak script/request mismatch')
    require(sys.platform.startswith('linux') and os.path.abspath(sys.executable) == r['python']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Qualified UM single-GPU runtime')
    require(sorted(os.sched_getaffinity(0)) == r['affinity'] and os.getpriority(os.PRIO_PROCESS,0) == r['nice'],
            'Frozen CPU resource budget')
    a3 = pin(r['original_a3_request']); records = manifest_records(pin(r['source_manifest']))
    points,policy,catalogue = frozen_points(a3)
    require(points == r['raw_points'] and policy == r['frozen_raw_policy'] and catalogue == r['frozen_raw_catalogue'],
            'Frozen policy descriptors changed')
    additions,old_count = source_extension(a3)
    require(additions == r['native_source_bindings_additions']
            and old_count == r['original_native_source_binding_count'], 'Frozen source inventory extension changed')
    for path, expected in a3['executable_bindings'].items():
        require(a3['bindings'][path] == expected == sha(path), 'Original dependency changed: '+path)
    sys.path[:0] = a3['pythonpath']
    h = load(a3['helper_module'],'_kodak_original_endpoint_helpers',HELPER_SHA)
    out = Path(r['out'])/method
    require(not out.exists(), 'Fresh method output required; no automatic retries or overwrites')
    out.mkdir(parents=True); started = time.monotonic(); rows = []
    def guard():
        require(not STOP and not h.STOP and time.time() < r['deadline_unix']
                and time.monotonic()-started < r['max_seconds_per_method']
                and not any(Path(p).exists() for p in r['stop_files']) and not (out/'STOP').exists(), 'Kodak STOP/deadline')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    calls = Calls(out,CAPS[method],guard)
    try:
        with h.exclusive(r['visual_lock']):
            guard(); items = preload(records)
            adapter = NeuralAdapter(method,r,a3,h,calls); t = adapter.t
            prior = h.descriptor(a3['prior_completed_timing'][OLD_NAMES[method]])
            runtime = h.runtime_flags(t)
            require(prior['status'] == 'INDEPENDENT_UNIFIED_UM_ENDPOINT_TIMING_COMPLETE_V1'
                    and prior['method'] == OLD_NAMES[method]
                    and runtime == prior['runtime'] and not t.is_autocast_enabled(),
                    'Previously qualified frozen numerical runtime')
            save(out/'attempt.json',dict(method=method,request=descriptor(request_path),runtime=runtime,
                status='STARTED_NOT_COMPLETE',source_count=24,frames=216,packet_cap=CAPS[method]))
            save(out/'model_storage.json',adapter.storage)
            with t.no_grad():
                for snr in SNRS:
                    for item in items:
                        for seed in SEEDS:
                            guard(); index = item['record']['source_index']
                            if hasattr(adapter.endpoint,'native'):
                                adapter.endpoint.native.common.check()
                            case = f'snr{snr}/source{index:02d}/noise{seed}'; calls.case = case
                            first_event = len(calls.rows)
                            wave = adapter.encode(item,snr,seed)
                            observed = adapter.observe(wave,item,snr,seed)
                            require(wave.shape == observed.shape == (1024,2)
                                    and np.isfinite(wave).all() and np.isfinite(observed).all(), 'Full finite IQ frames')
                            rgb = adapter.receive(observed,snr); t.cuda.synchronize()
                            require(rgb.dtype == np.float32 and rgb.shape == (3,256,256)
                                    and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, 'Exact endpoint float RGB')
                            status,receipt,resource,arrays = adapter.evidence(wave,observed)
                            frame_dir = out/'frames'/f's{snr:02d}_i{index:02d}_n{seed}'; frame_dir.mkdir(parents=True)
                            with (frame_dir/'rgb.npz').open('xb') as f:
                                np.savez_compressed(f,rgb=rgb)
                            with (frame_dir/'physical.npz').open('xb') as f:
                                np.savez_compressed(f,**arrays)
                            save(frame_dir/'reception.json',receipt)
                            row = dict(method=method,source_index=index,source_id=item['record']['source_id'],snr_db=snr,
                                noise_seed=seed,status=status,gray=('GRAY' in status),preprocessing_id=PREPROCESSING,
                                source=item['record']['png'],pixels_chw_uint8_sha256=item['record']['pixels_chw_uint8_sha256'],
                                reconstruction=dict(**descriptor(frame_dir/'rgb.npz'),key='rgb'),
                                physical=descriptor(frame_dir/'physical.npz'),reception=descriptor(frame_dir/'reception.json'),
                                resource=resource,packet_event_indices=list(range(first_event,len(calls.rows))),
                                packet_decode_count=len(calls.rows)-first_event, true_labels_used=False)
                            save(frame_dir/'frame.json',row); rows.append(row)
                            if hasattr(adapter.endpoint,'last_f'):
                                del adapter.endpoint.last_f
                            if len(rows)%24 == 0:
                                print(json.dumps(dict(method=method,completed_frames=len(rows),total_frames=216,
                                    packet_calls=len(calls.rows))),flush=True)
                            guard()
            adapter.finish(); guard()
            expected = {(i,s,n) for i in range(24) for s in SNRS for n in SEEDS}
            require(len(rows) == 216 and {(x['source_index'],x['snr_db'],x['noise_seed']) for x in rows} == expected
                    and all(x['status'] == 'COMPLETE' for x in calls.rows), 'Complete Kodak population and PHY closure')
            save(out/'frames.json',rows); save(out/'packet_events.json',calls.rows)
            closed = [out/name for name in ('attempt.json','model_storage.json','frames.json','packet_events.json')]
            closed += [p for name in ('frames','packet_events') for p in (out/name).rglob('*') if p.is_file()]
            outputs = {str(p):sha(p) for p in sorted(closed)}
            save(out/'completion.json',dict(status='COMPLETE',schema=SCHEMA,method=method,source_count=24,
                noise_count=3,SNRs=SNRS,frames=216,request=descriptor(request_path),outputs=outputs,
                packet_attempts=len(calls.rows),packet_cap=CAPS[method],runtime=runtime,
                VAR_null_embedding_calls=adapter.null_calls,true_labels_used=False,training_updates=0,
                policy_selection=False,original_ledger_mutated=False,gray_frames=sum(x['gray'] for x in rows),
                no_failed_working_points_removed=True,warmup_frames=0,timing_repetitions=0,
                decoder_batch_size=1,decoder_batch_scope='Qualified A3 B1 endpoints; no bitwise B3 equivalence to original common500 is claimed',
                original_native_source_binding_count=old_count,registered_additional_source_count=len(additions),
                seconds=time.monotonic()-started))
            return descriptor(out/'completion.json')
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',method=method,frames=len(rows),
            packet_attempts=len(calls.rows),packet_events=calls.rows,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--a3-request',required=True)
    p.add_argument('--source-manifest',required=True); p.add_argument('--out',required=True)
    p.add_argument('--request',required=True); p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--max-seconds',type=int,default=7200)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--method',choices=METHODS,required=True)
    args = parser.parse_args()
    result = (make_request(args.a3_request,args.source_manifest,args.out,args.request,args.deadline_unix,args.max_seconds)
              if args.command == 'prepare' else run(args.request,args.method))
    print(json.dumps(result,sort_keys=True))
