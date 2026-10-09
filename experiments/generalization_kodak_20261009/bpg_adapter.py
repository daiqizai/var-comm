"""Finite Kodak24 inference with the unchanged, previously frozen adaptive BPG.

This wrapper only changes the source population and its independent ledger.
The existing source candidate rule, byte container, PHY and received-byte
decoder are imported by their frozen hashes. No calibration or model is run.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = 'KODAK24_FROZEN_ADAPTIVE_BPG_REQUEST_V1'
METHOD = 'BPG_ADAPTIVE'
FROZEN_METHOD = 'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'
POPULATION = 'kodak24'
SNRS = [4, 10, 19]
SEEDS = [2001, 2002, 2003]
PROFILES = {'4': 3002, '10': 3007, '19': 3013}
PREPROCESSING = 'kodak_rgb_center_crop_256_v1'
NAMESPACE = 'KODAK24_GENERALIZATION_20261009_V1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    path = str(Path(path).resolve())
    return dict(path=path, sha256=sha(path))


def pin(item):
    require(sha(item['path']) == item['sha256'], 'Pinned input changed: ' + item['path'])
    return read(item['path'])


def save(path, value):
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == data, 'Immutable output differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify(outputs):
    for path, expected in outputs.items():
        require(sha(path) == expected, 'Completed output changed: ' + path)


def guard(request):
    require(time.time() < request['deadline_unix'], 'Kodak BPG deadline reached')
    require(not any(Path(p).exists() for p in request['stop_files']), 'Kodak BPG STOP requested')


def make_request(original_adaptive_request, source_manifest, out, request_path,
                 deadline_unix, affinities=((10, 11), (12, 13)), max_seconds=3600):
    """Prepare immutable inputs only; no codecs, PHY, workers or models start.

    ``source_manifest`` is a path to JSON with a ``records`` list. Each record
    has source_index, source_id, preprocessing_id, and png={path, sha256}.
    Paths in this function refer to files on the eventual execution host.
    """
    qref = (descriptor(original_adaptive_request) if isinstance(original_adaptive_request, (str, Path))
            else dict(original_adaptive_request))
    q = pin(qref)
    mref = (descriptor(source_manifest) if isinstance(source_manifest, (str, Path)) else dict(source_manifest))
    manifest = pin(mref)
    out = Path(out).resolve()
    request_path = Path(request_path).resolve()
    empty = request_path.parent / 'no_inherited_kodak_codec_cache.json'
    save(empty, dict(status='NO_KODAK_NATIVE_STREAM_CACHE_AVAILABLE', outputs={}, scientific_result=False))
    request = dict(schema=SCHEMA, worker_sha256=sha(__file__),
        original_adaptive_request=qref, source_manifest=mref, records=manifest['records'],
        original_adaptive_freeze=descriptor(Path(q['out']) / 'freeze/worker_0.json'),
        original_adaptive_freeze_completion=descriptor(Path(q['out']) / 'freeze/completion.json'),
        original_adaptive_qualification=descriptor(Path(q['out']) / 'qualification/worker_0.json'),
        original_adaptive_qualification_completion=descriptor(Path(q['out']) / 'qualification/completion.json'),
        empty_native_cache_index=descriptor(empty),
        disabled_native_cache_root=str(out / 'NO_INHERITED_NATIVE_CODEC_CACHE'),
        out=str(out), independent_ledger=str(out / 'bpg_phy_budget.sqlite'),
        phase_caps={POPULATION: 432}, source_count=24, frame_count=216,
        SNRs=SNRS, noise_seeds=SEEDS, noise_namespace=NAMESPACE,
        frozen_selected_profile_ids=PROFILES, source_case_cap_per_image=208, source_case_cap=4992,
        resources=dict(workers=2, threads=2, nice=15, affinities=[list(x) for x in affinities]),
        deadline_unix=float(deadline_unix), max_seconds=float(max_seconds),
        stop_files=list(dict.fromkeys([*q['stop_files'], str(out.parent / 'STOP'), str(out / 'STOP')])),
        training_updates=0, policy_selection=False, calibration_calls=0,
        new_qualification_calls=0, new_metric_calls=0,
        source_selection_truth_scope='Unchanged TX source encoding MSE only; no noisy-output selection.',
        source_truth_used_by_receiver=False, original_ledger_mutation=False,
        population=POPULATION, preprocessing_id=PREPROCESSING,
        actual_children_wait_zero_required=True)
    save(request_path, request)
    # Configuration validation is read-only and performs no scientific work.
    configure(request_path)
    return request


def configure(path):
    request = read(path)
    require(request['schema'] == SCHEMA and request['worker_sha256'] == sha(__file__),
            'Kodak wrapper/request identity changed')
    require(request['SNRs'] == SNRS and request['noise_seeds'] == SEEDS
            and request['noise_namespace'] == NAMESPACE and request['source_count'] == 24
            and request['frame_count'] == 216 and request['phase_caps'] == {POPULATION: 432}
            and request['frozen_selected_profile_ids'] == PROFILES
            and request['source_case_cap_per_image'] == 208 and request['source_case_cap'] == 4992,
            'Finite Kodak scope changed')
    require(request['population'] == POPULATION and request['preprocessing_id'] == PREPROCESSING
            and not request['policy_selection'] and not request['training_updates']
            and not request['calibration_calls'] and not request['new_qualification_calls']
            and not request['new_metric_calls'] and not request['source_truth_used_by_receiver']
            and not request['original_ledger_mutation'], 'Frozen inference-only scope changed')
    resources = request['resources']
    require(resources['workers'] == 2 and resources['threads'] == 2 and resources['nice'] == 15
            and len(resources['affinities']) == 2 and all(len(x) == 2 for x in resources['affinities'])
            and len(set(sum(resources['affinities'], []))) == 4
            and all(isinstance(x, int) and x >= 0 for x in sum(resources['affinities'], [])),
            'Two separate two-CPU workers required')
    require(0 < request['max_seconds'] <= 3600, 'At most one hour per owner invocation')
    manifest = pin(request['source_manifest'])
    records = request['records']
    require(manifest['records'] == records and len(records) == 24, 'Common Kodak records changed')
    for i, record in enumerate(records):
        require(record['source_index'] == i and record['source_id'] == f'kodak/kodim{i+1:02d}'
                and record['preprocessing_id'] == PREPROCESSING, 'Wrong common Kodak source identity')
        require(sha(record['png']['path']) == record['png']['sha256'], 'Common crop PNG changed')
    q = pin(request['original_adaptive_request'])
    freeze = pin(request['original_adaptive_freeze'])
    fn = pin(request['original_adaptive_freeze_completion'])
    qualification = pin(request['original_adaptive_qualification'])
    qn = pin(request['original_adaptive_qualification_completion'])
    require(fn['status'] == qn['status'] == 'ADAPTIVE_BPG_STAGE_COMPLETE_V1'
            and fn['actual_children_waited'] and qn['actual_children_waited']
            and fn['worker_exit_codes'] == qn['worker_exit_codes'] == [0]
            and fn['outputs'].get(request['original_adaptive_freeze']['path']) == request['original_adaptive_freeze']['sha256']
            and qn['outputs'].get(request['original_adaptive_qualification']['path']) == request['original_adaptive_qualification']['sha256'],
            'Actually completed original freeze/qualification required')
    require(freeze['request_sha256'] == fn['request_sha256'] == qualification['request_sha256']
            == qn['request_sha256'] == request['original_adaptive_request']['sha256']
            and freeze['calibration_source_count'] == 100 and freeze['noise_count'] == 3
            and not freeze['holdout_used_for_selection']
            and {s: freeze['selected_profile_ids'][s] for s in PROFILES} == PROFILES,
            'Original calibration-selected policy changed')
    for p, expected in q['source_bindings'].items():
        require(sha(p) == expected, 'Original source binding changed: ' + p)
    for p, expected in q['phy_dependency_bindings'].items():
        require(sha(p) == expected, 'Original PHY dependency changed: ' + p)
    for key in ['python', 'bpgenc', 'bpgdec']:
        require(sha(q[key]['path']) == q[key]['sha256'], 'Original executable changed: ' + key)
    old = pin(q['original_request'])
    rule = pin(q['source_rule'])
    require(old['bpgenc'] == q['bpgenc'] and old['bpgdec'] == q['bpgdec']
            and rule['status'] == 'SOURCE_RULE_FROZEN_MCS_NOT_SELECTED'
            and rule['resolutions'] == [256, 128, 64, 32]
            and rule['coarse_qps'] == [0, 8, 16, 24, 32, 40, 48, 51]
            and rule['capacity_bytes'] == sorted({p['capacity_bytes'] for p in q['catalogue']})
            and len(q['catalogue']) == 15, 'Frozen source-search rule/catalogue changed')
    empty = pin(request['empty_native_cache_index'])
    require(empty == dict(status='NO_KODAK_NATIVE_STREAM_CACHE_AVAILABLE', outputs={}, scientific_result=False)
            and not Path(request['disabled_native_cache_root']).exists(), 'Cannot borrow another population source cache')
    out = Path(request['out']).resolve()
    require(Path(request['independent_ledger']).resolve().parent == out
            and out != Path(q['out']).resolve()
            and not out.is_relative_to(Path(q['out']).resolve()), 'Output/ledger must be independent from original A1')
    dirs = [str(Path(p).parent) for p in q['source_bindings']
            if p.endswith(('/adaptive_codec.py', '/bpg_codec.py', '/bpg_phy.py'))]
    sys.path[:0] = list(dict.fromkeys(dirs))
    cfg = dict(q, out=str(out), deadline_unix=request['deadline_unix'], stop_files=request['stop_files'],
        records={POPULATION: records}, original_native_codec_completion={POPULATION: request['empty_native_cache_index']},
        original_native_root=request['disabled_native_cache_root'],
        _path=str(Path(path).resolve()), _sha256=sha(path))
    profiles = {int(s): next(p for p in q['catalogue'] if p['profile_id'] == pid) for s, pid in PROFILES.items()}
    require([profiles[s]['capacity_bytes'] for s in SNRS] == [155, 315, 593]
            and all(p['N'] == 1024 and p['header_symbols'] == 68 and p['body_symbols'] == 956 for p in profiles.values()),
            'Original paid working points differ')
    return request, cfg, qualification, profiles


def source_pixels(record):
    import numpy as np
    from PIL import Image
    require(sha(record['png']['path']) == record['png']['sha256'], 'Source crop changed')
    with Image.open(record['png']['path']) as im:
        require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (256, 256),
                'Require the common uint8 RGB256 crop; wrapper never resizes/crops input')
        pixels = np.asarray(im, dtype=np.uint8).copy()
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def worker(path, index):
    request, cfg, qualification, profiles = configure(path)
    require(index in (0, 1), 'Exactly two workers')
    os.sched_setaffinity(0, request['resources']['affinities'][index])
    os.setpriority(os.PRIO_PROCESS, 0, 15)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only process required')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '2'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from adaptive_codec import source_rows, ReceiverCodec
    from bpg_codec import digest
    from bpg_phy import Link, rgb_sha
    from supplement_ledger import Ledger
    cfg['_worker'] = index
    guard(request)
    link = Link(cfg)
    require(digest(link.identity) == digest(qualification['backend_identity'])
            and digest(link.plans) == digest(qualification['layouts']), 'Already-qualified PHY implementation/layout changed')
    ledger = Ledger(request['independent_ledger'], request['phase_caps'], sha(path))
    receiver = ReceiverCodec(cfg, POPULATION)
    out = Path(request['out'])
    outputs = {}
    for ordinal in range(index, 24, 2):
        guard(request)
        record = request['records'][ordinal]
        source_cp = out / 'sources' / f'{ordinal:04d}.json'
        if source_cp.exists():
            done = read(source_cp)
            require(done['request_sha256'] == sha(path) and done['source_id'] == record['source_id'], 'Existing source differs')
            verify(done['outputs'])
            outputs[str(source_cp)] = sha(source_cp)
            outputs.update(done['outputs'])
            continue
        pixels = source_pixels(record)
        target = pixels.astype(np.float32) / np.float32(255)
        coded = source_rows(cfg, POPULATION, ordinal, pixels)
        require(len(coded['rows']) <= 208 and coded['source_id'] == record['source_id'], 'Source search cap or identity changed')
        codec_cp = out / 'codec' / POPULATION / f'{ordinal:04d}' / 'completion.json'
        local = {str(codec_cp): sha(codec_cp)}
        rows = []
        for si, snr in enumerate(SNRS):
            profile = profiles[snr]
            selected = coded['fits'][str(profile['profile_id'])]['selected']
            for ni, seed in enumerate(SEEDS):
                guard(request)
                stem = f'{ordinal:04d}_snr{snr:02d}_noise{seed}'
                cp = out / 'frames' / (stem + '.json')
                if cp.exists():
                    row = read(cp)
                    require(row['request_sha256'] == sha(path) and row['source_id'] == record['source_id']
                            and row['snr_db'] == snr and row['noise_seed'] == seed, 'Existing frame differs')
                    verify(row['outputs'])
                else:
                    archive = out / 'reconstructions' / (stem + '.npz')
                    require(not archive.exists(), 'Unreceipted image retained; no automatic frame retry')
                    image = np.full((3, 256, 256), .5, dtype=np.float32)
                    status, outcome, calls = 'SOURCE_UNFIT', None, 0
                    different = energy = tx_sha = rx_sha = received_sha = None
                    frame_outputs = {}
                    frame = f'{POPULATION}:{ordinal:04d}:SNR{snr}:noise{seed}:MCS{profile["profile_id"]}'
                    counter = ordinal * 9 + si * 3 + ni
                    if selected is not None:
                        payload = Path(selected['stream']).read_bytes()
                        require(hashlib.sha256(payload).hexdigest() == selected['stream_sha256'], 'Selected complete stream changed')
                        tx = link.transmit(payload, profile, counter)
                        rx = link.noisy(tx, request['noise_namespace'], snr, ordinal, seed)
                        tx_sha, rx_sha = link.hp.array_sha(tx), link.hp.array_sha(rx)
                        energy = float(np.square(tx.astype(np.float64)).sum())
                        outcome = link.receive(rx, snr, counter, ledger, POPULATION, frame)
                        calls, status = outcome['packet_calls'], outcome['status']
                        if outcome['body'] and outcome['body']['parser_accepted']:
                            received = base64.b64decode(outcome['body']['payload_b64'])
                            received_sha = hashlib.sha256(received).hexdigest()
                            different = received != payload
                            decoded, status, receipt = receiver.decode(received)
                            frame_outputs[receipt] = sha(receipt)
                            if decoded is not None:
                                image = decoded
                    body = dict(outcome['body']) if outcome and outcome['body'] else None
                    if body and body.get('payload_b64') is not None:
                        body['received_BPG_sha256'] = hashlib.sha256(base64.b64decode(body.pop('payload_b64'))).hexdigest()
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open('xb') as f:
                        np.savez_compressed(f, rgb=image, source_rgb=target)
                        f.flush()
                        os.fsync(f.fileno())
                    frame_outputs[str(archive)] = sha(archive)
                    mse = float(np.square(image.astype(np.float64) - target.astype(np.float64)).mean())
                    row = dict(status='KODAK24_ADAPTIVE_BPG_FRAME_COMPLETE_V1', request_sha256=sha(path),
                        method=METHOD, frozen_method_identity=FROZEN_METHOD,
                        population=POPULATION, source_index=ordinal, source_id=record['source_id'],
                        preprocessing_id=PREPROCESSING, N=1024, snr_db=snr, noise_seed=seed,
                        noise_namespace=request['noise_namespace'], frame_id=frame, counter=counter,
                        profile_id=profile['profile_id'], q=profile['q'], rate=profile['rate'], k=profile['k'], n=profile['n'],
                        header_symbols=68, body_symbols=956, padding_symbols=0, capacity_bytes=profile['capacity_bytes'],
                        selected_resolution=selected['resolution'] if selected else None,
                        selected_qp=selected['qp'] if selected else None,
                        complete_BPG_bytes=selected['complete_BPG_bytes'] if selected else None,
                        transmitted_complete_BPG_sha256=selected['stream_sha256'] if selected else None,
                        received_complete_BPG_sha256=received_sha,
                        source_encoding_fit=selected is not None, source_unfit=selected is None,
                        actual_link_executed=selected is not None, link_status=status,
                        gray_substitution=status != 'BPG_DECODED', packet_decoder_calls=calls,
                        header=outcome['header'] if outcome else None, body=body,
                        transmitted_sha256=tx_sha, observation_sha256=rx_sha, actual_frame_energy=energy,
                        undetected_payload_difference=different, mse=mse,
                        psnr_db='Infinity' if mse == 0 else -10 * math.log10(mse),
                        image_sha256=rgb_sha(image), reference_sha256=rgb_sha(target),
                        float_reconstruction=dict(path=str(archive), sha256=sha(archive), image_key='rgb',
                                                  reference_key='source_rgb', dtype='float32', layout='CHW'),
                        original_frozen_source_rule=cfg['source_rule'], original_frozen_policy=request['original_adaptive_freeze'],
                        source_truth_used_by_receiver=False, policy_selection=False, training_updates=0, outputs=frame_outputs)
                    save(cp, row)
                rows.append(row)
                local[str(cp)] = sha(cp)
                local.update(row['outputs'])
        save(source_cp, dict(status='KODAK24_ADAPTIVE_BPG_SOURCE_COMPLETE_V1', request_sha256=sha(path),
            source_index=ordinal, source_id=record['source_id'], preprocessing_id=PREPROCESSING,
            source_codec_cases=len(coded['rows']), rows=rows, outputs=local))
        outputs[str(source_cp)] = sha(source_cp)
        outputs.update(local)
        print(json.dumps(dict(worker=index, source_index=ordinal, frames=len(rows),
            decoded=sum(x['link_status'] == 'BPG_DECODED' for x in rows),
            packet_calls=sum(x['packet_decoder_calls'] for x in rows))), flush=True)
    save(out / f'worker_{index}.json', dict(status='KODAK24_ADAPTIVE_BPG_SHARD_COMPLETE_V1',
        request_sha256=sha(path), worker=index, source_indices=list(range(index, 24, 2)), outputs=outputs))


def run(path):
    import fcntl
    request, cfg, _, _ = configure(path)
    out = Path(request['out'])
    out.mkdir(parents=True, exist_ok=True)
    from supplement_ledger import Ledger, root_snapshot
    with (out / 'owner.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completion = out / 'completion.json'
        if completion.exists():
            done = read(completion)
            require(done['request_sha256'] == sha(path), 'Completed request changed')
            verify(done['outputs'])
            print('Already complete; no new work')
            return
        guard(request)
        before = root_snapshot(cfg['original_root_ledger_readonly'])
        ledger = Ledger(request['independent_ledger'], request['phase_caps'], sha(path))
        ledger.assert_resume()
        attempt = out / ('attempt_' + str(time.time_ns()))
        attempt.mkdir()
        children, handles = [], []
        started = time.monotonic()
        def interrupted(signum, _):
            raise KeyboardInterrupt('Kodak BPG owner signal ' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            for index in (0, 1):
                argv = [cfg['python']['path'], '-B', str(Path(__file__).resolve()), '--request', str(Path(path).resolve()), '--worker', str(index)]
                log = attempt / f'worker_{index}.log'
                handle = log.open('xb')
                handles.append(handle)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, env=env)
                children.append(child)
                save(attempt / f'launch_{index}.json', dict(pid=child.pid, argv=argv, worker=index, log=str(log)))
            while any(p.poll() is None for p in children):
                guard(request)
                require(time.monotonic() - started < request['max_seconds'], 'Owner wall-time cap reached')
                require(not any(p.poll() not in (None, 0) for p in children), 'Child failed; stop sibling without successor')
                time.sleep(.5)
            codes = [p.wait() for p in children]
            require(codes == [0, 0], 'Both actual children must exit zero')
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            outputs, rows, source_cases = {}, [], 0
            for index in (0, 1):
                cp = out / f'worker_{index}.json'
                done = read(cp)
                require(done['request_sha256'] == sha(path), 'Worker request changed')
                verify(done['outputs'])
                outputs[str(cp)] = sha(cp)
                outputs.update(done['outputs'])
            for index in range(24):
                done = read(out / 'sources' / f'{index:04d}.json')
                source_cases += done['source_codec_cases']
                rows.extend(done['rows'])
            require(len(rows) == 216 and {(x['source_index'], x['snr_db'], x['noise_seed']) for x in rows}
                    == {(i, s, n) for i in range(24) for s in SNRS for n in SEEDS}
                    and all(x['source_id'] == request['records'][x['source_index']]['source_id'] for x in rows)
                    and source_cases <= 4992, 'Actual grid or source case cap differs')
            snapshot = ledger.assert_resume()
            require(snapshot['total'] == sum(x['packet_decoder_calls'] for x in rows) <= 432, 'Actual paid packet count differs')
            after = root_snapshot(cfg['original_root_ledger_readonly'])
            require(after == before, 'Original root ledger changed')
            # Small common consumer contract; failures retain their fixed RGB.
            frames = [dict(method=METHOD, frozen_method_identity=FROZEN_METHOD,
                source_index=x['source_index'], source_id=x['source_id'],
                snr_db=x['snr_db'], noise_seed=x['noise_seed'], status=x['link_status'],
                population=POPULATION, preprocessing_id=PREPROCESSING,
                reconstruction=dict(path=x['float_reconstruction']['path'],
                                    sha256=x['float_reconstruction']['sha256'], key='rgb'),
                reference_sha256=x['reference_sha256'], image_sha256=x['image_sha256'],
                source_unfit=x['source_unfit'], gray_substitution=x['gray_substitution'],
                packet_decoder_calls=x['packet_decoder_calls'], profile_id=x['profile_id'],
                selected_resolution=x['selected_resolution'], selected_qp=x['selected_qp'],
                complete_BPG_bytes=x['complete_BPG_bytes'], request_sha256=sha(path)) for x in rows]
            frame_index = out / 'frames.json'
            save(frame_index, frames)
            outputs[str(frame_index)] = sha(frame_index)
            for item in attempt.iterdir():
                outputs[str(item)] = sha(item)
            save(completion, dict(status='KODAK24_ADAPTIVE_BPG_ACTUAL_CHILDREN_WAIT_ZERO',
                request_sha256=sha(path), worker_sha256=sha(__file__), actual_children_waited=True, worker_exit_codes=codes,
                population=POPULATION, source_count=24, frame_count=216, SNRs=SNRS, noise_seeds=SEEDS,
                noise_namespace=NAMESPACE, selected_profile_ids=PROFILES, source_codec_cases=source_cases,
                source_unfit_frames=sum(x['source_unfit'] for x in rows),
                decoded=sum(x['link_status'] == 'BPG_DECODED' for x in rows), independent_ledger=snapshot,
                new_PHY_call_cap=432, new_qualification_calls=0, new_metric_calls=0, calibration_calls=0,
                training_updates=0, policy_selection=False, source_truth_used_by_receiver=False,
                root_budget_before=before, root_budget_after=after,
                original_adaptive_request=request['original_adaptive_request'], original_frozen_policy=request['original_adaptive_freeze'],
                source_manifest=request['source_manifest'], outputs=outputs, elapsed_seconds=time.monotonic() - started))
            print(json.dumps(dict(status='COMPLETE', completion=str(completion), PHY_calls=snapshot['total'])), flush=True)
        except BaseException:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for handle in handles:
                if not handle.closed:
                    handle.close()
            save(attempt / 'failure.json', dict(status='FAILED_NO_AUTOMATIC_SUCCESSOR', traceback=traceback.format_exc(),
                actual_children_waited=True, exit_codes=[p.returncode for p in children]))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--original-adaptive-request')
    parser.add_argument('--source-manifest')
    parser.add_argument('--out')
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--affinities', default='10,11;12,13')
    args = parser.parse_args()
    require(args.request, '--request required')
    if args.prepare:
        require(args.worker is None and not args.check_only and args.original_adaptive_request
                and args.source_manifest and args.out and args.deadline_unix, 'Incomplete prepare arguments')
        affinities = [[int(cpu) for cpu in part.split(',')] for part in args.affinities.split(';')]
        make_request(args.original_adaptive_request, args.source_manifest, args.out, args.request,
                     args.deadline_unix, affinities, args.max_seconds)
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', request=str(Path(args.request).resolve()), request_sha256=sha(args.request))))
    elif args.check_only:
        require(args.worker is None, 'Read-only check cannot select a worker')
        request, _, _, _ = configure(args.request)
        for record in request['records']:
            source_pixels(record)
        print(json.dumps(dict(status='PINNED_INPUTS_CHECKED_NO_SCIENTIFIC_EXECUTION', sources=24, planned_frames=216, max_PHY_calls=432)))
    elif args.worker is None:
        run(args.request)
    else:
        def interrupted(signum, _):
            raise KeyboardInterrupt('Kodak BPG worker signal ' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
        worker(args.request, args.worker)


if __name__ == '__main__':
    main()
