"""A2: frozen native/wrapper same-observation diagnostic, never a paid baseline.

prepare reads only original calibration assets; run calls the unmodified author
forward and projects its actual masked decoder input into the wrapper IQ layout.
No training, header decoder, PHY decoder, policy search, or holdout access occurs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
VERSION = 'A2-SWIN-NATIVE-SAME-OBSERVATION-20261009-V1'
CHECKPOINT_SHA = '8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21'
CHANNEL_SHA = '84572df9fb661d5e10cc0b00dbed1db473eef94bb1822833be06b33443dc74d9'
MANIFEST_SHA = '49f344e8cf72960b7c164af96e05c203e24f4ae1cbfb8be9f3f02f8842b8e8b8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Input changed: ' + path)


def arrsha(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def prepare(args):
    """Freeze first20 old-calibration identities, source/code hashes, tolerances."""
    import numpy as np
    import torch
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / 'request.json'
    if request_path.exists():
        old = read(request_path)
        verify(old['bindings'])
        print(json.dumps({'status': 'REUSE_PREPARED', 'request': str(request_path)}), flush=True)
        return
    base = root / 'outputs/EXTERNAL-COMPARISON-20261004'
    runtime = base / 'runtime'
    registration = base / 'fixed80k_revision/training_view/registration.json'
    reg = read(registration)
    cfg = reg['config']
    require(cfg['channels'] == [6, 13] and cfg['data_N'] == [768, 1664], 'Trained rates differ')
    require(cfg['training_snr_sampling'] == 'uniform integers1..13, one independent SNR per effective batch', 'Training SNR range differs')
    require(cfg['calibration_snrs'] == [1, 4, 7, 10, 13], 'Calibration SNR range differs')
    require(len(reg['calibration_ids']) == 1000, 'Original calibration population differs')
    checkpoint = base / 'swin_training/checkpoints/model_080000.pt'
    require(sha(checkpoint) == CHECKPOINT_SHA, 'Exact80k weights differ')
    vendor = root / 'experiments/external-baseline-positioning-20260916/vendor/SwinJSCC'
    bindings = {str(registration): sha(registration), str(checkpoint): CHECKPOINT_SHA,
                str(Path(__file__).resolve()): sha(__file__)}
    for p in sorted(vendor.rglob('*.py')):
        if '__pycache__' not in p.parts:
            expected = reg['bindings'][str(p)]
            require(sha(p) == expected, 'Registered author implementation changed')
            bindings[str(p)] = expected
    require(bindings[str(vendor / 'net/channel.py')] == CHANNEL_SHA, 'Native channel identity differs')
    for name in ['swin_model.py', 'swin_training_protocol.json']:
        p = runtime / name
        expected = reg['bindings'][str(p)]
        require(sha(p) == expected, 'Registered wrapper/protocol changed')
        bindings[str(p)] = expected
    manifests = [Path(p) for p, h in reg['bindings'].items() if h == MANIFEST_SHA]
    require(len(manifests) == 1, 'Original RGB manifest is ambiguous')
    manifest_path = manifests[0]
    require(sha(manifest_path) == MANIFEST_SHA, 'Original RGB manifest changed')
    bindings[str(manifest_path)] = MANIFEST_SHA
    manifest = read(manifest_path)
    populations = [p for p in manifest['populations'] if p['name'] == 'calibration']
    require(len(populations) == 1 and populations[0]['count'] == 1000, 'Not original calibration')
    images, sources = [], []
    for descriptor in populations[0]['shards']:
        p = (manifest_path.parent / descriptor['path']).resolve()
        require(p.is_relative_to(manifest_path.parent), 'Calibration shard escapes cache')
        require(sha(p) == descriptor['sha256'] == reg['bindings'][str(p)], 'Calibration shard changed')
        bindings[str(p)] = descriptor['sha256']
        payload = torch.load(p, map_location='cpu')
        x, ids = payload['targets_u8'], payload['image_ids']
        require(x.dtype == torch.uint8 and tuple(x.shape) == (len(ids), 3, 256, 256), 'Calibration RGB layout differs')
        for j, source_id in enumerate(ids):
            i = len(sources)
            require(source_id == reg['calibration_ids'][i], 'Calibration order changed')
            a = x[j].numpy().copy()
            images.append(a)
            sources.append(dict(source_index=i, source_id=source_id, shard=str(p),
                index_in_shard=j, source_u8_sha256=arrsha(a), native_noise_seed=4101 + i))
            if len(sources) == 20:
                break
        if len(sources) == 20:
            break
    require(len(sources) == 20, 'Missing20 calibration sources')
    archive = out / 'calibration_first20.npz'
    np.savez_compressed(archive, source_u8=np.stack(images), source_ids=np.asarray([s['source_id'] for s in sources]))
    bindings[str(archive)] = sha(archive)
    support = dict(status='COMPLETE', trained_channels=[6, 13], trained_snr_db=list(range(1, 14)),
        calibrated_snr_db=[1, 4, 7, 10, 13], checkpoint_step=80000, checkpoint_sha256=CHECKPOINT_SHA,
        budget_truncated=True, scientific_convergence_proven=False,
        N1024_trained_channel_choices=[6], original_N1024=dict(channels=6, body_N=768, header_N=256),
        C7_status='UNSUPPORTED', C7_reason='The code accepts C7 but the registered checkpoint trained C6/C13 only.',
        C13_at_N1024_status='UNSUPPORTED', C13_reason='1664 body complex symbols exceed total N1024 before header.',
        header_calibration_status='NOT_RUN',
        header_calibration_reason='No supported extra body channel at N1024. Reducing header protection while retaining C6 adds no source capacity; no such policy was adopted. This is not evidence that header256 is globally optimal.',
        final_common500_status='REUSE', strategy_changed=False, holdout_rerun=False,
        snr19_status='OUTSIDE_TRAINING_AND_CALIBRATION_RANGE',
        native_side_information='Known exact mask and float32 power are free diagnostic context, not a paid-information performance baseline.')
    write(out / 'support_scope.json', support)
    request = dict(version=VERSION, root=str(root), output=str(out), runtime=str(runtime), vendor=str(vendor),
        checkpoint=str(checkpoint), registration=str(registration), source_archive=str(archive),
        bindings=bindings, source_role='original calibration, fixed first20; no outcome selection', sources=sources,
        N=1024, snr_db=13, channels=6, batch_size=1, source_count=20,
        threshold=dict(decoder_input_max_abs=2e-6, rgb_max_abs=1e-4, rgb_mean_abs=1e-5, u8_max_abs=1, mse_relation_abs=0.02),
        comparison='unmodified author native.forward vs frozen SwinCodec.receive_data on native actual decoder-input projection',
        free_context_diagnostic=True, native_awgn_forwards=20, paid_phy_decodes=0,
        training_updates=0, policy_updates=0, holdout_read=False, development_read=False)
    write(request_path, request)
    print(json.dumps({'status': 'PREPARED', 'request': str(request_path), 'sources': len(sources), 'paid_phy_decodes': 0}), flush=True)


def run(args):
    import fcntl
    import numpy as np
    import torch
    request_path = Path(args.request).resolve()
    q = read(request_path)
    require(q['version'] == VERSION, 'Wrong request version')
    out = Path(q['output'])
    lock = (out / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(q['bindings'])
    if (out / 'completion.json').exists():
        done = read(out / 'completion.json')
        require(done['request_sha256'] == sha(request_path), 'Completed request changed')
        verify(done['outputs'])
        print(json.dumps({'status': 'REUSE_COMPLETE', 'sources': done['source_count']}), flush=True)
        return
    require(not (out / 'failure.json').exists(), 'Diagnose previous failure before a new versioned run')
    require(torch.__version__.startswith('1.12.1'), 'Use the original Torch1.12.1 runtime')
    active = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    require(not ({int(x.strip()) for x in active.splitlines() if x.strip().isdigit()} - {os.getpid()}), 'Another GPU process is active; defer serial diagnostic')
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, q['runtime'])
    from swin_model import build_official, pack_data, unpack_data
    model = build_official(q['vendor'], 'cuda:0')
    payload = torch.load(q['checkpoint'], map_location='cpu')
    require(payload['step'] == 80000 and payload['registration_sha256'] == sha(q['registration']), 'Checkpoint metadata mismatch')
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.eval().requires_grad_(False)
    native = model.native
    require(native.channel.__class__.__module__ == 'net.channel', 'Unmodified native Channel required')
    require(sha(sys.modules['net.channel'].__file__) == CHANNEL_SHA, 'Native channel source differs')
    with np.load(q['source_archive'], allow_pickle=False) as z:
        source_u8, ids = z['source_u8'].copy(), z['source_ids'].tolist()
    require(ids == [s['source_id'] for s in q['sources']], 'Frozen source identities changed')
    rows, cases = [], out / 'cases'
    cases.mkdir(exist_ok=True)
    began = time.time()
    try:
        for i, spec in enumerate(q['sources']):
            x = source_u8[i]
            require(arrsha(x) == spec['source_u8_sha256'], 'Source pixels changed')
            capture = {}
            def encoder_hook(module, inputs, outputs):
                capture['features'], capture['mask'] = (v.detach().clone() for v in outputs)
            def decoder_hook(module, inputs):
                capture['native_decoder_input'] = inputs[0].detach().clone()
            handles = [native.encoder.register_forward_hook(encoder_hook), native.decoder.register_forward_pre_hook(decoder_hook)]
            torch.manual_seed(spec['native_noise_seed'])
            torch.cuda.manual_seed_all(spec['native_noise_seed'])
            image = torch.from_numpy(x[None]).cuda().float().div(255)
            try:
                with torch.no_grad():
                    result = native(image, given_SNR=q['snr_db'], given_rate=q['channels'])
            finally:
                for h in handles:
                    h.remove()
            with torch.no_grad():
                native_rgb = result[0].clamp(0, 1)
                feature, mask, noisy = capture['features'], capture['mask'], capture['native_decoder_input']
                tx_iq, power, indices = pack_data(feature, mask)
                require(power.dtype == torch.float32 and indices.shape == (1, 6), 'Context layout changed')
                require(bool((noisy[mask == 0] == 0).all()), 'Native decoder input is not actually masked')
                active = noisy.gather(2, indices[:, None, :].expand(-1, 256, -1)).flatten(1)
                normalized = active / power.sqrt()[:, None]
                half = normalized.shape[1] // 2
                observed = torch.stack([normalized[:, :half], normalized[:, half:]], dim=-1)
                restored = unpack_data(observed, power, indices)
                wrapped_rgb = model.receive_data(observed, q['snr_db'], power, indices)
                diff = (native_rgb - wrapped_rgb).abs()
                u8_native = (native_rgb * 255).round().to(torch.int16)
                u8_wrapper = (wrapped_rgb * 255).round().to(torch.int16)
                mse01 = (native_rgb - image).square().mean()
                mse255 = float(result[3])
                row = dict(source_index=i, source_id=spec['source_id'], snr_db=13, channels=6,
                    noise_seed=spec['native_noise_seed'], context_power=float(power[0]),
                    native_decoder_input_max_abs=float((restored - noisy).abs().max()),
                    rgb_max_abs=float(diff.max()), rgb_mean_abs=float(diff.mean()),
                    uint8_max_abs=int((u8_native - u8_wrapper).abs().max()),
                    uint8_changed_values=int((u8_native != u8_wrapper).sum()),
                    native_mse_01=float(mse01), native_mse_255=mse255,
                    mse_scale_relation_abs=abs(mse255 - float(mse01) * 255 ** 2),
                    native_source_psnr_db=float(-10 * torch.log10(mse01)),
                    actual_body_symbols=int(observed.shape[1]),
                    tx_complex_energy_per_symbol=float(tx_iq.square().sum(-1).mean()),
                    native_rgb_min=float(native_rgb.min()), native_rgb_max=float(native_rgb.max()),
                    native_rgb_sha256=arrsha(native_rgb[0].cpu().numpy()),
                    wrapper_rgb_sha256=arrsha(wrapped_rgb[0].cpu().numpy()),
                    native_decoder_input_sha256=arrsha(noisy[0].cpu().numpy()),
                    observed_iq_sha256=arrsha(observed[0].cpu().numpy()))
                t = q['threshold']
                row['passed'] = bool(row['native_decoder_input_max_abs'] <= t['decoder_input_max_abs']
                    and row['rgb_max_abs'] <= t['rgb_max_abs'] and row['rgb_mean_abs'] <= t['rgb_mean_abs']
                    and row['uint8_max_abs'] <= t['u8_max_abs'] and row['mse_scale_relation_abs'] <= t['mse_relation_abs']
                    and abs(row['tx_complex_energy_per_symbol'] - 2) <= 1e-5)
                require(all(math.isfinite(v) for v in row.values() if isinstance(v, float)), 'Nonfinite diagnostic result')
                archive = cases / ('%04d.npz' % i)
                np.savez_compressed(archive, source_u8=x, native_rgb=native_rgb[0].cpu().numpy(),
                    wrapper_rgb=wrapped_rgb[0].cpu().numpy(), observed_iq=observed[0].cpu().numpy(),
                    indices=indices[0].cpu().numpy(), power=power.cpu().numpy())
                row['archive'], row['archive_sha256'] = str(archive), sha(archive)
                write(cases / ('%04d.json' % i), row)
                rows.append(row)
                print(json.dumps({'source': i, 'passed': row['passed'], 'rgb_max_abs': row['rgb_max_abs'],
                    'uint8_max_abs': row['uint8_max_abs']}), flush=True)
        require(len(rows) == 20, 'Incomplete20-source diagnostic')
        with (out / 'native_parity.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        environment = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), cpu_threads=6, interop_threads=2, batch_size=1, TF32=False,
            deterministic=True, native_channel_source_sha256=CHANNEL_SHA, checkpoint_sha256=CHECKPOINT_SHA)
        write(out / 'environment.json', environment)
        passed = all(row['passed'] for row in rows)
        summary = dict(status='PASS' if passed else 'FAIL', source_count=20, request_sha256=sha(request_path),
            maximum_rgb_absolute_difference=max(row['rgb_max_abs'] for row in rows),
            maximum_uint8_absolute_difference=max(row['uint8_max_abs'] for row in rows),
            maximum_decoder_input_absolute_difference=max(row['native_decoder_input_max_abs'] for row in rows),
            seconds=time.time() - began, free_context_diagnostic=True, performance_baseline=False,
            native_awgn_forwards=20, paid_phy_decodes=0, training_updates=0, policy_updates=0,
            holdout_read=False, development_read=False,
            normalization='Native complex Es1 and sigma=1/sqrt(2gamma), followed by sqrt(2P); wrapper complex Es2 and sigma=1/sqrt(gamma), followed by sqrt(P). Same actual noisy decoder coordinates are projected, no independent noise draw.',
            shared_context='Exact same per-source channel mask and actual float32 power; diagnostic-only free context.',
            limitations='Does not establish paid header reliability, optimal header length, checkpoint convergence, untrained channel support, or 19dB in-distribution performance.')
        write(out / 'native_parity_summary.json', summary)
        require(passed, 'Native/wrapper parity exceeded a preregistered tolerance; see actual diagnostic rows')
        outputs = {str(p): sha(p) for p in [out / 'native_parity.csv', out / 'native_parity_summary.json',
            out / 'environment.json', out / 'support_scope.json', *sorted(cases.glob('*'))] if p.is_file()}
        write(out / 'completion.json', dict(status='COMPLETE', request_sha256=sha(request_path),
            source_count=20, outputs=outputs, paid_phy_decodes=0, new_holdout_frames=0, training_updates=0))
        print(json.dumps(summary), flush=True)
    except Exception as e:
        write(out / 'failure.json', dict(status='FAILED', error=repr(e), finished_cases=len(rows),
            request_sha256=sha(request_path), paid_phy_decodes=0, training_updates=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--request', required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
