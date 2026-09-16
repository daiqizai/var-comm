"""Fixed-resource development evaluation of a sealed WeTok training milestone."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, configure_torch, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.digital import WeTokDigital
from wetok_comm.evaluation import LegacyReferences, metric_models, metrics, raw_noise
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, new_network, read_population


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: evaluate selected milestone models; never selects on development')
        return
    configure_torch()
    training = output_path(config, 'training')
    milestone_path = training / 'milestones' / f'step_{arguments.milestone:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if milestone['status'] != 'MILESTONE_COMPLETE' or milestone['updates_per_arm'] != arguments.milestone:
        raise RuntimeError('milestone is not complete')
    verify_sources(milestone['source_hashes'])
    preserved = training / 'milestones' / f'step_{arguments.milestone:07d}_optimizer.pt'
    if not preserved.exists() and sha256(training / 'resume.pt') == milestone['checkpoint_sha256']:
        os.link(training / 'resume.pt', preserved)
    if preserved.exists():
        if sha256(preserved) != milestone['checkpoint_sha256']:
            raise RuntimeError('preserved model/optimizer milestone checkpoint changed')
        write_json(training / 'milestones' / f'step_{arguments.milestone:07d}_optimizer.json',
                   {'checkpoint': str(preserved.relative_to(training)), 'sha256': milestone['checkpoint_sha256'],
                    'updates_per_arm': arguments.milestone, 'storage': 'hard_link_no_large_asset_copy'})
    output = (arguments.output_dir or output_path(config, 'evaluation') / f'step_{arguments.milestone:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('evaluation must stay in VAR_COMM outputs')
    if (output / 'completion.json').exists():
        raise FileExistsError('refusing to replace a frozen evaluation')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['milestone_sha256'] != sha256(milestone_path):
            raise RuntimeError('milestone changed after evaluation started')
    else:
        output.mkdir(parents=True, exist_ok=False)
        sources = snapshot(output, [Path(__file__), arguments.config, EXPERIMENT / 'docs/protocol.md',
            *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py')),
            WORKSPACE / 'var-next-scale-comm/src/var_comm/scale_channel.py', WORKSPACE / 'var-next-scale-comm/src/var_comm/token_trellis.cpp'])
        metadata = {'milestone_sha256': sha256(milestone_path), 'source_hashes': sources, 'created_local': now(),
                    'selection_population': 'calibration_not_development', 'legacy_latency_not_compared': True}
        write_json(output / 'metadata.json', metadata)
    started = time.time()
    device = torch.device('cuda:0')
    native = FrozenWeTok(device, 'both')
    perceptual, dino = metric_models(device)
    networks = {}
    for arm in config['arms']:
        selected = milestone['selected'][arm]
        if selected is None:
            raise RuntimeError('no selected calibrated model; do not substitute warmup as a strong control')
        path = training / selected['checkpoint']
        if sha256(path) != selected['checkpoint_sha256']:
            raise RuntimeError('selected model weights changed')
        network = new_network(config, arm, device)
        network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
        networks[arm] = network.eval().requires_grad_(False)
    model_objects = {'native': native.codec, 'lpips': perceptual, 'dino': dino, **networks}
    before = {name: module_sha256(model) for name, model in model_objects.items()}
    images, codes, identifiers = read_population(config, 'development')
    references = LegacyReferences()
    modem = WeTokDigital()
    rows, clean_rows, diagnostic_rows = [], [], []
    controls = ['digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
    max_power_error = 0.
    for index, identifier in enumerate(identifiers):
        directory = output / 'images' / f'{index:03d}'
        if (directory / 'receipt.json').exists():
            receipt = json.loads((directory / 'receipt.json').read_text())
            for relative, expected in receipt['output_hashes'].items():
                if sha256(directory / relative) != expected:
                    raise RuntimeError('resumed source-image evaluation is corrupted')
            with (directory / 'per_frame.csv').open() as handle:
                rows.extend(csv.DictReader(handle))
            with (directory / 'clean.csv').open() as handle:
                clean_rows.extend(csv.DictReader(handle))
            with (directory / 'noiseless.csv').open() as handle:
                diagnostic_rows.extend(csv.DictReader(handle))
            max_power_error = max(max_power_error, receipt['maximum_power_error'])
            continue
        directory.mkdir(parents=True, exist_ok=True)
        source = images[index:index + 1].to(device)
        grouped = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
        source_fq = indices_to_features(grouped)
        if index == 0:
            native.encode(source)
            native.decode(source_fq)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        fresh_codes = native.encode(source)
        torch.cuda.synchronize()
        encoding_seconds = time.perf_counter() - tick
        if not torch.equal(fresh_codes, grouped):
            raise RuntimeError('actual online Encoder differs from cached native source indices')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        clean = native.decode(source_fq)
        torch.cuda.synchronize()
        clean_decoder_seconds = time.perf_counter() - tick
        clean_metric = metrics(source, [clean[0].cpu()], perceptual, dino)[0]
        clean_row = {'image_index': index, 'image_id': identifier, 'reference': 'correct_native_Fq_not_a_wireless_method',
                     'raw_bits': 8192, 'online_encoder_seconds': encoding_seconds, 'native_decoder_seconds': clean_decoder_seconds, **clean_metric}
        clean_rows.append(clean_row)
        rendered, image_map, local_rows, local_diagnostic, predicted, waveforms = [], {}, [], [], {}, {}
        legacy_metrics = {}
        def remember(image):
            tensor = image.detach().cpu().contiguous()
            digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if digest not in image_map:
                image_map[digest] = len(rendered)
                rendered.append(tensor)
            return image_map[digest], digest
        native_u8 = grouped[0].cpu().numpy()
        tick = time.perf_counter()
        digital_signal = modem.transmit(native_u8)
        digital_tx_seconds = time.perf_counter() - tick
        truth_bits = np.unpackbits(native_u8.reshape(-1), bitorder='little')
        local_power = 0.
        for snr in config['evaluation']['snrs_db']:
            snrs = torch.tensor([snr], device=device)
            for seed in config['evaluation']['noise_seeds']:
                noise64 = raw_noise(identifier, seed)
                noise_hash = hashlib.sha256(noise64.tobytes()).hexdigest()
                noise = torch.tensor(noise64[None], dtype=torch.float32, device=device)
                common = {'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                          'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': noise_hash}
                for arm, network in networks.items():
                    if index == 0 and snr == config['evaluation']['snrs_db'][0] and seed == config['evaluation']['noise_seeds'][0]:
                        for warmup in range(2):
                            warm_signal = network.transmit(source_fq, snrs)
                            native.decode(network.receive(warm_signal + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None], snrs)['native_fq'])
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    transmitted = network.transmit(source_fq, snrs)
                    torch.cuda.synchronize()
                    tx_seconds = time.perf_counter() - tick
                    received = transmitted + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    estimate = network.receive(received, snrs)
                    image = native.decode(estimate['native_fq'])
                    torch.cuda.synchronize()
                    rx_seconds = time.perf_counter() - tick
                    reference, digest = remember(image[0])
                    power = float(transmitted.square().sum(-1).mean())
                    local_power = max(local_power, abs(power - 2))
                    key = f'{arm}_{snr}_{seed}'
                    predicted[key] = estimate['native_fq'][0].gt(0).cpu().numpy()
                    waveforms[key + '_tx'] = transmitted[0].cpu().numpy()
                    waveforms[key + '_rx'] = received[0].cpu().numpy()
                    local_rows.append({**common, 'arm': arm, 'header_uses': 0, 'data_uses': 3060, 'raw_bits': 8192,
                        'coded_bits': '', 'bit_error_rate': float(estimate['native_fq'].ne(source_fq).float().mean()), 'crc_accepted': '',
                        'image_ref': reference, 'image_sha256': digest, 'online_TX_seconds': encoding_seconds + tx_seconds,
                        'receiver_seconds': rx_seconds, 'latency_source': 'current_host_complete_RX_encoder_cost_in_TX',
                        'selected_step': milestone['selected'][arm]['step']})
                digital_received = digital_signal + noise64 / np.sqrt(10 ** (snr / 10))
                tick = time.perf_counter()
                decoded = modem.receive(digital_received, snr)
                candidate = torch.from_numpy(decoded['indices'][None]).to(device)
                image = native.decode(indices_to_features(candidate))
                torch.cuda.synchronize()
                digital_rx_seconds = time.perf_counter() - tick
                reference, digest = remember(image[0])
                local_rows.append({**common, 'arm': 'wetok_8PSK_FEC', 'header_uses': 0, 'data_uses': 3060, 'raw_bits': 8192,
                    'coded_bits': 9180, 'bit_error_rate': float(np.mean(decoded['decoded_bits'] != truth_bits)), 'crc_accepted': bool(decoded['crc_accepted']),
                    'image_ref': reference, 'image_sha256': digest, 'online_TX_seconds': encoding_seconds + digital_tx_seconds,
                    'receiver_seconds': digital_rx_seconds, 'latency_source': 'current_host_FEC_plus_native_RX', 'selected_step': ''})
                for arm in controls:
                    image, legacy = references.image(index, identifier, snr, seed, arm, noise_hash)
                    reference, digest = remember(image)
                    legacy_metrics[arm, snr, seed] = {'psnr_db': float(legacy['psnr_db']),
                        'lpips': float(legacy['lpips_alex']), 'dino': float(legacy['dino_cosine'])}
                    source_bits = '' if arm == 'perceptual_deepjscc' else (3060 if arm == 'digital_m8' else (1860 if snr < 2.5 else 3060 if snr < 5.5 else 5088))
                    local_rows.append({**common, 'arm': arm, 'header_uses': int(legacy['header_uses']), 'data_uses': int(legacy['data_uses']),
                        'raw_bits': source_bits, 'coded_bits': '' if arm == 'perceptual_deepjscc' else 6120,
                        'bit_error_rate': '', 'crc_accepted': '',
                        'image_ref': reference, 'image_sha256': digest, 'online_TX_seconds': '', 'receiver_seconds': '',
                        'latency_source': 'historical_not_ranked', 'selected_step': ''})
        for arm, network in networks.items():
            snrs = torch.tensor([19.], device=device)
            noiseless = network.receive(network.transmit(source_fq, snrs), snrs)
            reference, digest = remember(native.decode(noiseless['native_fq'])[0])
            local_diagnostic.append({'image_index': index, 'image_id': identifier, 'arm': arm,
                'channel': 'noiseless_with_nominal_19dB_condition_not_an_AWGN_ranking',
                'complex_uses': 3060, 'image_ref': reference, 'image_sha256': digest,
                'bit_error_rate': float(noiseless['native_fq'].ne(source_fq).float().mean())})
        scores = metrics(source, rendered, perceptual, dino)
        for row in local_rows + local_diagnostic:
            row.update(scores[int(row['image_ref'])])
            row['LPIPS_excess_from_native'] = row['lpips'] - clean_metric['lpips']
            row['severe_distortion'] = int(row['LPIPS_excess_from_native'] >= .15)
            if row['arm'] in controls:
                saved = legacy_metrics[row['arm'], row['snr_db'], row['seed']]
                for metric, tolerance in (('psnr_db', 1e-4), ('lpips', 1e-5), ('dino', 1e-5)):
                    if abs(row[metric] - saved[metric]) > tolerance:
                        raise RuntimeError(f'unified evaluator does not reproduce legacy {metric}: {row["arm"]}')
        if local_power > 1e-5:
            raise RuntimeError('neural energy budget violated')
        write_csv(directory / 'per_frame.csv', local_rows)
        write_csv(directory / 'clean.csv', [clean_row])
        write_csv(directory / 'noiseless.csv', local_diagnostic)
        np.savez_compressed(directory / 'reconstructions.npz', images=torch.stack(rendered).numpy())
        np.savez(directory / 'waveforms.npz', **waveforms)
        np.savez_compressed(directory / 'native_bits.npz', **predicted)
        write_json(directory / 'receipt.json', {'maximum_power_error': local_power, 'output_hashes': artifact_hashes(directory)})
        rows.extend(local_rows)
        diagnostic_rows.extend(local_diagnostic)
        max_power_error = max(max_power_error, local_power)
        write_csv(output / 'per_frame.csv', rows)
        write_json(output / 'status.json', {'status': 'EVALUATING_DEVELOPMENT', 'completed_images': index + 1, 'local_time': now()})
        print(f'development {index + 1}/100 rows={len(rows)}', flush=True)
    expected = 100 * 7 * 3 * 7
    if len(rows) != expected or len({(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in rows}) != expected:
        raise RuntimeError('incomplete or duplicate wireless comparison rows')
    after = {name: module_sha256(model) for name, model in model_objects.items()}
    if after != before:
        raise RuntimeError('evaluation changed model parameters')
    verify_sources(metadata['source_hashes'])
    write_csv(output / 'native_reference.csv', clean_rows)
    write_csv(output / 'noiseless_mapping.csv', diagnostic_rows)
    write_json(output / 'status.json', {'status': 'EVALUATION_COMPLETE', 'completed_images': 100, 'local_time': now()})
    write_json(output / 'completion.json', {'status': 'EVALUATION_COMPLETE', 'completed_local': now(), 'rows': len(rows),
        'source_hashes': metadata['source_hashes'], 'milestone_sha256': sha256(milestone_path),
        'frozen_before': before, 'frozen_after': after, 'maximum_power_error': max_power_error,
        'GPU_hours_this_session': (time.time() - started) / 3600, 'output_hashes': artifact_hashes(output)})
    print('MILESTONE_DEVELOPMENT_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
