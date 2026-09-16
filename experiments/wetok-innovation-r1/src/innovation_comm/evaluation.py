"""Shared-waveform receiver comparisons and read-only stronger-system references."""

import csv
import hashlib
import json

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, load_parent, new_system, output_path, settings
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.interface_evaluation import FrozenImageReferences, METRICS, SYSTEM_REFERENCES, load_evaluation_config
from wetok_comm.training import module_sha256


PARENT_VARIANTS = ('single_pass', 'multiscale_no_history', 'multiscale_conditioned')


def load_evaluation():
    evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
    config, base, parent_milestone = settings()
    if (evaluation['population'] != 'original_100_development_only' or evaluation['all_snrs'] != base['evaluation']['snrs_db'] or
        evaluation['primary_snrs'] != base['evaluation']['primary_snrs_db'] or evaluation['main_method_count'] != 13 or
        evaluation['main_rows'] != 27300 or evaluation['fixed_support_Deep_rows'] != 600):
        raise ValueError('receiver study cannot silently change the development population or SNR grid')
    return evaluation, config, base, parent_milestone


def receiver_name(variant):
    return 'receiver__' + variant


def validate_population(identifiers):
    if len(identifiers) != 100 or len(set(identifiers)) != 100:
        raise RuntimeError('evaluation requires the fixed 100 distinct development images')


def reference_names():
    return ['parent_geometry__' + name for name in PARENT_VARIANTS] + list(SYSTEM_REFERENCES)


def comparison_pairs(config):
    innovation = receiver_name('multiscale_innovation')
    pairs = [(innovation, receiver_name(name)) for name in config['variants'] if name != 'multiscale_innovation']
    pairs.extend([(receiver_name('multiscale_state_history'), receiver_name('multiscale_no_history')),
                  (receiver_name('full_grid_innovation'), receiver_name('single_pass')),
                  (receiver_name('multiscale_prediction_features'), receiver_name('multiscale_state_history')),
                  (receiver_name('multiscale_no_history'), receiver_name('single_pass'))])
    pairs.extend((receiver_name(name), 'parent_geometry__single_pass') for name in config['variants'])
    pairs.extend((receiver_name(name), control) for name in config['variants'] for control in SYSTEM_REFERENCES)
    pairs.extend((innovation, 'parent_geometry__' + name) for name in ('multiscale_no_history', 'multiscale_conditioned'))
    if len(pairs) != len(set(pairs)):
        raise RuntimeError('duplicate predeclared receiver contrasts')
    return pairs


def audited_milestone(config, step):
    if step not in config['calibration']['full_steps'] or step <= 0:
        raise ValueError('evaluation requires a registered complete-calibration milestone')
    training = output_path(config, 'training')
    milestone_path = training / 'milestones' / f'step_{step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if milestone['status'] != 'INNOVATION_MILESTONE_COMPLETE' or milestone['receiver_updates_per_arm'] != step:
        raise RuntimeError('receiver training milestone is incomplete')
    verify_sources(milestone['source_hashes'])
    if sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']:
        raise RuntimeError('receiver training/optimizer checkpoint changed')
    review_path = output_path(config, 'analysis') / f'calibration_{step:07d}/completion.json'
    review = json.loads(review_path.read_text())
    verify_sources(review['source_hashes'])
    if (review['shared_y_source_noise_frozen_encoder_Adam_selection_audit'] != 'PASS' or
        review['milestone_sha256'] != sha256(milestone_path) or review['receiver_updates'] != step or
        review['selected_steps'] != {name: choice['step'] for name, choice in milestone['selected'].items()}):
        raise RuntimeError('shared-waveform training/calibration review has not passed')
    for relative, expected in review['output_hashes'].items():
        if sha256(review_path.parent / relative) != expected:
            raise RuntimeError('frozen calibration review artifact changed')
    return milestone, milestone_path, review_path


def model_registry(config, base, milestone, device):
    parent = load_parent(config, base, device)
    encoder_hash = module_sha256(parent.encoder)
    if encoder_hash != milestone['frozen']['encoder'] or set(milestone['selected']) != set(config['variants']):
        raise RuntimeError('selected receivers do not share the pinned transmitter')
    training = output_path(config, 'training')
    networks, selections = {}, {}
    for variant in config['variants']:
        choice = milestone['selected'][variant]
        path = (training / choice['checkpoint']).resolve()
        if (not path.is_relative_to(training) or choice['scope'] != 'full' or choice['source_images'] != 1000 or
            choice['step'] not in config['calibration']['full_steps'] or choice['step'] > milestone['receiver_updates_per_arm'] or
            sha256(path) != choice['checkpoint_sha256']):
            raise RuntimeError('selected receiver checkpoint is not an eligible full-calibration choice')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['variant'] != variant or stored['step'] != choice['step']:
            raise RuntimeError('selected receiver variant or step changed')
        network = new_system(parent, variant, config, device)
        trainable = sum(value.numel() for value in network.parameters() if value.requires_grad)
        network.load_state_dict(stored['model'], strict=True)
        if module_sha256(network.encoder) != encoder_hash:
            raise RuntimeError('a selected receiver uses a different transmitter')
        name = receiver_name(variant)
        networks[name] = network.eval().requires_grad_(False)
        selections[name] = {**choice, 'absolute_checkpoint': str(path), 'receiver_trainable_parameters': trainable}
    return parent, networks, selections


class References:
    def __init__(self, evaluation):
        old_config, old_study, unused_base, unused_parent = load_evaluation_config()
        self.legacy = FrozenImageReferences(old_config, old_study)
        self.root = PROJECT / 'outputs' / evaluation['reference_geometry_evaluation']
        if sha256(self.root / 'completion.json') != evaluation['reference_receipt_sha256']:
            raise RuntimeError('frozen geometry reference receipt changed')
        self.receipt = json.loads((self.root / 'completion.json').read_text())
        if self.receipt['status'] != 'GEOMETRY_EVALUATION_COMPLETE':
            raise RuntimeError('parent geometry evaluation is incomplete')
        verify_sources(self.receipt['source_hashes'])
        if sha256(self.root / 'per_frame.csv') != self.receipt['output_hashes']['per_frame.csv']:
            raise RuntimeError('parent reference source/metrics changed')
        with (self.root / 'per_frame.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
        if len(self.rows) != 21000 or len(self.rows) != len(rows):
            raise RuntimeError('parent reference grid is incomplete')
        self.index, self.images, self.signals = None, None, None

    def _load_source(self, index):
        if self.index == index:
            return
        for filename in ('reconstructions.npz', 'waveforms.npz'):
            relative = f'images/{index:03d}/{filename}'
            if sha256(self.root / relative) != self.receipt['output_hashes'][relative]:
                raise RuntimeError('parent reference images or signal archive changed')
        with np.load(self.root / f'images/{index:03d}/reconstructions.npz', allow_pickle=False) as data:
            self.images = data['images'].copy()
        with np.load(self.root / f'images/{index:03d}/waveforms.npz', allow_pickle=False) as data:
            self.signals = {key: data[key].copy() for key in data.files if key.startswith('geometry204x30__single_pass__snr')}
        self.index = index

    def parent_signal(self, index, snr):
        self._load_source(index)
        return torch.from_numpy(self.signals[f'geometry204x30__single_pass__snr{float(snr)}'].copy())

    def image(self, index, identifier, snr, seed, name, noise_hash):
        if name in SYSTEM_REFERENCES:
            return self.legacy.image(index, identifier, snr, seed, name, noise_hash)
        if name not in reference_names():
            raise ValueError('unknown receiver-study reference')
        variant = name.split('__', 1)[1]
        row = self.rows[index, float(snr), int(seed), 'geometry204x30__' + variant]
        if row['image_id'] != identifier or row['noise_sha256'] != noise_hash:
            raise RuntimeError('parent source/noise pairing differs')
        if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, 0, 3060):
            raise RuntimeError('parent physical ledger differs')
        self._load_source(index)
        image = self.images[int(row['image_ref'])]
        if image.dtype != np.float32 or image.shape != (3, 256, 256) or hashlib.sha256(image.tobytes()).hexdigest() != row['image_sha256']:
            raise RuntimeError('parent image does not match its frozen result row')
        adapted = {**row, 'selected_step': row['selected_image_step'], 'bit_error_rate': row['bit_error_rate'], 'crc_accepted': ''}
        return torch.from_numpy(image.copy()), adapted, str(self.root / f'images/{index:03d}/reconstructions.npz')


def validate_rows(rows, config, base):
    methods = [receiver_name(name) for name in config['variants']]
    names = methods + reference_names()
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, snr, seed, name) for index in range(100) for snr in snrs for seed in seeds for name in names}
    if len(rows) != len(lookup) or set(lookup) != expected:
        raise RuntimeError('receiver comparison must contain all 27300 registered rows')
    distinct_sources, encoders, update_budgets = set(), set(), set()
    for index in range(100):
        source_ids = set()
        transmitted_by_snr = {snr: set() for snr in snrs}
        for seed in seeds:
            noise_hashes = set()
            for snr in snrs:
                signals, observations = set(), set()
                for name in names:
                    row = lookup[index, snr, seed, name]
                    source_ids.add(row['image_id'])
                    noise_hashes.add(row['noise_sha256'])
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header) or abs(float(row['total_energy']) - 6120) > 1e-5:
                        raise RuntimeError('receiver/reference physical N/E differs')
                    if name in methods:
                        signals.add(row['shared_transmitted_sha256'])
                        observations.add(row['shared_received_sha256'])
                        transmitted_by_snr[snr].add(row['shared_transmitted_sha256'])
                        encoders.add(row['shared_encoder_sha256'])
                        updates, selected = int(row['available_receiver_updates']), int(row['selected_receiver_step'])
                        update_budgets.add(updates)
                        if (updates not in config['calibration']['full_steps'] or updates <= 0 or
                            selected not in config['calibration']['full_steps'] or selected > updates or
                            int(row['selected_global_data_step']) != config['training']['global_data_start'] + selected):
                            raise RuntimeError('receiver updates or selected checkpoint eligibility differs')
                        if any(len(str(row[key])) != 64 for key in ('shared_transmitted_sha256', 'shared_received_sha256', 'shared_encoder_sha256')):
                            raise RuntimeError('missing per-frame shared-waveform evidence')
                        if row['decoder_interface'] != 'continuous_mean' or row['encoder_frozen'] not in (True, 'True', 1, '1'):
                            raise RuntimeError('receiver-study information or output interface changed')
                    if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                        raise RuntimeError('nonfinite outcome cannot be dropped')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('severe-distortion definition changed')
                if len(signals) != 1 or len(observations) != 1:
                    raise RuntimeError('trained receiver arms did not use exactly the same s/y')
            if len(noise_hashes) != 1:
                raise RuntimeError('reference channel draws differ')
        if len(source_ids) != 1:
            raise RuntimeError('source-image grouping changed')
        distinct_sources.update(source_ids)
        if any(len(values) != 1 for values in transmitted_by_snr.values()):
            raise RuntimeError('transmitter depends on receiver noise seed')
    if len(distinct_sources) != 100 or len(encoders) != 1 or len(update_budgets) != 1:
        raise RuntimeError('distinct-source count, common encoder or equal training opportunity changed')
    return lookup, names


def validate_selection(rows, milestone):
    for row in rows:
        if not row['arm'].startswith('receiver__'):
            continue
        choice = milestone['selected'][row['arm'].split('__', 1)[1]]
        if (int(row['selected_receiver_step']) != choice['step'] or
            row['selected_checkpoint_sha256'] != choice['checkpoint_sha256'] or
            int(row['available_receiver_updates']) != milestone['receiver_updates_per_arm'] or
            row['shared_encoder_sha256'] != milestone['frozen']['encoder']):
            raise RuntimeError('development result used a different selection or frozen transmitter')


def validate_native_reference(rows, clean_rows, noiseless_rows, config, base):
    clean = {int(row['image_index']): row for row in clean_rows}
    if len(clean_rows) != 100 or set(clean) != set(range(100)):
        raise RuntimeError('native reference must contain all source images')
    validate_population([row['image_id'] for row in clean_rows])
    expected = {(index, receiver_name(variant)) for index in range(100) for variant in config['variants']}
    if len(noiseless_rows) != len(expected) or {(int(row['image_index']), row['arm']) for row in noiseless_rows} != expected:
        raise RuntimeError('noiseless mapping diagnostic is incomplete')
    for row in clean_rows:
        if not all(np.isfinite(float(row[metric])) for metric in ('psnr_db', 'ssim', 'lpips', 'dino')):
            raise RuntimeError('native reference has nonfinite metrics')
    for row in rows + noiseless_rows:
        reference = clean[int(row['image_index'])]
        if (row['image_id'] != reference['image_id'] or not all(np.isfinite(float(row[metric])) for metric in METRICS) or
            abs(float(row['lpips']) - float(reference['lpips']) - float(row['LPIPS_excess_from_native'])) > 1e-10 or
            float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15)):
            raise RuntimeError('native-relative quality anchor or source grouping changed')
    for row in noiseless_rows:
        if row['channel'] != 'noiseless_nominal19_not_wireless_ranking' or int(row['complex_uses']) != 3060:
            raise RuntimeError('noiseless diagnostic was relabelled as a wireless outcome')


def statistics(rows, config, base):
    lookup, names = validate_rows(rows, config, base)
    sampled = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100,
        size=(base['evaluation']['bootstrap_resamples'], 100))
    summary, paired = [], []
    for snrs in [base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]]:
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, seed, name][metric])
            for snr in snrs for seed in base['evaluation']['noise_seeds']]) for index in range(100)]) for metric in METRICS} for name in names}
        for name in names:
            frames = [lookup[index, snr, seed, name] for index in range(100) for snr in snrs for seed in base['evaluation']['noise_seeds']]
            row = {'arm': name, 'snrs_db': label, 'source_images': 100, **{metric: float(value.mean()) for metric, value in values[name].items()}}
            for key in ('online_TX_seconds', 'receiver_seconds'):
                durations = [float(frame[key]) for frame in frames if frame[key] != '']
                row[key + '_mean'] = float(np.mean(durations)) if durations else ''
                row[key + '_p95'] = float(np.percentile(durations, 95)) if durations else ''
            summary.append(row)
        for method, reference in comparison_pairs(config):
            for metric in METRICS:
                difference = values[method][metric] - values[reference][metric]
                low, high = np.percentile(difference[sampled].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': method, 'control': reference, 'metric': metric,
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summary, paired


def supplement_statistics(rows, support_rows, config, base):
    from wetok_comm.deep_support import supplement_statistics as reduce_support

    return reduce_support(rows, support_rows, {'variants': config['variants'], 'interfaces': ['receiver']}, base)
