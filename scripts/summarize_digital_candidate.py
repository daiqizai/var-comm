"""Summarize only frozen digital VAR transmissions and produce convergence figures without model inference."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'outputs/CONVERGENCE-AUDIT-20260915'


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write(path, values):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    raw = rows(PROJECT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001/per_frame.csv')
    canonical = rows(PROJECT / 'outputs/WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000/per_frame.csv')
    canonical_lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in canonical}
    old_lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in raw}
    mapping = {'psnr_db': 'psnr_db', 'lpips_alex': 'lpips', 'dino_cosine': 'dino'}
    max_errors = {key: 0. for key in mapping}
    for row in raw:
        if row['arm'] not in ('whole_m8', 'whole_adaptive'):
            continue
        name = 'digital_m8' if row['arm'] == 'whole_m8' else 'digital_adaptive'
        reference = canonical_lookup[int(row['image_index']), float(row['snr_db']), int(row['seed']), name]
        if row['image_id'] != reference['image_id'] or int(row['complex_uses']) != 3060:
            raise RuntimeError('original digital and current reference source/ledger differ')
        for old_key, new_key in mapping.items():
            error = abs(float(row[old_key]) - float(reference[new_key]))
            max_errors[old_key] = max(max_errors[old_key], error)
            if error > (1e-4 if old_key == 'psnr_db' else 1e-5):
                raise RuntimeError('historical/current digital quality cannot be mixed')
    snrs = [1., 4., 7., 13., 19.]
    summary = []
    for snr in snrs:
        for name in ('whole_m7', 'whole_m8', 'whole_m9', 'whole_adaptive'):
            subset = [row for row in raw if row['arm'] == name and float(row['snr_db']) == snr]
            if len(subset) != 300:
                raise RuntimeError('digital mode comparison lost transmissions')
            summary.append({'arm': name, 'snr_db': snr, 'source_images': 100, 'transmissions': len(subset),
                'PSNR_dB': float(np.mean([float(row['psnr_db']) for row in subset])),
                'LPIPS': float(np.mean([float(row['lpips_alex']) for row in subset])),
                'DINO': float(np.mean([float(row['dino_cosine']) for row in subset])),
                'protocol_prefix_accepted_rate': float(np.mean([int(row['last_accepted_scale']) > 0 for row in subset])),
                'true_output_prefix_correct_rate_diagnostic_only': float(np.mean([int(row['output_prefix_correct']) for row in subset])),
                'header_false_acceptances': sum(int(row['header_false_acceptance']) for row in subset),
                'source_CRC_false_acceptances': sum(int(row['source_crc_false_acceptance_count']) for row in subset),
                'total_complex_uses': 3060, 'total_energy': 6120, 'header_uses': 68, 'data_uses': 2992})
    write(OUTPUT / 'tables/digital_fixed_modes_and_adaptive.csv', summary)
    base = yaml.safe_load((PROJECT / 'experiments/wetok-comm-v2-20260912/configs/study.yaml').read_text())
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100, size=(10000, 100))
    paired = []
    for selected_snrs in ([1., 4., 7.], *[[snr] for snr in snrs]):
        for control in ('whole_m7', 'whole_m8', 'whole_m9'):
            for metric in mapping:
                differences = np.array([np.mean([float(old_lookup[index, snr, seed, 'whole_adaptive'][metric]) -
                    float(old_lookup[index, snr, seed, control][metric]) for snr in selected_snrs for seed in base['evaluation']['noise_seeds']]) for index in range(100)])
                low, high = np.percentile(differences[draws].mean(1), (2.5, 97.5))
                paired.append({'method': 'whole_adaptive', 'control': control, 'snrs_db': '+'.join(map(str, selected_snrs)),
                    'metric': metric, 'delta': float(differences.mean()), 'ci_low': float(low), 'ci_high': float(high),
                    'scope': 'existing_development_transmissions_not_new_test'})
    write(OUTPUT / 'tables/digital_policy_vs_all_fixed_modes.csv', paired)
    figures = OUTPUT / 'figures'
    figures.mkdir(exist_ok=True)
    labels = {'whole_m7': 'Fixed VAR m7', 'whole_m8': 'Fixed VAR m8', 'whole_m9': 'Fixed VAR m9', 'whole_adaptive': 'Adaptive digital VAR'}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, ('PSNR_dB', 'LPIPS', 'DINO')):
        for name, label in labels.items():
            values = [row for row in summary if row['arm'] == name]
            axis.plot([row['snr_db'] for row in values], [row[metric] for row in values], marker='o', label=label)
        axis.set_xlabel('SNR dB')
        axis.set_ylabel(metric)
        axis.set_xticks(snrs)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle('Frozen digital VAR: N=3060 complex uses, E=6120; 100 development sources x 3 noises\nSame source/class/header protocol; no new inference or mode tuning')
    figure.tight_layout(rect=(0, 0, 1, .88))
    for suffix in ('png', 'pdf'):
        figure.savefig(figures / ('digital_policy_same_resource.' + suffix), dpi=170, bbox_inches='tight')
    plt.close(figure)
    latest = rows(OUTPUT / 'tables/quality_supported_snrs.csv')
    display = {'digital_adaptive': 'Adaptive digital VAR', 'digital_m8': 'Fixed VAR m8',
        'perceptual_deepjscc': 'Perceptual DeepJSCC', 'r2__full_grid_innovation': 'Best completed learned receiver',
        'wetok_8PSK_FEC': 'WeTok digital 8PSK+FEC'}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, ('psnr_db', 'lpips', 'dino')):
        for name, label in display.items():
            values = [row for row in latest if row['arm'] == name]
            axis.plot([float(row['snrs_db']) for row in values], [float(row[metric]) for row in values], marker='o', label=label)
        axis.set_xlabel('SNR dB')
        axis.set_ylabel(metric)
        axis.set_xticks(snrs)
        axis.grid(alpha=.2)
    handles, texts = axes[0].get_legend_handles_labels()
    figure.legend(handles, texts, loc='lower center', ncol=3, fontsize=8)
    figure.suptitle('System candidates at common N/E: different backbones and training histories\nAll failures retained; development evidence, not independent holdout')
    figure.tight_layout(rect=(0, .13, 1, .88))
    for suffix in ('png', 'pdf'):
        figure.savefig(figures / ('candidate_systems_same_resource.' + suffix), dpi=170, bbox_inches='tight')
    plt.close(figure)
    record = {'status': 'DIGITAL_CANDIDATE_EXISTING_DATA_AUDIT_AND_FIGURES_READY', 'maximum_current_vs_original_metric_error': max_errors,
        'new_transmissions': 0, 'new_model_inference': False, 'new_parameter_search': False, 'new_holdout': False,
        'digital_policy_origin': 'frozen_development_thresholds_not_new_independent_optimization',
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT / 'digital_candidate_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
