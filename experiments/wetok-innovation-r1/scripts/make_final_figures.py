"""Read-only fixed-budget result figures and predeclared-source reconstructions; no inference."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(JOINT / 'src'), str(EXPERIMENT / 'src'), str(BASE / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
from PIL import Image

from joint_sender.evaluation import load_evaluation
from joint_sender.references import References
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME
from wetok_comm.training import read_population


METHODS = {
    'receiver__single_pass': ('Single pass', 'C0'),
    'receiver__multiscale_innovation': ('Multiscale innovation', 'C4'),
    'receiver__full_grid_innovation': ('Full-grid innovation', 'C5'),
    'digital_m8': ('Digital fixed m8', 'C7'),
    'digital_adaptive': ('Digital adaptive', 'C2'),
    'perceptual_deepjscc': ('Perceptual DeepJSCC', 'C1'),
    'wetok_8PSK_FEC': ('WeTok digital 8PSK/FEC', 'C3'),
}


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: frozen result curves and source indices 0/37/99, SNR1/7/19, noise2001; no new inference')
        return
    evaluation, config, reference, base, parent = load_evaluation()
    references = References(evaluation)
    analysis = PROJECT / 'outputs' / evaluation['receiver_reference_analysis']
    receipt = json.loads((analysis / 'completion.json').read_text())
    if receipt['status'] != 'INNOVATION_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE':
        raise RuntimeError('only completed receiver results may be plotted')
    verify_sources(receipt['source_hashes'])
    for filename in ('summary.csv', 'paired.csv', 'deep_support_summary.csv', 'native_reference_summary.json'):
        if sha256(analysis / filename) != receipt['output_hashes'][filename]:
            raise RuntimeError('frozen receiver analysis input changed')
    summaries, paired = read_rows(analysis / 'summary.csv'), read_rows(analysis / 'paired.csv')
    support = read_rows(analysis / 'deep_support_summary.csv')
    native = json.loads((analysis / 'native_reference_summary.json').read_text())
    output = PROJECT / 'outputs/WETOK-INNOVATION-R1-FIGURES-20260914'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__)])
    snrs = base['evaluation']['snrs_db']
    figure, axes = pyplot.subplots(1, 3, figsize=(17, 5))
    for axis, metric, title in zip(axes, ('psnr_db', 'lpips', 'dino'), ('PSNR (higher better)', 'LPIPS (lower better)', 'DINO cosine (higher better)')):
        for name, (label, color) in METHODS.items():
            values = [float(next(row[metric] for row in summaries if row['arm'] == name and row['snrs_db'] == str(snr))) for snr in snrs]
            axis.plot(snrs, values, marker='o', linewidth=1.4, color=color, label=label)
        support_values = [float(next(row[metric] for row in support if row['arm'] == SUPPORT_NAME and row['snrs_db'] == str(snr))) for snr in (5., 6.)]
        axis.scatter([5., 6.], support_values, marker='x', s=60, color='black', label='Deep fixed-support supplement')
        axis.axhline(native[metric], color='.5', linestyle=':', label='Correct native codec (not wireless)')
        axis.set_xlabel('Actual SNR (dB)')
        axis.set_title(title)
        axis.set_xticks(snrs)
        axis.grid(alpha=.25)
    figure.suptitle('Development: N=3060 complex uses, total energy=6120; 100 sources x 3 noises')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    figure.tight_layout(rect=(0, .18, 1, .92))
    figure.savefig(output / 'all_snr_strong_comparisons.png', dpi=160)
    figure.savefig(output / 'all_snr_strong_comparisons.pdf')
    pyplot.close(figure)
    contrasts = [
        ('receiver__multiscale_innovation', 'receiver__single_pass', 'Multiscale - single'),
        ('receiver__multiscale_innovation', 'receiver__multiscale_state_history', 'Multiscale - state history'),
        ('receiver__multiscale_innovation', 'receiver__multiscale_prediction_features', 'Multiscale - prediction features'),
        ('receiver__multiscale_innovation', 'receiver__full_grid_innovation', 'Multiscale - full-grid'),
        ('receiver__full_grid_innovation', 'receiver__single_pass', 'Full-grid - single'),
        ('receiver__full_grid_innovation', 'digital_adaptive', 'Full-grid - digital adaptive'),
        ('receiver__full_grid_innovation', 'perceptual_deepjscc', 'Full-grid - perceptual Deep'),
    ]
    figure, axis = pyplot.subplots(figsize=(10, 5))
    for position, (method, control, label) in enumerate(contrasts):
        row = next(row for row in paired if row['method'] == method and row['control'] == control and row['metric'] == 'lpips' and row['snrs_db'] == '1.0+4.0+7.0')
        mean, low, high = (float(row[key]) for key in ('delta', 'ci_low', 'ci_high'))
        axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt='o', color='C0' if mean < 0 else 'C3', capsize=3)
    axis.axvline(0, color='black', linewidth=1)
    axis.set_yticks(range(len(contrasts)), [item[2] for item in contrasts])
    axis.invert_yaxis()
    axis.set_xlabel('Delta LPIPS (negative favors first method)')
    axis.set_title('Primary 1/4/7 dB: source-image paired bootstrap, pointwise 95% CI\nDevelopment only; no multiplicity or training-seed uncertainty correction')
    axis.grid(axis='x', alpha=.25)
    figure.tight_layout()
    figure.savefig(output / 'primary_paired_lpips.png', dpi=160)
    figure.savefig(output / 'primary_paired_lpips.pdf')
    pyplot.close(figure)
    images, codes, identifiers = read_population(base, 'development')
    image_manifest = []
    for index in (0, 37, 99):
        figure, axes = pyplot.subplots(3, 8, figsize=(20, 8.5))
        originals = output / f'originals/{index:03d}'
        originals.mkdir(parents=True)
        source = images[index].permute(1, 2, 0).numpy()
        Image.fromarray(np.round(np.clip(source, 0, 1) * 255).astype(np.uint8)).save(originals / 'source.png')
        for row_index, snr in enumerate((1., 7., 19.)):
            axes[row_index, 0].imshow(source, interpolation='nearest')
            axes[row_index, 0].set_title(f'Source / {snr:g} dB', fontsize=9)
            for column, (name, (label, color)) in enumerate(METHODS.items(), 1):
                key = (index, snr, 2001, name)
                old = references.rows[key]
                requested = name.replace('receiver__', 'frozen__')
                image, row = references.image(index, identifiers[index], snr, 2001, requested, old['noise_sha256'])
                values = image.permute(1, 2, 0).numpy()
                axes[row_index, column].imshow(values, interpolation='nearest')
                axes[row_index, column].set_title(f'{label}\nP={float(row["psnr_db"]):.2f} / L={float(row["lpips"]):.3f}', fontsize=8)
                filename = f'{name}__snr{snr:g}.png'
                Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(originals / filename)
                image_manifest.append({'source_index': index, 'image_id': identifiers[index], 'snr_db': snr, 'seed': 2001,
                    'method': name, 'display_png': str((originals / filename).relative_to(output)),
                    'actual_float32_archive': row['image_archive'], 'actual_float32_image_ref': int(row['image_ref']),
                    'actual_float32_sha256': row['image_sha256'], 'psnr_db': float(row['psnr_db']), 'lpips': float(row['lpips'])})
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f'Fixed source {index:03d}, noise 2001, N=3060/E=6120 (not quality-selected)\n{identifiers[index]}', fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .93))
        figure.savefig(output / f'fixed_source_{index:03d}.png', dpi=160)
        figure.savefig(output / f'fixed_source_{index:03d}.pdf')
        pyplot.close(figure)
    write_json(output / 'image_manifest.json', image_manifest)
    (output / 'README.md').write_text('# Frozen receiver result figures\n\n'
        'All figures are read-only derivations of the completed original development experiment. '
        'No training, new inference, or holdout access occurs. Sources 0/37/99, noise2001 and SNR1/7/19 are fixed, not chosen by image quality.\n\n'
        'The originals directory contains display PNGs, quantized from the retained float32 outputs. '
        'Scientific metrics were computed from float32, not these PNGs; image_manifest.json links every tile to the actual archive and hash.\n\n'
        'All compared wireless methods use 3060 complex symbols and total energy6120, but header allocation, model parameters and training history differ. '
        'The native horizontal line is not a wireless method; Deep support crosses are the predeclared fixed-support supplement.\n')
    write_json(output / 'completion.json', {'status': 'FROZEN_RECEIVER_FIGURES_COMPLETE', 'completed_local': now(),
        'source_hashes': sources, 'analysis_receipt_sha256': sha256(analysis / 'completion.json'),
        'reference_qualification_sha256': sha256(references.qualification_path), 'fixed_source_indices': [0, 37, 99],
        'qualitative_snrs': [1, 7, 19], 'qualitative_seed': 2001, 'GPU_used': False, 'new_inference': False,
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(str(output))


if __name__ == '__main__':
    main()
