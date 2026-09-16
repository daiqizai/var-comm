"""Readable views of frozen results; separate unsupported Deep conditioning from fair support points."""

import csv
import json
from pathlib import Path
import sys
import textwrap

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path.insert(0, str(BASE / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME


LABELS = {
    'receiver__single_pass': ('Single pass', 'C0'),
    'receiver__multiscale_innovation': ('Multiscale innovation', 'C4'),
    'receiver__full_grid_innovation': ('Full-grid innovation', 'C5'),
    'digital_m8': ('Digital fixed m8', 'C7'),
    'digital_adaptive': ('Digital adaptive', 'C2'),
    'perceptual_deepjscc': ('Perceptual DeepJSCC', 'C1'),
    'wetok_8PSK_FEC': ('WeTok digital 8PSK/FEC', 'C3'),
}


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    source = PROJECT / 'outputs/WETOK-INNOVATION-R1-FIGURES-20260914'
    source_receipt = json.loads((source / 'completion.json').read_text())
    verify_sources(source_receipt['source_hashes'])
    for name, expected in source_receipt['output_hashes'].items():
        if sha256(source / name) != expected:
            raise RuntimeError('frozen display source changed')
    analysis = PROJECT / 'outputs/WETOK-INNOVATION-R1-ANALYSIS/development_0005000'
    receipt = json.loads((analysis / 'completion.json').read_text())
    for name in ('summary.csv', 'deep_support_summary.csv', 'native_reference_summary.json'):
        if sha256(analysis / name) != receipt['output_hashes'][name]:
            raise RuntimeError('frozen quantitative source changed')
    summary = rows(analysis / 'summary.csv')
    support = rows(analysis / 'deep_support_summary.csv')
    native = json.loads((analysis / 'native_reference_summary.json').read_text())
    output = PROJECT / 'outputs/WETOK-INNOVATION-R1-PRESENTATION-20260914'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__)])
    snrs = [1., 4., 7., 13., 19.]
    figure, axes = pyplot.subplots(1, 3, figsize=(17, 5))
    for axis, metric, label in zip(axes, ('psnr_db', 'lpips', 'dino'), ('PSNR (higher better)', 'LPIPS (lower better)', 'DINO cosine (higher better)')):
        for method, (method_label, color) in LABELS.items():
            values = [float(next(row[metric] for row in summary if row['arm'] == method and row['snrs_db'] == str(snr))) for snr in snrs]
            axis.plot(snrs, values, marker='o', color=color, linewidth=1.4, label=method_label)
        support_values = [float(next(row[metric] for row in support if row['arm'] == SUPPORT_NAME and row['snrs_db'] == str(snr))) for snr in (5., 6.)]
        axis.scatter([5., 6.], support_values, marker='x', color='black', s=55, label='Deep: 5->4 / 6->7 support policy')
        axis.axhline(native[metric], color='.5', linestyle=':', label='Native codec reference (not wireless)')
        axis.set_title(label)
        axis.set_xlabel('Actual SNR (dB)')
        axis.set_xticks([1, 4, 5, 6, 7, 13, 19])
        axis.grid(alpha=.25)
    figure.suptitle('N=3060 complex uses, energy=6120; all methods at the five training-support SNR anchors\nActual Deep support-policy measurements at 5/6 dB are separate crosses; not a new trained model', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    figure.tight_layout(rect=(0, .18, 1, .89))
    figure.savefig(output / 'strong_comparison_training_support_anchors.png', dpi=160)
    figure.savefig(output / 'strong_comparison_training_support_anchors.pdf')
    pyplot.close(figure)
    figure, axis = pyplot.subplots(figsize=(8, 4))
    all_snrs = [1., 4., 5., 6., 7., 13., 19.]
    original = [float(next(row['lpips'] for row in summary if row['arm'] == 'perceptual_deepjscc' and row['snrs_db'] == str(snr))) for snr in all_snrs]
    axis.plot(all_snrs, original, marker='o', linestyle='--', color='C1', label='Archived original conditioning (diagnostic)')
    measured = [float(next(row['lpips'] for row in support if row['arm'] == SUPPORT_NAME and row['snrs_db'] == str(snr))) for snr in (5., 6.)]
    axis.scatter([5., 6.], measured, marker='x', color='black', s=80, label='Predeclared nearest-support policy')
    axis.set_title('Deep conditioning diagnostic: do not claim a method gain from the 5/6 dB dip')
    axis.set_xlabel('Actual SNR (dB)')
    axis.set_ylabel('LPIPS')
    axis.legend(fontsize=9)
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output / 'deep_conditioning_diagnostic.png', dpi=160)
    figure.savefig(output / 'deep_conditioning_diagnostic.pdf')
    pyplot.close(figure)
    manifest = json.loads((source / 'image_manifest.json').read_text())
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font, label_font, value_font = (ImageFont.truetype(font_path, size) for size in (20, 16, 15))
    tile, margin, column_gap, header, row_gap = 256, 20, 12, 64, 20
    width = margin * 2 + tile * 8 + column_gap * 7
    top, row_height = 86, header + tile + row_gap
    for index in (0, 37, 99):
        canvas = Image.new('RGB', (width, top + row_height * 3 + margin), 'white')
        draw = ImageDraw.Draw(canvas)
        image_id = next(row['image_id'] for row in manifest if row['source_index'] == index)
        draw.text((margin, 12), f'Fixed source {index:03d}; N=3060 / E=6120; noise 2001; not selected by quality', font=title_font, fill='black')
        draw.text((margin, 44), image_id, font=label_font, fill='black')
        for row_index, snr in enumerate((1, 7, 19)):
            top_row = top + row_index * row_height
            for column, method in enumerate([None, *LABELS]):
                left = margin + column * (tile + column_gap)
                if method is None:
                    filename = source / f'originals/{index:03d}/source.png'
                    caption, values = f'Source / {snr} dB', ''
                else:
                    record = next(row for row in manifest if row['source_index'] == index and row['snr_db'] == snr and row['method'] == method)
                    filename = source / record['display_png']
                    caption = LABELS[method][0]
                    values = f'PSNR {record["psnr_db"]:.2f} | LPIPS {record["lpips"]:.3f}'
                lines = textwrap.wrap(caption, width=26)
                for line_index, text in enumerate(lines):
                    draw.text((left, top_row + 17 * line_index), text, font=label_font, fill='black')
                draw.text((left, top_row + 39), values, font=value_font, fill='black')
                with Image.open(filename) as image:
                    if image.size != (tile, tile):
                        raise RuntimeError('source tiles must remain their original 256x256 size')
                    canvas.paste(image.convert('RGB'), (left, top_row + header))
        canvas.save(output / f'fixed_source_{index:03d}.png')
        canvas.save(output / f'fixed_source_{index:03d}.pdf', resolution=120.)
    (output / 'README.md').write_text('# Readable presentation views\n\n'
        'Use strong_comparison_training_support_anchors for the five common trained SNR anchors; actual Deep fixed-support 5/6 measurements are separate crosses. '
        'The archived off-support conditioning dip is retained only in the explicitly labelled diagnostic figure, not used to manufacture a new-method advantage. '
        'All underlying metric rows/rankings remain unchanged.\n\n'
        'Montages paste the original 256x256 display PNG tiles without resampling. Labels occupy separate margins. '
        'Source selection, image hashes and float32 archives remain in the original figures directory image_manifest.json. '
        'Original per-method PNGs remain there; this directory contains presentation layouts only.\n')
    write_json(output / 'completion.json', {'status': 'READABLE_FROZEN_RESULT_VIEWS_COMPLETE', 'completed_local': now(),
        'source_hashes': source_hashes, 'frozen_figures_receipt_sha256': sha256(source / 'completion.json'),
        'analysis_receipt_sha256': sha256(analysis / 'completion.json'), 'new_inference': False, 'GPU_used': False,
        'source_tiles_resampled': False, 'original_metrics_changed': False, 'research_goal_complete': False,
        'output_hashes': artifact_hashes(output)})
    print(output)


if __name__ == '__main__':
    main()
