#!/usr/bin/env python3
"""Plot sealed T1 statistics only: no inference, channel calls or bootstrap."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SNRS = (4, 10, 19)
COMPLETION_SHA = '69a25187ee60fb4f749d4c290360903c661107574788e4b2cc4f0b282ffba4b7'
ORIGINAL_SHA = '5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859'
PARTIAL = 'RAW64_PARTIAL_VAR_COMPLETION'
METHODS = (
    (PARTIAL, 'Partial-scale raw + VAR (NeST-Com)', '#0072B2', '-', 'o'),
    ('EC_STATIC_WHOLE', 'Static entropy, complete-scale + VAR', '#D55E00', '--', 's'),
    ('EC_VAR_WHOLE', 'VAR entropy, complete-scale + VAR', '#CC79A7', '-.', '^'),
)
METRICS = (
    ('psnr_db', 'PSNR', 'PSNR (dB) ↑', 'ΔPSNR (dB) ↑', 1, 'psnr'),
    ('lpips_alex', 'LPIPS-Alex', 'LPIPS ↓', 'ΔLPIPS ↓', 1, 'lpips'),
    ('dinov2_vitl14_cosine', 'DINOv2 ViT-L/14', 'Cosine similarity ↑', 'ΔDINOv2-L ↑', 1, 'dinov2_l'),
    ('convnext_top1_source_prediction', 'ConvNeXt prediction agreement',
     'Prediction agreement (%) ↑', 'ΔAgreement (percentage points) ↑', 100, 'convnext_agreement'),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def readcsv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def csvout(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def one(rows):
    require(len(rows) == 1, f'Expected precisely one source row; found {len(rows)}')
    return rows[0]


def gather(stats, original, freeze):
    require(sha(stats / 'completion.json') == COMPLETION_SHA, 'Unrecognized completed statistics')
    done = read(stats / 'completion.json')
    require(done['status'] == 'T1_POSTHOC_COMMON500_NEW_PAIRED_STATISTICS_COMPLETE'
            and done['source_count'] == 500 and done['noise_count'] == 3
            and len(set(done['source_ids'])) == 500 and done['post_hoc_supplement'] is True
            and done['source_mean_first'] is True and done['old_intervals_recomputed'] is False
            and done['frame_level_noise_pairing_claimed'] is False
            and done['bootstrap_seed'] == 2026100701 and done['bootstrap_replicates'] == 10000,
            'Statistics population or interpretation mismatch')
    inputs = {str(stats / 'completion.json'): COMPLETION_SHA}
    for name in ('summary.csv', 'paired.csv', 'raw_reference_summary_unchanged.json'):
        expected = one([h for p, h in done['outputs'].items() if p.endswith('/' + name)])
        require(sha(stats / name) == expected, 'Statistics SHA differs: ' + name)
        inputs[str(stats / name)] = expected
    require(sha(original) == ORIGINAL_SHA, 'Original published summary differs')
    require(sha(freeze) == done['freeze']['sha256'], 'Frozen EC policy differs')
    inputs[str(original)] = ORIGINAL_SHA
    inputs[str(freeze)] = sha(freeze)
    summary, paired, old = readcsv(stats / 'summary.csv'), readcsv(stats / 'paired.csv'), readcsv(original)
    raw_copied = read(stats / 'raw_reference_summary_unchanged.json')
    require(len(raw_copied) == done['old_summary_rows_copied'] == 24, 'Original raw copy count differs')
    require(len(summary) == done['summary_rows'] == 48 and len(paired) == done['paired_rows'] == 132,
            'Full CSV row counts differ from completion')
    require(len({(r['point_id'], r['snr_db'], r['metric']) for r in summary}) == len(summary),
            'Duplicate summary rows')
    require(len({(r['method'], r['reference'], r['snr_db'], r['metric']) for r in paired}) == len(paired),
            'Duplicate paired rows')
    result = []
    for is_pair, source_rows in ((False, summary), (True, paired)):
        for family, label, *_ in METHODS[1:] if is_pair else METHODS:
            for metric, _, _, _, scale, _ in METRICS:
                for snr in SNRS:
                    point, reference = f'{family}_SNR_{snr}', f'{PARTIAL}_SNR_{snr}'
                    candidates = old if not is_pair and family == PARTIAL else source_rows
                    row = one([r for r in candidates if r['metric'] == metric and int(r['snr_db']) == snr
                               and r['method' if is_pair else 'point_id'] == point
                               and (not is_pair or r['reference'] == reference)])
                    require((int(row['source_count']), int(row['noise_count']), int(row['frame_count'])) == (500, 3, 1500)
                            and row['metric_identity'] == done['metric_identity'][metric], 'Population/evaluator mismatch')
                    require(all(Decimal(row[k]).is_finite() for k in ('mean', 'ci_low', 'ci_high'))
                            and Decimal(row['ci_low']) <= Decimal(row['mean']) <= Decimal(row['ci_high']), 'Invalid CI')
                    if is_pair:
                        require(row['delta_definition'] == 'method minus reference'
                                and row['frame_level_noise_pairing_claimed'] == 'False', 'Paired direction mismatch')
                    elif family == PARTIAL:
                        previous = one([r for r in raw_copied if r['point_id'] == point and r['metric'] == metric])
                        require(all(Decimal(row[k]) == Decimal(str(previous[k])) for k in ('mean', 'ci_low', 'ci_high')),
                                'Original raw values or CIs changed')
                    result.append(dict(row, figure='paired' if is_pair else 'quality', family=family,
                                       public_method=label, source_csv='paired.csv' if is_pair else ('original_published/summary.csv' if family == PARTIAL else 'summary.csv'),
                                       display_scale=scale, display_unit=('percentage points' if is_pair else 'percent') if scale == 100 else '',
                                       **{f'plot_{k}': str(Decimal(row[k]) * scale) for k in ('mean', 'ci_low', 'ci_high')}))
    require(len(result) == 60, 'Expected 36 summary and 24 paired rows')
    return result, inputs, done


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
                        'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                        'legend.fontsize': 8, 'axes.linewidth': .6, 'grid.linewidth': .5,
                        'grid.color': '#E3E3E3', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                        'svg.fonttype': 'none', 'svg.hashsalt': 'wcl-t5-entropy-quality-v1',
                        'figure.facecolor': 'white', 'axes.facecolor': 'white',
                        'savefig.facecolor': 'white', 'axes.unicode_minus': True, 'path.simplify': False})


def panel(ax, rows, metric, paired):
    key, title, label, delta, scale, _ = metric
    values = [0] if paired else []
    for family, public, color, linestyle, marker in METHODS[1:] if paired else METHODS:
        chosen = [one([r for r in rows if r['figure'] == ('paired' if paired else 'quality')
                       and r['family'] == family and r['metric'] == key and int(r['snr_db']) == snr]) for snr in SNRS]
        y, lo, hi = [np.asarray([float(r[f'plot_{k}']) for r in chosen]) for k in ('mean', 'ci_low', 'ci_high')]
        values.extend(lo); values.extend(hi)
        # Open triangle keeps the VAR curve visible when it overlaps static entropy.
        face = 'white' if family == 'EC_VAR_WHOLE' else color
        if paired:
            ax.errorbar(SNRS, y, yerr=[y - lo, hi - y], color=color, linestyle=linestyle,
                        marker=marker, markerfacecolor=face, lw=1.25, ms=4.3, capsize=3,
                        elinewidth=.95, zorder=3)
        else:
            ax.fill_between(SNRS, lo, hi, color=color, alpha=.12, linewidth=0)
            ax.plot(SNRS, y, color=color, linestyle=linestyle, marker=marker,
                    markerfacecolor=face, lw=1.35, ms=4.3, zorder=3)
    if paired:
        ax.axhline(0, color='#777777', ls=':', lw=.8, zorder=1)
    lower, upper = min(values), max(values)
    pad = max((upper - lower) * .12, 1e-5)
    ax.set_ylim(lower - pad, upper + pad)
    ax.set_xlim(3.2, 19.8); ax.set_xticks(SNRS)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel(delta if paired else label)
    ax.set_title(title, pad=7); ax.grid(True); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, out, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and text.get_clip_on() is False:
            box = text.get_window_extent(renderer)
            require(box.x0 >= bounds.x0 - 1 and box.y0 >= bounds.y0 - 1
                    and box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1,
                    f'Text outside figure: {text.get_text()}')
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{stem}.{extension}', dpi=600 if extension == 'png' else 150)


def figures(rows, out):
    for paired, stem in ((False, 'fig_t5_entropy_quality_N1024'), (True, 'fig_t5_entropy_minus_partial_N1024')):
        methods = METHODS[1:] if paired else METHODS
        handles = [Line2D([], [], color=c, ls=l, marker=m, lw=1.3, ms=4,
                          markerfacecolor='white' if f == 'EC_VAR_WHOLE' else c, label=p)
                   for f, p, c, l, m in methods]
        heading = 'Entropy-coded complete-scale minus NeST-Com' if paired else 'Same-prior source-coding comparison'
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.6))
        for ax, metric in zip(axes.flat, METRICS): panel(ax, rows, metric, paired)
        fig.suptitle(heading + ', N = 1024', y=.985, fontsize=10)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .947), ncol=1, frameon=False,
                   handlelength=3, labelspacing=.4)
        fig.subplots_adjust(left=.112, right=.985, bottom=.12, top=.775, wspace=.4, hspace=.61)
        fig.text(.5, .025, 'Post-hoc same-source comparison · 500 sources × 3 noises · existing 95% CIs',
                 ha='center', va='center', fontsize=7.5)
        save(fig, out, stem); plt.close(fig)
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(3.5, 3.45))
            panel(ax, rows, metric, paired)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995), ncol=1,
                       frameon=False, fontsize=7, handlelength=2.8, labelspacing=.4)
            fig.subplots_adjust(left=.2, right=.97, bottom=.22, top=.68)
            fig.text(.53, .055, 'N = 1024 · 500 sources × 3 noises\nPost-hoc same-source comparison',
                     ha='center', fontsize=7)
            save(fig, out, stem + '_' + metric[-1]); plt.close(fig)


CAPTIONS = r'''% Directly consumed completed statistics; no new bootstrap or evaluation.
\newcommand{\TfiveEntropyQualityCaption}{Same-prior source-coding comparison at $N=1024$.
The original partial-scale raw-token system (NeST-Com) is compared with complete-scale transmission using static or VAR-conditional entropy coding and the same frozen VAR completion model.
This is a post-hoc same-source supplement on the previously examined 500-source holdout, with three registered noises per source; all scheduled frames, including failures, remain in the statistics.
Lines show CSV means and shaded bands reproduce the existing pointwise 95\% source-bootstrap confidence intervals after averaging the three noises, with no multiple-comparison adjustment.
The original raw means and intervals are unchanged. ConvNeXt reports source-prediction agreement in percent, not classification accuracy.
Each method uses its calibration-frozen deployed protocol; raw KEEP and entropy-code rejection/fallback rules differ, so the contrast is not compression alone. Source pairing does not imply identical received observations for different waveforms.}

\newcommand{\TfiveEntropyPairedCaption}{Source-paired differences at $N=1024$: each complete-scale entropy-coded system minus the original partial-scale raw-token system (NeST-Com).
Markers and error bars reproduce the existing paired CSV means and pointwise 95\% source-bootstrap intervals on 500 common sources, each averaged over three noises.
The horizontal line is zero. Positive differences favor the entropy-coded system for PSNR, DINOv2-L cosine similarity and prediction agreement; negative LPIPS differences favor it.
Agreement differences are percentage points, not relative percent changes. Zero means and intervals crossing zero are retained, including the VAR-conditional entropy system's 19~dB agreement result.
These are post-hoc same-source comparisons of the complete deployed protocols, including their distinct registered failure rules; the differences are not obtained by subtracting marginal intervals.}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('statistics', 'original-summary', 'freeze', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), 'Use a fresh output directory; never overwrite previous figures')
    rows, inputs, done = gather(args.statistics.resolve(), args.original_summary.resolve(), args.freeze.resolve())
    args.out.mkdir(parents=True)
    style(); figures(rows, args.out)
    csvout(args.out / 'plot_data.csv', rows)
    (args.out / 'captions.tex').write_text(CAPTIONS, encoding='utf-8')
    methodmap = {f: {'public_label': p, 'color': c, 'linestyle': l, 'marker': m} for f, p, c, l, m in METHODS}
    write(args.out / 'method_mapping.json', methodmap)
    (args.out / 'README.md').write_text(
        '# Completed entropy-code quality figures\n\n'
        'N1024 only; SNR4/10/19. Post-hoc same-source comparison on the original500 sources,3 noises each. '
        'This plotting run performs zero model, PHY and bootstrap calls. The two figure families have four independently laid-out single-column panels each, exported as vector PDF, editable-text SVG and600dpi PNG.\n\n'
        'Inputs are authenticated by the completed T1 statistics SHA and output seals. '
        'The original raw partial means/CIs additionally match the published252176e041758ecb2d3e81b6fde5b587e7e17bb7 CSV. '
        'Only the displayed agreement values are multiplied by100. LPIPS signs are unchanged. All three registered SNRs and all selected rows are retained. '
        'The source CSV precision is preserved in plot_data.csv; no values are copied from prose.\n\n'
        'Method mapping and appearance are in method_mapping.json; captions.tex explains pairing and the statistical/failure-rule limitations. '
        'The complete-scale static and VAR-conditional entropy curves are separate methods; no result is selected per source or noise.\n\n'
        'Reproduce into a fresh output directory with Python, numpy and matplotlib:\n\n```text\n'
        f'python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_entropy_quality.py --statistics "{args.statistics}" '
        f'--original-summary "{args.original_summary}" --freeze "{args.freeze}" --out "<fresh-output-directory>"\n```\n', encoding='utf-8')
    completion = dict(status='T5_ENTROPY_QUALITY_EXISTING_CI_PLOTS_COMPLETE', N=1024, source_count=500, noise_count=3,
                      snrs=SNRS, plot_rows=60, paired_delta='entropy-coded complete-scale minus original raw partial-scale',
                      post_hoc_same_source=True, frame_level_noise_pairing_claimed=False,
                      failure_frames_included=True, agreement_scale=100, lpips_sign_unchanged=True,
                      new_model_calls=0, new_packet_calls=0, new_bootstrap_calls=0,
                      raw_published_summary_verified=True, code_sha256=sha(__file__), input_bindings=inputs,
                      input_policy_remote_descriptor=done['freeze'], methods=methodmap,
                      output_hashes={p.name: sha(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    write(args.out / 'completion.json', completion)
    print(json.dumps({k: completion[k] for k in ('status', 'plot_rows', 'new_bootstrap_calls')}))


if __name__ == '__main__':
    main()
