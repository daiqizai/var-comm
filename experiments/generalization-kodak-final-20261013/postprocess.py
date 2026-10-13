"""Add final frozen P100k/Swin200k Kodak statistics; preserve historical anchors.

Consumes completed metric CSVs only. This entry has no model, reconstruction,
channel, calibration, or training calls. The original Kodak interval function is
imported unchanged; it is called only for the two new methods and their new
source-paired contrasts. Historical raw/BPG interval rows are copied verbatim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'experiments/generalization_kodak_20261009/analyze.py'
ANCHORS = ('VAR_UNCONDITIONAL', 'BPG_ADAPTIVE')
NEW = ('P100K', 'SWIN200K')
METHODS = ('VAR_UNCONDITIONAL', 'P100K', 'SWIN200K', 'BPG_ADAPTIVE')
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine',
           'convnext_top1_source_prediction')
SNRS = (4, 10, 19)
NOISES = (2001, 2002, 2003)
SEED = 2026100901
REPLICATES = 10000
DISPLAY = {
    'VAR_UNCONDITIONAL': ('NeST-Com (raw)', '#0072B2', '-', 'o'),
    'P100K': ('Latent-space JSCC (100k)', '#D55E00', '--', 's'),
    'SWIN200K': ('SwinJSCC-200k (adapted)', '#009E73', '-.', '^'),
    'BPG_ADAPTIVE': ('Adaptive-resolution BPG + LDPC', '#CC79A7',
                     (0, (5, 1, 1, 1)), 'v'),
}
PANELS = (
    ('PSNR (dB) \u2191', 1), ('LPIPS \u2193', 1),
    ('DINOv2-L cosine similarity \u2191', 1),
    ('ConvNeXt prediction agreement (%) \u2191', 100),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), 'Cannot export an empty table')
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_union_csv(path, rows):
    """The portable scorer adds an explicit historical-identity column."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def bound_file(path, completion, group='outputs'):
    matches = [h for p, h in completion[group].items()
               if Path(p).name == Path(path).name]
    require(len(matches) == 1 and sha(path) == matches[0],
            f'Completion SHA does not match {path}')


def load_original():
    spec = importlib.util.spec_from_file_location('frozen_kodak_analyze', ORIGINAL)
    require(spec is not None and spec.loader is not None, 'Original Kodak analysis entry missing')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SEED == SEED and module.REPLICATES == REPLICATES
            and tuple(module.METRICS) == METRICS and tuple(module.SNRS) == SNRS,
            'Original Kodak statistics settings changed')
    return module


def csv_subset_bytes(path, predicate):
    """Preserve complete original header/data lines, including number precision."""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    header = next(csv.reader([lines[0].decode('utf-8-sig').rstrip('\r\n')]))
    kept = []
    for line in lines[1:]:
        fields = next(csv.reader([line.decode('utf8').rstrip('\r\n')]))
        row = dict(zip(header, fields))
        if predicate(row):
            kept.append(line)
    return lines[0] + b''.join(kept), len(kept)


def append_csv_preserving_old(path, original_bytes, new_rows):
    fields = next(csv.reader([original_bytes.splitlines()[0].decode('utf-8-sig')]))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\n')
    writer.writerows(new_rows)
    Path(path).write_bytes(original_bytes + stream.getvalue().encode('utf8'))


def validate_new(rows, completion, ids, old_rows):
    require(completion['status'] == 'KODAK24_FINAL_BASELINES_FOUR_METRICS_COMPLETE_V1',
            'Actual final-baseline scoring completion required')
    require(completion['source_count'] == 24 and completion['frame_count'] == 432
            and completion['metric_rows'] == 1728 and completion['N'] == 1024
            and completion['source_ids'] == ids and completion['metrics'] == list(METRICS)
            and set(completion['methods']) == set(NEW), 'New score scope must be exact')
    require(len(rows) == 1728, 'Expected 432 logical frames times four metrics')
    source_identity = {}
    for row in old_rows:
        index = int(row['source_index'])
        value = (row['reference_sha256'], row['source_prediction'],
                 row['preprocessing_id'], row['metric_backend_identity'])
        require(source_identity.setdefault(index, value) == value,
                'Historical source/metric identities disagree')
    grouped, frames, seen = {}, {}, set()
    for row in rows:
        method, metric = row['method'], row['metric']
        snr, index, noise = int(row['snr_db']), int(row['source_index']), int(row['noise_seed'])
        key = method, snr, index, noise, metric
        require(method in NEW and metric in METRICS and snr in SNRS
                and 0 <= index < 24 and noise in NOISES and key not in seen,
                f'Duplicate or unexpected final score row: {key}')
        seen.add(key)
        require(row['source_id'] == ids[index] and int(row['N']) == 1024
                and row['population'] == 'Kodak24', 'Different source population/crop/bandwidth')
        identity = (row['reference_sha256'], row['source_prediction'],
                    row['preprocessing_id'], row['original_metric_backend_identity'])
        require(identity == source_identity[index], 'New and old source/metric identities must match')
        require(row['metric_backend_identity'] == completion['metric_backend_identity']
                and row['original_metric_backend_identity'] == completion['original_metric_backend_identity'],
                'Current runtime and original mathematical identities must both be explicit')
        require(row['true_class_label_available'].lower() == 'false', 'No true class labels for Kodak')
        value = float(row['value'])
        require(np.isfinite(value), 'Nonfinite metric must be investigated; never cap or drop it')
        if metric == 'lpips_alex':
            require(value >= 0, 'LPIPS cannot be negative')
        elif metric == 'dinov2_vitl14_cosine':
            require(-1.000001 <= value <= 1.000001, 'Cosine outside numerical tolerance')
        elif metric == 'convnext_top1_source_prediction':
            require(value in (0, 1) and value == int(int(row['source_prediction']) == int(row['reconstruction_prediction'])),
                    'Agreement must equal saved source/reconstruction prediction equality')
        metadata = (row['source_id'], row['status'], row['reconstruction_sha256'],
                    row['reconstruction_archive_sha256'], row['reconstruction_prediction'])
        frame_key = method, snr, index, noise
        require(frames.setdefault(frame_key, metadata) == metadata,
                'Four metrics must describe the same logical output, including failure status')
        grouped.setdefault((method, snr, metric), {}).setdefault(index, {})[noise] = value
    expected = {(m, s, i, n) for m in NEW for s in SNRS for i in range(24) for n in NOISES}
    require(set(frames) == expected, 'Incomplete new 24 x 3 SNR x 3 noise x 2 method frame grid')
    arrays = {}
    for method in NEW:
        for snr in SNRS:
            for metric in METRICS:
                group = grouped[method, snr, metric]
                require(set(group) == set(range(24)) and all(set(v) == set(NOISES) for v in group.values()),
                        'All fixed sources/noises, including every failure, are mandatory')
                arrays[method, snr, metric] = np.asarray([
                    np.mean([group[i][n] for n in NOISES], dtype=np.float64) for i in range(24)
                ], dtype=np.float64)
    return arrays, frames


def validate_new_math_proof(completion_path, completion, old_done):
    """Same metric math/weights, explicitly different H800 runtime identity."""
    proof_path = Path(completion_path).parent / 'metric_math_identity_proof.json'
    bound_file(proof_path, completion)
    require(completion['metric_math_identity_proof']['sha256'] == sha(proof_path),
            'Explicit metric proof SHA differs from the completion output binding')
    proof = read_json(proof_path)
    require(proof['status'] == 'PASS_ORIGINAL_KODAK_MATH_AND_WEIGHT_BINDINGS'
            and proof['original_Kodak_evaluator_identity'] == old_done['original_metric_backend_identity']
            == completion['original_metric_backend_identity']
            and proof['DINO_feature_and_preprocess_source_unchanged'] is True
            and proof['original_FullSuite_identity_reused_as_current_runtime'] is False
            and proof['old_host_bitwise_equivalence_claimed'] is False
            and completion['old_host_bitwise_equivalence_claimed'] is False
            and completion['source_and_reconstruction_preprocessing_identical'] is True
            and completion['metric_backend_identity'] != completion['original_metric_backend_identity'],
            'Historical mathematics/preprocessing/weights proof and distinct runtime identity required')
    require(set(proof['Kodak_expression_AST_sha256']) == {
        'lpips', 'cosine', 'mse', 'source_feature', 'feature', 'source_prediction', 'reconstruction_prediction'
    } and proof['actual_frozen_weights'] and proof['original_computation_source_bindings'],
            'Complete seven-expression AST proof and frozen weight/source bindings required')
    return proof_path


def validate_reuse(receipt):
    require(receipt['status'] == 'PASS_FROZEN_KODAK_ANCHOR_REUSE'
            and set(receipt['approved_methods']) == set(ANCHORS)
            and receipt['source_count'] == 24 and receipt['N'] == 1024
            and receipt['snr_db'] == list(SNRS) and receipt['noise_seeds'] == list(NOISES)
            and receipt['source_pixels_match'] is True and receipt['model_identity_match'] is True
            and receipt['configuration_identity_match'] is True and receipt['metric_identity_match'] is True,
            'Independent source/model/configuration/metric anchor reuse audit required')


def validate_summary(rows):
    seen = set()
    for row in rows:
        key = row['method'], int(row['snr_db']), row['metric']
        require(key not in seen and key[0] in METHODS and key[1] in SNRS and key[2] in METRICS,
                'Duplicate or unexpected summary point')
        seen.add(key)
        lo, mean, hi = (float(row[k]) for k in ('ci_low', 'mean', 'ci_high'))
        require(np.isfinite([lo, mean, hi]).all() and lo <= mean <= hi,
                f'CI bounds do not contain mean: {key}')
        require(int(row['source_count']) == 24 and int(row['noise_count']) == 3
                and int(row['bootstrap_seed']) == SEED and int(row['bootstrap_replicates']) == REPLICATES,
                'Summary population/bootstrap differs from frozen Kodak settings')
    require(seen == {(m, s, k) for m in METHODS for s in SNRS for k in METRICS},
            'Complete four-method by three-SNR by four-metric summary required')


def new_statistics(original, ids, arrays):
    draws = np.random.default_rng(SEED).integers(0, 24, size=(REPLICATES, 24))
    summary, paired, means = [], [], []
    for method in NEW:
        for snr in SNRS:
            for metric in METRICS:
                vector = arrays[method, snr, metric]
                summary.append(dict(method=method, snr_db=snr, metric=metric,
                                    **original.interval(vector, draws)))
                means.extend(dict(method=method, snr_db=snr, metric=metric,
                                  source_index=i, source_id=sid, mean=float(vector[i]), noise_count=3)
                             for i, sid in enumerate(ids))
    for reference in NEW:
        for snr in SNRS:
            for metric in METRICS:
                paired.append(dict(method='VAR_UNCONDITIONAL', reference=reference,
                                   snr_db=snr, metric=metric, delta_definition='method minus reference',
                                   frame_level_noise_pairing_claimed=False,
                                   **original.interval(arrays['VAR_UNCONDITIONAL', snr, metric]
                                                       - arrays[reference, snr, metric], draws)))
    require(len(summary) == 24 and len(paired) == 24 and len(means) == 576, 'Exact new statistics scope')
    return summary, paired, means, hashlib.sha256(draws.tobytes()).hexdigest()


def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.labelsize': 8.5, 'axes.titlesize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'path.simplify': False, 'axes.unicode_minus': True})
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.15), dpi=150)
    fig.subplots_adjust(left=.105, right=.985, bottom=.105, top=.85, wspace=.43, hspace=.54)
    handles = [Line2D([], [], label=DISPLAY[m][0], color=DISPLAY[m][1],
                      linestyle=DISPLAY[m][2], marker=DISPLAY[m][3], markersize=4, linewidth=1.3)
               for m in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .99), ncol=2,
               fontsize=7.7, frameon=False, handlelength=2.8, columnspacing=1.3, labelspacing=.5)
    plot_rows = []
    for ax, metric, (ylabel, scale), letter in zip(axes.flat, METRICS, PANELS, 'abcd'):
        lows, highs = [], []
        for method in METHODS:
            group = sorted([r for r in summary if r['method'] == method and r['metric'] == metric],
                           key=lambda r: int(r['snr_db']))
            require([int(r['snr_db']) for r in group] == list(SNRS), 'Three numerical SNR points per curve')
            y, low, high = (np.array([float(r[k]) for r in group]) * scale
                            for k in ('mean', 'ci_low', 'ci_high'))
            lows.extend(low)
            highs.extend(high)
            _, color, line, marker = DISPLAY[method]
            ax.fill_between(SNRS, low, high, color=color, alpha=.12, linewidth=0, zorder=1)
            ax.plot(SNRS, y, color=color, linestyle=line, marker=marker, linewidth=1.3,
                    markersize=4, zorder=3)
            for row in group:
                plot_rows.append(dict(**row, display_scale=scale,
                                      plot_mean=float(row['mean']) * scale,
                                      plot_ci_low=float(row['ci_low']) * scale,
                                      plot_ci_high=float(row['ci_high']) * scale))
        lo, hi = min(lows), max(highs)
        span = hi - lo if hi > lo else max(abs(hi) * .1, .01)
        ax.set_ylim(lo - .08 * span, hi + .08 * span)
        ax.set_xlim(3.3, 19.7)
        ax.set_xticks(SNRS)
        ax.set_xlabel('SNR (dB)')
        # Long metric names stay as panel titles, avoiding unreadable vertical labels.
        ax.set_ylabel(ylabel.replace('DINOv2-L ', '').replace('ConvNeXt ', ''))
        ax.set_title(f'({letter}) {("PSNR", "LPIPS", "DINOv2-L", "ConvNeXt")[METRICS.index(metric)]}',
                     loc='left', pad=7)
        ax.grid(color='#DFE3E6', linewidth=.55)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.spines[['top', 'right']].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for ax in axes.flat for t in
              [ax.xaxis.label, ax.yaxis.label, ax._left_title, *ax.get_xticklabels(), *ax.get_yticklabels()]
              if t.get_visible() and t.get_text()]
    boxes = [t.get_window_extent(renderer) for t in labels]
    boxes.extend(legend.get_window_extent(renderer) for legend in fig.legends)
    require(all(b.x0 >= -.5 and b.y0 >= -.5 and b.x1 <= fig.bbox.x1 + .5 and b.y1 <= fig.bbox.y1 + .5
                for b in boxes), 'Publication figure text or legend would be clipped')
    outputs = []
    for ext in ('pdf', 'svg', 'png'):
        path = out / f'kodak24_N1024_final_four_methods.{ext}'
        fig.savefig(path, dpi=600 if ext == 'png' else 150)
        outputs.append(path)
    plt.close(fig)
    write_csv(out / 'plot_data.csv', plot_rows)
    return outputs


def write_prose(out, summary, paired):
    caption = r'''\newcommand{\KodakFinalBaselinesCaption}{Frozen-method transfer to all 24 Kodak images, each represented by its previously fixed $256\times256$ center crop, at $N=1024$ and SNRs of 4, 10, and 19 dB. Three original method-specific noise realizations are averaged within each source; curves and bands show source means and pointwise 95\% percentile source-bootstrap intervals (10,000 replicates, seed 2026100901). The latent-space JSCC baseline uses the frozen 100k checkpoint and SwinJSCC the adapted 200k checkpoint, both selected before Kodak evaluation. Their actual transmitted waveforms, received observations, and reconstructions are newly generated; the source-, model-, policy-, and metric-matched NeST-Com raw and adaptive-resolution BPG + LDPC results and their original intervals are reused without recomputation. All prescribed failure outputs remain included. ConvNeXt measures agreement with the source-image prediction, not classification accuracy; higher is better except LPIPS. Swin200k's training range includes 19 dB. No training, checkpoint selection, or strategy calibration uses Kodak. This fixed 24-crop test provides limited cross-collection evidence and does not certify non-overlap with all pretrained-model corpora.}
'''
    (out / 'captions.tex').write_text(caption, encoding='utf8')
    pairs = {(r['reference'], int(r['snr_db']), r['metric']): r for r in paired}
    fragments = []
    for reference, name in (('P100K', 'latent-space JSCC (100k)'), ('SWIN200K', 'adapted SwinJSCC (200k)')):
        values = []
        for metric, unit in zip(METRICS, ('dB', '', '', 'percentage points')):
            row = pairs[reference, 19, metric]
            scale = 100 if metric == METRICS[-1] else 1
            values.append(f"{float(row['mean']) * scale:+.3f}" + (f' {unit}' if unit else ''))
        fragments.append(f"relative to {name}, its source-paired differences at 19 dB are "
                         + ', '.join(values) + ' for PSNR, LPIPS, DINOv2-L cosine similarity, and '
                         'ConvNeXt prediction agreement, respectively')
    paragraph = ('We evaluated the frozen systems on all 24 previously fixed Kodak center crops '
                 '($256\\times256$), using $N=1024$, SNRs of 4, 10, and 19 dB, and three prescribed '
                 'noise realizations per source. The final latent-space JSCC and SwinJSCC checkpoints '
                 'were transferred without Kodak training or calibration. NeST-Com (raw) is compared '
                 'with both final baselines and adaptive-resolution BPG + LDPC; ' + '; '.join(fragments)
                 + '. Positive differences favor NeST-Com except LPIPS, where negative is better. '
                 'The complete three-SNR results and pointwise source-bootstrap intervals are reported '
                 'in the accompanying table; this small fixed-crop benchmark supports a limited '
                 'cross-collection comparison.\n')
    (out / 'manuscript_results.tex').write_text(paragraph, encoding='utf8')
    report = ['# Frozen final-baseline Kodak24 comparison', '',
              'All 24 fixed 256x256 center crops; N1024; SNR4/10/19; original noise labels2001/2002/2003. '
              'Every failure frame is included. Equal noise labels do not claim identical received observations across different waveforms.', '',
              'Two new baselines contribute432 actual logical frames and1728 metric cells. '
              'The existing raw and adaptive BPG anchors contribute432 verified reused frames. '
              'There are24 new single-method intervals and24 new NeST-minus-baseline intervals; '
              'the original raw/BPG summary rows and12 NeST-minus-BPG intervals are copied verbatim. '
              'No old interval is recomputed, and no Kodak selection, training, or calibration occurs.', '',
              'The original analyze.py interval function and original shared10000x24 bootstrap index array '
              'construction are reused with seed2026100901. Three noises are averaged within each source '
              'before source-level inference. LPIPS deltas keep their original sign; agreement is a source-prediction proxy.', '',
              '| Method | SNR (dB) | Metric | Mean | 95% CI |',
              '|---|---:|---|---:|---|']
    for row in summary:
        scale = 100 if row['metric'] == METRICS[-1] else 1
        report.append(f"| {DISPLAY[row['method']][0]} | {row['snr_db']} | {row['metric']} | "
                      f"{float(row['mean']) * scale:.6g} | [{float(row['ci_low']) * scale:.6g}, "
                      f"{float(row['ci_high']) * scale:.6g}] |")
    report.extend(['', 'All CSV values retain the original precision; the Markdown table is for reading. '
                   'The four-metric figure uses every mean and interval at true numerical SNR positions. '
                   'Its editable SVG and vector PDF do not contain rasterized data curves. PNG visual review must be recorded separately.'])
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf8')


def finish_render(out, checkpoint, no_plots=False):
    """Layout only: use immutable completed statistics, with no resampling."""
    require(checkpoint['status'] == 'KODAK24_FINAL_BASELINE_SOURCE_STATISTICS_COMPLETE_V1',
            'Complete source-statistics checkpoint required for rendering')
    require(checkpoint['script_sha256'] == sha(__file__), 'Rendering entry changed after its statistics binding')
    require(not (out / 'completion.json').exists(), 'Already completed delivery must not be rendered again')
    require(all(sha(p) == h for p, h in checkpoint['inputs'].items()),
            'An input changed after completed statistics')
    require(all(sha(p) == h for p, h in checkpoint['outputs'].items()),
            'Completed statistical tables changed before rendering')
    summary = read_csv(out / 'summary.csv')
    validate_summary(summary)
    paired = read_csv(out / 'paired_new.csv')
    require(len(paired) == 24, 'Exact completed new paired table required')
    figure_outputs = [] if no_plots else plot(out, summary)
    write_prose(out, summary, paired)
    require(all(sha(p) == h for p, h in checkpoint['inputs'].items())
            and all(sha(p) == h for p, h in checkpoint['outputs'].items()),
            'Layout modified scientific inputs or completed statistical tables')
    result = dict(checkpoint)
    result.update(status='KODAK24_FINAL_BASELINE_SOURCE_STATISTICS_AND_PLOT_COMPLETE_V1',
                  outputs={str(p.resolve()): sha(p) for p in out.iterdir() if p.is_file()},
                  statistics_checkpoint_sha256=sha(out / 'statistics_completion.json'),
                  figure_formats=len(figure_outputs), actual_PNG_visual_review='PENDING',
                  layout_new_bootstrap_calls=0)
    write_json(out / 'completion.json', result)
    return result


def render_existing(out):
    """Explicit recovery after a layout failure; never restart completed statistics."""
    out = Path(out)
    require((out / 'statistics_completion.json').is_file(),
            'No completed statistics checkpoint; automatic retry is forbidden')
    return finish_render(out, read_json(out / 'statistics_completion.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path)
    parser.add_argument('--completion', type=Path)
    parser.add_argument('--old-root', type=Path, default=ROOT / 'results/generalization_kodak_20261009')
    parser.add_argument('--reuse-audit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--render-only', action='store_true',
                        help='Recover layout from a SHA-verified completed statistics checkpoint; no bootstrap')
    args = parser.parse_args()
    if args.render_only:
        require(not args.no_plots and args.rows is None and args.completion is None and args.reuse_audit is None,
                '--render-only requires only --out, and never accepts new scoring/statistics inputs')
        result = render_existing(args.out)
        print(json.dumps(dict(status='LAYOUT_COMPLETE_FROM_STATISTICS_CHECKPOINT',
                              layout_new_bootstrap_calls=0, figure_formats=result['figure_formats'])))
        return
    require(args.rows is not None and args.completion is not None and args.reuse_audit is not None,
            'Normal execution requires --rows, --completion, and --reuse-audit')
    require(not args.out.exists(), 'Use a new independent output directory; never repeat completed statistics')
    original = load_original()
    old_metrics = args.old_root / 'metrics_v1'
    old_analysis = args.old_root / 'analysis_v1'
    old_done = read_json(old_metrics / 'completion.json')
    analysis_done = read_json(old_analysis / 'completion.json')
    bound_file(old_metrics / 'rows.csv', old_done)
    for name in ('summary.csv', 'source_means.csv', 'paired.csv'):
        bound_file(old_analysis / name, analysis_done)
    new_done = read_json(args.completion)
    bound_file(args.rows, new_done)
    math_proof_path = validate_new_math_proof(args.completion, new_done, old_done)
    validate_reuse(read_json(args.reuse_audit))
    old_rows = read_csv(old_metrics / 'rows.csv')
    ids, old_arrays, old_frames, old_noises = original.validate(old_rows, old_done)
    require(all(old_noises[m] == list(NOISES) for m in ANCHORS), 'Original anchor noises must remain unchanged')
    new_rows = read_csv(args.rows)
    new_arrays, new_frames = validate_new(new_rows, new_done, ids, old_rows)
    arrays = {k: v for k, v in old_arrays.items() if k[0] in ANCHORS}
    arrays.update(new_arrays)
    input_paths = [ORIGINAL, args.rows, args.completion, math_proof_path, args.reuse_audit, old_metrics / 'rows.csv',
                   old_metrics / 'completion.json', old_analysis / 'completion.json',
                   *(old_analysis / n for n in ('summary.csv', 'source_means.csv', 'paired.csv'))]
    inputs = {str(p.resolve()): sha(p) for p in input_paths}
    old_summary_bytes, count = csv_subset_bytes(old_analysis / 'summary.csv', lambda r: r['method'] in ANCHORS)
    require(count == 24, 'Exactly24 historical anchor summary rows')
    old_means_bytes, count = csv_subset_bytes(old_analysis / 'source_means.csv', lambda r: r['method'] in ANCHORS)
    require(count == 576, 'Exactly576 historical anchor source mean rows')
    old_paired_bytes, count = csv_subset_bytes(old_analysis / 'paired.csv', lambda r: r['reference'] == 'BPG_ADAPTIVE')
    require(count == 12, 'Exactly12 historical NeST-minus-BPG rows')
    for row in read_csv(old_analysis / 'source_means.csv'):
        if row['method'] in ANCHORS:
            key = row['method'], int(row['snr_db']), row['metric']
            require(row['source_id'] == ids[int(row['source_index'])]
                    and abs(float(row['mean']) - arrays[key][int(row['source_index'])]) < 1e-12,
                    'Historical source mean differs from original metric rows')
    args.out.mkdir(parents=True)
    write_json(args.out / 'statistics_reserved.json', dict(
        status='RESERVED_SOURCE_STATISTICS_NO_AUTOMATIC_RETRY', inputs=inputs,
        script_sha256=sha(__file__), bootstrap_seed=SEED, bootstrap_replicates=REPLICATES,
        new_summary_intervals=24, new_paired_intervals=24, old_interval_recomputations=0))
    new_summary, new_paired, new_means, draw_sha = new_statistics(original, ids, arrays)
    require(draw_sha == analysis_done['shared_draws_sha256'], 'Bootstrap index bytes differ from original Kodak draw array')
    append_csv_preserving_old(args.out / 'summary.csv', old_summary_bytes, new_summary)
    append_csv_preserving_old(args.out / 'source_means.csv', old_means_bytes, new_means)
    append_csv_preserving_old(args.out / 'paired.csv', old_paired_bytes, new_paired)
    write_csv(args.out / 'paired_new.csv', new_paired)
    write_csv(args.out / 'summary_new.csv', new_summary)
    write_csv(args.out / 'source_means_new.csv', new_means)
    write_union_csv(args.out / 'rows_all_four_methods.csv',
                    [dict(r, original_metric_backend_identity=r['metric_backend_identity'])
                     for r in old_rows if r['method'] in ANCHORS] + new_rows)
    summary = read_csv(args.out / 'summary.csv')
    validate_summary(summary)
    require(len(read_csv(args.out / 'source_means.csv')) == 1152
            and len(read_csv(args.out / 'paired.csv')) == 36, 'Complete four-method/source/contrast exports')
    require((args.out / 'summary.csv').read_bytes().startswith(old_summary_bytes)
            and (args.out / 'source_means.csv').read_bytes().startswith(old_means_bytes)
            and (args.out / 'paired.csv').read_bytes().startswith(old_paired_bytes), 'Historical row bytes changed')
    frames = {k: v for k, v in old_frames.items() if k[0] in ANCHORS}
    frames.update(new_frames)
    require(len(frames) == 864, 'Combined grid must contain864 logical method frames')
    states = []
    for method in METHODS:
        for snr in SNRS:
            statuses = [v[1] for k, v in frames.items() if k[0] == method and k[1] == snr]
            require(len(statuses) == 72, 'Every method/SNR requires72 frames including failures')
            for state in sorted(set(statuses)):
                count = statuses.count(state)
                states.append(dict(method=method, snr_db=snr, status=state,
                                   frames=count, denominator=72, fraction=count / 72))
    write_csv(args.out / 'status_counts.csv', states)
    require(all(sha(p) == h for p, h in inputs.items()), 'An input changed during postprocessing')
    outputs = {str(p.resolve()): sha(p) for p in args.out.iterdir() if p.is_file()}
    checkpoint = dict(
        status='KODAK24_FINAL_BASELINE_SOURCE_STATISTICS_COMPLETE_V1',
        source_count=24, noise_count=3, frame_count=864, new_frame_count=432,
        reused_frame_count=432, summary_rows=48, summary_new_rows=24,
        paired_rows=36, paired_new_rows=24, paired_reused_rows=12, source_mean_rows=1152,
        bootstrap_seed=SEED, bootstrap_replicates=REPLICATES, draws_calls=1,
        interval_function_calls=48, original_interval_function_sha256=sha(ORIGINAL),
        shared_draws_sha256=draw_sha, source_ids=ids, methods=list(METHODS),
        inputs=inputs, outputs=outputs, script_sha256=sha(__file__),
        old_interval_recomputations=0, historical_rows_byte_preserved=True,
        new_model_calls=0, new_PHY_calls=0, new_channel_draws=0,
        old_statistics_modified=False, Kodak_policy_selection=False, training_updates=0)
    write_json(args.out / 'statistics_completion.json', checkpoint)
    result = finish_render(args.out, checkpoint, no_plots=args.no_plots)
    print(json.dumps(dict(status='COMPLETE', summary_rows=48, paired_rows=36,
                          new_intervals=48, figure_formats=result['figure_formats'])))


if __name__ == '__main__':
    main()
