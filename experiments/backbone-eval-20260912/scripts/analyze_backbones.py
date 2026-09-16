#!/usr/bin/env python3
"""CPU-only paired analysis of the bounded backbone evaluation.

No generative FID and no physical-channel claims. Unequal natural source rates
are explicitly labelled, never interpolated/rebranded as old VAR 'm8'.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROJECT = EXPERIMENT.parents[1]
METRICS = ('psnr_db', 'ssim', 'lpips_alex', 'dino_cosine')
HIGHER = {'psnr_db': True, 'ssim': True, 'lpips_alex': False, 'dino_cosine': True}
MODELS = ('old-official', 'old-fidelity', 'xq', 'wetok')


def write_json(path: Path, data: Any):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, *, seed=20260912, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty image vector required')
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), (resamples, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [.025, .975])
    return float(values.mean()), float(low), float(high)


def load_runs(root: Path, *, require_all=True):
    result, receipts, paths = [], {}, {}
    for model in MODELS:
        directory = root / model
        complete_path = directory / 'completion.json'
        if not complete_path.exists():
            if require_all:
                raise FileNotFoundError(f'missing complete run: {complete_path}')
            continue
        receipt = json.loads(complete_path.read_text())
        if receipt.get('status') != 'COMPLETE':
            raise RuntimeError(f'{model} is not complete')
        metadata = json.loads((directory / 'metadata.json').read_text())
        source = directory / 'per_image.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha != receipt['per_image_jsonl_sha256']:
            raise RuntimeError(f'{model} per-image artifact hash changed')
        declared_names = {condition['name'] for condition in metadata['conditions']}
        if set(row['name'] for row in rows) != declared_names:
            raise RuntimeError('actual conditions differ from declared frozen protocol')
        if len(rows) != receipt['images'] * len(declared_names):
            raise RuntimeError('incomplete image-by-condition Cartesian product')
        if len(rows) != receipt['rows']:
            raise RuntimeError('row count differs from completion receipt')
        keys = [(row['image_id'], row['name']) for row in rows]
        if len(set(keys)) != len(keys):
            raise RuntimeError('duplicate image/condition keys')
        for row in rows:
            if row['model'] != model or not all(np.isfinite(row[metric]) for metric in METRICS):
                raise RuntimeError('invalid model or nonfinite metrics')
            if row['kind'] == 'completion':
                if row['prefix_audit'].get('prefix_preserved') is not True or row['prefix_audit'].get('true_suffix_passed') is not False:
                    raise RuntimeError('completion receiver prefix boundary is unaudited')
        if len({row['image_id'] for row in rows}) != receipt['images']:
            raise RuntimeError('image population count mismatch')
        result.extend(rows)
        receipts[model] = {'completion': receipt, 'metadata': metadata}
        paths[model] = directory
    if not result:
        raise RuntimeError('no complete model runs')
    manifests = {r['completion']['manifest_sha256'] for r in receipts.values()}
    image_counts = {r['completion']['images'] for r in receipts.values()}
    smoke = {r['completion']['smoke'] for r in receipts.values()}
    if len(manifests) != 1 or len(image_counts) != 1 or len(smoke) != 1:
        raise RuntimeError('runs do not share manifest, population size and smoke/full role')
    sample_sources = defaultdict(set)
    for row in result:
        sample_sources[row['image_id']].add((row['source_file_sha256'], row['source_tensor_sha256'], row['class_index']))
    if len(sample_sources) != next(iter(image_counts)):
        raise RuntimeError('models have different source image identities')
    if any(len(signatures) != 1 for signatures in sample_sources.values()):
        raise RuntimeError('sources/preprocessing/classes differ between models')
    return result, receipts, paths


def group_rows(rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['name'])].append(row)
    for selected in grouped.values():
        selected.sort(key=lambda row: row['image_id'])
        if len({row['image_id'] for row in selected}) != len(selected):
            raise RuntimeError('condition has duplicate images')
    return dict(grouped)


def image_averaged_sampling(groups: dict):
    """Sensitivity seeds are repeated predictions, not extra independent images."""
    groups = dict(groups)
    for prefix in (8, 9):
        selections = [rows for (model, name), rows in groups.items()
                      if model == 'xq' and name.startswith(f'p{prefix}_sample_s')]
        if not selections:
            continue
        ids = [tuple(row['image_id'] for row in rows) for rows in selections]
        if len(set(ids)) != 1:
            raise RuntimeError('sampling seeds have different image populations')
        means = []
        for position, image_id in enumerate(ids[0]):
            records = [rows[position] for rows in selections]
            result = dict(records[0])
            result['name'] = f'p{prefix}_sample_mean'
            result['seed'] = None
            result['sampling_seeds'] = [row['seed'] for row in records]
            result['sampling_aggregation'] = 'mean_metrics_across_seeds_within_image_not_best_seed'
            for metric in METRICS:
                result[metric] = float(np.mean([row[metric] for row in records]))
            result['inference_measure'] = dict(records[0]['inference_measure'])
            result['inference_measure']['seconds'] = float(np.mean([row['inference_measure']['seconds'] for row in records]))
            means.append(result)
        groups[('xq', f'p{prefix}_sample_mean')] = means
    return groups


def summarize(groups: dict, resamples: int):
    summaries = []
    for (model, name), rows in sorted(groups.items()):
        bits = {int(row['raw_bits']) for row in rows}
        if len(bits) != 1:
            raise RuntimeError('a natural rate point has variable raw bit count')
        summary = {'model': model, 'condition': name, 'raw_bits': bits.pop(), 'n_images': len(rows),
            'primary': rows[0]['primary'], 'kind': rows[0]['kind'],
            'class_side_bits_if_sent': rows[0]['class_side_bits_if_sent'],
            'encode_seconds_mean': float(np.mean([row['encode_measure']['seconds'] for row in rows])),
            'inference_seconds_mean': float(np.mean([row['inference_measure']['seconds'] for row in rows])),
            'inference_seconds_median': float(np.median([row['inference_measure']['seconds'] for row in rows])),
            'inference_seconds_p95': float(np.quantile([row['inference_measure']['seconds'] for row in rows], .95)),
            'peak_codec_allocated_bytes': max(max(row['encode_measure']['peak_allocated_bytes'],
                                                  row['inference_measure']['peak_allocated_bytes']) for row in rows),
            'peak_codec_reserved_bytes': max(max(row['encode_measure']['peak_reserved_bytes'],
                                                 row['inference_measure']['peak_reserved_bytes']) for row in rows)}
        summary['encode_plus_inference_seconds_mean'] = summary['encode_seconds_mean'] + summary['inference_seconds_mean']
        for metric in METRICS:
            mean, low, high = bootstrap([row[metric] for row in rows], resamples=resamples)
            summary[metric + '_mean'] = mean
            summary[metric + '_ci95_low'] = low
            summary[metric + '_ci95_high'] = high
        summaries.append(summary)
    return summaries


def pair_comparisons(groups: dict):
    comparisons = []
    for model in ('xq', 'wetok'):
        for reference in ('old-official', 'old-fidelity'):
            comparisons.append(((model, 'full'), (reference, 'full'), 'full_codec_unequal_raw_bits'))
    comparisons.append((('old-fidelity', 'full'), ('old-official', 'full'), 'existing_fidelity_tradeoff_control'))
    for model in ('old-official', 'xq'):
        for prefix in (8, 9):
            comparisons.append(((model, f'p{prefix}_argmax'), (model, f'p{prefix}_direct'), 'same_prefix_completion_value'))
    for xq_prefix, old_prefix in ((8, 8), (9, 8), (9, 9)):
        for suffix in ('argmax', 'direct'):
            comparisons.append((('xq', f'p{xq_prefix}_{suffix}'), ('old-official', f'p{old_prefix}_{suffix}'),
                                'natural_prefix_unequal_raw_bits'))
        comparisons.append((('xq', f'p{xq_prefix}_sample_mean'), ('old-official', f'p{old_prefix}_argmax'),
                            'sampling_sensitivity_unequal_rates_and_decoding_rules'))
    for prefix in (8, 9):
        comparisons.append((('xq', f'p{prefix}_sample_mean'), ('xq', f'p{prefix}_argmax'),
                            'sampling_sensitivity_same_prefix'))
    return [comparison for comparison in comparisons if comparison[0] in groups and comparison[1] in groups]


def paired(groups: dict, resamples: int):
    results, per_image = [], []
    for candidate_key, reference_key, role in pair_comparisons(groups):
        candidate, reference = groups[candidate_key], groups[reference_key]
        if [row['image_id'] for row in candidate] != [row['image_id'] for row in reference]:
            raise RuntimeError(f'nonidentical paired image sets: {candidate_key} {reference_key}')
        description = {'candidate_model': candidate_key[0], 'candidate_condition': candidate_key[1],
            'reference_model': reference_key[0], 'reference_condition': reference_key[1], 'comparison_role': role,
            'candidate_raw_bits': int(candidate[0]['raw_bits']), 'reference_raw_bits': int(reference[0]['raw_bits']),
            'raw_bit_delta': int(candidate[0]['raw_bits']) - int(reference[0]['raw_bits']),
            'n_images': len(candidate), 'delta_convention': 'candidate_minus_reference'}
        for metric in METRICS:
            delta = np.array([a[metric] - b[metric] for a, b in zip(candidate, reference)], dtype=np.float64)
            mean, low, high = bootstrap(delta, resamples=resamples)
            oriented = delta if HIGHER[metric] else -delta
            results.append({**description, 'metric': metric, 'delta_mean': mean, 'delta_ci95_low': low,
                'delta_ci95_high': high, 'candidate_wins': int((oriented > 1e-12).sum()),
                'ties': int((np.abs(oriented) <= 1e-12).sum()), 'candidate_losses': int((oriented < -1e-12).sum())})
        for a, b in zip(candidate, reference):
            per_image.append({**description, 'image_id': a['image_id'], 'image_index': a['image_index'],
                **{metric + '_delta': a[metric] - b[metric] for metric in METRICS}})
    return results, per_image


def evidence_flags(pair_rows: list[dict]):
    """Descriptive flags, not arbitrary 5% gates or an automatic migration claim."""
    grouped = defaultdict(dict)
    for row in pair_rows:
        key = (row['candidate_model'], row['candidate_condition'], row['reference_model'], row['reference_condition'])
        grouped[key][row['metric']] = row
    flags = []
    for key, rows in grouped.items():
        if key[0] != 'xq' or key[2] != 'old-official' or key[1] not in ('p8_argmax', 'p9_argmax', 'full'):
            continue
        p, l = rows['psnr_db'], rows['lpips_alex']
        flags.append({'candidate': '/'.join(key[:2]), 'reference': '/'.join(key[2:]),
            'raw_bit_delta': p['raw_bit_delta'],
            'psnr_paired_ci_strictly_positive': p['delta_ci95_low'] > 0,
            'lpips_paired_ci_strictly_worse': l['delta_ci95_low'] > 0,
            'lpips_paired_ci_no_worse_upper_bound': l['delta_ci95_high'] <= 0,
            'note': 'CI flags are descriptive; a CI crossing zero is not proof of equivalence. Migration also needs instance inspection and meaningful rate/quality gain.'})
    return flags


def contact_sheets(groups: dict, paths: dict, output: Path):
    from PIL import Image, ImageDraw
    if ('old-official', 'full') not in groups:
        return []
    official = groups[('old-official', 'full')]
    indices = []
    reasons = defaultdict(list)
    def add(rows, reason):
        for row in rows:
            index = row['image_index']
            if index not in indices:
                indices.append(index)
            reasons[index].append(reason)
    add(sorted(official, key=lambda row: row['psnr_db'])[:6], 'six_lowest_original_full_PSNR')
    if ('xq', 'full') in groups:
        xq = {row['image_id']: row for row in groups[('xq', 'full')]}
        add(sorted(official, key=lambda row: xq[row['image_id']]['psnr_db'] - row['psnr_db'])[:3],
            'three_worst_XQ_full_PSNR_deltas')
        add(sorted(official, key=lambda row: xq[row['image_id']]['lpips_alex'] - row['lpips_alex'], reverse=True)[:3],
            'three_worst_XQ_full_LPIPS_deltas')
    sets = {
        'full_reconstruction': [('old-official', 'full'), ('old-fidelity', 'full'), ('xq', 'full'), ('wetok', 'full')],
        'prefix_completion': [('old-official', 'p8_argmax'), ('xq', 'p8_argmax'), ('xq', 'p9_argmax'), ('old-official', 'p9_argmax')],
        'xq_same_prefix': [('xq', 'p8_direct'), ('xq', 'p8_argmax'), ('xq', 'p9_direct'), ('xq', 'p9_argmax')],
    }
    lookup = {(model, name, row['image_index']): row for (model, name), rows in groups.items() for row in rows}
    for label, entries in sets.items():
        entries = [key for key in entries if key in groups]
        if not entries:
            continue
        cell_w, cell_h = 258, 295
        canvas = Image.new('RGB', ((1 + len(entries)) * cell_w, len(indices) * cell_h), 'white')
        draw = ImageDraw.Draw(canvas)
        for y, index in enumerate(indices):
            reference = lookup[('old-official', 'full', index)]
            with Image.open(paths['old-official'] / 'images' / f'{index:03d}' / 'source.png') as source:
                canvas.paste(source.convert('RGB'), (0, y * cell_h + 37))
            draw.text((2, y * cell_h + 2), f"Source #{index} class{reference['class_index']}", fill='black')
            for x, (model, name) in enumerate(entries, 1):
                row = lookup[(model, name, index)]
                with Image.open(paths[model] / row['png_relative_path']) as image:
                    canvas.paste(image.convert('RGB'), (x * cell_w, y * cell_h + 37))
                draw.text((x * cell_w + 2, y * cell_h + 2), f"{model}/{name} {row['raw_bits']}bit\nPSNR {row['psnr_db']:.2f} LPIPS {row['lpips_alex']:.4f}", fill='black')
        canvas.save(output / f'{label}_difficult_cases.png')
    chosen = [{'image_index': index, 'selection_reasons': reasons[index],
               'image_id': lookup[('old-official', 'full', index)]['image_id']} for index in indices]
    write_json(output / 'difficult_case_selection.json', chosen)
    return chosen


def rate_plot(groups: dict, summaries: list[dict], output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'old-official': '#555555', 'old-fidelity': '#9d6200', 'xq': '#126faa', 'wetok': '#0b875b'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, metric in zip(axes, ('psnr_db', 'lpips_alex', 'ssim')):
        for row in summaries:
            if not row['primary'] or row['kind'] == 'direct':
                continue
            marker = 's' if row['kind'] == 'full' else 'o'
            axis.errorbar([row['raw_bits']], [row[metric + '_mean']],
                yerr=[[row[metric + '_mean'] - row[metric + '_ci95_low']], [row[metric + '_ci95_high'] - row[metric + '_mean']]],
                fmt=marker, color=colors[row['model']], capsize=3)
            axis.annotate(row['model'] + '/' + row['condition'], (row['raw_bits'], row[metric + '_mean']),
                          xytext=(3, 4), textcoords='offset points', fontsize=7)
        axis.set_xlabel('Actual fixed-length source index bits / image')
        axis.set_ylabel(metric + (' (lower better)' if metric == 'lpips_alex' else ' (higher better)'))
        axis.grid(alpha=.25)
    fig.suptitle('Natural discrete source-rate points; no channel, no interpolation, class pre-shared for completion')
    fig.tight_layout()
    fig.savefig(output / 'rate_quality_discrete.png', dpi=180)
    plt.close(fig)


def historical_replay(groups: dict, receipts: dict, output: Path):
    """Report new-vs-archived numerical replay; do not impose a scientific gate."""
    historical = receipts.get('old-official', {}).get('metadata', {}).get('config', {}).get('historical_source')
    if not historical:
        write_json(output / 'historical_replay_receipt.json', {'status': 'NOT_CONFIGURED'})
        return []
    path = Path(historical) / 'outputs/analysis/ANALYSIS-VAR-SEMANTIC-RATE-CONTRIBUTION-IMAGENET100-001/per_image_metrics.csv'
    if not path.is_file():
        raise FileNotFoundError(f'archived baseline table missing: {path}')
    with path.open(newline='') as handle:
        archived = {(row['image_id'], row['decoder'], row['budget'], row['arm']): row for row in csv.DictReader(handle)}
    correspondence = {'full': ('full', 'full'), 'p8_direct': ('m8', 'prefix_only'),
        'p8_argmax': ('m8', 'var_completed'), 'p9_direct': ('m9', 'prefix_only'),
        'p9_argmax': ('m9', 'var_completed')}
    image_deltas, summary = [], []
    for (model, condition), rows in groups.items():
        if model not in ('old-official', 'old-fidelity') or condition not in correspondence:
            continue
        decoder = 'official' if model == 'old-official' else 'tuned'
        budget, arm = correspondence[condition]
        references = [archived[(row['image_id'], decoder, budget, arm)] for row in rows]
        for row, reference in zip(rows, references):
            if int(reference['class_index']) != row['class_index'] or int(reference['raw_bits']) != row['raw_bits']:
                raise RuntimeError('archived class/raw-rate pairing mismatch')
            image_deltas.append({'model': model, 'condition': condition, 'image_id': row['image_id'],
                'image_index': row['image_index'], **{metric + '_new_minus_archived': row[metric] - float(reference[metric]) for metric in METRICS}})
        for metric in METRICS:
            new = np.array([row[metric] for row in rows], dtype=np.float64)
            old = np.array([float(row[metric]) for row in references], dtype=np.float64)
            delta = new - old
            summary.append({'model': model, 'condition': condition, 'metric': metric, 'n_images': len(rows),
                'new_mean': float(new.mean()), 'archived_mean': float(old.mean()),
                'mean_delta': float(delta.mean()), 'max_abs_delta': float(np.abs(delta).max()),
                'minimum_delta': float(delta.min()), 'maximum_delta': float(delta.max())})
    write_csv(output / 'historical_replay_summary.csv', summary)
    write_csv(output / 'historical_replay_per_image.csv', image_deltas)
    write_json(output / 'historical_replay_receipt.json', {'status': 'REPLAY_DIFFERENCES_REPORTED_NO_HARD_FLOAT_GATE',
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'paired_rows': len(image_deltas),
        'note': 'New batch-one/no-TF32/deterministic-CuDNN evaluation versus archived batches may differ numerically; report rather than transplant old means or silently relabel a protocol change.'})
    return summary


def report(summaries, pair_rows, receipts, flags, replay, output: Path):
    main = [row for row in summaries if row['primary']]
    lines = ['# Bounded no-training backbone evaluation', '',
        '**Scope:** the original 100-image development population; noiseless source indices, not a channel experiment. '
        'Ground-truth ImageNet class is pre-shared for completion (optional 10-bit label shown separately). '
        'The XQ and original VAR scale numbers denote different token counts and are never rate-equated.', '',
        '## Primary paired-image means', '',
        '| Model | Natural point | Raw bits | PSNR | SSIM | LPIPS-Alex | DINO cosine | Encode ms | Decode/completion ms | Peak codec allocated MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if 'xq' in receipts:
        pair_meta = receipts['xq']['metadata']['adapter_metadata']
        if pair_meta.get('paired_tokenizer_matches_standalone') is not True:
            lines[2:2] = ['**CHECKPOINT IDENTITY WARNING:** The official paired-generator embedded tokenizer differs from the standalone model-zoo tokenizer candidates. All XQ rows below use the embedded tokenizer required by the paired generator, not an unqualified result for the standalone checkpoint. Detailed tensor comparisons are in `inputs.json`.', '']
    for row in main:
        lines.append(f"| {row['model']} | {row['condition']} | {row['raw_bits']} | {row['psnr_db_mean']:.4f} | "
            f"{row['ssim_mean']:.5f} | {row['lpips_alex_mean']:.5f} | {row['dino_cosine_mean']:.5f} | "
            f"{1000*row['encode_seconds_mean']:.2f} | {1000*row['inference_seconds_mean']:.2f} | "
            f"{row['peak_codec_allocated_bytes']/2**20:.1f} |")
    if replay:
        lines += ['', '## Replay of the known original-VAR reference', '',
            'These are new minus archived per-image differences, not a model improvement. Values may shift slightly under batch-one deterministic/no-TF32 arithmetic; all comparisons above use the new unified evaluator. Large differences require investigation before interpreting migration.', '',
            '| Model | Condition | Metric | Mean delta | Maximum absolute per-image delta |',
            '|---|---|---|---:|---:|']
        for row in replay:
            lines.append(f"| {row['model']} | {row['condition']} | {row['metric']} | {row['mean_delta']:+.8g} | {row['max_abs_delta']:.8g} |")
    lines += ['', '## Interpretation boundaries', '',
        '- Full reconstruction is a codec result, not sufficient evidence for a next-scale migration.',
        '- Primary completions use true transmitted prefixes and deterministic class-conditioned argmax (no unconditional CFG mixing).',
        '- XQ official-style sampling is a sensitivity analysis only, not a per-image output selector; seed metrics are averaged within image before bootstrapping.',
        '- Compare `paired_deltas.csv` and `per_image_deltas.csv`: deltas are candidate minus reference. Positive LPIPS is worse.',
        '- Source-rate mismatches are explicit. No FEC/header/pilots/noise/complex-channel-use claim is made.',
        '- XQ uses DINO-related training; DINO similarity is reported, not independent evidence of recovered semantic information.',
        '- Lack of significant LPIPS degradation does not establish perceptual equivalence. Inspect paired intervals and difficult source-instance details.',
        '- Codec timing excludes file I/O, preprocessing and metric inference. Completion timing includes its final RGB decode. Sender and receiver model weights are resident together. Metrics are loaded only after releasing the codec.',
        '- A formal migration recommendation requires examining stable PSNR/LPIPS improvements at prefix+completion points and whether the actual rate advantage warrants adaptation cost. These descriptive flags are not a new arbitrary threshold.', '',
        '## Descriptive evidence flags', '', '```json', json.dumps(flags, indent=2), '```', '',
        '## Artifacts', '',
        '- `summary.csv`: per-condition image means/95% bootstrap intervals, batch-one timings and measured GPU peaks.',
        '- `paired_deltas.csv`, `per_image_deltas.csv`: source-paired comparisons and all per-image changes.',
        '- `rate_quality_discrete.png`: natural rates plotted as points only; no false equal-rate interpolation.',
        '- `*_difficult_cases.png`, `difficult_case_selection.json`: fixed worst-case selection rules, not best-looking examples.',
        '- `inputs.json`: immutable result hashes and per-model metadata.',
        '- `historical_replay_summary.csv` and `historical_replay_per_image.csv`: new-vs-archived numerical baseline audit.', '',
        '**Communication attribution remains open:** any adopted visual backbone needs its own digital baseline and re-adapted communication E/D; a better visual codec is not itself a JSCC contribution.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/eval.local.json')
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-incomplete-models', action='store_true', help='explicit partial descriptive analysis, never migration verdict')
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    root = args.results_root or Path(config.get('output_root', EXPERIMENT / 'outputs')) / ('smoke' if args.smoke else 'full')
    output = args.output_dir or root.parent / ('analysis-smoke' if args.smoke else 'analysis')
    output = output.resolve()
    if not (output.is_relative_to(EXPERIMENT.resolve()) or output.is_relative_to((PROJECT / 'outputs').resolve())):
        raise RuntimeError('analysis output must remain in VAR_COMM, never a historical project')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    rows, receipts, paths = load_runs(root, require_all=not args.allow_incomplete_models)
    n_images = next(iter(receipts.values()))['completion']['images']
    is_smoke = next(iter(receipts.values()))['completion']['smoke']
    if is_smoke != args.smoke or n_images != (2 if args.smoke else 100):
        raise RuntimeError('smoke/full role or original population count mismatch')
    groups = image_averaged_sampling(group_rows(rows))
    summaries = summarize(groups, args.bootstrap_resamples)
    pair_rows, per_image = paired(groups, args.bootstrap_resamples)
    flags = evidence_flags(pair_rows)
    output.mkdir(parents=True)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired_deltas.csv', pair_rows)
    write_csv(output / 'per_image_deltas.csv', per_image)
    write_json(output / 'inputs.json', receipts)
    write_json(output / 'evidence_flags.json', flags)
    contact_sheets(groups, paths, output)
    rate_plot(groups, summaries, output)
    replay = historical_replay(groups, receipts, output)
    report(summaries, pair_rows, receipts, flags, replay, output)
    write_json(output / 'completion.json', {'status': 'COMPLETE_DESCRIPTIVE_ANALYSIS', 'smoke': args.smoke,
        'images': n_images, 'models': list(receipts), 'complete_four_model_protocol': len(receipts) == 4,
        'bootstrap_unit': 'image', 'bootstrap_resamples': args.bootstrap_resamples, 'bootstrap_seed': 20260912,
        'source_only': True, 'migration_verdict': 'REQUIRES_PAIRED_PREFIX_QUALITY_AND_INSTANCE_REVIEW',
        'training_updates': 0, 'summary_sha256': hashlib.sha256((output / 'summary.csv').read_bytes()).hexdigest()})
    print(json.dumps({'status': 'COMPLETE_DESCRIPTIVE_ANALYSIS', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
