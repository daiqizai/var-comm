"""Read-only audit and publication report for the closed baseline supplement.

No model, channel or bootstrap calls. Original result files stay unchanged.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/baseline_strength_20261011'
CHAIN = BASE / 'automatic_followthrough_v1'
STATS = BASE / 'paired_supplement_v1'
FIGS = BASE / 'final_comparison_figures_v1'
OUT = BASE / 'formal_delivery_v1'
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine',
           'convnext_top1_source_prediction')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    assert not OUT.exists(), 'Publication audit already exists'
    assert not list(STATS.glob('*failure.json')), 'Preserve and resolve failures first'
    assert read(STATS / 'completion.json')['status'] == 'COMPLETE_BOTH_SOURCE_PAIRED_SUPPLEMENTS'
    waits = {}
    for name in ('actual_wait.json', 'plots_diagnostics_actual_wait.json',
                 'plots_final_actual_wait.json'):
        value = read(STATS / name)
        assert value['exit_code'] == 0 and value['actual_child_waited'], name
        assert not value.get('timed_out', False), name
        waits[name] = value
    assert read(CHAIN / 'completion.json')['status'] == 'ALL_DIAGNOSTIC_AND_FROZEN_FRAME_SCORING_COMPLETE'
    figure = read(FIGS / 'completion.json')
    for name, digest in figure['output_sha256'].items():
        assert sha(FIGS / name) == digest
    audits = {}; bindings = {}; pairs = {}
    for cohort, count, snrs in [('common500', 500, [1, 4, 7, 10, 13, 19]),
                               ('new100', 100, [4, 10, 19])]:
        folder = STATS / cohort
        admission = read(folder / 'admission.json')
        assert admission['source_count'] == count and admission['noise_count'] == 3
        assert admission['old_intervals_recomputed'] == 0
        assert admission['all_failure_rows_retained'] and not admission['selection_feedback']
        done = read(folder / 'completion.json')
        for name, digest in done['outputs'].items():
            assert sha(folder / name) == digest
            bindings[str(folder / name)] = digest
        summary = rows(folder / 'summary.csv'); contrasts = rows(folder / 'paired.csv')
        assert len(summary) == 2 * len(snrs) * len(METRICS)
        assert len(contrasts) == (4 if cohort == 'common500' else 2) * len(snrs) * len(METRICS)
        for data in (summary, contrasts):
            keys = set(); groups = {}
            for row in data:
                key = (row['method'], row.get('reference'), row['metric'], int(row['snr_db']))
                assert key not in keys; keys.add(key)
                assert int(row['source_count']) == count and int(row['noise_count']) == 3
                assert float(row['ci_low']) <= float(row['mean']) <= float(row['ci_high'])
                assert row['improvement_direction'] == ('negative' if row['metric'] == 'lpips_alex' else 'positive')
                assert float(row['display_scale']) == (100 if row['metric'] == METRICS[-1] else 1)
                if 'reference' in row:
                    assert row['delta_definition'] == 'method - reference'
                groups.setdefault(key[:3], []).append(key[-1])
            assert all(sorted(values) == snrs for values in groups.values())
        failures = rows(folder / 'failures.csv')
        assert sum(int(r['frame_count']) for r in failures) == count * len(snrs) * 3 * 2
        assert all(int(r['duplicate_count']) == int(r['missing_count']) == 0 for r in failures)
        audits[cohort] = dict(source_count=count, noise_count=3,
                             frames=count * len(snrs) * 3 * 2,
                             new_summary_rows=len(summary), paired_rows=len(contrasts),
                             admission=admission, failure_counts=failures)
        pairs[cohort] = contrasts
    OUT.mkdir()
    report = ['# Frozen baseline supplement: formal evaluation and paired statistics', '',
              'The selected P100000 and Swin200000 checkpoints completed 18,000 original-common500 and 1,800 existing-new100 method frames. The independent scoring and statistics/plot processes all returned actual exit zero. Every source, SNR and registered noise appears exactly once per new method; failure rows remain in the denominator. No training or checkpoint reselection followed these scores.', '',
              'Common500 uses the original continuous-baseline noises 2001/2002/2003; new100 uses 9301/9302/9303. Digital common500 anchors retain their original 6201/6202/6203 noises. Pairing is by source after averaging each method over its own registered three noises, not by identical received observations. The two cohorts are never combined.', '',
              'Only new single-method intervals and the registered contrasts are computed, using the original source-level 10,000-replicate bootstrap and seed 2026100701. All old intervals are copied unchanged. These are subsequent baseline comparisons on previously used evaluation cohorts, not newly blind tests. Frozen metric definitions, model identities and numerical settings are retained; cross-host bitwise equivalence is not claimed.', '',
              '## Source-paired PSNR contrasts', '',
              'Values below are method minus reference in dB, with direct paired 95% intervals. Full-precision CSVs contain all four metrics, including negative and zero values.', '']
    for cohort, data in pairs.items():
        report += [f'### {cohort}', '', '| Method | Reference | SNR (dB) | Delta PSNR (dB) | 95% interval |',
                   '|---|---|---:|---:|---|']
        for row in data:
            if row['metric'] == 'psnr_db':
                report.append(f"| {row['method']} | {row['reference']} | {row['snr_db']} | {float(row['mean']):.6f} | [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] |")
        report.append('')
    report += ['## Reading the figures', '',
               'PSNR and DINOv2-L are higher-is-better. LPIPS retains its original sign and is lower-is-better. ConvNeXt source-prediction agreement is not classification accuracy; its single-method values and intervals display percent, while paired differences display percentage points. All six common500 SNR positions use their true numerical spacing.', '',
               'The old adapted Swin80k point at 19 dB remains outside its training/calibration range and is shown with an open marker and dashed final segment. The new adapted Swin200k model combines additional training with expanded 1–19 dB coverage and the explicitly authorized fresh optimizer, sampler and RNG from the 80k weights. It reached the 120k additional-update budget while still improving; convergence and official optimality are not claimed. P100k resumed its full training state and met the registered empirical plateau rule. No capacity expansion was run.', '',
               'Four final figure groups are available under `../final_comparison_figures_v1/` as vector PDF, editable-text SVG and 600 dpi PNG, with exact plot rows and English captions. PNG visual review is recorded separately; generation alone is not review.', '',
               '## Remaining delivery', '',
               'Same-machine timing of the four digital modes and two selected baselines is still pending implementation qualification and an exclusive resource window. No old 4090D timing is mixed into an H800 speed comparison. The final training recovery states and these lightweight result increments depend on the existing original model/data/runtime recovery archives.']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    settings = r'''\paragraph{Continued adapted baselines.}
We keep the tokenizer, unconditional VAR and shared image decoder frozen.
The latent continuous baseline retains its original 495,208-parameter communication
module and resumes the complete 40k training state. Its calibration-selected 100k
checkpoint satisfies the preregistered empirical plateau criterion under the tested
learning-rate schedule. The adapted SwinJSCC baseline starts from the 80k weights
with newly initialized optimizer, sampling and random states, extends training SNRs
to the integers 1--19 dB, and retains the original C6/C13 training interfaces and paid
side information. Its selected 200k checkpoint reaches the additional 120k-update
budget while the calibration objective is still improving; convergence is not claimed.
The old 80k result at 19 dB remains outside its original training/calibration range.
After freezing these selections, we evaluate N1024 on the original 500 sources at
1/4/7/10/13/19 dB and, separately, on the previously published 100-source cohort at
4/10/19 dB, using three registered noise replicates per source. These are subsequent
same-source baseline supplements, not newly blind tests. We average within source
before applying the original 10,000-replicate paired bootstrap (seed 2026100701).
All failure cases remain in the denominators, and no evaluation score feeds back
to checkpoint selection. ConvNeXt prediction agreement measures agreement with
the original image prediction, not ground-truth classification accuracy.
'''
    (OUT / 'settings.tex').write_text(settings)
    (OUT / 'audit.json').write_text(json.dumps(dict(status='PASS_CLOSED_FORMAL_SUPPLEMENT',
        cohorts=audits, actual_waits=waits, input_sha256=bindings,
        extra_inference_calls=0, extra_bootstrap_calls=0, timing_complete=False), indent=2) + '\n')
    print('PASS_CLOSED_FORMAL_SUPPLEMENT')


if __name__ == '__main__':
    main()
