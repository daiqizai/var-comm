"""Import verified immutable N512 and older resource references; no inference."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs/EXTREME-BW-20261001-R1-N1024'
RESULT = ROOT / 'results/extreme_bandwidth_20261001_R1_N1024'
PRIOR_RESULT = ROOT / 'results/extreme_bandwidth_20260930_R1'
PRIOR_TRAIN = ROOT / 'outputs/EXTREME-BW-20260930-R1/training/p512_2026093001'
PRIOR_RAW_SHA = '8b6cdab96bd7549dc6935e16fe6b8c68317c49d5ec5d674682a189de343561ab'
PRIOR_CONFIG_SHA = 'ef0a4c0d694d8078f551c096f8a7abd44222082dcedcebc4d4df7626118fae3a'
PRIOR_POLICY_SHA = '2db4c88ec65da4ef4ef5151d1985a850b411c8a4a2951591560f50aec7c4601f'
PRIOR_CHECKPOINT_SHA = '28009f4263905bf307abec341ff4aff147140e3543ec3eae231781ca5ff0036b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_identity(rows):
    values = {(int(r['source_index']), r['source_id'], r['preprocessing_id']) for r in rows}
    if len(values) != 100 or {r[0] for r in values} != set(range(100)):
        raise RuntimeError('Expected the original 100 source identities')
    return sorted(values)


def verify_shared_visuals(previous, current):
    for name in ('vae', 'var', 'decoder', 'lpips', 'dino'):
        if previous['identity']['models'][name] != current['identity']['models'][name]:
            raise RuntimeError('N512/N1024 visual or quality asset differs: ' + name)
    for field in ('stats_sha256', 'quality'):
        if previous['identity'][field] != current['identity'][field]:
            raise RuntimeError('N512/N1024 common asset/quality preprocessing differs: ' + field)
    if previous['mismatch_permutation'] != current['mismatch_permutation']:
        raise RuntimeError('N512/N1024 mismatch source permutations differ')
    if previous['snrs'] != current['snrs'] or previous['development_seeds'] != current['development_seeds']:
        raise RuntimeError('N512/N1024 paired SNR/noise-repeat scope differs')


def main():
    identity = read(HERE / 'compatible_budget_reference_identity.json')
    source = HERE / 'compatible_budget_references.csv'
    if sha(source) != identity['reference_csv_sha256']:
        raise RuntimeError('Prepared historical reference table changed')
    for path, expected in identity['source_bindings'].items():
        if sha(path) != expected:
            raise RuntimeError('Historical reference evidence changed: ' + path)
    plugin = read(RESULT / 'plugin_config.json')
    digital = read(RESULT / 'digital_config.json')
    # Historical identities were checked against the original development set.
    # The digital calibration config contains1000 calibration identities; use
    # its separately frozen development identity for the source-level equality.
    development = read(OUT / 'digital/development_identity.json')
    if not plugin['no_development_tuning'] or not digital['no_development_tuning']:
        raise RuntimeError('Development selection boundary changed')
    if development['sources'] != identity['source_ids']:
        raise RuntimeError('Historical and current development sources differ')
    if development['preprocessing_ids'] != identity['preprocessing_ids']:
        raise RuntimeError('Historical and current development preprocessing differ')
    from csv import DictReader
    with (RESULT / 'plugin_per_frame.csv').open(newline='') as handle:
        rows = list(DictReader(handle))
    observed = sorted({(int(r['source_index']), r['source_id'], r['preprocessing_id']) for r in rows})
    expected = [(i, sid, pre) for i, (sid, pre) in enumerate(zip(identity['source_ids'], identity['preprocessing_ids']))]
    if observed != expected:
        raise RuntimeError('Historical source/preprocessing order differs from measured P1024')
    for name in ('compatible_budget_references.csv', 'compatible_budget_reference_identity.json'):
        shutil.copyfile(HERE / name, RESULT / name)

    pinned = {PRIOR_RESULT / 'per_frame.csv': PRIOR_RAW_SHA,
              PRIOR_RESULT / 'config.json': PRIOR_CONFIG_SHA,
              PRIOR_RESULT / 'selected_policy.json': PRIOR_POLICY_SHA}
    for path, expected_hash in pinned.items():
        if sha(path) != expected_hash:
            raise RuntimeError('Immutable N512 comparison asset changed: ' + str(path))
    previous = read(PRIOR_RESULT / 'config.json')
    completed = read(PRIOR_RESULT / 'analysis_completion.json')
    training = read(PRIOR_TRAIN / 'completion.json')
    selected = read(PRIOR_TRAIN / 'selected_P512.json')
    if completed['status'] != 'N512_SOURCE_PAIRED_ANALYSIS_COMPLETE' or completed['metric_rows'] != 45000:
        raise RuntimeError('N512 comparison was not fully measured')
    for name in ('per_frame.csv', 'config.json', 'selected_policy.json'):
        if completed['files'][name] != sha(PRIOR_RESULT / name):
            raise RuntimeError('N512 analysis receipt differs from measured evidence')
    if selected != training['selected']['P512'] or selected != previous['plugin']['identity']['selected']:
        raise RuntimeError('N512 selected model identity changed')
    if selected['step'] != 40000 or selected['checkpoint_sha256'] != PRIOR_CHECKPOINT_SHA or selected['N'] != 512:
        raise RuntimeError('N512 comparison is not the executed selected40k model')
    if sha(PRIOR_TRAIN / 'registration.json') != selected['registration_sha256']:
        raise RuntimeError('N512 training registration changed')
    if sha(selected['checkpoint']) != PRIOR_CHECKPOINT_SHA:
        raise RuntimeError('N512 selected checkpoint changed')
    if not training['state']['finished'] or training['development_read'] or training['convergence_claimed']:
        raise RuntimeError('N512 training completion boundary changed')
    verify_shared_visuals(previous['plugin'], plugin)
    with (PRIOR_RESULT / 'per_frame.csv').open(newline='') as handle:
        previous_rows = list(csv.DictReader(handle))
    if source_identity(previous_rows) != expected:
        raise RuntimeError('N512/N1024 source/preprocessing order differs')
    files = [*pinned, PRIOR_RESULT / 'analysis_completion.json', PRIOR_RESULT / 'summary.csv',
             PRIOR_RESULT / 'decisions.json', PRIOR_TRAIN / 'completion.json',
             PRIOR_TRAIN / 'selected_P512.json', PRIOR_TRAIN / 'registration.json']
    archive = RESULT / 'N512_reference'; archive.mkdir(exist_ok=True)
    for path in files:
        if path.name == 'per_frame.csv':
            continue  # Original N512 table shards retain the complete measured table.
        name = 'training_' + path.name if path.parent == PRIOR_TRAIN else path.name
        shutil.copyfile(path, archive / name)
    budget = dict(N=512, training_seed=selected['training_seed'], selected_updates=selected['step'], completed_updates=training['state']['step'],
        budget_truncated=training['budget_truncated'], stop_reason=training['state']['decisions'][-1]['reason'],
        final_relative_improvements=training['state']['decisions'][-1]['relative_improvements'],
        convergence_claimed=training['convergence_claimed'])
    write(OUT / 'prior_budget_import.json', dict(status='IMMUTABLE_N512_COMPARISON_IMPORTED', N=512,
        source_bindings={str(path): sha(path) for path in files}, original_result=str(PRIOR_RESULT),
        copied_light_receipts=str(archive), source_ids=identity['source_ids'], preprocessing_ids=identity['preprocessing_ids'],
        visual_models={name: plugin['identity']['models'][name] for name in ('vae', 'var', 'decoder', 'lpips', 'dino')},
        mismatch_permutation=plugin['mismatch_permutation'], budget=budget,
        pairing='same source/preprocessing/SNR/noise-repeat index; separate N/E waveforms and observations',
        source_reuse_correlated=True, training_seed_variance_estimated=False, gpu_work=False))
    # This is the supervisor's completion receipt. Write it only after every
    # historical/N512 binding, model and paired-source check has succeeded.
    write(OUT / 'reference_import.json', dict(status='COMPATIBLE_HISTORICAL_REFERENCES_IMPORTED',
        rows=15, gpu_work=False, reference_csv_sha256=sha(source),
        identity_sha256=sha(HERE / 'compatible_budget_reference_identity.json'),
        prior_budget_receipt_sha256=sha(OUT / 'prior_budget_import.json'),
        N512_comparison_verified=True))


if __name__ == '__main__':
    main()
