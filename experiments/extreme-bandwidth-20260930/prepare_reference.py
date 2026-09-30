"""Verify and copy previously completed pure-P resource references; no GPU work."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs/EXTREME-BW-20260930-R1'
RESULT = ROOT / 'results/extreme_bandwidth_20260930_R1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    identity = json.loads((HERE / 'compatible_budget_reference_identity.json').read_text())
    source = HERE / 'compatible_budget_references.csv'
    if sha(source) != identity['reference_csv_sha256']:
        raise RuntimeError('Prepared historical reference table changed')
    for path, expected in identity['source_bindings'].items():
        if sha(path) != expected:
            raise RuntimeError('Historical reference evidence changed: ' + path)
    plugin = json.loads((RESULT / 'plugin_config.json').read_text())
    digital = json.loads((RESULT / 'digital_config.json').read_text())
    # Historical identities were checked against the original development set.
    # The digital calibration config contains1000 calibration identities; use
    # its separately frozen development identity for the source-level equality.
    development = json.loads((OUT / 'digital/development_identity.json').read_text())
    if not plugin['no_development_tuning'] or not digital['no_development_tuning']:
        raise RuntimeError('Development selection boundary changed')
    if 'sources' in development and development['sources'] != identity['source_ids']:
        raise RuntimeError('Historical and current development sources differ')
    if 'preprocessing_ids' in development and development['preprocessing_ids'] != identity['preprocessing_ids']:
        raise RuntimeError('Historical and current development preprocessing differ')
    from csv import DictReader
    with (RESULT / 'plugin_per_frame.csv').open(newline='') as handle:
        rows = list(DictReader(handle))
    observed = sorted({(int(r['source_index']), r['source_id'], r['preprocessing_id']) for r in rows})
    expected = [(i, sid, pre) for i, (sid, pre) in enumerate(zip(identity['source_ids'], identity['preprocessing_ids']))]
    if observed != expected:
        raise RuntimeError('Historical source/preprocessing order differs from measured P512')
    for name in ('compatible_budget_references.csv', 'compatible_budget_reference_identity.json'):
        shutil.copyfile(HERE / name, RESULT / name)
    (OUT / 'reference_import.json').write_text(json.dumps(dict(status='COMPATIBLE_HISTORICAL_REFERENCES_IMPORTED',
        rows=15, gpu_work=False, reference_csv_sha256=sha(source),
        identity_sha256=sha(HERE / 'compatible_budget_reference_identity.json')), indent=2) + '\n')


if __name__ == '__main__':
    main()
