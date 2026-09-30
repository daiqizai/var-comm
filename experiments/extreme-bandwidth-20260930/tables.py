"""Reuse the byte-preserving Step2 table archiver for the N512 result tables."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/extreme_bandwidth_20260930_R1'
RUN = 'EXTREME-BW-20260930-R1'
SOURCE = ROOT / 'experiments/rx-posterior-step2-B-20260930/tables.py'


def engine(results):
    spec = importlib.util.spec_from_file_location('extreme_bw_original_archiver', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = json.loads((results / 'table_registry.json').read_text())
    if registry['run'] != RUN:
        raise RuntimeError('Wrong table registry run')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != registry['archiver_sha256']:
        raise RuntimeError('Registered archiver changed')
    module.TABLES = registry['tables']
    return module


def export(results=RESULT):
    module = engine(results)
    manifest = results / 'table_shards/manifest.json'
    if not manifest.exists():
        module.export(results)
        record = json.loads(manifest.read_text())
        record['run'] = RUN
        manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    verify(results)


def verify(results=RESULT, restore=False):
    record = json.loads((results / 'table_shards/manifest.json').read_text())
    if record['run'] != RUN:
        raise RuntimeError('Wrong archived result run')
    engine(results).verify_or_restore(results, restore)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['export', 'verify', 'restore'])
    parser.add_argument('--results', type=Path, default=RESULT)
    args = parser.parse_args()
    export(args.results) if args.action == 'export' else verify(args.results, args.action == 'restore')
