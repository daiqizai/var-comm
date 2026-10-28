"""Restore the exact original catalogue bytes from the compact publication copy.

This only restores a configuration file; it does not run a scientific stage.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
R = Path(__file__).resolve().parents[2]
O = R / 'results/nestcom_n512_20261028_v1'
identity = json.loads((O / 'public_catalogue_archive.json').read_text())
compact = (O / 'public_catalogue.compact.json').read_bytes()
assert hashlib.sha256(compact).hexdigest() == identity['compact_sha256']
original = (json.dumps(json.loads(compact), indent=2, allow_nan=False) + '\n').encode('utf-8')
assert hashlib.sha256(original).hexdigest() == identity['original_sha256']
assert len(original) == identity['original_bytes']
target = O / 'public_catalogue.json'
if target.exists():
    assert target.read_bytes() == original
elif not args.check:
    with target.open('xb') as f:
        f.write(original)
print('PASS: compact export reproduces the exact original catalogue bytes.')
