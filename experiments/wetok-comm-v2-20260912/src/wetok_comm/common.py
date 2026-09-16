"""Local assets, immutable inputs and output accounting."""

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[2]
PROJECT = EXPERIMENT.parents[1]
WORKSPACE = PROJECT.parent
REFERENCE = PROJECT / 'experiments/backbone-eval-20260912'


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(path=None):
    config = yaml.safe_load(Path(path or EXPERIMENT / 'configs/study.yaml').read_text())
    if sha256(WORKSPACE / config['brief']) != config['brief_sha256']:
        raise RuntimeError('research brief changed; review and register a new configuration')
    if config['channel']['data_uses'] * 2 != config['model']['channel_positions'] * config['model']['channel_features']:
        raise ValueError('learned real-coordinate budget is not exact')
    if config['source']['raw_bits'] != 8192 or config['channel']['complex_uses'] != 3060:
        raise ValueError('this registered A0/A1 configuration has a fixed source and physical anchor')
    return config


def output_path(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('outputs must remain physically inside VAR_COMM/outputs')
    return path


def assets():
    visual = json.loads((REFERENCE / 'configs/eval.local.json').read_text())
    binding = yaml.safe_load((WORKSPACE / 'var-next-scale-comm/outputs/remote_m8_handoff_2026-09-11/assets.source.local.yaml').read_text())
    return {**binding, **{key: value for key, value in visual.items() if key.startswith('wetok_')},
            'native_reference': str(PROJECT / 'outputs/ei-liulu-xqvar-eval-20260912-v1/full/wetok')}


def configure_torch():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)


def snapshot(directory, paths):
    records = {}
    for path in paths:
        path = Path(path).resolve()
        relative = path.relative_to(WORKSPACE)
        destination = Path(directory) / 'snapshots' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        records[str(relative)] = sha256(path)
    return records


def verify_sources(records):
    for relative, expected in records.items():
        if sha256(WORKSPACE / relative) != expected:
            raise RuntimeError(f'frozen source changed: {relative}')


def artifact_hashes(path):
    root = Path(path)
    return {str(item.relative_to(root)): sha256(item) for item in sorted(root.rglob('*')) if item.is_file()}
