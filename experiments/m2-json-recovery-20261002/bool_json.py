"""Serialize exact NumPy booleans without changing scientific objects or math."""
from __future__ import annotations
import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/home/liulu/projects/VAR_COMM')
OUT = ROOT / 'outputs/M2-JSON-RECOVERY-20261002'
ENTRY = ROOT / 'outputs/METRIC-CONCURRENT-R4-20261002/runtime/scheduled_process.py'
ADMISSION = ROOT / 'outputs/METRIC-CONCURRENT-R4-20261002/admission.json'
ENV = 'M2_JSON_RECOVERY_MANIFEST'
STAGES = ('calibration', 'evaluation', 'actual', 'timing')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def source_bindings(folder):
    files = sorted(p for p in Path(folder).resolve().iterdir() if p.suffix in ('.py', '.md'))
    if not files or any(p.is_symlink() or not p.is_file() for p in files): raise RuntimeError('Invalid recovery source inventory')
    return {str(p): sha(p) for p in files}


def verify(mapping):
    if not isinstance(mapping, dict) or not mapping: raise RuntimeError('Missing recovery bindings')
    for path, digest in mapping.items():
        if not re.fullmatch('[0-9a-f]{64}', str(digest)) or sha(path) != digest:
            raise RuntimeError('Recovery bound file changed: ' + path)


def eligible(argv, entry=ENTRY):
    return bool(argv) and Path(argv[0]).resolve() == Path(entry).resolve()


def validate_manifest(path, argv, here, *, root=ROOT, out=OUT, entry=ENTRY, admission=ADMISSION):
    root, out, here = Path(root).resolve(), Path(out).resolve(), Path(here).resolve()
    path = Path(path).resolve(); entry, admission = Path(entry).resolve(), Path(admission).resolve()
    if path != out / 'manifest.json' or here != out / 'runtime': raise RuntimeError('Unexpected recovery manifest/runtime location')
    if (len(argv) != 7 or not eligible(argv, entry) or argv[1:3] != ['--root', str(root)]
            or argv[3:5] != ['--admission', str(admission)] or argv[5] != '--stage' or argv[6] not in STAGES):
        raise RuntimeError('Recovery applies only to the registered R4 scheduled entry and arguments')
    manifest = read(path); own = source_bindings(here)
    if (manifest.get('status') != 'REGISTERED_M2_BOOLEAN_JSON_RECOVERY' or manifest.get('entrypoint') != str(entry)
            or manifest.get('runtime_source_bindings') != own or manifest.get('training_updates') != 0
            or manifest.get('policy_selection_updates') != 0 or manifest.get('inference_functions_changed') is not False):
        raise RuntimeError('Recovery manifest scope/source differs')
    original = manifest.get('original_source_bindings'); verify(original); verify(own)
    required = {str(root / 'experiments/scale-causal-partial-residual-20261002' / n) for n in ('common.py', 'm2_runner.py')}
    required.add(str(root / 'experiments/metric-speed-20261002/wrapper.py'))
    registered = source_bindings(entry.parent)
    if not required.issubset(original) or any(original.get(p) != digest for p, digest in registered.items()):
        raise RuntimeError('Original M2/common/wrapper/full R4 runtime are not pinned')
    return manifest


def process_identity(pid=None, proc=Path('/proc')):
    pid = os.getpid() if pid is None else pid
    raw = (Path(proc) / str(pid) / 'stat').read_text(); fields = raw[raw.rfind(') ') + 2:].split()
    if len(fields) < 20 or not fields[19].isdigit(): raise RuntimeError('Cannot bind recovery process start ticks')
    return dict(pid=int(pid), start_ticks=fields[19])


def install_default(numpy_module, record=lambda status, counts: None):
    """The original encoder handles every type except exact numpy.bool_."""
    original = json.JSONEncoder.default
    if getattr(original, '_m2_boolean_json_recovery', False): raise RuntimeError('Boolean JSON recovery already installed')
    counts = dict(numpy_bool_conversions=0, converted_true=0, converted_false=0)
    def default(encoder, value):
        if type(value) is numpy_module.bool_:
            answer = bool(value); counts['numpy_bool_conversions'] += 1
            counts['converted_true' if answer else 'converted_false'] += 1
            if counts['numpy_bool_conversions'] == 1: record('BOOLEAN_CONVERSION_RECORDED', dict(counts))
            return answer
        return original(encoder, value)
    default._m2_boolean_json_recovery = True
    json.JSONEncoder.default = default
    return counts, original


def install(path, argv, here, *, root=ROOT, out=OUT, entry=ENTRY, admission=ADMISSION):
    manifest = validate_manifest(path, argv, here, root=root, out=out, entry=entry, admission=admission)
    # No Torch import or GPU initialization occurs in this compatibility adapter.
    import numpy as np
    who = process_identity(); destination = Path(out) / 'executions' / f'{who["pid"]}_{who["start_ticks"]}.json'
    bound = {str(Path(path).resolve()): sha(path), **manifest['runtime_source_bindings'], **manifest['original_source_bindings']}
    base = dict(schema_version=1, **who, stage=argv[6], entrypoint=str(Path(entry).resolve()), inputs=bound,
        manifest_sha256=sha(path), runtime_source_bindings=manifest['runtime_source_bindings'],
        original_source_bindings=manifest['original_source_bindings'],
        adapter='exact_numpy_bool_JSONEncoder_default_only', training_updates=0, policy_selection_updates=0,
        inference_functions_changed=False, scientific_values_or_policy_choices_changed=False,
        scientific_completion_claimed=False, started_time=time.time())
    def record(status, counts):
        verify(bound); value = dict(base, status=status, **counts, time=time.time())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, destination)
    counts, _ = install_default(np, record)
    record('REGISTERED', dict(counts))
    atexit.register(lambda: record('PROCESS_EXIT_RECORDED', dict(counts)))
    return destination
