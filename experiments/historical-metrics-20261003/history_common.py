"""Shared, CPU-only receipts for additive evaluation of completed old studies."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path

NEW_METRICS = ('dinov2_vitl14_cosine', 'clip_image_cosine', 'dists', 'dreamsim',
               'ms_ssim', 'resnet50_top1_label', 'resnet50_top1_source_prediction')


@contextmanager
def native_interop_setup(torch, registered_threads):
    """Initialize the shared pool once; old repeated identical requests are safe."""
    setter = torch.set_num_interop_threads
    if torch.get_num_interop_threads() != registered_threads:
        setter(registered_threads)
    def same_value_only(value):
        if type(value) is not int or value != registered_threads:
            raise RuntimeError('Native adapter requires a different interop pool; separate process protocol required')
        if torch.get_num_interop_threads() != registered_threads:
            raise RuntimeError('Registered interop pool changed')
    torch.set_num_interop_threads = same_value_only
    try:
        yield
        same_value_only(registered_threads)
    finally:
        torch.set_num_interop_threads = setter


def set_numerical_runtime(torch, flags, *, warn_only=False):
    if torch.get_num_interop_threads() != flags['interop_threads']:
        raise RuntimeError('Cannot switch an initialized interop pool during metric evaluation')
    torch.set_float32_matmul_precision(flags['precision'])
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic'], warn_only=warn_only)
    if torch.get_num_threads() != flags['threads']:
        torch.set_num_threads(flags['threads'])


@contextmanager
def metric_numerical_context(torch, registered_flags, read_flags):
    """Every new metric uses the six-study flags and restores native inference."""
    before = read_flags(torch)
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        set_numerical_runtime(torch, registered_flags)
        if read_flags(torch) != registered_flags:
            raise RuntimeError('Frozen metric numerical settings could not be established')
        yield
        if read_flags(torch) != registered_flags:
            raise RuntimeError('Metric computation changed its registered numerical settings')
    finally:
        set_numerical_runtime(torch, before, warn_only=warn_only)
        if read_flags(torch) != before:
            raise RuntimeError('Metric computation did not restore native numerical settings')


def canonical(value):
    """JSON container normalization preserves registered numeric values."""
    return json.loads(json.dumps(value, allow_nan=False))


def original_fields(row):
    return {k: v for k, v in row.items() if not k.startswith(('history_', 'new_'))}


def normalize_metadata(row, metadata):
    """Explicit compatibility for the frozen receiver's registered output roles."""
    if not isinstance(metadata, dict):
        raise RuntimeError('Historical scientific metadata missing')
    value = canonical(metadata)
    if 'reference_only' not in value:
        role = value.get('output_role')
        if type(value.get('oracle')) is not bool or role not in ('main', 'diagnostic', 'oracle_reference', 'source_reference', 'reference'):
            raise RuntimeError('No registered role from which to derive reference annotation')
        value['reference_only'] = value['oracle'] or role in ('oracle_reference', 'source_reference', 'reference') or value.get('scope') == 'source_only_reference'
        value['metadata_derivations'] = {'reference_only': 'registered oracle/output_role/scope'}
    for key in ('label_conditioned', 'reference_only', 'oracle'):
        if type(value.get(key)) is not bool:
            raise RuntimeError('Explicit Boolean historical annotation required: '+key)
    value['method_id'] = value.get('method_id') or value.get('method') or row.get('method') or row.get('control')
    if not value['method_id']:
        raise RuntimeError('Registered historical method identity is missing')
    if value.get('model_identity'):
        value['model_identity_sha256'] = identity(value['model_identity'])
        if not (value.get('model_id') or value.get('checkpoint_sha256')):
            value['model_id'] = value['model_identity_sha256']
    if value.get('channel_uses_not_applicable') is True:
        # Old source references used textual sentinels; new references use
        # blanks. Preserve their raw rows and normalize only analysis metadata.
        value.update(N='', E='', snr_db='')
    elif value.get('N') in (None, ''):
        for column in ('N', 'N_paid', 'complex_uses'):
            if row.get(column) not in (None, ''):
                value['N'] = row[column]
                break
    if not value.get('checkpoint_sha256') and row.get('checkpoint_sha256'):
        value['checkpoint_sha256'] = row['checkpoint_sha256']
    if not value.get('snr_definition') and value.get('snr_field', 'snr_db') == 'snr_db':
        value['snr_definition'] = 'physical_channel_snr_db'
    value.setdefault('original_metric_models', {'dino_cosine': 'DINOv2 ViT-S/14', 'lpips_alex': 'LPIPS AlexNet v0.1'})
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError('Frozen historical input changed: ' + str(path))


def row_ids(study, source_index, rows):
    """Retain exact duplicates with an occurrence index; never discard rows."""
    seen = Counter()
    result = []
    for row in rows:
        raw = identity(row)
        result.append(identity([study, source_index, raw, seen[raw]]))
        seen[raw] += 1
    return result


def checkpoint_valid(payload, binding, expected_ids, *, expected_rows, study,
                     source_index, evaluation_identity, qualified_batch_sizes):
    if payload.get('binding') != binding:
        raise RuntimeError('Historical checkpoint registration differs')
    if payload.get('payload_sha256') != identity({k: v for k, v in payload.items() if k != 'payload_sha256'}):
        raise RuntimeError('Historical checkpoint checksum differs')
    rows = payload.get('rows', [])
    ids = [r['history_row_id'] for r in rows]
    if len(ids) != len(set(ids)) or ids != list(expected_ids):
        raise RuntimeError('Historical checkpoint row coverage differs')
    if row_ids(study, source_index, expected_rows) != list(expected_ids):
        raise RuntimeError('Expected row identity is inconsistent')
    if payload.get('source_index') != source_index:
        raise RuntimeError('Historical checkpoint source index differs')
    if payload.get('evaluation_identity') != evaluation_identity:
        raise RuntimeError('Historical checkpoint evaluator/batch identity differs')
    used = payload.get('metric_batch_sizes_used', [])
    unique = payload.get('unique_images')
    if (not used or any(type(n) is not int or n not in qualified_batch_sizes for n in used)
            or type(unique) is not int or unique < 1 or sum(used) != unique or unique > len(rows)):
        raise RuntimeError('Historical checkpoint used an unqualified or incomplete metric batch')
    parity = payload.get('parity', [])
    if len(parity) != len(rows) or not all(p.get('replay_parity_passed') is True and p.get('synthetic') is False for p in parity):
        raise RuntimeError('Historical checkpoint requires actual replay parity')
    if not all(r.get('history_replay_parity_passed') is True for r in rows):
        raise RuntimeError('Historical row lacks actual parity')
    for row, proof, original in zip(rows, parity, expected_rows):
        if proof.get('history_row_id') != row['history_row_id']:
            raise RuntimeError('Historical row and parity identities differ')
        if row.get('history_original_row_sha256') != identity(original_fields(row)):
            raise RuntimeError('Original historical row fields changed')
        if original_fields(row) != original or identity(original_fields(row)) != identity(original):
            raise RuntimeError('Historical row differs from the exact original registration')
        if (row.get('history_study') != study or row.get('history_source_index') != source_index
                or row.get('history_modelmanifest_sha256') != evaluation_identity['modelmanifest_sha256']):
            raise RuntimeError('Historical row source/model identity differs')
        for metric in NEW_METRICS:
            value = row.get('new_'+metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError('Missing/nonfinite additional metric: '+metric)
    return payload


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('Refuse empty historical metric table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def source_bindings(directory):
    return {str(p.resolve()): sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file() and p.suffix in ('.py', '.md')}


def proc(pid):
    p = Path('/proc') / str(pid)
    try:
        raw = (p / 'stat').read_text()
        rest = raw[raw.rfind(')') + 2:].split()
        return dict(pid=int(pid), start_ticks=rest[19], state=rest[0], command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(who):
    actual = proc(who['pid'])
    return actual is not None and actual['state'] != 'Z' and str(actual['start_ticks']) == str(who['start_ticks'])


def verify_parent_complete(root):
    """Use the single strict actual-publication/process-exit implementation."""
    from history_controller import parent_gate, NotReady
    try:
        return parent_gate(root)['inputs']
    except NotReady:
        return None
