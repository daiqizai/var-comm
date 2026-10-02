"""Restore the historical digital latent-error alias before unchanged parity.

Both historical analyses assign latent_sq_error = latent_sq_err_final. The
frozen digital replay computes the latter from the actual receiver latent but
omits the former. This adapter adds only that alias; it never reads an expected
measurement, changes a reconstructed pixel, or replaces the parity checker.
"""
from __future__ import annotations
import ast
import functools
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_computed_alias(computed):
    if 'latent_sq_err_final' not in computed:
        raise RuntimeError('Digital replay did not compute its original-coordinate latent error')
    final = computed['latent_sq_err_final']
    valid = computed.get('latent_valid')
    if type(valid) is not bool:
        raise RuntimeError('Digital replay latent validity must be explicit')
    if valid:
        if isinstance(final, bool) or not isinstance(final, (int, float)) or not math.isfinite(final) or final < 0:
            raise RuntimeError('Digital replay valid latent error must be finite and nonnegative')
    elif final != '':
        raise RuntimeError('Header-erased latent error must remain blank')
    if 'latent_sq_error' in computed and computed['latent_sq_error'] != final:
        raise RuntimeError('Digital replay existing latent alias disagrees')
    return dict(computed, latent_sq_error=final)


def adapt_digital_source(original):
    @functools.wraps(original)
    def replay_digital(self, index, rows):
        if self.N not in (512, 1024):
            raise RuntimeError('Latent alias compatibility is scoped to historical N512/N1024')
        for row, rgb, computed in original(self, index, rows):
            yield row, rgb, add_computed_alias(computed)
    replay_digital._historical_latent_alias = True
    return replay_digital


def has_historical_alias(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_rows')
    def field(node, name):
        return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'row'
                and isinstance(node.slice, ast.Constant) and node.slice.value == name)
    return any(isinstance(n, ast.Assign) and len(n.targets) == 1 and field(n.targets[0], 'latent_sq_error')
               and field(n.value, 'latent_sq_err_final') for n in ast.walk(function))


def install(replay, root):
    root = Path(root).resolve()
    source = root / 'experiments/unified-metrics-20261002/replay.py'
    if Path(replay.__file__).resolve() != source:
        raise RuntimeError('Compatibility adapter requires the frozen original replay module')
    registration = root / 'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json'
    registered = json.loads(registration.read_text(encoding='utf-8'))['source_bindings']
    if registered.get(str(source)) != sha(source):
        raise RuntimeError('Frozen original replay source changed')
    bindings = {str(source): sha(source), str(registration): sha(registration)}
    for folder in ('extreme-bandwidth-20260930', 'extreme-bandwidth-20261001-N1024'):
        analysis = root / 'experiments' / folder / 'analysis.py'
        if not has_historical_alias(analysis):
            raise RuntimeError('Historical analysis does not define the exact latent alias')
        for name in ('analysis.py', 'digital.py'):
            path = analysis.parent / name
            bindings[str(path)] = sha(path)
    original = replay._OldAdapter._digital_source
    if getattr(original, '_historical_latent_alias', False):
        raise RuntimeError('Historical digital compatibility was already installed')
    if Path(original.__code__.co_filename).resolve() != source:
        raise RuntimeError('Unexpected original digital adapter implementation')
    parity = replay.parity_check
    replay._OldAdapter._digital_source = adapt_digital_source(original)
    if replay.parity_check is not parity:
        raise RuntimeError('Original parity checker must remain unchanged')
    receipt = dict(schema_version=1, name='HISTORICAL_DIGITAL_LATENT_ERROR_ALIAS',
        scope=['N512', 'N1024'], added_field='latent_sq_error',
        computed_from='actual replay latent_sq_err_final; double-precision sum of squared latent differences',
        header_erasure='preserve blank latent error', expected_measurement_copied=False,
        original_parity_checker_unchanged=True, original_tolerances_unchanged=True,
        reconstruction_changed=False, training_updates=0, policy_selection_updates=0,
        source_bindings=bindings)
    replay._historical_latent_alias_receipt = receipt
    return receipt
