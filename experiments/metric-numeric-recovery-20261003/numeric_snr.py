"""Read the frozen M2 actual-link SNR text without changing its representation.

The actual CSV and policy use ``1.0`` while the original replay also calls
``int(row['snr_db'])``. This narrowly scoped str subclass preserves that exact
CSV text, equality, JSON, and row hashes; only its explicit integer conversion
is supplied. Every original grid, source, policy, and parity check still runs.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path


FROZEN_REPLAY_SHA256 = '9582ce2f89c430369b7b19fd7810d1f90e56240f6d7b9845506fc57ded00bcbd'
SNR_VALUES = (1, 4, 7, 13, 19)
_INTEGER_BY_TEXT = {text: value for value in SNR_VALUES
                    for text in (str(value), str(value) + '.0')}


class IntSNR(str):
    """An unchanged registered SNR string with a strict __int__ conversion."""
    __slots__ = ()

    def __new__(cls, value):
        if type(value) not in (str, IntSNR):
            raise ValueError('M2 actual SNR must be original CSV string text')
        raw = str(value)
        if raw not in _INTEGER_BY_TEXT:
            raise ValueError('Unregistered M2 actual SNR literal: ' + repr(raw))
        return str.__new__(cls, raw)

    def __int__(self):
        # No float parsing, rounding, truncation, alternate spelling, or clean.
        return _INTEGER_BY_TEXT[str(self)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt_prepare_rows(original):
    @functools.wraps(original)
    def prepare_rows(self, study, rows):
        prepared = original(self, study, rows)
        if study != 'M2_ACTUAL':
            return prepared
        # The caller's input dictionaries are unchanged, including on failure.
        return [dict(row, snr_db=IntSNR(row['snr_db'])) for row in prepared]
    prepare_rows._m2_actual_integer_snr = True
    return prepare_rows


def install(replay, root=None):
    """Install once, returning stable source-bound operational provenance.

    The startup owner adds this receipt and its own qualification/manifest to
    final registration bindings. This function neither writes nor binds files
    on an engine, changes scientific globals, or touches the R4 cache adapter.
    ``root`` additionally requires the production source path and registration.
    """
    source = Path(replay.__file__).resolve()
    if sha(source) != FROZEN_REPLAY_SHA256:
        raise RuntimeError('Numeric SNR compatibility requires the pinned frozen replay')
    bindings = {str(source): sha(source), str(Path(__file__).resolve()): sha(__file__)}
    if root is not None:
        root = Path(root).resolve()
        if source != root / 'experiments/unified-metrics-20261002/replay.py':
            raise RuntimeError('Unexpected frozen replay source path')
        registration = root / 'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json'
        registered = json.loads(registration.read_text(encoding='utf-8'))['source_bindings']
        if registered.get(str(source)) != FROZEN_REPLAY_SHA256:
            raise RuntimeError('Frozen replay registration differs')
        bindings[str(registration)] = sha(registration)
    original = replay.ReplayEngine._prepare_rows
    if (getattr(original, '_m2_actual_integer_snr', False)
            or hasattr(replay, '_numeric_snr_receipt')):
        raise RuntimeError('Numeric SNR compatibility is already installed')
    if (Path(original.__code__.co_filename).resolve() != source
            or original.__qualname__ != 'ReplayEngine._prepare_rows'
            or original.__globals__ is not replay.__dict__):
        raise RuntimeError('Unexpected original row preparation implementation')
    receipt = dict(schema_version=1, name='M2_ACTUAL_ORIGINAL_TEXT_INTEGER_SNR',
        scope=['M2_ACTUAL'], field='snr_db', allowed_literals=sorted(_INTEGER_BY_TEXT),
        integer_values=list(SNR_VALUES), original_string_values_retained=True,
        original_csv_unchanged=True, original_row_hashes_unchanged=True,
        original_row_ids_unchanged=True, original_policy_unchanged=True,
        original_grid_checks_unchanged=True, original_parity_checker_unchanged=True,
        original_tolerances_unchanged=True, reconstruction_changed=False,
        training_updates=0, policy_selection_updates=0, source_bindings=bindings)
    replay.ReplayEngine._prepare_rows = adapt_prepare_rows(original)
    replay._numeric_snr_receipt = receipt
    return receipt
