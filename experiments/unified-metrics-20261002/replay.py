"""Read-only replay of frozen communication outputs for additive image metrics.

No launcher, training, calibration, policy selection, old cache writer, PNG
reader or image quantization is used here. The native adapters call the original
receivers and renderers, preserving their original batching. Importing this file
needs only NumPy; Torch and the original scientific dependencies load at setup.

API::

    engine = ReplayEngine(repository_root, loaded, development_data)
    engine.setup(['N512', 'N1024', 'M1', 'M2_ORACLE', 'M2_ACTUAL'])
    for original_row_plus_metadata, rgb, parity in engine.iterate_source('M1', 0):
        # rgb is the original float32 CHW decoder output, before PNG conversion.
        ...

All original rows, including duplicated output fingerprints and D0 references,
are retained. M2 clean oracle rows have snr_db='clean', noise_seed=0 only.
The caller owns scheduling, new metric models, checkpoints, and sidecar writes.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

STUDIES = ('N512', 'N1024', 'M1', 'M1_RATE', 'M2_ORACLE', 'M2_ACTUAL')
SNRS = (1, 4, 7, 13, 19)
DEV_SEEDS = (2001, 2002, 2003)
M2_CONTROLS = ('UNGUIDED', 'LIKELIHOOD', 'STATIC', 'VAR_GUIDED', 'DIRECT')
PROJECTIONS = ('g8_c32', 'g8_c8', 'g6_c8', 'g4_c32', 'g4_c16', 'g4_c8')
PARITY_TOLERANCES = {
    'psnr_db': (1e-5, 0.), 'lpips_alex': (2e-6, 0.),
    'dino_cosine': (2e-6, 0.), 'dino_mismatched': (2e-6, 0.),
    'dino_specificity': (4e-6, 0.),
    'latent_sq_err_base': (1e-5, 1e-7),
    'latent_sq_err_post': (1e-5, 1e-7),
    'latent_sq_err_final': (1e-5, 1e-7), 'latent_sq_error': (1e-5, 1e-7),
    'zero_erasure_proxy_sq_error': (1e-5, 1e-7),
}
HASH_FIELDS = ('waveform_sha256', 'observation_sha256',
               'measurement_wave_sha256', 'measurement_y_sha256',
               'tx_position_sha256', 'rx_position_sha256')
EXACT_FIELDS = ('latent_valid', 'decoder_applied', 'header_ok', 'header_crc_ok',
    'header_fields_legal', 'body_crc_ok', 'prefix_crc_ok', 'partial_crc_ok',
    'gain_crc_ok', 'gain_ok', 'gain_fields_valid', 'gain_numeric_valid',
    'analog_used', 'decoded_label', 'decoded_mode',
    'decoded_m', 'decoded_q', 'decoded_order', 'decoded_header_code',
    'trusted_prefix_scales', 'hard_candidate_prefix_scales', 'partial_used',
    'raw_candidate_used_after_crc_failure', 'true_class_sent', 'padding_ignored',
    'known_tokens_fixed', 'mismatch_source_id', 'action_m', 'm', 'q', 'order',
    'action_id', 'projection', 'control', 'lambda', 'policy_action')


class ReplayMismatch(RuntimeError):
    """A frozen identity, grid, observation or reconstructed metric differs."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def arraysha(value):
    """Original waveform convention: SHA256 of native contiguous bytes."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rgb_fingerprint(value):
    a = rgb_array(value)
    return hashlib.sha256(b'float32:3,256,256:RGB\0' + a.tobytes()).hexdigest()


def metric_cache_key(rgb, target, metric_identity):
    """Only exact float RGB/target/model identity can share new metric scores."""
    h = hashlib.sha256()
    h.update(rgb_fingerprint(rgb).encode())
    h.update(rgb_fingerprint(target).encode())
    h.update(json.dumps(metric_identity, sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


def rgb_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.shape != (3, 256, 256) or a.dtype != np.float32:
        raise ReplayMismatch('Replay requires original float32 RGB [3,256,256]')
    if not np.isfinite(a).all() or a.min() < -1e-6 or a.max() > 1.000001:
        raise ReplayMismatch('Replayed RGB is nonfinite or outside frozen decoder range')
    # Copying does not change precision; ownership must not leak mutable caches.
    return np.array(a, dtype=np.float32, order='C', copy=True)


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ReplayMismatch('Expected an explicit boolean, received ' + repr(value))


def blank(value):
    return value is None or value == ''


def norm(value):
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def row_key(study, row):
    """A row key includes method, resource facet and clean/noisy scope."""
    return tuple(norm(row.get(k, '')) for k in
        ('source_id', 'source_index', 'preprocessing_id', 'N', 'phy_family',
         'snr_db', 'noise_seed', 'method', 'stage', 'projection', 'control'))


def row_id(study, row):
    payload = [study, *row_key(study, row)]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def source_row_hash(row):
    """Bind the exact original CSV string fields before adding absent metadata."""
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rate_scopes():
    return {(m, q, order) for m, L in ((5, 36), (6, 64))
            for q in sorted(set(range(0, L, 4)) | {L})
            for order in ('raster', 'random', 'entropy', 'oracle')}


def old_methods(N):
    plugin = {f'P{N}', 'V_lambda1'}
    plugin.update(f'{p}_{suffix}' for p in ('A1', 'A2', 'V')
                  for suffix in ('policy', 'raw_fused', 'common', 'tok'))
    digital = {f'D_{c}_{p}{suffix}' for p in ('QPSK', '16QAM')
               for c in ('C', 'U') for suffix in ('', '_D0')}
    digital.update(f'D_prefix_{c}_{p}' for p in ('QPSK', '16QAM') for c in ('C', 'U'))
    digital.update(f'D_{c}_{p}_at_{selector}_action' for p in ('QPSK', '16QAM')
                   for c, selector in (('C', 'U'), ('U', 'C')))
    return plugin | digital


def metadata(row, study):
    method = row.get('method', row.get('control', ''))
    decoder = row.get('decoder_id') or 'Dc'
    historical = study == 'M1' and boolean(row.get('historical_reference', False))
    digital = study in ('N512', 'N1024') and str(method).startswith('D_')
    if historical:
        digital = method == 'legacy_policy'
    condition = row.get('class_condition', '')
    paid_class = digital
    class_regime = digital and (condition == 'C' or str(method).startswith('D_C_'))
    used_class = class_regime and (blank(row.get('header_ok')) or boolean(row['header_ok']))
    true_class_tx = paid_class  # old protocol pays ten TX class bits, even for U.
    clean = study == 'M2_ORACLE' and str(row.get('snr_db')) == 'clean'
    if study.startswith('M2_'):
        paid_class = used_class = true_class_tx = False
    return dict(replay_study=study, replay_row_id=row_id(study, row),
        replay_decoder_id=decoder, replay_reference_only=decoder == 'D0',
        replay_historical_reference=historical,
        replay_true_class_sent=true_class_tx,
        replay_paid_class_header=paid_class,
        replay_received_class_used_by_generator=used_class,
        replay_semantic_side_information=class_regime,
        replay_class_condition=('received_paid_class' if class_regime else
            'prefix_without_class_generation' if str(method).startswith('D_prefix_') else
            'unconditional1000'),
        replay_oracle_side_information=study == 'M2_ORACLE',
        replay_clean_oracle=clean,
        replay_source_only=study == 'M1_RATE',
        replay_clean_source_diagnostic=study == 'M1_RATE',
        replay_source_row_sha256=row.get('replay_source_row_sha256', ''),
        replay_paid_oracle_mask=(study == 'M1' and not historical and
                                 str(row.get('order')) == 'oracle'),
        replay_selection_unchanged=True, replay_training_updates=0)


def parity_check(original, computed, *, required_quality=True):
    """Fail closed on old quality/F values and every available original hash.

    The tolerances account for scalar rendering roundoff, never token or bit
    changes. No tolerance is permitted on waveform/observation SHA256.
    Header erasures must retain their blank latent errors, not a zero latent.
    """
    checked, missing, deltas = [], [], {}
    names = list(PARITY_TOLERANCES)
    for name in names:
        if name not in original:
            continue
        expected = original[name]
        if blank(expected):
            if name in computed and not blank(computed[name]):
                raise ReplayMismatch('Blank original field became numeric: ' + name)
            continue
        if name not in computed:
            missing.append(name)
            continue
        got = computed[name]
        if blank(got):
            raise ReplayMismatch('Numeric original field became blank: ' + name)
        expected, got = float(expected), float(got)
        if not math.isfinite(expected) or not math.isfinite(got):
            raise ReplayMismatch('Nonfinite parity field: ' + name)
        atol, rtol = PARITY_TOLERANCES[name]
        delta = got - expected
        if abs(delta) > atol + rtol * abs(expected):
            raise ReplayMismatch(f'{name} differs: expected={expected!r}, replay={got!r}, delta={delta!r}')
        deltas[name] = delta; checked.append(name)
    for name in HASH_FIELDS + EXACT_FIELDS:
        if name not in original or blank(original[name]):
            continue
        if name not in computed:
            missing.append(name); continue
        numeric = name in ('decoded_label', 'decoded_mode', 'decoded_m', 'decoded_q',
            'decoded_header_code', 'trusted_prefix_scales', 'hard_candidate_prefix_scales',
            'action_m', 'm', 'q', 'lambda')
        if name in HASH_FIELDS:
            equal = str(original[name]) == str(computed[name])
        elif numeric:
            equal = not blank(computed[name]) and float(original[name]) == float(computed[name])
        else:
            equal = norm(original[name]) == norm(computed[name])
        if not equal:
            raise ReplayMismatch(f'{name} differs: expected={original[name]!r}, replay={computed[name]!r}')
        checked.append(name)
    if required_quality:
        # Every original nonblank quality/F/hash/event field is mandatory.
        if missing:
            raise ReplayMismatch('Replay omitted frozen parity fields: ' + ', '.join(missing))
        for name in ('psnr_db', 'lpips_alex', 'dino_cosine'):
            if name not in checked:
                raise ReplayMismatch('Original quality comparison unavailable: ' + name)
    return dict(status='PASS', checked_fields=checked, metric_deltas=deltas,
                missing_fields=missing, tolerances=PARITY_TOLERANCES)


@contextmanager
def module_aliases(aliases):
    """Scope plain-name imports made by immutable legacy source files."""
    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_file_module(name, path, aliases=None):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(getattr(module, '__file__', '')).resolve() != path:
            raise ReplayMismatch('Module identity collision: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayMismatch('Cannot load frozen source: ' + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with module_aliases(aliases or {}):
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class Layout:
    source: Path
    result: Path
    output: Path
    rows: Path


def default_layouts(root):
    root = Path(root)
    new_result = root / 'results/scale_causal_partial_residual_20261002'
    new_output = root / 'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
    new_source = root / 'experiments/scale-causal-partial-residual-20261002'
    result = {}
    for N, date in ((512, '20260930'), (1024, '20261001')):
        suffix = '' if N == 512 else '_N1024'
        r = root / f'results/extreme_bandwidth_{date}_R1{suffix}'
        o = root / ('outputs/EXTREME-BW-20260930-R1' if N == 512 else
                    'outputs/EXTREME-BW-20261001-R1-N1024')
        s = root / ('experiments/extreme-bandwidth-20260930' if N == 512 else
                    'experiments/extreme-bandwidth-20261001-N1024')
        result[f'N{N}'] = Layout(s, r, o, r / 'per_frame.csv')
    for study, filename in (('M1', 'm1_per_frame.csv'), ('M1_RATE', 'm1_rate_curve.csv'),
                            ('M2_ORACLE', 'm2_oracle_per_frame.csv'),
                            ('M2_ACTUAL', 'm2_actual_per_frame.csv')):
        result[study] = Layout(new_source, new_result, new_output, new_result / filename)
    return result


class ReplayEngine:
    """One read-only native replay runtime and source-at-a-time API.

    ``loaded`` and ``data`` are the frozen methods common.setup()/data() outputs;
    this class does not change their runtime precision or scheduler. For CPU
    engineering tests only, adapters may be injected with synthetic=True.
    """
    def __init__(self, root, loaded, data, layouts=None, *, legacy_check=True,
                 adapters=None, synthetic=False):
        self.root = Path(root).resolve()
        self.loaded, self.data = loaded, data
        self.layouts = default_layouts(self.root)
        if layouts:
            self.layouts.update({k: v if isinstance(v, Layout) else Layout(**v)
                                 for k, v in layouts.items()})
        self.legacy_check, self.synthetic = bool(legacy_check), bool(synthetic)
        if adapters and not synthetic:
            raise ReplayMismatch('Injected replay adapters are engineering fixtures only')
        self.adapters = dict(adapters or {})
        self.rows, self.by_source, self.artifacts = {}, {}, {}
        self._modules, self._old = {}, {}
        self.records = data['records']
        if not synthetic and (not legacy_check or len(self.records) != 100):
            raise ReplayMismatch('Formal replay requires all100 sources and original quality parity')
        ids = [r['image_id'] for r in self.records]
        if len(set(ids)) != len(ids):
            raise ReplayMismatch('Duplicate input sources')
        self.mismatch = self._derangement()

    @staticmethod
    def _derangement():
        rng = np.random.default_rng(20260930)
        while True:
            p = rng.permutation(100)
            if (p != np.arange(100)).all():
                return p.tolist()

    def bind(self, path):
        path = Path(path).resolve()
        expected = sha256_file(path)
        if str(path) in self.artifacts and self.artifacts[str(path)] != expected:
            raise ReplayMismatch('Input changed while binding: ' + str(path))
        self.artifacts[str(path)] = expected
        return expected

    def read(self, path):
        self.bind(path)
        return read_json(path)

    def verify_bindings(self, bindings):
        for path, expected in bindings.items():
            if self.bind(path) != expected:
                raise ReplayMismatch('Frozen source/artifact differs: ' + path)

    def validate_registration(self, path):
        registration = self.read(path)
        if registration['identity'] != self.loaded['identity']:
            raise ReplayMismatch('Frozen visual model identity changed')
        ids = [r['image_id'] for r in self.records]
        pre = [r['preprocessing_id'] for r in self.records]
        if registration['source_ids'] != ids or registration['preprocessing_ids'] != pre:
            raise ReplayMismatch('Frozen development image/preprocessing order changed')
        if registration['data_bindings'] != self.data['bindings']:
            raise ReplayMismatch('Frozen development source bytes/latents changed')
        self.verify_bindings(registration['source_bindings'])
        self.verify_bindings(registration.get('input_artifacts', {}))
        return registration

    def setup(self, studies=None):
        for study in studies or STUDIES:
            if study not in STUDIES:
                raise ValueError('Unknown replay study: ' + study)
            if study in self.rows:
                continue
            layout = self.layouts[study]
            self.bind(layout.rows)
            with Path(layout.rows).open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            rows = self._prepare_rows(study, rows)
            self._validate_rows(study, rows)
            self.rows[study] = rows
            grouped = defaultdict(list)
            for row in rows:
                grouped[int(row['source_index'])].append(row)
            self.by_source[study] = dict(grouped)
            if study not in self.adapters:
                if study in ('N512', 'N1024'):
                    self.adapters[study] = self._old_adapter(int(study[1:]))
                elif study == 'M1':
                    self.adapters[study] = _M1Adapter(self, layout)
                elif study == 'M1_RATE':
                    self.adapters[study] = _M1RateAdapter(self, layout)
                else:
                    self.adapters[study] = _M2Adapter(self, layout, study)
            self.adapters[study].validate_rows(rows)
        return self

    def _prepare_rows(self, study, rows):
        if study != 'M1_RATE':
            return rows
        result = []
        for original in rows:
            i = int(original['source_index'])
            if not 0 <= i < len(self.records):
                raise ReplayMismatch('Rate-curve source index outside original population')
            m, q, order = int(original['m']), int(original['q']), original['order']
            added = dict(preprocessing_id=self.records[i]['preprocessing_id'],
                snr_db='clean', noise_seed='0', method='rate_curve_' + order,
                projection=f'm{m}_K{q}', N='', phy_family='source_only',
                stage='source_only_rate_curve', decoder_id='Dc',
                class_condition='unconditional1000',
                replay_source_row_sha256=source_row_hash(original))
            if set(added) & set(original):
                raise ReplayMismatch('Original rate-curve schema unexpectedly contains added identity fields')
            result.append(dict(original, **added))
        return result

    def _validate_rows(self, study, rows):
        seen = set()
        for row in rows:
            i = int(row['source_index']); key = row_key(study, row)
            if key in seen:
                raise ReplayMismatch('Duplicate frozen row: ' + repr(key))
            seen.add(key)
            if not 0 <= i < len(self.records):
                raise ReplayMismatch('Invalid frozen source index')
            r = self.records[i]
            if row['source_id'] != r['image_id'] or row['preprocessing_id'] != r['preprocessing_id']:
                raise ReplayMismatch('Frozen source/preprocessing differs from replay input')
            if not blank(row.get('mismatch_source_id')):
                expected = self.records[self.mismatch[i]]['image_id']
                if row['mismatch_source_id'] != expected:
                    raise ReplayMismatch('Original specificity derangement changed')
            if str(row['snr_db']) == 'clean':
                if study not in ('M2_ORACLE', 'M1_RATE') or int(row['noise_seed']) != 0:
                    raise ReplayMismatch('Clean source/oracle diagnostics require clean SNR and seed0')
            elif int(row['snr_db']) not in SNRS or int(row['noise_seed']) not in DEV_SEEDS:
                raise ReplayMismatch('Frozen development SNR/noise grid changed')
            if study == 'M2_ORACLE':
                full = row['projection'] == 'g8_c32'
                clean = row['snr_db'] == 'clean'
                expected_stage = ('full_noiseless' if full else 'reduced_noiseless') if clean else (
                    'full_noisy_infeasible_oracle' if full else 'reduced_noisy')
                if row['stage'] != expected_stage or not boolean(row.get('oracle_side_information', True)):
                    raise ReplayMismatch('M2 oracle scope/stage changed')
            elif study == 'M2_ACTUAL':
                if row['stage'] != 'actual_link' or boolean(row.get('oracle_side_information', False)):
                    raise ReplayMismatch('M2 actual link scope/stage changed')
            elif study in ('N512', 'N1024'):
                if int(row['N']) != int(study[1:]):
                    raise ReplayMismatch('Old row belongs to another resource budget')
                if row.get('decoder_id') != ('D0' if row['method'].endswith('_D0') else 'Dc'):
                    raise ReplayMismatch('Old decoder/reference scope changed')
            elif study == 'M1_RATE':
                if (row['stage'] != 'source_only_rate_curve' or row['N'] != ''
                        or row['phy_family'] != 'source_only' or row['snr_db'] != 'clean'
                        or boolean(row['wireless_claim'])):
                    raise ReplayMismatch('M1 rate curve cannot become a wireless/noisy result')
        if self.synthetic:
            return
        if study in ('N512', 'N1024'):
            methods = old_methods(int(study[1:]))
            expected = {(i, m, str(s), str(seed)) for i in range(100)
                        for m in methods for s in SNRS for seed in DEV_SEEDS}
            actual = {(int(r['source_index']), r['method'], r['snr_db'], r['noise_seed']) for r in rows}
            if len(rows) != 45000 or actual != expected:
                raise ReplayMismatch('Old full30-method45000-row grid incomplete')
        elif study == 'M2_ORACLE':
            expected = {(i, p, c, str(s), str(seed)) for i in range(100)
                        for p in PROJECTIONS for c in M2_CONTROLS
                        for s in ('clean', *SNRS) for seed in ((0,) if s == 'clean' else DEV_SEEDS)}
            actual = {(int(r['source_index']), r['projection'], r['control'], r['snr_db'], r['noise_seed']) for r in rows}
            if len(rows) != 48000 or actual != expected:
                raise ReplayMismatch('Six-projection clean/noisy oracle grid incomplete')
        elif study == 'M1_RATE':
            expected = {(i, m, q, order) for i in range(100) for m, q, order in rate_scopes()}
            actual = {(int(x['source_index']), int(x['m']), int(x['q']), x['order']) for x in rows}
            if len(rows) != 10800 or actual != expected:
                raise ReplayMismatch('M1 source-only dense10800-row grid incomplete')

    def _old_adapter(self, N):
        if N not in self._old:
            self._old[N] = _OldAdapter(self, self.layouts[f'N{N}'], N)
        return self._old[N]

    def methods_modules(self, layout):
        key = str(layout.source)
        if key in self._modules:
            return self._modules[key]
        old = self._old_adapter(1024)
        source = Path(layout.source)
        c = load_file_module('metrics_replay_methods_common', source / 'common.py',
                             {'assets': old.assets, 'digital': old.digital})
        p = load_file_module('metrics_replay_partial_phy', source / 'partial_phy.py')
        pr = load_file_module('metrics_replay_partial_receiver', source / 'partial_receiver.py',
                              {'partial_phy': p})
        m1 = load_file_module('metrics_replay_m1_runner', source / 'm1_runner.py',
                              {'common': c, 'partial_phy': p, 'partial_receiver': pr})
        residual = load_file_module('metrics_replay_residual_receiver', source / 'residual_receiver.py')
        m2 = load_file_module('metrics_replay_m2_runner', source / 'm2_runner.py',
                              {'common': c, 'residual_receiver': residual})
        value = dict(common=c, partial_phy=p, partial_receiver=pr, m1=m1,
                     residual=residual, m2=m2)
        self._modules[key] = value
        return value

    def target(self, index):
        pixels = np.asarray(self.records[index]['pixels'])
        if pixels.shape != (3, 256, 256) or pixels.dtype != np.uint8:
            raise ReplayMismatch('Original source must be uint8 CHW256')
        return pixels.astype(np.float32) / np.float32(255.)

    def metadata(self, row, study):
        value = metadata(row, study)
        i = int(row['source_index'])
        value.update(replay_source_class_index=int(self.records[i]['class_index']),
                     replay_mismatch_source_id=self.records[self.mismatch[i]]['image_id'])
        return value

    def iterate_source(self, study, index):
        if study not in self.rows:
            self.setup([study])
        if not 0 <= index < len(self.records):
            raise ValueError('Invalid development source index')
        rows = self.by_source[study].get(index, [])
        expected = {row_key(study, r): r for r in rows}
        seen = set()
        adapter = self.adapters[study]
        # Avoid inference_mode: original receivers deliberately used no_grad.
        context = _null_context() if self.synthetic else self._torch().no_grad()
        with context:
            for row, image, computed in adapter.iterate_source(index, rows):
                key = row_key(study, row)
                if key not in expected or key in seen or row != expected[key]:
                    raise ReplayMismatch('Replay changed/duplicated an original row')
                seen.add(key)
                rgb = rgb_array(image)
                parity = parity_check(row, computed, required_quality=self.legacy_check)
                parity.update(rgb_sha256=rgb_fingerprint(rgb),
                    target_sha256=rgb_fingerprint(self.target(index)),
                    original_row_sha256=row_id(study, row), synthetic=self.synthetic,
                    replay_selection_unchanged=True, original_png_used=False,
                    replay_parity_passed=True)
                augmented = dict(row, **self.metadata(row, study))
                yield augmented, rgb, parity
        if seen != set(expected):
            raise ReplayMismatch('Replay omitted original frozen rows')

    @staticmethod
    def _torch():
        import torch
        return torch

    def manifest(self):
        return dict(status='FROZEN_REPLAY_INPUTS_BOUND', synthetic=self.synthetic,
            studies=list(self.rows), rows={s: len(r) for s, r in self.rows.items()},
            source_ids=[r['image_id'] for r in self.records],
            preprocessing_ids=[r['preprocessing_id'] for r in self.records],
            model_identity=self.loaded.get('identity'), input_sha256=dict(self.artifacts),
            legacy_parity_required=self.legacy_check, parity_tolerances=PARITY_TOLERANCES,
            original_policy_unchanged=True, training_updates=0,
            no_original_cache_writes=True, no_png_reconstruction=True,
            clean_oracle_rule='M2 oracle and M1 source-only clean SNR with seed0; never replicated across three noise seeds')

    def verify_frozen(self):
        self.verify_bindings(dict(self.artifacts))
        if not self.synthetic and self._old:
            next(iter(self._old.values())).assets.assert_frozen(self.loaded)
        return self.manifest()

    def setup_bound_files(self):
        """Return an independent path->SHA snapshot for sidecar receipts."""
        return dict(self.artifacts)


@contextmanager
def _null_context():
    yield


class _OldAdapter:
    """Original continuous3-seed batches plus original digital render groups."""
    def __init__(self, engine, layout, N):
        self.e, self.layout, self.N = engine, layout, N
        torch = engine._torch(); self.torch = torch
        if str(engine.loaded['device']) != 'cuda:0':
            raise ReplayMismatch('Old native replay must preserve original cuda:0 device')
        self.assets = load_file_module(f'metrics_replay_assets_N{N}', layout.source / 'assets.py')
        self.phy = load_file_module(f'metrics_replay_phy_N{N}', layout.source / 'digital_protocol.py')
        self.digital = load_file_module(f'metrics_replay_digital_N{N}', layout.source / 'digital.py',
                                       {'assets': self.assets, 'digital_protocol': self.phy})
        self.plugin = load_file_module(f'metrics_replay_plugin_N{N}', layout.source / 'plugin.py')
        self.pcfg = engine.read(layout.result / 'plugin_config.json')
        self.dcfg = engine.read(layout.result / 'digital_config.json')
        self.ppolicy = engine.read(layout.result / 'plugin_selected_policy.json')
        self.dpolicy = engine.read(layout.result / 'digital_selected_policy.json')
        for cfg in (self.pcfg, self.dcfg):
            engine.verify_bindings(cfg['source_bindings'])
            if cfg['mismatch_permutation'] != engine.mismatch:
                raise ReplayMismatch('Original DINO permutation differs')
            for name, digest in engine.loaded['identity']['models'].items():
                if cfg['identity']['models'][name] != digest:
                    raise ReplayMismatch('Old/shared visual identity differs: ' + name)
        if self.dcfg['identity'] != engine.loaded['identity']:
            raise ReplayMismatch('Old digital visual asset identity differs')
        for kind, policy in (('plugin', self.ppolicy), ('digital', self.dpolicy)):
            path = layout.result / f'{kind}_selected_policy.json'
            receipt = engine.read(layout.output / kind / 'calibration_complete.json')
            cfgpath = layout.result / f'{kind}_config.json'
            if (receipt['policy_sha256'] != engine.bind(path)
                    or policy['config_sha256'] != engine.bind(cfgpath)
                    or receipt['config_sha256'] != engine.bind(cfgpath)
                    or policy['development_read'] is not False):
                raise ReplayMismatch('Old frozen selection receipt differs')
        ids = [r['image_id'] for r in engine.records]
        preprocessing = [r['preprocessing_id'] for r in engine.records]
        pdev = engine.read(layout.output / 'plugin/development_identity.json')
        ddev = engine.read(layout.output / 'digital/development_identity.json')
        if (pdev['source_ids'] != ids or pdev['preprocessing_ids'] != preprocessing
                or pdev['bindings'] != engine.data['bindings']
                or pdev['config_sha256'] != engine.bind(layout.result / 'plugin_config.json')
                or pdev['policy_sha256'] != engine.bind(layout.result / 'plugin_selected_policy.json')):
            raise ReplayMismatch('Old plugin development input/policy bytes changed')
        if (ddev['sources'] != ids or ddev['preprocessing_ids'] != preprocessing
                or ddev['bindings'] != engine.data['bindings']
                or ddev['assets'] != engine.loaded['identity']
                or ddev['mismatch_permutation'] != engine.mismatch
                or ddev['config_sha256'] != engine.bind(layout.result / 'digital_config.json')
                or ddev['policy_sha256'] != engine.bind(layout.result / 'digital_selected_policy.json')):
            raise ReplayMismatch('Old digital development input/policy bytes changed')
        selected = layout.output / f'training/p{N}_2026093001/selected_P{N}.json'
        prefix = '_extreme_bw_train' if N == 512 else '_extreme_bw_n1024_train'
        train = load_file_module(prefix, layout.source / 'train.py')
        self.p, pmeta = train.load_for_evaluation(selected, self.plugin.scale_statistics(), engine.loaded['device'])
        self.pmeta = pmeta; engine.bind(selected)
        engine.bind(selected.parent / 'registration.json')
        engine.bind(selected.parent / 'completion.json')
        engine.verify_bindings(pmeta['registration']['bindings'])
        if pmeta['selected_sha256'] != self.pcfg['identity']['selected_sha256']:
            raise ReplayMismatch('Old selected continuous checkpoint differs')
        engine.verify_bindings({pmeta['checkpoint']: pmeta['selected']['checkpoint_sha256']})
        chosen = pmeta['selected']
        engine.verify_bindings({chosen['calibration_csv']: chosen['calibration_sha256']})
        if pmeta['decoder_sha256'] != engine.loaded['identity']['models']['decoder']:
            raise ReplayMismatch('Old continuous decoder identity differs')
        if self.plugin.old.b.state_sha256(self.p) != self.pcfg['identity']['models'][f'P{N}']:
            raise ReplayMismatch('Old continuous transmitter/receiver weights differ')
        self.models = {k: engine.loaded[k] for k in ('vae', 'var', 'decoder', 'lpips', 'dino')}
        self.models[f'P{N}'] = self.p
        path = layout.output / 'plugin/development_observations.pt'
        descriptor = engine.read(Path(str(path) + '.json'))
        if engine.bind(path) != descriptor['sha256']:
            raise ReplayMismatch('Sealed old observations tensor differs')
        self.obs = torch.load(path, map_location='cpu', weights_only=True)
        expected = dict(config_sha256=engine.bind(layout.result / 'plugin_config.json'),
            role='development', seeds=list(DEV_SEEDS), source_ids=[r['image_id'] for r in engine.records],
            preprocessing_ids=[r['preprocessing_id'] for r in engine.records])
        if self.obs['identity'] != expected or tuple(self.obs['Z'].shape) != (1500, 32, 16, 16):
            raise ReplayMismatch('Sealed old observation/source identity differs')
        self.obs_index = {}
        for j, row in enumerate(self.obs['rows']):
            key = (int(row['source_index']), int(row['snr_db']), int(row['noise_seed']))
            if key in self.obs_index or int(self.obs['source_indices'][j]) != key[0]:
                raise ReplayMismatch('Sealed observations contain duplicate/misaligned source')
            self.obs_index[key] = j
        expected_keys = {(i, s, seed) for i in range(100) for s in SNRS for seed in DEV_SEEDS}
        if set(self.obs_index) != expected_keys or not torch.isfinite(self.obs['Z']).all():
            raise ReplayMismatch('Sealed original observations grid incomplete/nonfinite')
        self.cache = self.digital.ReceivedCache()

    def validate_rows(self, rows):
        return None

    def _plugin_source(self, index, rows):
        e, p, torch = self.e, self.plugin, self.torch
        record = e.records[index]
        wave = self.p.transmit(e.data['F'][index:index + 1].to(e.loaded['device']))[0].cpu().numpy()
        wave_hash = p.waveform_sha(wave)
        for snr in SNRS:
            relevant = [r for r in rows if int(r['snr_db']) == snr and not r['method'].startswith('D_')]
            if not relevant:
                continue
            inds = [self.obs_index[(index, snr, seed)] for seed in DEV_SEEDS]
            Z = self.obs['Z'][inds]; ii = self.obs['source_indices'][inds]
            f = e.data['F'][ii].to(e.loaded['device'])
            truth = p.old.b.split(e.data['T'][ii].to(e.loaded['device']))
            estimates = {f'P{self.N}': Z}; post_for = {}; candidates = {}; info = {}
            level = self.ppolicy['levels'][str(snr)]
            for prior in ('A1', 'A2', 'V'):
                selected = level['methods'][prior]; lam = selected['raw_selected_lambda']
                key = p.candidate_key(prior, lam)
                if key not in candidates:
                    candidates[key] = p.rx.infer(e.loaded['vae'], e.loaded['var'], Z.to(e.loaded['device']),
                        prior, level['variance_by_scale'], e.loaded['static'], lam=lam,
                        clean=f, truth=truth, details=True)
                Q = candidates[key]['fhat'].cpu(); post_for[prior] = Q
                alpha = torch.tensor(selected['alpha'], dtype=torch.float32).reshape(1, 32, 1, 1)
                common = torch.tensor(level['alpha_common'], dtype=torch.float32).reshape(1, 32, 1, 1)
                for suffix, value in (('policy', Z if selected['policy_action'] == 'BYPASS' else p.rx.fusion(Z, Q, alpha)),
                        ('raw_fused', p.rx.fusion(Z, Q, alpha)), ('common', p.rx.fusion(Z, Q, common)), ('tok', Q)):
                    name = prior + '_' + suffix; estimates[name] = value
                    info[name] = (prior, lam, selected['policy_action'] if suffix == 'policy' else 'DIAGNOSTIC')
            key = 'V_lambda1.0'
            if key not in candidates:
                candidates[key] = p.rx.infer(e.loaded['vae'], e.loaded['var'], Z.to(e.loaded['device']),
                    'V', level['variance_by_scale'], e.loaded['static'], lam=1., details=False)
            q1 = candidates[key]['fhat'].cpu()
            alpha1 = torch.tensor(level['V_lambda1']['alpha'], dtype=torch.float32).reshape(1, 32, 1, 1)
            estimates['V_lambda1'] = p.rx.fusion(Z, q1, alpha1); info['V_lambda1'] = ('V', 1., 'CORRECT')
            rendered = {}
            # Preserve original dict order and exact tensor alias reuse.
            for name, value in estimates.items():
                alias = next((x for x in rendered if torch.equal(value, estimates[x])), None)
                if alias is None:
                    rendered[name] = p.images_metrics(value, ii, e.records, self.models,
                        e.data['reference'].to(e.loaded['device']), torch.tensor(e.mismatch), save_images=True)
                else:
                    rendered[name] = rendered[alias]
            channels = {}
            for seed in DEV_SEEDS:
                y = p.apply_channel(wave, snr, record['image_id'], seed, p.CELL)
                oj = self.obs_index[(index, snr, seed)]; original = self.obs['rows'][oj]
                if original['waveform_sha256'] != wave_hash or original['observation_sha256'] != p.waveform_sha(y):
                    raise ReplayMismatch('Continuous TX/AWGN differs from sealed observation')
                regenerated = self.p.receive(torch.tensor(y[None], device=e.loaded['device']),
                    torch.tensor([snr], dtype=torch.float32, device=e.loaded['device']))[0].cpu()
                if not torch.equal(regenerated, self.obs['Z'][oj]):
                    raise ReplayMismatch('Continuous receiver differs from sealed Z')
                channels[seed] = dict(waveform_sha256=wave_hash, observation_sha256=p.waveform_sha(y))
            for row in relevant:
                seed = int(row['noise_seed']); j = DEV_SEEDS.index(seed); name = row['method']
                value = estimates[name]; metrics, images = rendered[name]
                prior, lam, action = info.get(name, (f'P{self.N}', '', 'BYPASS'))
                Q = q1 if name == 'V_lambda1' else post_for.get(prior, Z)
                computed = dict(metrics[j], **channels[seed], latent_valid=True, decoder_applied=True,
                    policy_action=action, **{'lambda': lam},
                    latent_sq_err_base=float((e.data['F'][index].double() - Z[j].double()).square().sum()),
                    latent_sq_err_post=float((e.data['F'][index].double() - Q[j].double()).square().sum()),
                    latent_sq_err_final=float((e.data['F'][index].double() - value[j].double()).square().sum()))
                computed['latent_sq_error'] = computed['latent_sq_err_final']
                yield row, images[j], computed

    def _digital_source(self, index, rows):
        e, d, phy = self.e, self.digital, self.phy
        record = e.records[index]; scales = d.split_tokens(e.data['T'][index].numpy())
        for snr in SNRS:
            for family in phy.PHY_FAMILIES:
                relevant = [r for r in rows if int(r['snr_db']) == snr and
                            r['method'].startswith('D_') and r['phy_family'] == family]
                if not relevant:
                    continue
                choices = self.dpolicy['levels'][str(snr)][family]
                waves = {m: phy.transmit(scales, int(record['class_index']), phy.Action(m, family))
                         for m in {c['action_m'] for c in choices.values()}}
                for seed in DEV_SEEDS:
                    received = {}
                    for m, (wave, ledger) in waves.items():
                        y = phy.apply_channel(wave, snr, record['image_id'], seed, family)
                        event = phy.receive(y, snr, family)
                        latent = d.render_received(event, e.loaded, self.cache)
                        received[m] = (event, latent, dict(ledger,
                            waveform_sha256=phy.waveform_sha(wave), observation_sha256=phy.waveform_sha(y)))
                    outputs, information = d.selected_outputs(family, choices, received)
                    scored = d.quality_outputs(e.loaded, record, outputs, e.data['reference'][index],
                        e.data['reference'][e.mismatch[index]], images=True)
                    for row in relevant:
                        if int(row['noise_seed']) != seed:
                            continue
                        name = row['method']; metrics, image = scored[name]
                        condition, event, ledger, m = information[name]
                        computed = dict(ledger, **d.event_fields(event), **metrics,
                            **d.latent_fields(e.data['F'][index], outputs[name][0]), action_m=m)
                        computed['mismatch_source_id'] = e.records[e.mismatch[index]]['image_id']
                        if 'dino_specificity' in row:
                            computed['dino_specificity'] = metrics['dino_cosine'] - metrics['dino_mismatched']
                        yield row, image, computed

    def iterate_source(self, index, rows):
        # Source-local received-prefix memoization has no target/row leakage.
        self.cache = self.digital.ReceivedCache()
        yield from self._plugin_source(index, rows)
        yield from self._digital_source(index, rows)


class _M1Adapter:
    def __init__(self, engine, layout):
        self.e, self.layout = engine, layout
        self.mods = engine.methods_modules(layout); self.m = self.mods['m1']
        engine.setup(['N512', 'N1024'])
        self.policy = engine.read(layout.result / 'm1_policy.json')
        receipt = engine.read(layout.output / 'm1_calibration_complete.json')
        if self.policy['development_read'] is not False or receipt['policy_sha256'] != engine.bind(layout.result / 'm1_policy.json'):
            raise ReplayMismatch('M1 frozen policy changed')
        engine.validate_registration(layout.output / 'm1_development_registration.json')
        done = engine.read(layout.output / 'm1_development_complete.json')
        if done['status'] != 'COMPLETE' or done['policy_sha256'] != receipt['policy_sha256']:
            raise ReplayMismatch('M1 development completion differs from policy')
        self.expected_actions = {}
        phy = self.mods['partial_phy']
        for cell in self.policy['cells']:
            key = (cell['N'], cell['phy_family'], cell['snr_db'], cell['method'] + '_policy')
            self.expected_actions[key] = self.m.action_from(cell['action'])
        for key, action in list(self.expected_actions.items()):
            if key[3] == 'entropy_policy':
                for order in ('raster', 'random', 'oracle'):
                    try:
                        a = phy.Action(action.N, action.phy, action.m, action.q, order)
                    except ValueError:
                        continue
                    self.expected_actions[(*key[:3], order + '_at_entropy')] = a

    def validate_rows(self, rows):
        counts = defaultdict(int); main_keys = set(); reference_keys = set()
        for row in rows:
            N, snr, seed = int(row['N']), int(row['snr_db']), int(row['noise_seed'])
            if boolean(row['historical_reference']):
                if row['method'] not in (f'P{N}', 'legacy_policy'):
                    raise ReplayMismatch('Unknown M1 historical reference method')
                reference_keys.add((int(row['source_index']), N, row['phy_family'], snr, seed, row['method']))
                continue
            key = (N, row['phy_family'], snr, row['method'])
            if key not in self.expected_actions or self.m.action_id(self.expected_actions[key]) != row['action_id']:
                raise ReplayMismatch('M1 row action differs from frozen policy/sameK control')
            counts[key] += 1
            main_keys.add((int(row['source_index']), seed, *key))
        if set(counts) != set(self.expected_actions) or any(n != 300 for n in counts.values()):
            raise ReplayMismatch('M1 policy/sameK development grid incomplete')
        if sum(boolean(r['historical_reference']) for r in rows) != 12000:
            raise ReplayMismatch('M1 historical resource facets incomplete')
        expected_main = {(i, seed, *key) for i in range(100) for seed in DEV_SEEDS
                         for key in self.expected_actions}
        expected_references = {(i, N, fam, snr, seed, method) for i in range(100)
            for N in (512, 1024) for fam in ('QPSK', '16QAM') for snr in SNRS
            for seed in DEV_SEEDS for method in (f'P{N}', 'legacy_policy')}
        if main_keys != expected_main or reference_keys != expected_references:
            raise ReplayMismatch('M1 complete source/noise/resource/reference grid differs')

    def iterate_source(self, index, rows):
        e, m = self.e, self.m
        txmemo = dict(waves={}, logits={}); render_memo = {}; metric_memo = {}
        references = [r for r in rows if boolean(r['historical_reference'])]
        for row in rows:
            if boolean(row['historical_reference']):
                continue
            key = (int(row['N']), row['phy_family'], int(row['snr_db']), row['method'])
            action = self.expected_actions[key]
            computed, image = m.score_one(e.loaded, e.data, index, action,
                int(row['snr_db']), int(row['noise_seed']), txmemo, render_memo, metric_memo, images=True)
            if 'dino_specificity' in row:
                computed['dino_specificity'] = computed['dino_cosine'] - computed['dino_mismatched']
            yield row, image, computed
        for N in (512, 1024):
            relevant = [r for r in references if int(r['N']) == N]
            if not relevant:
                continue
            old = e._old_adapter(N)
            # Delegate using original old table rows; facets can duplicate P.
            old_study = f'N{N}'
            if old_study not in e.rows:
                e.setup([old_study])
            wanted = {f'P{N}', 'D_U_QPSK', 'D_U_16QAM'}
            oldrows = [r for r in e.by_source[old_study][index] if r['method'] in wanted]
            mapped = {(r['method'], int(r['snr_db']), int(r['noise_seed'])): (image, comp)
                      for r, image, comp in old.iterate_source(index, oldrows)}
            for row in relevant:
                name = f'P{N}' if row['method'] == f'P{N}' else 'D_U_' + row['phy_family']
                image, comp = mapped[(name, int(row['snr_db']), int(row['noise_seed']))]
                comp = dict(comp, mismatch_source_id=e.records[e.mismatch[index]]['image_id'])
                yield row, image, comp


class _M1RateAdapter:
    """All original source-only m5/m6 dense points; no AWGN or paid N claim."""
    def __init__(self, engine, layout):
        self.e, self.layout = engine, layout
        self.mods = engine.methods_modules(layout)
        engine.validate_registration(layout.output / 'm1_rate_curve_registration.json')
        receipt = engine.read(layout.output / 'm1_rate_curve_complete.json')
        if receipt['status'] != 'COMPLETE' or receipt['rows'] != 10800 or receipt['wireless_claim'] is not False:
            raise ReplayMismatch('M1 dense source-only completion differs')
        self.registration_sha = engine.bind(layout.output / 'm1_rate_curve_registration.json')

    def validate_rows(self, rows):
        for index in range(100):
            cell = self.e.read(self.layout.output / 'm1_rate_curve/cells' / f'{index:03d}.json')
            if cell['identity'] != self.registration_sha:
                raise ReplayMismatch('Original rate-curve cell registration differs')
            originals = {(int(x['m']), int(x['q']), x['order']): x for x in cell['rows']}
            selected = [x for x in rows if int(x['source_index']) == index]
            if len(cell['rows']) != 108 or set(originals) != rate_scopes() or len(selected) != 108:
                raise ReplayMismatch('Original rate-curve cell scope incomplete/duplicated')
            for row in selected:
                original = originals[(int(row['m']), int(row['q']), row['order'])]
                csv_fields = {k: row[k] for k in original}
                if source_row_hash(csv_fields) != row['replay_source_row_sha256']:
                    raise ReplayMismatch('Added rate metadata changed original CSV identity')
                for k, v in original.items():
                    if isinstance(v, bool):
                        equal = boolean(row[k]) == v
                    elif isinstance(v, (int, float)):
                        equal = float(row[k]) == v
                    else:
                        equal = row[k] == v
                    if not equal:
                        raise ReplayMismatch('Rate original cell/CSV differs: ' + k)

    def iterate_source(self, index, rows):
        e = self.e; c = self.mods['common']; phy = self.mods['partial_phy']; rx = self.mods['partial_receiver']
        record = e.records[index]; scales = c.split_tokens(e.data['T'][index].numpy())
        logits = {}
        for row in rows:
            m, q, order = int(row['m']), int(row['q']), row['order']
            if m not in logits:
                logits[m] = rx.prefix_logits(e.loaded['vae'], e.loaded['var'], scales[:m], e.loaded['device'])
            L = phy.SIZES[m] ** 2
            if q == L:
                latent = c.legacy.complete_latent(e.loaded['vae'], e.loaded['var'], scales[:m + 1], 1000, e.loaded['device']).cpu()
            elif q == 0:
                latent = c.legacy.complete_latent(e.loaded['vae'], e.loaded['var'], scales[:m], 1000, e.loaded['device']).cpu()
            else:
                action = phy.Action(1024, '16QAM', m, q, order)
                positions = rx.order_positions(scales[:m], order, q, logits=logits[m],
                    truth_next=scales[m] if order == 'oracle' else None)
                wave, _ = phy.transmit(scales, action, positions=positions)
                event = phy.receive(wave, 100., 1024, '16QAM')
                if not event['header_ok'] or not event['prefix_crc_ok'] or not event['partial_usable']:
                    raise ReplayMismatch('Original clean rate-curve PHY roundtrip failed')
                if (not all(np.array_equal(x, y) for x, y in zip(event['prefix'], scales[:m]))
                        or not np.array_equal(event['partial_values'], scales[m][positions])):
                    raise ReplayMismatch('Clean rate-curve source tokens changed')
                received = rx.complete_partial(e.loaded['vae'], e.loaded['var'], event, e.loaded['device'])
                if received['diagnostics']['known_positions'] != positions.tolist():
                    raise ReplayMismatch('Clean rate-curve known positions changed')
                latent = received['fhat'].cpu()
            metrics, image = c.quality(e.loaded, record, latent, e.data['F'][index],
                e.data['reference'][index], e.data['reference'][e.mismatch[index]], images=True)
            computed = dict(metrics, m=m, q=q, order=order, projection=row['projection'])
            if 'dino_specificity' in row:
                computed['dino_specificity'] = metrics['dino_cosine'] - metrics['dino_mismatched']
            yield row, image, computed


class _M2Adapter:
    def __init__(self, engine, layout, study):
        self.e, self.layout, self.study = engine, layout, study
        self.mods = engine.methods_modules(layout); self.m = self.mods['m2']; self.rx = self.mods['residual']
        self.policy, self.gate = self.m.frozen_policy()
        if self.policy.get('development_read') is not False or self.gate.get('development_read') is not False:
            raise ReplayMismatch('M2 calibration policy/gate development boundary differs')
        for filename in ('policy.json', 'gate.json', 'actual_policy.json'):
            engine.bind(layout.output / 'm2/calibration' / filename)
        self.projections, self.stats, self.operators, self.static = self.m.load_state(engine.loaded)
        receipt = engine.read(layout.output / 'm2/statistics_complete.json')
        engine.verify_bindings(receipt['files']); engine.verify_bindings(receipt['source_bindings'])
        engine.verify_bindings(self.policy['source_bindings'])
        self.tokens = self.m.split_batch(engine.data['T'])
        if study == 'M2_ORACLE':
            engine.validate_registration(layout.output / 'm2_evaluation_registration.json')
            receipt = engine.read(layout.output / 'm2_evaluation_complete.json')
            if receipt['status'] != 'M2_EVALUATION_COMPLETE' or receipt['metric_rows'] != 48000:
                raise ReplayMismatch('M2 oracle evaluation incomplete')
        else:
            self.actual_policy = engine.read(layout.output / 'm2/calibration/actual_policy.json')
            engine.verify_bindings(self.actual_policy.get('source_bindings', {}))
            receipt = engine.read(layout.output / 'm2_actual_complete.json')
            if receipt['status'] != 'M2_ACTUAL_COMPLETE':
                raise ReplayMismatch('M2 actual evaluation incomplete')
            self.actual_receipt = receipt
            if receipt['branch'] == 'ACTUAL_LINK_EVALUATED':
                engine.validate_registration(layout.output / 'm2_actual_registration.json')
            elif receipt['branch'] != 'SKIPPED_GATE_NOT_MET' or self.gate['passed']:
                raise ReplayMismatch('M2 actual skip branch inconsistent')
        if receipt['synthetic'] is not False or receipt['training_updates'] != 0:
            raise ReplayMismatch('M2 replay requires completed real zero-training study')
        engine.verify_bindings(receipt['files'])

    def validate_rows(self, rows):
        if self.study == 'M2_ORACLE':
            return
        if self.actual_receipt['branch'] == 'SKIPPED_GATE_NOT_MET':
            if rows:
                raise ReplayMismatch('Gate-skipped M2 must have no actual result rows')
            return
        selected = [c for c in self.actual_policy['choices'] if c['status'] == 'SELECTED']
        expected = {(i, str(c['N']), c['phy_family'], str(c['snr_db']), str(seed), c['projection'], control)
                    for i in range(100) for c in selected for seed in DEV_SEEDS for control in M2_CONTROLS}
        actual = {(int(r['source_index']), r['N'], r['phy_family'], r['snr_db'], r['noise_seed'],
                   r['projection'], r['control']) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ReplayMismatch('M2 frozen actual policy/control grid incomplete')

    def _score(self, index, latent):
        metrics, image = self.m.score(self.e.loaded, self.e.data, index, latent, images=True)
        return metrics, image

    def iterate_source(self, index, rows):
        e, m, torch = self.e, self.m, self.e._torch()
        if not rows:
            return
        record = e.records[index]
        prefix = [t[index:index + 1].to(e.loaded['device']) for t in self.tokens[:4]]
        pre = self.rx.prefix_latent(e.loaded['vae'].quantize, prefix)
        clean = e.data['F'][index:index + 1].to(e.loaded['device'])
        if self.study == 'M2_ORACLE':
            base = m.greedy(e.loaded, prefix)
            base_metrics, base_image = self._score(index, base)
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row['projection'], row['snr_db'], int(row['noise_seed']))].append(row)
            for (name, snr, seed), group in grouped.items():
                p = self.projections[name]; snr = 'clean' if snr == 'clean' else int(snr)
                obs, variance, info = m.measure(clean, pre, p, record, snr, seed)
                for row in group:
                    control = row['control']
                    lam = self.policy['var'][name]['selected_lambda'] if control == 'VAR_GUIDED' else self.policy['static'][name]['selected_lambda'] if control == 'STATIC' else 0.
                    if control == 'UNGUIDED':
                        metrics, image = base_metrics, base_image
                    else:
                        latent = m.run_control(e.loaded, prefix, obs, p, self.stats[name],
                            self.operators[name], self.static, variance, control, lam, base)
                        metrics, image = self._score(index, latent)
                    yield row, image, dict(metrics, **info, projection=name, control=control, **{'lambda': lam})
        else:
            scales = self.mods['common'].split_tokens(e.data['T'][index].numpy())
            grouped = defaultdict(list); waves = {}
            for row in rows:
                grouped[(int(row['N']), row['phy_family'], int(row['snr_db']),
                         int(row['noise_seed']), row['projection'])].append(row)
            for (N, phy, snr, seed, name), group in grouped.items():
                p = self.projections[name]; wave_key = (N, phy, name)
                if wave_key not in waves:
                    coordinates = p.forward(clean - pre)[0].cpu().numpy()
                    waves[wave_key] = m.actual_transmit(scales, coordinates, N, p, phy)
                wave, ledger = waves[wave_key]
                perturbation = m.seeded_noise(f'{self.mods["common"].RUN}/M2/actual/N{N}|{record["image_id"]}', seed, (N, 2))
                y = wave + perturbation * 10. ** (-float(snr) / 20)
                event = m.actual_receive(y, snr, N, p, phy)
                outputs = m.actual_outputs(e.loaded, event, p, self.stats[name],
                    self.operators[name], self.static, self.policy)
                for row in group:
                    control = row['control']; metrics, image = self._score(index, outputs[control])
                    lam = self.policy['var'][name]['selected_lambda'] if control == 'VAR_GUIDED' else self.policy['static'][name]['selected_lambda'] if control == 'STATIC' else 0.
                    yield row, image, dict(ledger, **m.event_meta(event), **metrics,
                        projection=name, control=control, **{'lambda': lam})


def create_engine(root, studies=None, *, layouts=None, loaded=None, data=None):
    """Load frozen shared assets/data and return a fully set up native engine.

    Scheduling remains the caller's responsibility. No result, policy, source,
    cache, registration, or completion file is created by this factory. Passing
    already loaded assets avoids a duplicate model allocation. ``layouts`` may
    override source/result/output paths without changing any frozen selection.
    """
    root = Path(root).resolve()
    all_layouts = default_layouts(root)
    if layouts:
        all_layouts.update({k: v if isinstance(v, Layout) else Layout(**v)
                           for k, v in layouts.items()})
    old_source = all_layouts['N1024'].source
    assets = load_file_module('metrics_replay_assets_N1024', old_source / 'assets.py')
    phy = load_file_module('metrics_replay_phy_N1024', old_source / 'digital_protocol.py')
    digital = load_file_module('metrics_replay_digital_N1024', old_source / 'digital.py',
                              {'assets': assets, 'digital_protocol': phy})
    common = load_file_module('metrics_replay_methods_common', all_layouts['M1'].source / 'common.py',
                             {'assets': assets, 'digital': digital})
    if loaded is None:
        loaded = common.setup()
    if data is None:
        data = common.data('development', loaded)
    engine = ReplayEngine(root, loaded, data, all_layouts)
    return engine.setup(studies)
