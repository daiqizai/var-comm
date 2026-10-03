"""Read-only native replay of completed large-budget and selected C grids.

No historical main, selector, cache writer, training or timing study is called.
Import and construction require only NumPy; setup loads the original GPU models.
The caller owns scheduling, supplementary metrics and all new output files.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np

STUDIES = ('CONTINUOUS_GRID', 'C_SELECTED_GRID', 'C_SELECTED_BASELINES')
SNRS = (1, 4, 7, 13, 19)
SEEDS = (2001, 2002, 2003)
PARITY_TOLERANCES = {'psnr_db': (1e-5, 0.), 'lpips_alex': (2e-6, 0.),
                     'dino_cosine': (2e-6, 0.), 'mse': (1e-8, 1e-6)}
EXACT_FIELDS = ('waveform_sha256', 'observation_sha256', 'header_ok',
                'body_crc_ok', 'source_overflow_erasure', 'N', 'N_header',
                'N_data', 'N_continuous', 'mcs', 'energy_constraint', 'm_actual')


class ReplayMismatch(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rgb(value):
    a = np.asarray(value)
    if a.dtype != np.float32 or a.shape != (3, 256, 256):
        raise ReplayMismatch('Original float32 CHW RGB required')
    if not np.isfinite(a).all() or a.min() < -1e-6 or a.max() > 1.000001:
        raise ReplayMismatch('Invalid original decoder RGB range')
    return np.array(a, copy=True, order='C')


def integer(value):
    if isinstance(value, bool):
        raise ReplayMismatch('Boolean is not an integer identity')
    x = float(value)
    if not math.isfinite(x) or int(x) != x:
        raise ReplayMismatch('Nonintegral grid identity')
    return int(x)


def csv_value(value):
    return '' if value is None else str(value)


def parity_check(original, computed):
    """All four old metrics and available channel fields must be reproduced."""
    deltas = {}
    for key, (atol, rtol) in PARITY_TOLERANCES.items():
        if key not in original or key not in computed:
            raise ReplayMismatch('Missing original quality field: ' + key)
        a, b = float(original[key]), float(computed[key])
        if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > atol + rtol*abs(a):
            raise ReplayMismatch('Historical metric changed: ' + key)
        deltas[key] = b-a
    _channel_parity(original, computed)
    return {'passed': True, 'replay_parity_passed': True, 'synthetic': False,
            'metric_deltas': deltas,
            'exact_fields_checked': [k for k in EXACT_FIELDS if k in original and original[k] != ''],
            'original_row_sha256': identity(original), 'parity_origin': 'fresh_native_replay'}


def _channel_parity(original, computed):
    for key in EXACT_FIELDS:
        if key in original and original[key] != '':
            if key not in computed or csv_value(original[key]) != csv_value(computed[key]):
                raise ReplayMismatch('Historical channel field changed: ' + key)
    if 'E' in original:
        if 'E' not in computed or not math.isclose(float(original['E']), float(computed['E']),
                                                  rel_tol=1e-7, abs_tol=1e-5):
            raise ReplayMismatch('Historical transmitted energy changed')


def _imports(root):
    roots = [root/'src', root/'experiments/token_channel_efficiency_20260923/src',
             root/'experiments/var-short-prefix-hybrid-20260923/src']
    base = root/'experiments/var-latent-enhancement-20260917'
    roots += [base/'src', *[base/p/'src' for p in ('phase_b', 'evaluation', 'followup', 'research')]]
    for path in reversed(roots):
        if not path.is_dir():
            raise ReplayMismatch('Missing original import directory: ' + str(path))
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in (
        'token_efficiency.common', 'token_efficiency.continuous_grid',
        'token_efficiency.C_evaluate', 'token_efficiency.digital_grid')}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('token_efficiency', 'short_prefix', 'latent_enhancement',
                                  'latent_enhancement_b', 'latent_enhancement_eval',
                                  'latent_followup', 'latent_research', 'var_comm'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ReplayMismatch('Original module imported from another checkout: ' + name)
    return modules


class BudgetAdapter:
    def __init__(self, root, study, loaded=None, data=None):
        if study not in STUDIES:
            raise ValueError('Unsupported historical study: ' + str(study))
        self.root, self.study = Path(root).resolve(), study
        self._loaded, self._data = loaded, data
        self.bindings, self.rows, self.records = {}, [], []
        self.by_source = defaultdict(list)
        self._locations, self._cells, self._main = {}, {}, {}
        self.ready = False
        base = self.root/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
        self.folder = base/('continuous_grid_v1' if study == 'CONTINUOUS_GRID'
                           else 'C_priority_selected_v1/selected_grid')
        self._read_frozen()

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ReplayMismatch('Frozen file hash differs: ' + str(path))
        key = str(path)
        if key in self.bindings and self.bindings[key] != actual:
            raise ReplayMismatch('File changed while loading: ' + key)
        self.bindings[key] = actual
        return actual

    def _read_frozen(self):
        regpath = self.folder/'registration.json'
        self.registration_sha256 = self.bind(regpath)
        self.registration = reg = read(regpath)
        self.bind(self.folder/'completion.json')
        done = read(self.folder/'completion.json')
        expected = ('REAL_CONTINUOUS_GRID_COMPLETE' if self.study == 'CONTINUOUS_GRID'
                    else 'REAL_C_SELECTED_GRID_COMPLETE')
        if (done.get('status') != expected or done.get('synthetic') is not False
                or reg.get('synthetic') is not False
                or done.get('registration_sha256') != self.registration_sha256):
            raise ReplayMismatch('A completed real original grid is required')
        if tuple(reg['snrs']) != SNRS or tuple(reg['seeds']) != SEEDS or len(reg['sources']) != 100:
            raise ReplayMismatch('Original population/SNR/seed scope differs')
        if identity(reg['context']) != reg['context_sha256']:
            raise ReplayMismatch('Original context hash differs')
        self.bind(self.folder/'qualification.json')
        qual = read(self.folder/'qualification.json')
        qstatus = ('REAL_SELECTED_CONTINUOUS_REPLAY_PASS' if self.study == 'CONTINUOUS_GRID'
                   else 'REAL_C_SELECTED_CACHE_ONLINE_REPLAY_PASS')
        if (qual.get('status') != qstatus or qual.get('synthetic') is not False
                or qual.get('registration_sha256') != self.registration_sha256):
            raise ReplayMismatch('Original selected replay qualification required')
        self.context = reg['context']
        if self.study == 'CONTINUOUS_GRID':
            self.methods = ['P2048', 'P3060', 'P4084']
            self.model_metadata = {'P'+n: meta for n, meta in self.context['selected'].items()}
            if set(self.model_metadata) != set(self.methods):
                raise ReplayMismatch('Three distinct original continuous models required')
        else:
            self.model_metadata = self.context['models']
            self.methods = list(self.model_metadata)
            if len(self.methods) != 14:
                raise ReplayMismatch('All 14 original selected C models required')
        table = self.folder/'per_frame.csv'
        self.table_sha256 = self.bind(table, done['files']['per_frame.csv'])
        with table.open(newline='', encoding='utf-8') as f:
            main_rows = list(csv.DictReader(f))
        expected_keys = {(i, method, snr, seed) for i in range(100)
                         for method in self.methods for snr in SNRS for seed in SEEDS}
        source_ids = {}
        for lineno, row in enumerate(main_rows):
            key = (integer(row['source_index']), row['method'], integer(row['snr_db']), integer(row['noise_seed']))
            if key not in expected_keys or key in self._main:
                raise ReplayMismatch('Duplicate/extra original main row')
            i, method, _, _ = key
            if source_ids.setdefault(i, row['source_id']) != row['source_id']:
                raise ReplayMismatch('Source index identity changed')
            if (reg['sources'].get(row['source_id']) != row['preprocessing_id']
                    or row['context_sha256'] != reg['context_sha256'] or row['population'] != 'development'):
                raise ReplayMismatch('Row escaped registered image/context')
            self._main[key] = row
            if self.study != 'C_SELECTED_BASELINES':
                self._append(row, i, table, '/csv_rows/'+str(lineno), self.table_sha256)
        if set(self._main) != expected_keys or done['frame_rows'] != len(main_rows):
            raise ReplayMismatch('Incomplete original main grid')
        self.source_ids = [source_ids[i] for i in range(100)]
        if set(self.source_ids) != set(reg['sources']) or len(set(self.source_ids)) != 100:
            raise ReplayMismatch('Source roster differs')
        # Completion hashes already seal the original cell bytes. Do not call
        # load_cell: its interrupted-write recovery would mutate historical files.
        expected_cells = set()
        for i in range(100):
            for mi, method in enumerate(self.methods):
                suffix = method[1:] if self.study == 'CONTINUOUS_GRID' else f'{mi:02d}'
                name = f'{i:04d}_{suffix}.json'; expected_cells.add(name)
                path = self.folder/'cells'/name
                cell_sha = self.bind(path, done['cell_sha256'][name])
                cell = read(path)
                if (cell['registration_sha256'] != self.registration_sha256
                        or cell['source_id'] != self.source_ids[i] or cell['method'] != method):
                    raise ReplayMismatch('Original cell identity mismatch')
                self._cells[i, method] = cell
                expected_main = [self._main[i, method, snr, seed] for snr in SNRS for seed in SEEDS]
                if len(cell['rows']) != 15:
                    raise ReplayMismatch('Incomplete original cell rows')
                for raw, typed in zip(expected_main, cell['rows']):
                    if any(csv_value(typed.get(k)) != v for k, v in raw.items()):
                        raise ReplayMismatch('Original CSV and sealed cell disagree')
                if self.study == 'C_SELECTED_BASELINES':
                    base_rows = cell['base_rows']
                    hybrid = self.model_metadata[method]['kind'] == 'hybrid'
                    grid = [(s, seed, condition) for s in SNRS for seed in SEEDS
                            for condition in ('B_RX', 'C_RX')] if hybrid else []
                    actual = [(integer(r['snr_db']), integer(r['noise_seed']), r['condition']) for r in base_rows]
                    if actual != grid:
                        raise ReplayMismatch('Incomplete or reordered original baseline grid')
                    for j, row in enumerate(base_rows):
                        if (row['method'] != method or row['source_id'] != self.source_ids[i]
                                or row.get('enhancement_removed') is not True
                                or row['N_paid'] != self.model_metadata[method]['N']):
                            raise ReplayMismatch('Baseline cell identity changed')
                        self._append(row, i, path, '/base_rows/'+str(j), cell_sha)
        if expected_cells != set(done['cell_sha256']):
            raise ReplayMismatch('Original sealed cell inventory differs')
        for i, image_id in enumerate(self.source_ids):
            self.records.append({'source_index': i, 'source_id': image_id, 'image_id': image_id,
                                 'preprocessing_id': reg['sources'][image_id]})

    def _append(self, row, index, path, pointer, sha):
        self.rows.append(row); self.by_source[index].append(row)
        self._locations[identity(row)] = {'path': str(path.resolve()), 'sha256': sha,
                                         'pointer': pointer, 'original_row_sha256': identity(row)}

    def setup(self):
        if self.ready:
            return self
        # Loading fresh frozen original models avoids trusting an unrelated
        # caller's in-memory state. The optional slots are reserved for a future
        # independently qualified sharing protocol, not an unchecked shortcut.
        if self._loaded is not None:
            raise ReplayMismatch('Unqualified injected models are not accepted; use loaded=None')
        for path, expected in self.context['bindings'].items():
            self.bind(path, expected)
        modules = _imports(self.root)
        common = modules['token_efficiency.common']
        self.native = modules['token_efficiency.continuous_grid' if self.study == 'CONTINUOUS_GRID'
                              else 'token_efficiency.C_evaluate']
        if common.ROOT.resolve() != self.root:
            raise ReplayMismatch('Original runtime repository root mismatch')
        common.configure_runtime(); self.native.require_available()
        self.safe = self.native.SafeEvaluation()
        torch = self.native.torch
        self.torch = torch
        self.device = torch.device('cuda:0')
        paths = self.native.model_paths()
        self.bind(self.root/'configs/next_scale_prior_diagnostic.yaml')
        for name in ('vae_checkpoint', 'var_checkpoint'):
            self.bind(paths[name], paths[name+'_sha256'])
        self.vae, self.var = self.native.load_models(paths, self.device)
        self.decoder = self.native.load_decoder(self.vae, self.device)
        if self.native.state_sha256(self.decoder) != self.context['decoder_sha256']:
            raise ReplayMismatch('Original frozen Dc differs')
        self.scale = self.native.scale_statistics(self.device)
        decoder_common = importlib.import_module('latent_enhancement_b.common')
        gate = decoder_common.load_gate()
        self.bind(decoder_common.decoder_gate_path())
        self.bind(gate['selection']['checkpoint'], gate['selection']['checkpoint_sha256'])
        self.bind(decoder_common.CACHE/'training_statistics.json', gate['statistics_sha256'])
        for group in ('stage_A_bindings', 'frozen_stage_A_sources'):
            for path, expected in gate[group].items():
                self.bind(path, expected)
        for value in self.context.get('finalizations', {}).values():
            self.bind(value['path'], value['sha256'])
        self.models = {}
        for method in self.methods:
            meta = self.model_metadata[method]
            if self.study == 'CONTINUOUS_GRID':
                if method == 'P4084':
                    model, actual = self.native.load_legacy(self.scale, self.device)
                else:
                    model, actual = self.native.load_selected_budget(Path(meta['selected']), self.scale, self.device)
                keys = ('selected_sha256', 'checkpoint_sha256', 'N', 'step', 'decoder_sha256')
            else:
                desc = {k: meta[k] for k in ('method', 'training', 'arm', 'N', 'training_seed')}
                model, actual = self.native.load_selected(desc, self.scale, self.device)
                keys = ('selected', 'selected_sha256', 'kind', 'N', 'arm', 'training_seed',
                        'decoder_sha256', 'training_completed_step')
            if any(actual[k] != meta[k] for k in keys):
                raise ReplayMismatch('Selected model metadata differs: ' + method)
            selected_path = Path(meta['selected'] if self.study == 'CONTINUOUS_GRID' else meta['selected_file'])
            self.bind(selected_path, meta['selected_sha256'])
            selected = read(selected_path)
            registration_path = selected_path.parent/'registration.json'
            self.bind(registration_path, selected['registration_sha256'])
            training_registration = read(registration_path)
            for path, expected in training_registration['bindings'].items():
                self.bind(path, expected)
            completion_path = selected_path.parent/'completion.json'
            if completion_path.is_file():
                self.bind(completion_path)
            self.models[method] = model
        records, image_bindings = self.native.population('development')
        if identity(image_bindings) != identity(self.registration['image_bindings']):
            raise ReplayMismatch('Original image file bindings differ')
        if self._data is not None:
            # A supplied population must match the native loader, including labels.
            if identity(self._data['image_bindings']) != identity(image_bindings):
                raise ReplayMismatch('Injected data bindings differ')
            supplied = self._data['records']
            if len(supplied) != len(records):
                raise ReplayMismatch('Injected population size differs')
            for a, b in zip(supplied, records):
                if any(a[k] != b[k] for k in ('image_id', 'class_index', 'preprocessing_id')) or not np.array_equal(a['pixels'], b['pixels']):
                    raise ReplayMismatch('Injected population differs')
        for i, record in enumerate(records):
            if (record['image_id'] != self.source_ids[i]
                    or record['preprocessing_id'] != self.records[i]['preprocessing_id']
                    or hashlib.sha256(record['pixels'].tobytes()).hexdigest() != record['preprocessing_id']):
                raise ReplayMismatch('Native source order or raw pixels changed')
            record.update(source_index=i, source_id=record['image_id'])
        self.records = records
        qpath = self.root/'configs/progressive_channel.yaml'; self.bind(qpath)
        qcfg = self.native.yaml.safe_load(qpath.read_text())['quality']
        self.lp, self.dino, linear = self.native.load_quality_models(qcfg, self.device)
        for path in (qcfg['alexnet_checkpoint'], qcfg['dino_checkpoint'], linear):
            key = str(Path(path).resolve())
            expected = next((v for p, v in self.context['bindings'].items() if Path(p).resolve() == Path(path).resolve()), None)
            if expected is None:
                raise ReplayMismatch('Quality weight not present in original bindings: ' + key)
            self.bind(path, expected)
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(self.root):
                self.bind(path)
        precision = self.context['precision']
        if (torch.backends.cuda.matmul.allow_tf32 != precision['matmul_tf32']
                or torch.backends.cudnn.allow_tf32 != precision['cudnn_tf32']
                or torch.backends.cudnn.deterministic != precision['cudnn_deterministic']
                or torch.are_deterministic_algorithms_enabled() != precision['deterministic_algorithms']):
            raise ReplayMismatch('Original numerical backend differs')
        self.ready = True
        return self

    def metadata(self, row):
        method = row['method']; meta = self.model_metadata[method]
        hybrid = meta.get('kind') == 'hybrid'
        baseline = self.study == 'C_SELECTED_BASELINES'
        i = self.source_ids.index(row['source_id'])
        location = self._locations[identity(row)]
        selected = meta.get('selected')
        step = meta['step'] if self.study == 'CONTINUOUS_GRID' else selected['step']
        checkpoint_sha = meta['checkpoint_sha256'] if self.study == 'CONTINUOUS_GRID' else selected['checkpoint_sha256']
        return {'study': self.study, 'source_index': i, 'image_id': row['source_id'],
                'preprocessing_id': self.registration['sources'][row['source_id']],
                'condition': row.get('condition', 'selected_model'), 'decoder': 'Dc',
                'decoder_sha256': self.context['decoder_sha256'], 'N': meta['N'],
                'selected_step': step, 'checkpoint_sha256': checkpoint_sha,
                'training_seed': meta.get('training_seed'), 'oracle': False,
                'label_conditioned': hybrid, 'true_class_sent': hybrid,
                'classification_main_eligible': not hybrid,
                'classification_caveat': 'paid true class in original digital header' if hybrid else '',
                'reference_only': baseline, 'enhancement_removed': baseline,
                'row_id': identity({'study': self.study, **location}), 'original_location': location,
                'original_quality_model': {'dino': 'DINOv2 ViT-S/14', 'lpips': 'AlexNet v0.1'},
                'original_registration_sha256': self.registration_sha256}

    def describe(self):
        return {'study': self.study, 'sources': 100, 'rows': len(self.rows), 'methods': self.methods,
                'snrs_db': list(SNRS), 'noise_seeds': list(SEEDS), 'population': 'development',
                'original_table_sha256': self.table_sha256,
                'original_registration_sha256': self.registration_sha256,
                'baseline_role': self.study == 'C_SELECTED_BASELINES',
                'training_updates': 0, 'policy_updates': 0, 'new_holdout': False,
                'old_metric_tolerances': PARITY_TOLERANCES}

    def expected_rows(self, source_index):
        """Return untouched original rows, in the same order as iterate_source."""
        index = integer(source_index)
        if not 0 <= index < 100:
            raise IndexError(index)
        return [dict(row) for row in self.by_source[index]]

    def iterate_source(self, index):
        if not self.ready:
            raise ReplayMismatch('Call setup before real reconstruction')
        index = integer(index)
        if not 0 <= index < 100:
            raise IndexError(index)
        record = self.records[index]
        target = rgb(record['pixels'].astype(np.float32)/255)
        baseline = self.study == 'C_SELECTED_BASELINES'
        with self.torch.no_grad():
            for method in self.methods:
                meta, model = self.model_metadata[method], self.models[method]
                if baseline and meta['kind'] != 'hybrid':
                    continue
                images, infos = [], []
                for snr in SNRS:
                    for seed in SEEDS:
                        self.safe.check()
                        if self.study == 'CONTINUOUS_GRID':
                            image, info = self.native.execute(record, self.native.Cell('continuous', meta['N']),
                                snr, seed, self.vae, self.var, self.decoder, self.device, model)
                        else:
                            image, info = self.native.execute(record, model, meta, snr, seed,
                                self.vae, self.var, self.decoder, self.device)
                        image = rgb(image)
                        if baseline:
                            _channel_parity(self._main[index, method, snr, seed],
                                            {**info['ledger'], **info['rx'],
                                             'waveform_sha256': info['waveform_sha256'],
                                             'observation_sha256': info['observation_sha256']})
                            received = info['received_conditions']; event = received['phy']
                            for condition in ('B_RX', 'C_RX'):
                                if event['header_ok']:
                                    status = self.torch.tensor([[1., float(event['body_crc_ok']), model.ledger['m']]], device=self.device)
                                    base = self.native.render_received(self.decoder, received[condition], status)[0].cpu().numpy()
                                else:
                                    base = np.full_like(image, .5)
                                images.append(rgb(base)); infos.append({'header_ok': bool(event['header_ok']), 'body_crc_ok': bool(event['body_crc_ok'])})
                        else:
                            images.append(image)
                            infos.append({**info['ledger'], **info['rx'],
                                          'waveform_sha256': info['waveform_sha256'],
                                          'observation_sha256': info['observation_sha256']})
                metrics, *_ = self.native.quality_metrics(target, images, self.lp, self.dino, self.device)
                rows = ([r for r in self.by_source[index] if r['method'] == method] if baseline else
                        [self._main[index, method, snr, seed] for snr in SNRS for seed in SEEDS])
                if len(metrics) != len(rows) or len(images) != len(rows):
                    raise ReplayMismatch('Original quality batching changed')
                for original, image, quality, info in zip(rows, images, metrics, infos):
                    # Preserve original continuous-vs-C MSE rounding convention.
                    mse = (float(np.mean(np.square(image-target), dtype=np.float64))
                           if self.study == 'CONTINUOUS_GRID' else
                           float(np.square(image-target, dtype=np.float64).mean()))
                    proof = parity_check(original, {**quality, **info, 'mse': mse})
                    proof.update(row_id=self.metadata(original)['row_id'],
                                 original_location=self.metadata(original)['original_location'],
                                 rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                 target_sha256=hashlib.sha256(target.tobytes()).hexdigest())
                    yield dict(original), image, target.copy(), proof


def create_adapter(root, study, loaded=None, data=None):
    return BudgetAdapter(root, study, loaded=loaded, data=data)
