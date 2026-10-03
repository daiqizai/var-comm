"""Read exact historical float RGB archives; never run a historical main.

Admission manifest: outputs/HISTORICAL-METRICS-20261003/cached_studies.json.
Each study binds original complete receipts, unmodified CSVs, source records,
targets, and every referenced archive to an original receipt hash member.
Saved PNGs and newly inferred archive hashes are not admission evidence.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')
LIMITS = {'psnr_db': 1e-4, 'lpips_alex': 1e-5, 'dino_cosine': 1e-5}
FORMATS = ('indexed_npz', 'latent_matrix_npz', 'indexed_torch', 'direct_npy')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def require(value, message):
    if not value:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def pointer(value, path):
    require(path.startswith('/'), 'Receipt hash proof requires a JSON pointer')
    for part in path[1:].split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value

def rgb(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    require(value.dtype == np.float32 and value.ndim == 3 and value.shape[0] == 3,
            'Only native CHW float32 RGB caches are accepted')
    require(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1,
            'Cached float RGB is nonfinite or outside [0,1]')
    return np.ascontiguousarray(value)

def pixel_sha(value):
    return hashlib.sha256(rgb(value).tobytes()).hexdigest()

def parity(original, actual, columns, tolerances=None):
    limits = dict(LIMITS if tolerances is None else tolerances)
    require(set(limits) == set(METRICS), 'Three registered legacy parity tolerances required')
    delta = {}
    for key in METRICS:
        old_key = columns[key]
        require(old_key in original and key in actual, 'Legacy metric missing: ' + old_key)
        old, new = float(original[old_key]), float(actual[key])
        require(math.isfinite(old) and math.isfinite(new), 'Nonfinite legacy quality metric')
        require(0 <= float(limits[key]) <= LIMITS[key], 'Parity tolerance exceeds original cache replay limits')
        delta[key] = abs(old - new)
        require(delta[key] <= float(limits[key]), 'Cached legacy metric differs: ' + key)
    return dict(replay_parity_passed=True, synthetic=False, metric_differences=delta,
                tolerances=limits, parity_basis='receipt-bound native float RGB and fresh original metric scores',
                training_updates=0, policy_selection_updates=0)

def _module(path):
    name = '_historical_cached_quality_' + sha(path)[:16]
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == Path(path).resolve(), 'Quality module collision')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class CachedAdapter:
    def __init__(self, root, study, *, manifest=None, quality_checker=None,
                 engineering=False):
        self.root, self.study = Path(root).resolve(), study
        self.manifest_path = Path(manifest) if manifest else self.root / 'outputs/HISTORICAL-METRICS-20261003/cached_studies.json'
        document = read(self.manifest_path)
        require(document.get('status') == 'REGISTERED_ORIGINAL_FLOAT_CACHES', 'Cache admission manifest is not registered')
        self.spec = document['studies'][study]
        require(self.spec.get('synthetic') is False or engineering, 'Scientific cache manifest cannot be synthetic')
        require(self.spec.get('format') in FORMATS, 'Unsupported native float cache format')
        require(self.spec.get('training_updates') == 0 and self.spec.get('policy_selection_updates') == 0,
                'Cache supplement cannot train or select policies')
        self._checker = quality_checker
        require(quality_checker is None or engineering, 'Injected quality checker is engineering-only')
        self.bindings = {str(self.manifest_path.resolve()): sha(self.manifest_path)}
        self._proven = set()
        proof_json, proof_csv = {}, {}
        for proof in self.spec['proofs']:
            path, receipt = self.path(proof['path']), self.path(proof['receipt'])
            self.bind(receipt, proof['receipt_sha256'])
            if proof.get('kind', 'json') == 'csv':
                require(str(receipt) in self._proven, 'Cache proof CSV must itself be an original receipt-bound table')
                if str(receipt) not in proof_csv:
                    with receipt.open(newline='', encoding='utf-8-sig') as stream:
                        proof_csv[str(receipt)] = list(csv.DictReader(stream))
                proof_row = proof_csv[str(receipt)][proof['row_index']]
                member = proof_row[proof['column']]
            else:
                if str(receipt) not in proof_json:
                    proof_json[str(receipt)] = read(receipt)
                member = pointer(proof_json[str(receipt)], proof['pointer'])
            require(member == proof['sha256'],
                    'Cache/table hash is not the recorded original receipt member')
            self.bind(path, proof['sha256'])
            self._proven.add(str(path))
        # Some early evaluators recorded tensor hashes instead of container
        # hashes. Admit those exact pixels only when an original bound row
        # supplies their hash; a fresh container hash alone is insufficient.
        self._file_proven = set(self._proven)
        self._pixel_proofs = {}
        for proof in self.spec.get('pixel_proofs', []):
            path, receipt = self.path(proof['path']), self.path(proof['receipt'])
            self.bind(receipt, proof['receipt_sha256'])
            require(str(receipt) in self._proven, 'Pixel proof table lacks an original receipt binding')
            if str(receipt) not in proof_csv:
                if proof.get('receipt_format') == 'jsonl':
                    proof_csv[str(receipt)] = [json.loads(line) for line in receipt.read_text().splitlines()]
                else:
                    with receipt.open(newline='', encoding='utf-8-sig') as stream:
                        proof_csv[str(receipt)] = list(csv.DictReader(stream))
            row = proof_csv[str(receipt)][proof['row_index']]
            require(row[proof['column']] == proof['pixel_sha256'], 'Original tensor hash member differs')
            self.bind(path, proof['sha256'])
            self._pixel_proofs.setdefault(str(path), []).append(proof)
            self._proven.add(str(path))
        for path, expected in self.spec.get('evidence_bindings', {}).items():
            self.bind(self.path(path), expected)
        completion = self.path(self.spec['completion']['path'])
        self.bind(completion, self.spec['completion']['sha256'])
        require(read(completion).get('status') in self.spec['completion']['allowed_statuses'],
                'Original historical evaluation is incomplete')
        self.records = self.spec['records']
        require(len(self.records) == (self.spec.get('source_count') if engineering else 100),
                'Historical cache requires the original complete 100-source population')
        require(len({r['image_id'] for r in self.records}) == len(self.records), 'Repeated historical source ID')
        self.rows = []
        self._by_source = [[] for _ in self.records]
        self._origins = {}
        self._table_settings = {}
        selected = self.spec.get('selected_method_keys')
        if selected is not None:
            require(isinstance(selected, list) and selected and len(set(selected)) == len(selected)
                    and set(selected) == set(self.spec['methods']), 'Explicit unique selected cache methods required')
        id_to_index = {r['image_id']: i for i, r in enumerate(self.records)}
        for i, record in enumerate(self.records):
            require(int(record['source_index']) == i and 0 <= int(record['class_index']) < 1000,
                    'Historical source index/class registration differs')
            require(record.get('preprocessing_id') and len(record.get('target_rgb_sha256', '')) == 64,
                    'Original source preprocessing/float target binding required')
            self.proven_path(record['target']['path'])
        for table in self.spec['tables']:
            path = self.proven_path(table['path'])
            table_sha = self.bindings[str(path)]
            self._table_settings[str(path)] = table
            with path.open(newline='', encoding='utf-8-sig') as stream:
                iterable = (json.loads(line) for line in stream) if table.get('table_format') == 'jsonl' else csv.DictReader(stream)
                for position, row in enumerate(iterable):
                    if selected is not None and self.method_key(row) not in selected:
                        continue
                    source_id = row[self.spec['source_id_column']]
                    require(source_id in id_to_index, 'Cache table includes an unregistered source')
                    index = id_to_index[source_id]
                    if self.spec.get('source_index_column'):
                        require(int(row[self.spec['source_index_column']]) == index, 'Row/source index differs')
                    self._by_source[index].append(row)
                    self.rows.append(row)
                    self._origins.setdefault(identity(row), []).append(dict(table=str(path), table_sha256=table_sha, row_index=position))
        require(len(self.rows) == self.spec['expected_rows'], 'Complete historical row count differs')
        for index, rows in enumerate(self._by_source):
            require(len(rows) == self.spec['rows_per_source'][index], 'Historical source row coverage differs')
            for row in rows:
                self.locator(index, row)
                self.metadata(row)
                require(all(c in row for c in self.spec['metric_columns'].values()), 'Original metric columns missing')
        self._setup = False

    def path(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.root / path).resolve()

    def bind(self, path, expected):
        require(isinstance(expected, str) and len(expected) == 64, 'Invalid frozen SHA256')
        if str(path) in self.bindings:
            require(self.bindings[str(path)] == expected, 'Conflicting cache bindings')
            return
        actual = sha(path)
        require(actual == expected, 'Original cache input changed: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == expected, 'Conflicting cache bindings')
        self.bindings[str(path)] = expected

    def proven_path(self, value):
        path = self.path(value)
        require(str(path) in self._proven, 'Archive/table/target lacks original receipt hash proof: ' + str(path))
        return path

    def method_key(self, row):
        return '|'.join(str(row[k]) for k in self.spec['method_key_columns'])

    def locator(self, index, row):
        origin = self._origins.get(identity(row), [])
        overrides = self._table_settings[origin[0]['table']] if origin else {}
        settings = {**self.spec, **overrides}
        fmt = settings['format']
        if settings.get('locator_by_row_sha256'):
            locator = settings['locator_by_row_sha256'][identity(row)]
            return self.proven_path(locator['path']), locator.get('key'), locator.get('indices', [])
        if fmt == 'latent_matrix_npz':
            path = self.proven_path(self.records[index]['matrix_archive'])
            method = row[settings['method_key_columns'][0]]
            snr = float(row[settings['snr_column']])
            seed = int(row[settings['seed_column']])
            grid = [(float(s), int(n)) for s, n in settings['frame_order']]
            require((snr, seed) in grid and method in settings['method_order'], 'Matrix cache grid differs')
            return path, 'images', [settings['method_order'].index(method), grid.index((snr, seed))]
        if fmt == 'direct_npy':
            value = row[settings['archive_column']]
            if not Path(value).is_absolute():
                value = str(self.path(settings['archive_base']) / value)
            return self.proven_path(value), None, []
        slot = row[settings.get('slot_column', 'image_ref')]
        if ':' in slot:
            prefix, slot = slot.rsplit(':', 1)
            require(prefix in settings['reference_roots'], 'Unknown historical image reference prefix')
            path = self.proven_path(settings['reference_roots'][prefix].format(source_index=index))
        else:
            if settings.get('archive_column'):
                path = self.proven_path(row[settings['archive_column']])
            else:
                path = self.proven_path(settings['archive_template'].format(source_index=index))
        if settings.get('slot_kind') == 'string':
            require(bool(slot), 'Empty keyed cache image slot')
            return path, settings.get('image_key', 'images'), [slot]
        require(str(int(slot)) == str(slot).lstrip('0') or int(slot) == 0, 'Invalid cache image slot')
        require(int(slot) >= 0, 'Negative cache image slot')
        return path, settings.get('image_key', 'images'), [int(slot)]

    def metadata(self, row):
        key = self.method_key(row)
        require(key in self.spec['methods'], 'Missing historical method decoder/information metadata: ' + key)
        meta = dict(self.spec['methods'][key])
        meta.update(self.spec.get('row_metadata', {}).get(identity(row), {}))
        require(all(type(meta.get(k)) is bool for k in ('label_conditioned', 'reference_only', 'oracle')),
                'Explicit historical label/oracle/reference facets are required')
        require(meta.get('decoder') and meta.get('model_id'), 'Original decoder/model identity required')
        meta.update(method=meta.get('method') or row[self.spec['method_key_columns'][0]],
                    original_metric_columns=dict(self.spec['metric_columns']),
                    original_locations=self._origins.get(identity(row), []),
                    synthetic=False, cache_layout=self.spec['format'])
        for destination, column in self.spec.get('metadata_columns', {}).items():
            if column in row:
                meta[destination] = row[column]
        return meta

    def setup(self):
        for path, expected in self.bindings.items():
            require(sha(path) == expected, 'Frozen cache binding changed before setup')
        if self._checker is None:
            import torch
            q = self.spec['quality']
            if 'numerical_runtime' in q:
                from history_common import set_numerical_runtime
                set_numerical_runtime(torch, q['numerical_runtime'])
            module_path = self.path(q['module_path'])
            require(str(module_path) in self.bindings, 'Native legacy quality source is not bound')
            paths = {k: str(self.path(v)) for k, v in q['paths'].items()}
            for key in ('alexnet_checkpoint', 'dino_checkpoint'):
                require(paths[key] in self.bindings, 'Legacy metric weights not bound')
            module = _module(module_path)
            self.device = torch.device('cuda:0')
            self.lpips, self.dino, linear = module.load_quality_models(paths, self.device)
            require(sha(linear) == q['lpips_linear_sha256'], 'LPIPS linear weights differ from historical registration')
            self.bindings[str(Path(linear).resolve())] = q['lpips_linear_sha256']
            if q.get('batch_size', 8) == 8:
                self._checker = lambda target, images: module.quality_metrics(target, images, self.lpips, self.dino, self.device)[0]
            else:
                # Calling the original quality routine on each original-sized
                # chunk preserves the historic forward batch size.
                size = int(q['batch_size'])
                self._checker = lambda target, images: [row for start in range(0, len(images), size)
                    for row in module.quality_metrics(target, images[start:start+size], self.lpips, self.dino, self.device)[0]]
        self._setup = True
        return self

    def expected_rows(self, index):
        return [dict(row) for row in self._by_source[index]]

    def load_array(self, path, key, slots, cache):
        token = (str(path), key)
        if token not in cache:
            require(sha(path) == self.bindings[str(path)], 'Archive changed after registration')
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    cache[token] = data[key].copy()
            elif path.suffix == '.npy' and key is None:
                cache[token] = np.load(path, allow_pickle=False)
            elif path.suffix == '.pt':
                import torch
                data = torch.load(path, map_location='cpu', weights_only=True)
                cache[token] = data[key]
            else:
                raise RuntimeError('PNG or unregistered cache container cannot replace float RGB')
        value = cache[token]
        require(str(path) in self._file_proven or any(
            p.get('key') == key and p.get('indices', []) == slots
            for p in self._pixel_proofs.get(str(path), [])),
            'Unbound pixels within an admitted archive')
        # Verify original pixel proofs before any CHW normalization. Typed
        # hashes include the original dtype and shape (often a batch axis).
        for proof in self._pixel_proofs.get(str(path), []):
            if proof.get('key') != key or proof.get('indices', []) != slots:
                continue
            candidate = value
            for slot in slots:
                candidate = candidate[slot]
            array = candidate.detach().cpu().numpy() if hasattr(candidate, 'detach') else np.asarray(candidate)
            digest = hashlib.sha256()
            if proof.get('hash_kind') == 'typed_tensor':
                digest.update(proof['dtype'].encode())
                digest.update(json.dumps(proof['shape']).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
            require(digest.hexdigest() == proof['pixel_sha256'], 'Original pixel proof failed')
        for slot in slots:
            value = value[slot]
        return rgb(value)

    def describe(self):
        return dict(study=self.study, rows=len(self.rows), sources=len(self.records),
                    methods=len(self.spec['methods']), native_float_cache=True,
                    training_updates=0, policy_selection_updates=0)

    def _load_pixels(self, index, row, cache=None):
        path, key, slots = self.locator(index, row)
        image = self.load_array(path, key, slots, {} if cache is None else cache)
        col = self.spec.get('image_hash_column')
        if col and row.get(col):
            expected = self.bindings[str(path)] if self.spec.get('image_hash_kind') == 'file_sha256' else pixel_sha(image)
            require(row[col] == expected, 'Per-row original image bytes differ')
        return image

    def _load_target(self, index, cache):
        record = self.records[index]
        t = record['target']
        target = self.load_array(self.proven_path(t['path']), t['key'], t.get('indices', []), cache)
        if t.get('transform'):
            require(t['transform']=='original_VAR_uint8_normalization_roundtrip', 'Unknown target preprocessing')
            target=np.rint(target*np.float32(255)).astype(np.uint8).astype(np.float32)
            target=(((target/np.float32(127.5))-np.float32(1))+np.float32(1))*np.float32(.5)
        require(pixel_sha(target) == record['target_rgb_sha256'], 'Original target/preprocessing pixels differ')
        return target

    def load_image(self, index, row, cache=None):
        """CPU-only pixel proof; caller must still check its own original metrics.

        This entry never labels metric parity as passed. It permits exact policy
        aliases to reuse a sealed candidate image and recompute their own scores.
        """
        cache = {} if cache is None else cache
        image, target = self._load_pixels(index, row, cache), self._load_target(index, cache)
        path, key, slots = self.locator(index, row)
        proof = dict(cached_pixels_verified=True, synthetic=False,
            original_candidate_row_sha256=identity(row), image_sha256=pixel_sha(image),
            target_sha256=self.records[index]['target_rgb_sha256'], archive=str(path),
            archive_sha256=self.bindings[str(path)], array_key=key, indices=slots,
            frozen_manifest_sha256=self.bindings[str(self.manifest_path.resolve())])
        return image, target, proof

    def iterate_source(self, index):
        require(self._setup, 'Historical cache adapter.setup() must precede scoring')
        record, cache = self.records[index], {}
        target = self._load_target(index, cache)
        images, locators, rows = [], [], self._by_source[index]
        if self.spec['format'] == 'latent_matrix_npz':
            path = self.proven_path(record['matrix_archive'])
            with np.load(path, allow_pickle=False) as a:
                require(a['method_order'].tolist() == self.spec['method_order'], 'Native matrix method order differs')
        seen, unique, positions = {}, [], []
        for row in rows:
            path, key, slots = self.locator(index, row)
            image = self._load_pixels(index, row, cache)
            digest = pixel_sha(image)
            if digest not in seen:
                seen[digest] = len(unique)
                unique.append(image)
            positions.append(seen[digest])
            images.append(image)
            locators.append(dict(archive=str(path), archive_sha256=self.bindings[str(path)], array_key=key, indices=slots))
        scored = self._checker(target, unique)
        require(len(scored) == len(unique), 'Legacy quality checker omitted an image')
        for row, image, position, location in zip(rows, images, positions, locators):
            checked = parity(row, scored[position], self.spec['metric_columns'], self.spec.get('tolerances'))
            checked.update(original_row_sha256=identity(row), image_sha256=pixel_sha(image),
                           target_sha256=record['target_rgb_sha256'], original_cache_location=location,
                           original_locations=self._origins[identity(row)],
                           frozen_manifest_sha256=self.bindings[str(self.manifest_path.resolve())])
            yield dict(row), image, target, checked

def create_adapter(root, study, loaded=None, data=None):
    require(loaded is None and data is None, 'Historical cache adapter loads only registered original inputs')
    return CachedAdapter(root, study)
