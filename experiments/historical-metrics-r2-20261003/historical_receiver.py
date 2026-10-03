"""Read-only native replay of completed Step 1 v3 and Step 2 receivers.

Construction validates historical receipts and row inventories without importing
Torch. setup/iterate_source load frozen models and execute inference only. No old
main, calibration, training, output writer or launcher is called. The caller owns
resource admission. RGB is the actual decoder float output, never a saved PNG.
"""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import signal
import sys
import types

STUDIES = ('RX_STEP2_A', 'RX_STEP2_B', 'RX_STEP1_V3_A', 'RX_STEP1_V3_B')
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')
TOLERANCES = {'psnr_db': 1e-5, 'lpips_alex': 2e-6,
              'dino_cosine': 2e-6, 'dino_mismatched': 2e-6}
PREFIX = '/home/liulu/projects/VAR_COMM/'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parity(original, actual, tolerances=None):
    """Historical numerical score parity, not a claim of archived pixel equality."""
    limits = tolerances or TOLERANCES
    differences = {}
    for key in METRICS + ('dino_mismatched',):
        if key not in original:
            continue
        require(key in actual, 'missing replay metric ' + key)
        a, b = float(original[key]), float(actual[key])
        require(math.isfinite(a) and math.isfinite(b), 'nonfinite metric ' + key)
        differences[key] = abs(a - b)
        require(differences[key] <= limits[key],
                'historical metric parity failed: ' + key + ' ' + str(differences[key]))
    require(all(k in differences for k in METRICS), 'incomplete image parity metrics')
    return dict(parity_passed=True, metric_differences=differences,
                tolerances=dict(limits), original_pixel_tensor_available=False,
                parity_basis='native float RGB + original per-frame score parity',
                synthetic=False, training_updates=0, policy_selection_updates=0)


def image_branches(cell, cell_path, cell_sha, study):
    """Flatten only existing image branches; retain exact nested identity."""
    output = []
    for j, row in enumerate(cell['rows']):
        if study != 'RX_STEP1_V3_B':
            output.append(dict(row))
            continue
        if row.get('mode') != 'CL':
            continue
        for branch, suffix, latent_key in (
                ('token_image', 'tok', 'latent_squared_error'),
                ('fused_image', 'fuse', 'fused_latent_squared_error')):
            require(branch in row, 'CL row lacks image branch')
            flat = {k: v for k, v in row.items()
                    if k not in ('token_image', 'fused_image')}
            flat.update(row[branch])
            flat.update(method=row['profile'] + '_' + suffix,
                        latent_sq_err_final=row[latent_key],
                        original_cell_path=str(cell_path), original_cell_sha256=cell_sha,
                        original_nested_row_index=j, original_image_branch=branch,
                        original_nested_row_sha256=canonical_hash(row))
            output.append(flat)
    if study == 'RX_STEP1_V3_A':
        # The original source-only reference was published inside the same cell.
        ref = dict(cell['source_A_Dc_true_Fq'])
        ref.update(method='Source_A_Dc_true_Fq', source_id=cell['source_id'],
                   source_index=cell['source_index'], noise_seed=0,
                   population='original_development_100',
                   preprocessing_sha256=cell['source_identity']['rgb_sha256'],
                   original_cell_path=str(cell_path), original_cell_sha256=cell_sha,
                   original_image_branch='source_A_Dc_true_Fq',
                   original_nested_row_sha256=canonical_hash(cell['source_A_Dc_true_Fq']))
        output.append(ref)
    return output


def _csv_value(value):
    # This is exactly the old csv.DictWriter spelling, including bools/empty.
    return '' if value is None else str(value)


def validate_csv_rows(cell_rows, csv_rows):
    require(len(cell_rows) == len(csv_rows), 'cell/CSV row count differs')
    for a, b in zip(cell_rows, csv_rows):
        converted = {('lambda' if k == 'lambda_' else k): _csv_value(v)
                     for k, v in a.items()}
        require(converted == b, 'cell differs from sealed original CSV row')


class _Base:
    def __init__(self, root, study, loaded=None, data=None):
        self.root = Path(root).resolve()
        self.study = study
        self.loaded = loaded
        self.data = data
        self.bindings = {}
        self.rows = []
        self._source_rows = {}
        self._aliases = {}
        self._ready = False
        self._records = None

    def _path(self, path):
        text = str(path)
        return self.root / text[len(PREFIX):] if text.startswith(PREFIX) else Path(path)

    def _bind(self, path, expected=None):
        path = self._path(path)
        require(path.is_file(), 'required historical input missing: ' + str(path))
        actual = sha(path)
        require(expected is None or actual == expected, 'historical SHA differs: ' + str(path))
        old = self.bindings.get(str(path))
        require(old is None or old == actual, 'bound file changed: ' + str(path))
        self.bindings[str(path)] = actual
        return path

    def _json(self, path, expected=None):
        return json.loads(self._bind(path, expected).read_text(encoding='utf-8'))

    def _source_code(self, names, registered=None):
        registered = registered or {}
        for path in names:
            key = PREFIX + Path(path).relative_to(self.root).as_posix()
            self._bind(path, registered.get(key))

    def _bind_weight_metadata(self, value):
        if isinstance(value, list):
            for v in value:
                self._bind_weight_metadata(v)
        elif isinstance(value, dict):
            if 'checkpoint' in value and 'checkpoint_sha256' in value:
                self._bind(value['checkpoint'], value['checkpoint_sha256'])
            for key in ('files', 'model_files'):
                if key in value and isinstance(value[key], dict):
                    for p, h in value[key].items():
                        if isinstance(h, str) and len(h) == 64 and str(p).endswith(('.pt', '.pth')):
                            self._bind(p, h)
            for key in ('dino_checkpoint', 'alexnet_checkpoint'):
                if key in value:
                    self._bind(value[key], value[key + '_sha256'])
            for v in value.values():
                self._bind_weight_metadata(v)

    def _bind_population_files(self, identities):
        from latent_enhancement_eval import runner
        self._bind(runner.TOKENS_PATH)
        self._bind(runner.DIGITAL_ROOT / 'population.json')
        for i, identity in enumerate(identities):
            p = runner.SOURCE_RECON_ROOT / f'images/{i:03d}/reconstructions.npz'
            self._bind(p, identity['source_npz_sha256'])

    def expected_rows(self, source_index):
        # Return fresh dictionaries; no adapter can mutate its sealed inventory.
        return [dict(r) for r in self._source_rows[int(source_index)]]

    def describe(self):
        return dict(study=self.study, sources=len(self._source_rows), rows=len(self.rows),
                    methods=sorted({r['method'] for r in self.rows}),
                    source_indexes=sorted(self._source_rows),
                    training_updates=0, policy_selection_updates=0,
                    original_scope_preserved=True, synthetic=False,
                    diagnostics_not_rendered=self.diagnostics_not_rendered,
                    model_identity=self.model_identity)

    @property
    def records(self):
        if self._records is None:
            raise RuntimeError('setup() must load and validate historical source records first')
        return self._records

    @contextmanager
    def _context(self):
        """Confine generic historical import aliases and launcher import effects."""
        keys = ('run_preflight', 'probe', 'rx_v3_common', 'rx_v3_b_helpers',
                'receiver', 'train', 'protocol', 'qualification')
        before = {k: sys.modules.get(k) for k in keys}
        path_before = list(sys.path)
        env_before = {k: os.environ.get(k) for k in
                      ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'CUBLAS_WORKSPACE_CONFIG', 'VAR_COMM_DECODER_GATE')}
        signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        sys.path[:0] = [str(p) for p in self._python_paths()]
        for k, v in self._aliases.items():
            sys.modules[k] = v
        # run_preflight's only role on import was environment/path mutation. Its
        # delivery_chain.CHAIN assignment is deliberately not executed by replay.
        sys.modules['run_preflight'] = types.ModuleType('historical_inert_preflight')
        try:
            yield
        finally:
            for k, v in before.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            sys.path[:] = path_before
            for k, v in env_before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for s, v in signals.items():
                signal.signal(s, v)

    def _python_paths(self):
        e = self.root / 'experiments/var-latent-enhancement-20260917'
        return [self.here, self.oldhere, self.root / 'src',
                *[e / s for s in ('src', 'phase_b/src', 'evaluation/src',
                                  'followup/src', 'research/src', 'mechanisms/src')],
                self.root / 'experiments/var-short-prefix-hybrid-20260923/src',
                self.root / 'experiments/token_channel_efficiency_20260923/src']

    def _module(self, name, path, alias=None):
        spec = importlib.util.spec_from_file_location('_historical_' + self.study + '_' + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if alias:
            self._aliases[alias] = module
            sys.modules[alias] = module
        return module

    def _common(self):
        self.probe = self._module('probe', self.oldhere / 'probe.py', 'probe')
        self.old = self._module('common', self.oldhere / 'rx_v3_common.py', 'rx_v3_common')

    def _validate_records(self, records, identities):
        require(len(records) == 100 and len(identities) == 100, 'historical sources must be exactly 100')
        import numpy as np
        normalized = []
        for i, (rec, expected) in enumerate(zip(records, identities)):
            sid = rec.get('image_id', rec.get('target', {}).get('image_id'))
            require(sid == expected['source_id'], 'source ordering changed')
            pixels = np.asarray(rec['pixels'])
            require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256), 'source pixel domain changed')
            for key in ('rgb_sha256', 'source_npz_sha256'):
                if key in expected:
                    require(rec.get(key) == expected[key], 'source binding differs: ' + key)
            prep = rec.get('preprocessing_id', rec.get('rgb_sha256'))
            if 'preprocessing_id' in expected:
                require(prep == expected['preprocessing_id'], 'preprocessing differs')
            class_index = rec.get('class_index', rec.get('target', {}).get('class_index'))
            require(class_index is not None, 'missing true evaluation class index')
            normalized.append(dict(source_id=sid, source_index=i, image_id=sid,
                                   preprocessing_id=prep, class_index=int(class_index),
                                   true_class_index=int(class_index), pixels=pixels,
                                   native_record=rec))
        self._records = normalized

    def _before_source(self, index):
        require(self._ready, 'setup() must precede replay')
        # All cells were verified during inventory construction. Recheck this
        # source plus shared inputs here rather than rehashing 100 large cells
        # for each source. The final runner must bind the complete input map.
        for p, h in self.bindings.items():
            if ('_cells/' in Path(p).as_posix() or '/source_' in Path(p).as_posix()) and \
                    Path(p).stem not in (f'{int(index):03d}', f'source_{int(index):03d}'):
                continue
            require(sha(p) == h, 'historical input changed after setup: ' + p)
        self.old.b.boundary()

    def _result(self, row, image, target, actual, **extra):
        import numpy as np
        image = np.asarray(image, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        require(image.shape == target.shape == (3, 256, 256), 'RGB shape changed')
        require(np.isfinite(image).all() and np.isfinite(target).all(), 'nonfinite RGB')
        evidence = parity(row, actual)
        latent_actual = extra.pop('latent_actual', None)
        if latent_actual is not None:
            key = ('latent_sq_err_final' if 'latent_sq_err_final' in row else 'latent_squared_error')
            if key in row:
                wanted = float(row[key])
                require(math.isfinite(float(latent_actual)) and
                        math.isclose(wanted, float(latent_actual), rel_tol=1e-6, abs_tol=1e-5),
                        'historical latent parity failed')
                evidence.update(latent_metric=key, latent_metric_difference=abs(wanted - float(latent_actual)),
                                latent_parity_passed=True)
        evidence.update(original_row_sha256=canonical_hash(row),
                        native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        native_target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                        **extra)
        return dict(row), image, target, evidence


class Step2Adapter(_Base):
    def __init__(self, root, study, loaded=None, data=None):
        super().__init__(root, study, loaded, data)
        self.arm = 'A' if study == 'RX_STEP2_A' else 'B'
        self.here = self.root / ('experiments/rx-posterior-step2-' + self.arm + '-20260930')
        self.oldhere = self.root / 'experiments/rx-posterior-step1-20260929'
        self.out = self.root / ('outputs/RX-POSTERIOR-STEP2-' + self.arm + '-20260930-R1')
        if self.arm == 'B':
            self.out /= 'evaluation'
        self.result = self.root / ('results/rx_posterior_step2_' + self.arm + '_20260930_R1')
        self.cfg = self._json(self.result / 'config.json')
        self.policy = self._json(self.result / 'selected_policy.json')
        done = self._json(self.out / 'completion.json')
        cal = self._json(self.out / 'calibration_complete.json')
        self.dev_identity = self._json(self.out / 'development_identity.json')
        require(done['status'] in ('A_COMPLETE_NO_TRAINING_AWAIT_USER_DECISION',
                                  'B_SELECTED_P_LOW_EVALUATION_COMPLETE'), 'Step2 is not complete')
        for receipt in (done, cal, self.dev_identity):
            require(receipt['config_sha256'] == sha(self.result / 'config.json'), 'config binding differs')
            require(receipt['policy_sha256'] == sha(self.result / 'selected_policy.json'), 'policy binding differs')
        require(self.policy['development_read'] is False, 'policy was not sealed before development')
        require(done.get('training_updates', done.get('evaluation_training_updates')) == 0
                and done.get('frozen_weights_unchanged') is True,
                'evaluation changed weights')
        self.snrs = self.cfg['snrs']
        self.seeds = self.cfg['development_seeds']
        require(self.snrs == ([-5, -2, 1, 4, 7, 13] if self.arm == 'A' else [-5, -2, 1, 4, 13]),
                'original Step2 SNR scope differs')
        require(self.seeds == [2001, 2002, 2003], 'original noise seeds changed')
        self.model_identity = self.cfg['identity']
        self.base = 'B2' if self.arm == 'A' else 'P_low'
        self.native_base = 'P4084' if self.arm == 'A' else 'P_low'
        self.diagnostics_not_rendered = ['TF token scores', 'CL per-scale token scores']
        table = self._table('per_frame.csv')
        table_groups = {i: [] for i in range(100)}
        for row in table:
            table_groups[int(row['source_index'])].append(row)
        self.cells = {}
        for i in range(100):
            cell = self._json(self.out / 'development_cells' / f'{i:03d}.json')
            require(cell['identity'] == self.dev_identity, 'development cell identity differs')
            validate_csv_rows(cell['rows'], table_groups[i])
            require(len(cell['rows']) == len(self.snrs) * 3 * 14, 'incomplete 14-output source grid')
            require(all(r['source_index'] == i for r in cell['rows']), 'source index differs')
            methods = {self.base, 'V_lambda1'} | {p + '_' + s for p in ('A1', 'A2', 'V')
                       for s in ('policy', 'raw_fused', 'common', 'tok')}
            require({(r['method'], r['snr_db'], r['noise_seed']) for r in cell['rows']} ==
                    {(m, s, seed) for m in methods for s in self.snrs for seed in self.seeds},
                    'Step2 method/SNR/noise coverage differs')
            self.cells[i] = cell
            self._source_rows[i] = [dict(r) for r in cell['rows']]
            self.rows.extend(self._source_rows[i])
        require(len(self.rows) == done['files'], 'historical completion count differs')
        self._source_code([self.here / p for p in
                           (('run.py', 'receiver.py') if self.arm == 'A' else
                            ('evaluate.py', 'receiver.py', 'train.py', 'protocol.py', 'qualification.py'))]
                          + [self.oldhere / p for p in ('probe.py', 'rx_v3_common.py')]
                          + [self.root / p for p in ('src/var_comm/quality.py', 'src/var_comm/next_scale_prior.py',
                             'experiments/token_channel_efficiency_20260923/src/token_efficiency/execution.py',
                             'experiments/token_channel_efficiency_20260923/src/token_efficiency/C_evaluate.py',
                             'experiments/var-latent-enhancement-20260917/research/src/latent_research/models.py')],
                          self.cfg.get('source_bindings'))

    def _table(self, name):
        manifest_path = self.result / 'table_shards/manifest.json'
        manifest = self._json(manifest_path) if manifest_path.exists() else None
        registered = manifest['tables'].get(name) if manifest else None
        p = self.result / name
        if p.exists():
            self._bind(p, registered['sha256'] if registered else None)
            raw = p.read_bytes()
        else:
            require(registered is not None, 'original CSV and archived manifest missing')
            header = None
            chunks = []
            for part in registered['parts']:
                q = self._bind(self.result / part['path'], part['sha256'])
                chunk = q.read_bytes()
                head, _, body = chunk.partition(b'\n')
                head += b'\n'
                require(header is None or head == header, 'shard CSV headers differ')
                header = head
                chunks.append(body)
            raw = header + b''.join(chunks)
            require(hashlib.sha256(raw).hexdigest() == registered['sha256'], 'restored CSV SHA differs')
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'), newline='')))
        require(registered is None or len(rows) == registered['rows'], 'CSV count differs')
        return rows

    def setup(self):
        if self._ready:
            return self
        with self._context():
            self._common()
            self.rx = self._module('receiver', self.here / 'receiver.py', 'receiver')
            if self.arm == 'B':
                self._module('protocol', self.here / 'protocol.py', 'protocol')
                self._module('qualification', self.here / 'qualification.py', 'qualification')
                self._module('train', self.here / 'train.py', 'train')
            self.engine = self._module('engine', self.here / ('run.py' if self.arm == 'A' else 'evaluate.py'))
            if self.loaded is None:
                self.models, self.stats, self.static, identity = self.engine.setup()
            else:
                self.models, self.stats, self.static, identity = (
                    self.loaded[k] for k in ('models', 'stats', 'static', 'identity'))
            require(identity == self.model_identity, 'loaded Step2 model identity differs')
            self._bind_weight_metadata(identity)
            models_receipt = self._json(self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/models.json')
            self._bind_weight_metadata(models_receipt)
            self._bind(self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/calibration_statistics.pt',
                       identity['stats_sha256'])
            records, population_bindings = self.engine.population('development') if self.data is None else self.data
            require(population_bindings == self.dev_identity['bindings'], 'source population bindings differ')
            self._bind_population_files(population_bindings)
            expected = [dict(source_id=sid, preprocessing_id=prep)
                        for sid, prep in zip(self.dev_identity['source_ids'], self.dev_identity['preprocessing_ids'])]
            self._validate_records(records, expected)
            import torch
            for model in self.models.values():
                require(not model.training and not any(p.requires_grad for p in model.parameters()),
                        'historical models must remain frozen')
            with torch.no_grad():
                self.references = torch.stack([
                    self.engine.dino_features(self.models['dino'],
                        (torch.tensor(r['pixels'][None], device='cuda', dtype=torch.float32) / 127.5 - 1 + 1) / 2)[0]
                    for r in records])
            self.permutation = torch.tensor(self.cfg['mismatch_permutation'], dtype=torch.long)
            self._ready = True
        return self

    def metadata(self, row):
        snr = row['snr_db']
        return dict(study=self.study, scope='historical_development', decoder_id='Dc',
                    label_conditioned=False, class_condition='U', oracle=False,
                    output_role='main' if row['method'] in (self.base, 'A1_policy', 'A2_policy', 'V_policy') else 'diagnostic',
                    is_main_conclusion=row['method'] in (self.base, 'A1_policy', 'A2_policy', 'V_policy'),
                    model_identity=self.model_identity, original_policy_sha256=sha(self.result / 'selected_policy.json'),
                    original_metric_columns={k: k for k in METRICS + ('dino_mismatched',)},
                    snr_field='snr_db',
                    snr_role=('out_of_original_training_range' if self.arm == 'A' and snr < 0 else
                              'low_snr_main' if self.arm == 'B' and snr in (-5, -2, 1) else
                              'support' if self.arm == 'B' and snr == 4 else
                              'side_effect' if self.arm == 'B' and snr == 13 else 'original_main'),
                    training_updates=0, policy_selection_updates=0)

    def iterate_source(self, index):
        import torch
        with self._context(), torch.no_grad():
            self._before_source(index)
            index = int(index)
            rec = self.records[index]
            records = [r['native_record'] for r in self.records]
            target = rec['pixels'].astype('float32') / 255.
            vae, var, p = (self.models[k] for k in ('vae', 'var', self.native_base))
            f = vae.quant_conv(vae.encoder(torch.tensor(rec['pixels'][None], device='cuda', dtype=torch.float32) / 127.5 - 1))
            wave = p.transmit(f)[0].cpu().numpy()
            require(wave.shape == (4084, 2), 'continuous waveform budget changed')
            emitted = []
            for snr in self.snrs:
                self.old.b.boundary()
                level = self.policy['levels'][str(snr)]
                observations = []
                original_rows = [r for r in self._source_rows[index] if r['snr_db'] == snr]
                for seed in self.seeds:
                    y = self.engine.apply_channel(wave, snr, rec['source_id'], seed, self.engine.CELL)
                    same = [r for r in original_rows if r['noise_seed'] == seed]
                    require(all(r['waveform_sha256'] == self.engine.waveform_sha(wave)
                                and r['observation_sha256'] == self.engine.waveform_sha(y) for r in same),
                            'actual waveform/noise differs from original')
                    observations.append(p.receive(torch.tensor(y[None], device='cuda'),
                        torch.tensor([snr], dtype=torch.float32, device='cuda'))[0].cpu())
                z = torch.stack(observations)
                estimates = {self.base: z}
                candidates = {}
                for prior in ('A1', 'A2', 'V'):
                    selected = level['methods'][prior]
                    lam = selected['raw_selected_lambda']
                    key = self.engine.candidate_key(prior, lam)
                    if key not in candidates:
                        candidates[key] = self.rx.infer(vae, var, z.cuda(), prior,
                            level['variance_by_scale'], self.static, lam=lam, details=False)['fhat'].cpu()
                    q = candidates[key]
                    alpha = torch.tensor(selected['alpha'], dtype=torch.float32).reshape(1, 32, 1, 1)
                    common = torch.tensor(level['alpha_common'], dtype=torch.float32).reshape(1, 32, 1, 1)
                    estimates.update({prior + '_policy': z if selected['policy_action'] == 'BYPASS' else self.rx.fusion(z, q, alpha),
                                      prior + '_raw_fused': self.rx.fusion(z, q, alpha),
                                      prior + '_common': self.rx.fusion(z, q, common), prior + '_tok': q})
                key = 'V_lambda1.0'
                if key not in candidates:
                    candidates[key] = self.rx.infer(vae, var, z.cuda(), 'V', level['variance_by_scale'],
                        self.static, lam=1., details=False)['fhat'].cpu()
                estimates['V_lambda1'] = self.rx.fusion(z, candidates[key],
                    torch.tensor(level['V_lambda1']['alpha'], dtype=torch.float32).reshape(1, 32, 1, 1))
                rendered = {}
                ii = torch.tensor([index] * 3)
                for name, value in estimates.items():
                    identical = next((other for other in rendered if torch.equal(value, estimates[other])), None)
                    if identical is None:
                        rendered[name] = self.engine.images_metrics(value, ii, records, self.models,
                            self.references, self.permutation, save_images=True)
                    else:
                        rendered[name] = rendered[identical]
                for row in original_rows:
                    j = self.seeds.index(row['noise_seed'])
                    actual, images = rendered[row['method']]
                    emitted.append(self._result(row, images[j].numpy(), target, actual[j],
                        decoder_batch_size=3, observation_hash_verified=True,
                        latent_actual=float((f[0].cpu().double() - estimates[row['method']][j].double()).square().sum()),
                        true_bypass_verified=(row['policy_action'] == 'BYPASS' and torch.equal(estimates[row['method']], z))))
            require([r[0] for r in emitted] == self.expected_rows(index), 'replay row order differs')
            require(self.engine.dump_hashes(self.models) == self.model_identity['models'], 'frozen weights changed')
            yield from emitted


class Step1Adapter(_Base):
    def __init__(self, root, study, loaded=None, data=None):
        super().__init__(root, study, loaded, data)
        self.arm = 'A' if study == 'RX_STEP1_V3_A' else 'B'
        self.here = self.oldhere = self.root / 'experiments/rx-posterior-step1-20260929'
        self.out = self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/revision_v3'
        self.result = self.root / ('results/rx_posterior_step1_20260929/revision_v3_' + self.arm)
        self.done = self._receipt(self.arm + '_completion.json')
        require(self.done['status'] == ('REAL_ORACLE_AND_LMMSE_TASK_A_COMPLETE' if self.arm == 'A'
                                       else 'REAL_V3_B_DEVELOPMENT_COMPLETE'), 'Step1 not complete')
        require(self.done['synthetic'] is False and self.done['sources'] == 100, 'not complete real original population')
        registration_name = 'A_registration.json' if self.arm == 'A' else 'B_evaluation_registration.json'
        registration = self._receipt(registration_name)
        require(sha(self.receipt_paths[registration_name]) == self.done['registration_sha256'],
                'Step1 original registration SHA differs')
        self.cfg = self._json(self.here / 'design_v3.json')
        design_key = PREFIX + (self.here / 'design_v3.json').relative_to(self.root).as_posix()
        self._bind(self.here / 'design_v3.json', registration['bindings'].get(design_key))
        self.models_receipt = self._json(self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/models.json')
        self.quality_identity = self._receipt('quality_identity.json', fallback_arm='A')
        self.population = self._receipt('A_population.json', fallback_arm='A')['records']
        self.model_identity = dict(vae=self.models_receipt['vae_state_sha256'],
                                   var=self.models_receipt['var_state_sha256'],
                                   quality=self.quality_identity)
        self.frozen = None
        if self.arm == 'B':
            self.frozen = self._receipt('B_frozen_config.json')
            cal = self._receipt('B_calibration_completion.json')
            frozen_path = self.receipt_paths['B_frozen_config.json']
            require(cal['frozen_config_sha256'] == sha(frozen_path), 'Step1 calibration seal differs')
            require(self.done['frozen_config_sha256'] == sha(frozen_path), 'Step1 completion policy differs')
            require(self.frozen['development_read'] is False, 'Step1 development tuning')
            require(self.frozen['algorithm_sha256'] == sha(self.here / 'B_algorithm_v3.json'), 'Step1 algorithm differs')
            self._bind(self.here / 'B_algorithm_v3.json')
        self.cells = {}
        archive = None
        if self.arm == 'B' and (self.result / 'source_archive_manifest.json').exists():
            archive = self._json(self.result / 'source_archive_manifest.json')
            self.archive_helper = self._module('archive', self.here / 'rx_v3_archive.py')
            self._bind(self.here / 'rx_v3_archive.py')
            archive_index = self._json(self.result / 'index.json')['files']
        raw_count = 0
        for i in range(100):
            original = self.out / (self.arm + '_cells') / f'{i:03d}.json'
            published = self.result / f'source_{i:03d}.json'
            expected = self.done['cells'][PREFIX + original.relative_to(self.root).as_posix()]
            path = original if original.exists() else published
            if not original.exists() and self.arm == 'B':
                require(archive is not None, 'missing original and archived B source cell')
                entry = archive[i]
                require(entry['file'] == published.name and entry['original_sha256'] == expected,
                        'packed B original identity differs')
                cell = self.archive_helper.unpack(self._json(published, archive_index[published.name]['sha256']))
                restored = self.archive_helper.original_bytes(cell)
                require(len(restored) == entry['original_bytes'] and
                        hashlib.sha256(restored).hexdigest() == expected, 'lossless B unpack failed')
            else:
                cell = self._json(path, expected)
            require(cell['synthetic'] is False and cell['source_index'] == i, 'Step1 cell identity differs')
            require(cell['registration_sha256'] == self.done['registration_sha256'], 'Step1 registration differs')
            require(cell['source_identity'] == self.population[i], 'Step1 source differs')
            if self.arm == 'A':
                require({(r['method'], r['snr_equiv_db'], r['noise_seed']) for r in cell['rows']} ==
                        {(m, s, seed) for m in ('B1', 'O1') for s in self.cfg['oracle_snrs_db']
                         for seed in self.cfg['development_noise_seeds']} and len(cell['rows']) ==
                        2 * len(self.cfg['oracle_snrs_db']) * len(self.cfg['development_noise_seeds']),
                        'Step1 A original grid differs')
            else:
                profiles = ('A1_decision', 'A2_decision', 'A2_fixed', 'V_decision', 'V_fixed')
                required = {(level['name'], seed, mode, profile) for level in self.frozen['levels']
                            for seed in self.frozen['algorithm']['development_noise_seeds']
                            for mode in (('TF', 'CL') if level['role'] == 'decision' else ('TF',))
                            for profile in profiles}
                require({(r['level'], r['noise_seed'], r['mode'], r['profile']) for r in cell['rows']} ==
                        required and len(cell['rows']) == len(required), 'Step1 B original diagnostic grid differs')
                require(cell['frozen_config_sha256'] == sha(self.receipt_paths['B_frozen_config.json']),
                        'Step1 B cell policy differs')
            self.cells[i] = cell
            rows = image_branches(cell, path, expected, study)
            self._source_rows[i] = rows
            self.rows.extend(rows)
            raw_count += len(cell['rows'])
        require(raw_count == self.done['rows'], 'Step1 raw diagnostic count differs')
        self.diagnostics_not_rendered = (['TF', 'probability_only', 'extra 30dB TF gate'] if self.arm == 'B' else [])
        self._source_code([self.here / p for p in ('probe.py', 'rx_v3_common.py', 'rx_v3_b_helpers.py')]
                          + [self.root / p for p in ('src/var_comm/quality.py', 'src/var_comm/next_scale_prior.py')],
                          registration['bindings'])

    def _receipt(self, name, fallback_arm=None):
        if not hasattr(self, 'receipt_paths'):
            self.receipt_paths = {}
        path = self.out / name
        if not path.exists():
            result = self.result if fallback_arm is None else self.result.parent / ('revision_v3_' + fallback_arm)
            path = result / name
            index = self._json(result / 'index.json')
            expected = index['files'][name]['sha256']
        else:
            seal = Path(str(path) + '.sha256')
            expected = seal.read_text().strip() if seal.exists() else None
            if seal.exists():
                self._bind(seal)
        self.receipt_paths[name] = path
        return self._json(path, expected)

    def setup(self):
        if self._ready:
            return self
        with self._context():
            self._common()
            self.helper = self._module('helpers', self.here / 'rx_v3_b_helpers.py', 'rx_v3_b_helpers')
            import torch
            if self.loaded is None:
                vae, var, self.stats, self.static = self.old.start()
                dec = self.old.b.load_decoder(vae, torch.device('cuda:0'))
                lp, dino, quality = self.old.load_quality()
                self.models = dict(vae=vae, var=var, decoder=dec, lpips=lp, dino=dino)
            else:
                self.models, self.stats, self.static = (self.loaded[k] for k in ('models', 'stats', 'static'))
                quality = self.loaded['quality']
            require(quality == self.quality_identity, 'original quality evaluator differs')
            self._bind_weight_metadata(self.models_receipt)
            self._bind_weight_metadata(quality)
            stats_receipt = self._json(self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/calibration_statistics.json')
            self._bind(self.root / 'outputs/RX-POSTERIOR-STEP1-20260929/calibration_statistics.pt',
                       stats_receipt['sha256'])
            self.model_hashes = {k: self.old.b.state_sha256(v) for k, v in self.models.items()}
            require(self.model_hashes['vae'] == self.model_identity['vae'] and
                    self.model_hashes['var'] == self.model_identity['var'], 'original VAE/VAR differs')
            a_done = self.done if self.arm == 'A' else self._receipt('A_completion.json', fallback_arm='A')
            require(self.model_hashes['decoder'] == a_done['decoder_state_sha256'], 'historical Dc differs')
            for model in self.models.values():
                model.eval().requires_grad_(False)
            if self.data is None:
                from latent_enhancement_eval.runner import load_targets
                records = load_targets()
            else:
                records = self.data
            self._validate_records(records, self.population)
            self._bind_population_files(self.population)
            self._ready = True
        return self

    def metadata(self, row):
        oracle = row['method'] in ('O1', 'Source_A_Dc_true_Fq')
        return dict(study=self.study, scope=('source_only_reference' if row['method'] == 'Source_A_Dc_true_Fq'
                                            else 'historical_equivalent_latent_noise'),
                    decoder_id='Dc', label_conditioned=False, class_condition='U',
                    oracle=oracle, output_role='oracle_reference' if oracle else 'main',
                    is_main_conclusion=not oracle, finite_channel_budget=False,
                    noise_domain='standardized latent noise, not paid AWGN symbols',
                    snr_field='snr_equiv_db',
                    original_metric_columns={k: k for k in METRICS},
                    model_identity=self.model_identity, training_updates=0, policy_selection_updates=0)

    def iterate_source(self, index):
        import torch
        with self._context(), torch.no_grad():
            self._before_source(index)
            index = int(index)
            rec = self.records[index]['native_record']
            target = rec['pixels'].astype('float32') / 255.
            vae, var, dec, lp, dino = (self.models[k] for k in ('vae', 'var', 'decoder', 'lpips', 'dino'))
            f, truth, fq = self.old.dev_source(vae, rec)
            expected = self.expected_rows(index)
            produced = {}
            if self.arm == 'A':
                mu, sigma, rho = (self.stats[k].cuda() for k in ('mu', 'sigma', 'rho'))
                for snr in self.cfg['oracle_snrs_db']:
                    eta = 10 ** (-snr / 20)
                    for seed in self.cfg['development_noise_seeds']:
                        self.old.b.boundary()
                        z = self.old.observe(f, self.stats, self.records[index]['source_id'], seed, eta)
                        zz = (z - mu) / sigma
                        b1 = mu + sigma * zz / (1 + eta ** 2)
                        o1 = self.old.fusion(fq, z, self.stats, rho, eta)
                        images = dec(torch.cat([b1, o1], 0)).cpu().numpy()
                        scores = self.old.quality(target, list(images), lp, dino)
                        for j, method in enumerate(('B1', 'O1')):
                            latent = (b1, o1)[j]
                            produced[(method, snr, seed)] = (images[j], scores[j],
                                dict(latent_actual=float((f.double() - latent.double()).square().sum())))
                clean = dec(fq)[0].cpu().numpy()
                produced[('Source_A_Dc_true_Fq', None, 0)] = (clean, self.old.quality(target, [clean], lp, dino)[0], {})
            else:
                for level in self.frozen['levels']:
                    if level['role'] != 'decision':
                        continue
                    eta = level['eta']
                    variance = self.old.variances(self.stats, eta)
                    for seed in self.frozen['algorithm']['development_noise_seeds']:
                        self.old.b.boundary()
                        z = self.old.observe(f, self.stats, self.records[index]['source_id'], seed, eta)
                        dedup = {}
                        for prior, version, lam in self.helper.profiles(level):
                            key = (prior, lam)
                            if key not in dedup:
                                dedup[key] = self.old.infer(vae, var, z, truth, prior, 'CL', variance,
                                    self.static, lam, details=False)['fhat']
                            q = dedup[key]
                            profile = self.helper.key(prior, version)
                            rho = torch.tensor(self.frozen['rho'][level['name'] + '/' + profile]['variance'],
                                device='cuda', dtype=torch.float32).reshape(1, 32, 1, 1)
                            fused = self.old.fusion(q, z, self.stats, rho, eta)
                            images = dec(torch.cat([q, fused], 0)).cpu().numpy()
                            scores = self.old.quality(target, list(images), lp, dino)
                            for j, suffix in enumerate(('tok', 'fuse')):
                                produced[(profile + '_' + suffix, level['name'], seed)] = (images[j], scores[j],
                                    dict(offline_truth_used_for_decisions=False, closed_loop=True,
                                         latent_actual=float((f.double() - (q, fused)[j].double()).square().sum())))
            for row in expected:
                key = (row['method'], row.get('level') if self.arm == 'B' else row.get('snr_equiv_db'), row['noise_seed'])
                image, score, extra = produced[key]
                yield self._result(row, image, target, score,
                                   decoder_batch_size=1 if row['method'] == 'Source_A_Dc_true_Fq' else 2,
                                   **extra)
            require({k: self.old.b.state_sha256(v) for k, v in self.models.items()} == self.model_hashes,
                    'historical frozen model weights changed')


def create_adapter(root, study, loaded=None, data=None):
    require(study in STUDIES, 'unsupported completed receiver study: ' + study)
    return (Step2Adapter if study.startswith('RX_STEP2') else Step1Adapter)(root, study, loaded, data)
