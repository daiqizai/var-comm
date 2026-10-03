"""Frozen final-budget views; select original rows before any model execution.

No training, calibration, old main, full candidate evaluation or new selection.
Original CSV dictionaries are returned unchanged. Old quality and PHY identity
must pass on every selected image before supplementary metrics are accepted.
"""
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

import historical_budget as budget
from history_common import python_source_in_repo

BASE = 'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
P4084_METHODS = ('P4084_N4084_seed2026092304', 'P4084_N4084_seed2026092404')
STUDIES = ('FINAL_P2048_P3060', 'FINAL_P4084_SELECTED_SEEDS',
           'FINAL_DIGITAL_QPSK', 'FINAL_DIGITAL_16QAM', 'FINAL_PHASE2_N4084_DIGITAL')


def require(ok, message):
    if not ok:
        raise budget.ReplayMismatch(message)


def digital_events(row, event):
    """Keep separately recorded TX erasure and received-header erasure fields."""
    converted = dict(event)
    for field in ('header_ok', 'header_crc_ok', 'body_crc_ok', 'source_complete'):
        if field in converted:
            converted[field] = bool(converted[field])
    converted['rx_source_overflow_erasure'] = bool(event.get('source_overflow_erasure', False))
    for field in ('header_ok', 'header_crc_ok', 'body_crc_ok', 'source_complete',
                  'source_error', 'decoded_label', 'decoded_mode', 'rx_source_overflow_erasure'):
        if field in row:
            require(field in converted and budget.csv_value(converted[field]) == row[field],
                    'selected receiver event differs: '+field)
    return converted


def registered_quality_weight(adapter, path):
    """Allow only byte-identical LPIPS linear weights in a different venv."""
    key = str(Path(path).resolve())
    registered = adapter.context['bindings']
    expected = registered.get(key)
    if expected is None:
        suffix = '/lpips/weights/v0.1/alex.pth'
        require(key.replace('\\', '/').endswith(suffix), 'unregistered quality asset path')
        matches = [(p, digest) for p, digest in registered.items() if p.replace('\\', '/').endswith(suffix)]
        require(len(matches) == 1, 'ambiguous original LPIPS linear weights')
        origin, expected = matches[0]
        adapter.bind(origin, expected); adapter.bind(path, expected)
        adapter.weight_path_relocations = {key: dict(original=origin, sha256=expected,
                                                     content_identical=True)}
    else:
        adapter.bind(path, expected)
    return expected


def select_policy_rows(rows, choices, mcs, context_sha):
    """Pure, strict join of calibration choices to unchanged development rows."""
    actions = {}
    for choice in choices:
        key = (choice['family'], budget.integer(choice['N']), budget.integer(choice['snr_db']))
        require(key not in actions, 'duplicate frozen final policy action')
        require(choice['mcs'] == mcs and choice['context_sha256'] == context_sha,
                'final policy context differs')
        expected = 'per_frame_2N' if mcs == 'QPSK' else 'fixed_constellation_average_2_per_symbol'
        require(choice['energy_constraint'] == expected, 'policy energy protocol differs')
        actions[key] = choice['method']
    require(set(actions) == {(f, n, s) for f in ('raw', 'arithmetic')
            for n in (2048, 3060, 4084) for s in budget.SNRS}, 'final policy scope differs')
    selected, seen = [], set()
    for row in rows:
        key = (row['family'], budget.integer(row['N']), budget.integer(row['snr_db']))
        if row['method'] != actions.get(key):
            continue
        identity = (budget.integer(row['source_index']), *key, budget.integer(row['noise_seed']))
        require(identity not in seen, 'duplicate final policy development row')
        require(row['mcs'] == mcs and row['context_sha256'] == context_sha,
                'selected development context differs')
        seen.add(identity); selected.append(row)
    expected = {(i, f, n, s, z) for i in range(100) for f in ('raw', 'arithmetic')
                for n in (2048, 3060, 4084) for s in budget.SNRS for z in budget.SEEDS}
    require(seen == expected, 'incomplete final policy development grid')
    return selected, actions


class SelectedContinuous(budget.BudgetAdapter):
    def __init__(self, root, study, loaded=None, data=None):
        self.selected_study = study
        old = 'CONTINUOUS_GRID' if study == 'FINAL_P2048_P3060' else 'C_SELECTED_GRID'
        super().__init__(root, old, loaded, data)
        methods = ('P2048', 'P3060') if old == 'CONTINUOUS_GRID' else P4084_METHODS
        require(set(methods) <= set(self.methods), 'final continuous model roster absent')
        # Parent setup and iteration both read these narrowed collections. No
        # excluded P4084-10k or hybrid model is loaded or called.
        self.methods = list(methods)
        self.model_metadata = {k: self.model_metadata[k] for k in methods}
        self.rows = [r for r in self.rows if r['method'] in methods]
        self.by_source = {i: [r for r in self.by_source[i] if r['method'] in methods] for i in range(100)}
        self._main = {k: r for k, r in self._main.items() if k[1] in methods}
        expected_steps = {'P2048': 30000, 'P3060': 37500,
                          P4084_METHODS[0]: 27500, P4084_METHODS[1]: 30000}
        for name, meta in self.model_metadata.items():
            step = meta['step'] if old == 'CONTINUOUS_GRID' else meta['selected']['step']
            require(step == expected_steps[name], 'final selected checkpoint step differs')
            require(meta.get('kind', 'continuous') == 'continuous', 'noncontinuous model in final P roster')

    def setup(self):
        if self.ready:
            return self
        require(self._loaded is None, 'Unqualified injected models are not accepted')
        import lpips
        linear = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
        expected = registered_quality_weight(self, linear)
        # A runtime copy preserves the exact old registration bytes. Parent
        # loader can then check the actual environment's identical weight file.
        self.context = deepcopy(self.context)
        self.context['bindings'][str(linear.resolve())] = expected
        return super().setup()

    def metadata(self, row):
        value = super().metadata(row)
        name = row['method']
        initialization = 2026092303 if name == P4084_METHODS[0] else 2026092404 if name == P4084_METHODS[1] else 2026092304
        training = 2026092404 if name == P4084_METHODS[1] else 2026092304
        value.update(study=self.selected_study, method_id=name, training_seed=training,
                     initialization_seed=initialization, condition='U', model_id=name,
                     parent_updates=10000 if name == P4084_METHODS[0] else 0,
                     training_completed_step=40000 if name == 'P3060' else 30000,
                     original_study=self.study, selected_before_computation=True)
        value['weight_path_relocations'] = getattr(self, 'weight_path_relocations', {})
        return value

    def describe(self):
        return {**super().describe(), 'study': self.selected_study,
                'selected_before_computation': True, 'obsolete_P4084_10000_included': False}


class FinalDigital:
    def __init__(self, root, study, loaded=None, data=None):
        require(loaded is None and data is None, 'unqualified injected digital state is forbidden')
        self.root, self.study = Path(root).resolve(), study
        self.mcs = 'QPSK' if study == 'FINAL_DIGITAL_QPSK' else '16QAM'
        self.bindings, self.ready = {}, False
        self.folder = self.root/BASE/'digital_grid_v1'/self.mcs/'development'
        self.calibration = self.folder.parent/'calibration'
        self.registration_sha256 = self.bind(self.folder/'registration.json')
        self.registration = reg = budget.read(self.folder/'registration.json')
        self.bind(self.folder/'completion.json'); done = budget.read(self.folder/'completion.json')
        require(done['status'] == 'REAL_DIGITAL_GRID_COMPLETE' and done['synthetic'] is False
                and done['registration_sha256'] == self.registration_sha256,
                'completed real final digital grid required')
        require(reg['role'] == done['role'] == 'development' and reg['mcs'] == done['mcs'] == self.mcs
                and reg['synthetic'] is False and reg['new_holdout'] is False,
                'digital population/protocol differs')
        require(tuple(reg['snrs']) == budget.SNRS and tuple(reg['seeds']) == budget.SEEDS
                and len(reg['sources']) == 100, 'original digital source scope differs')
        self.context = reg['context']
        require(budget.identity(self.context) == reg['context_sha256'], 'digital context hash differs')
        self.policy_sha256 = self.bind(self.calibration/'policy.json', reg['policy_sha256'])
        require(done['policy_sha256'] == self.policy_sha256, 'development policy seal differs')
        self.policy = policy = budget.read(self.calibration/'policy.json')
        require(policy['status'] == 'FROZEN_CALIBRATION_POLICY', 'final calibration policy required')
        self.bind(self.calibration/'completion.json'); caldone = budget.read(self.calibration/'completion.json')
        require(caldone['status'] == 'REAL_DIGITAL_GRID_COMPLETE' and caldone['synthetic'] is False
                and caldone['policy_sha256'] == self.policy_sha256, 'completed frozen calibration required')
        table = self.folder/'per_frame.csv'
        self.table_sha256 = self.bind(table, done['files']['per_frame.csv'])
        with table.open(newline='', encoding='utf-8') as f:
            all_rows = list(csv.DictReader(f))
        require(len(all_rows) == done['frame_rows'], 'original digital full table count differs')
        self.rows, self.actions = select_policy_rows(all_rows, policy['choices'], self.mcs, reg['context_sha256'])
        self.by_source, self._locations = defaultdict(list), {}
        linenos = {id(r): i for i, r in enumerate(all_rows)}
        for row in self.rows:
            i = budget.integer(row['source_index'])
            require(reg['sources'].get(row['source_id']) == row['preprocessing_id'] and
                    row['population'] == 'development', 'selected source identity differs')
            self.by_source[i].append(row)
            self._locations[budget.identity(row)] = dict(path=str(table), sha256=self.table_sha256,
                pointer='/csv_rows/'+str(linenos[id(row)]), original_row_sha256=budget.identity(row))
        self.records = []
        for i in range(100):
            ids = {r['source_id'] for r in self.by_source[i]}
            require(len(ids) == 1, 'source index identity differs')
            sid = ids.pop()
            self.records.append(dict(source_index=i, image_id=sid, source_id=sid,
                                     preprocessing_id=reg['sources'][sid]))
        require(len({r['image_id'] for r in self.records}) == 100, 'duplicate source identity')
        # Validate only the cells that actually contribute selected output rows.
        ledger = [r for r in self.context['candidate_ledger'] if r['status'] == 'ELIGIBLE']
        ordinals = {r['method']: i for i, r in enumerate(ledger)}
        require(len(ordinals) == done['eligible_methods'], 'registered candidate ledger differs')
        self.ledger = {r['method']: r for r in ledger}
        for i, group in self.by_source.items():
            for method in dict.fromkeys(r['method'] for r in group):
                require(method in ordinals, 'frozen policy selected an illegal candidate')
                name = f'{i:04d}_{ordinals[method]:02d}.json'
                path = self.folder/'cells'/name
                self.bind(path, done['cell_sha256'][name]); cell = budget.read(path)
                require(cell['registration_sha256'] == self.registration_sha256 and
                        cell['source_id'] == self.records[i]['image_id'] and cell['method'] == method,
                        'selected original cell identity differs')
                for row in (r for r in group if r['method'] == method):
                    matches = [r for r in cell['rows'] if budget.integer(r['snr_db']) == budget.integer(row['snr_db'])
                               and budget.integer(r['noise_seed']) == budget.integer(row['noise_seed'])]
                    require(len(matches) == 1 and all(budget.csv_value(matches[0].get(k)) == v for k, v in row.items()),
                            'selected CSV and sealed cell disagree')

    def bind(self, path, expected=None):
        return budget.BudgetAdapter.bind(self, path, expected)

    def setup(self):
        if self.ready:
            return self
        for path, digest in self.context['bindings'].items():
            self.bind(path, digest)
        mods = budget._imports(self.root)
        self.native = n = mods['token_efficiency.digital_grid']
        require(n.ROOT.resolve() == self.root, 'wrong original repository root')
        mods['token_efficiency.common'].configure_runtime(); n.require_available()
        self.safe, self.torch = n.SafeEvaluation(), n.torch
        self.device = n.torch.device('cuda:0')
        acceptance = self.root/BASE/'execution_qualification_v1/acceptance.json'
        self.bind(acceptance, self.context['qualification_sha256'])
        accepted = n.require_real_execution_acceptance()
        require(accepted['frozen_model_states']['Dc'] == self.context['decoder_sha256'], 'qualified Dc differs')
        paths = n.model_paths()
        for key in ('vae_checkpoint', 'var_checkpoint'):
            self.bind(paths[key], paths[key+'_sha256'])
        self.vae, self.var = n.load_models(paths, self.device)
        self.decoder = n.load_decoder(self.vae, self.device)
        require(n.state_sha256(self.decoder) == self.context['decoder_sha256'], 'frozen digital Dc differs')
        qcfg = n.yaml.safe_load((self.root/'configs/progressive_channel.yaml').read_text())['quality']
        self.lp, self.dino, linear = n.load_quality_models(qcfg, self.device)
        for path in (qcfg['alexnet_checkpoint'], qcfg['dino_checkpoint'], linear):
            registered_quality_weight(self, path)
        records, image_bindings = n.population('development')
        require(budget.identity(image_bindings) == budget.identity(self.registration['image_bindings']), 'source archive bindings differ')
        for saved, actual in zip(self.records, records):
            require(all(actual[k] == saved[k] for k in ('image_id', 'preprocessing_id')) and
                    hashlib.sha256(actual['pixels'].tobytes()).hexdigest() == saved['preprocessing_id'],
                    'digital source pixels/order differ')
            actual.update(source_index=saved['source_index'], source_id=actual['image_id'])
        require(len(records) == 100, 'native source count differs')
        self.records = records
        for module in tuple(sys.modules.values()):
            path = python_source_in_repo(module, self.root)
            if path is not None:
                self.bind(path)
        precision = self.context['precision']
        require(n.torch.backends.cuda.matmul.allow_tf32 == precision['matmul_tf32']
                and n.torch.backends.cudnn.allow_tf32 == precision['cudnn_tf32']
                and n.torch.backends.cudnn.deterministic == precision['cudnn_deterministic']
                and n.torch.are_deterministic_algorithms_enabled() == precision['deterministic_algorithms'],
                'original digital numerical backend differs')
        self.ready = True
        return self

    def expected_rows(self, index):
        return [dict(r) for r in self.by_source[budget.integer(index)]]

    def metadata(self, row):
        family, n = row['family'], budget.integer(row['N'])
        return dict(study=self.study, method_id=f'final_{family}_{self.mcs}_N{n}_Dc',
            model_id=f'final_{family}_{self.mcs}_N{n}_Dc', decoder_id='Dc',
            decoder_sha256=self.context['decoder_sha256'], N=n, phy=self.mcs,
            energy_constraint=row['energy_constraint'], condition='C', label_conditioned=True,
            true_class_sent=True, class_condition='paid_class_header', classification_main_eligible=False,
            classification_caveat='true class is transmitted in the charged header',
            reference_only=False, oracle=False, original_location=self._locations[budget.identity(row)],
            original_method=row['method'], original_study='token_efficiency_digital_grid_v1',
            selected_policy_sha256=self.policy_sha256, selected_before_computation=True,
            weight_path_relocations=getattr(self, 'weight_path_relocations', {}),
            original_metric_columns={'psnr_db': 'psnr_db', 'lpips_alex': 'lpips_alex', 'dino_cosine': 'dino_cosine'},
            snr_field='snr_db', source_index_field='source_index', noise_seed_field='noise_seed',
            training_updates=0, policy_selection_updates=0)

    def describe(self):
        return dict(study=self.study, sources=100, rows=len(self.rows),
            methods=sorted({self.metadata(r)['method_id'] for r in self.rows}),
            selected_policy_sha256=self.policy_sha256, selected_before_computation=True,
            full_candidate_evaluation=False, original_table_sha256=self.table_sha256,
            snrs_db=list(budget.SNRS), noise_seeds=list(budget.SEEDS),
            training_updates=0, policy_selection_updates=0, new_holdout=False)

    def iterate_source(self, index):
        require(self.ready, 'setup must precede digital replay')
        index = budget.integer(index); record = self.records[index]
        target = budget.rgb(record['pixels'].astype(np.float32)/255)
        rows = self.by_source[index]; grouped = defaultdict(list)
        for row in rows:
            grouped[row['method']].append(row)
        prepared, output = {}, {}
        with self.torch.no_grad():
            for method, selected in grouped.items():
                ledger = self.ledger[method]
                cell = self.native.Cell(ledger['family'], ledger['N'], ledger['m_requested'], self.mcs)
                require(cell.name == method, 'native frozen cell identity differs')
                if cell.family not in prepared:
                    prepared[cell.family] = self.native.prepare_digital(record['pixels'], record['class_index'],
                        cell.family, self.vae, self.var, self.device)
                wave, resource = self.native.transmit_prepared(prepared[cell.family], record['class_index'], cell)
                # Preserve the original old-metric batch dimensions (8+7), but
                # reconstruct only selected rows. Unselected metric slots are
                # placeholders and never become results or parity evidence.
                slots = [target] * 15; infos = {}
                for row in selected:
                    self.safe.check()
                    snr, seed = budget.integer(row['snr_db']), budget.integer(row['noise_seed'])
                    slot = budget.SNRS.index(snr)*3 + budget.SEEDS.index(seed)
                    observed = self.native.apply_channel(wave, snr, record['image_id'], seed, cell)
                    image, event = self.native.receive(observed, snr, cell, self.vae, self.var, self.decoder, self.device)
                    slots[slot] = budget.rgb(image)
                    infos[slot] = {**digital_events(row, event), **resource,
                        'waveform_sha256': self.native.waveform_sha(wave),
                        'observation_sha256': self.native.waveform_sha(observed)}
                metrics = self.native.quality_metrics(target, slots, self.lp, self.dino, self.device)[0]
                require(len(metrics) == 15, 'native quality batch shape differs')
                for row in selected:
                    slot = budget.SNRS.index(budget.integer(row['snr_db']))*3 + budget.SEEDS.index(budget.integer(row['noise_seed']))
                    image = slots[slot]
                    mse = float(np.mean(np.square(image-target), dtype=np.float64))
                    proof = budget.parity_check(row, {**metrics[slot], **infos[slot], 'mse': mse})
                    proof.update(original_location=self._locations[budget.identity(row)],
                        rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                        frozen_policy_sha256=self.policy_sha256, selected_before_computation=True)
                    output[budget.identity(row)] = (dict(row), image, target.copy(), proof)
            for row in rows:
                yield output[budget.identity(row)]


class FinalPhase2:
    """Use only selected strict candidate images, expose exact adaptive rows."""
    def __init__(self, root, study, loaded=None, data=None):
        require(loaded is None and data is None, 'unqualified injected Phase2 state is forbidden')
        import historical_latent as latent
        self.root, self.study = Path(root).resolve(), study
        self.native = latent.create_adapter(root, 'PHASE2_STRICT_DIGITAL')
        self.bindings = self.native.bindings
        folder = self.root/'results/review_20260923_phase2/system_policy'
        policy_path = self.native._resolve(folder/'digital_adaptive_policies.json')
        self.policy_sha256 = budget.sha256(policy_path)
        self.policy = budget.read(policy_path)
        require(self.policy['calibration_sha256'] == self.native.done['calibration_sha256'],
                'Phase2 final policy calibrated on a different numerical protocol')
        table = self.native._resolve(folder/'digital_adaptive_per_frame.csv')
        self.table_sha256 = budget.sha256(table)
        with table.open(newline='', encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))
        require({r['method'] for r in self.rows} == {'raw_adaptive_m789_Dc', 'arithmetic_adaptive_m789_Dc'},
                'Phase2 final method roster differs')
        latent.validate_grid('PHASE2_SYSTEM_LOOKUP', self.rows,
                             {'raw_adaptive_m789_Dc', 'arithmetic_adaptive_m789_Dc'})
        candidates = {latent.row_key('PHASE2_STRICT_DIGITAL', r): r for r in self.native.rows}
        self.aliases, selected = {}, []
        self.by_source = defaultdict(list)
        for row in self.rows:
            family = row['method'].split('_')[0]; snr = float(row['snr_db'])
            mode = self.policy['actions'][family]['4084']['Dc'][str(snr)]['quality']
            require(int(row['selected_mode']) == mode, 'Phase2 final policy action differs')
            key = (f'{family}_N4084_m{mode}_Dc', int(row['source_index']), snr, int(row['seed']))
            require(key in candidates, 'missing selected strict candidate')
            candidate = candidates[key]
            for field in ('psnr_db', 'lpips_alex', 'dino_cosine', 'image_id'):
                require(candidate.get(field, candidate.get('lpips') if field == 'lpips_alex' else None) == row[field],
                        'Phase2 final alias metric/source differs: '+field)
            require(budget.identity(candidate) not in self.aliases, 'duplicate selected candidate alias')
            self.aliases[budget.identity(candidate)] = row
            selected.append(candidate); self.by_source[int(row['source_index'])].append(row)
        self.native.rows = selected
        self.native._groups = {i: [r for r in selected if latent.source_index(r) == i] for i in range(100)}
        self.native.methods = {latent.method_id('PHASE2_STRICT_DIGITAL', r) for r in selected}
        self.native_numeric_flags = self.native.native_numeric_flags

    @property
    def records(self):
        return self.native.records

    def setup(self):
        self.native.setup()
        return self

    def expected_rows(self, index):
        return [dict(r) for r in self.by_source[int(index)]]

    def metadata(self, row):
        return dict(study=self.study, method_id=row['method'], model_id=row['method'],
            decoder_id='Dc', N=4084, phy='QPSK', condition='C', label_conditioned=True,
            true_class_sent=True, class_condition='paid_class_header', oracle=False,
            reference_only=False, classification_main_eligible=False,
            selected_policy_sha256=self.policy_sha256, original_study='PHASE2_STRICT_DIGITAL',
            original_metric_columns={'psnr_db': 'psnr_db', 'lpips_alex': 'lpips_alex', 'dino_cosine': 'dino_cosine'},
            snr_field='snr_db', source_index_field='source_index', noise_seed_field='seed',
            selected_before_computation=True, training_updates=0, policy_selection_updates=0)

    def describe(self):
        return dict(study=self.study, sources=100, rows=len(self.rows),
            methods=sorted({r['method'] for r in self.rows}), selected_before_computation=True,
            full_candidate_evaluation=False, selected_policy_sha256=self.policy_sha256,
            original_table_sha256=self.table_sha256, training_updates=0, policy_selection_updates=0)

    def iterate_source(self, index):
        import torch
        import historical_latent as latent
        n = self.native
        with n._context(), torch.no_grad():
            require(n._ready, 'setup must precede Phase2 final replay')
            require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                    and not torch.backends.cudnn.benchmark, 'Phase2 strict precision changed')
            n.runtime.require_available()
            record = n.targets[int(index)]
            target = budget.rgb(record['pixels'].astype(np.float32)/255)
            source = n.progressive.split_prefix(record['tokens'], 10)
            grouped = defaultdict(list)
            for candidate in n.expected_rows(index):
                grouped[latent.method_id('PHASE2_STRICT_DIGITAL', candidate)].append(candidate)
            cache, output = {}, {}
            for selected in grouped.values():
                slots, infos = [target]*15, {}
                for candidate in selected:
                    snr, seed = int(latent.number(candidate, 'snr_db')), latent.noise_seed(candidate)
                    slot = budget.SNRS.index(snr)*3 + budget.SEEDS.index(seed)
                    image, info = n._digital_image(record, source, candidate, cache)
                    n._verify_info(candidate, info)
                    slots[slot], infos[slot] = budget.rgb(image), info
                scores = n.q.quality_metrics(target, slots, n.lp, n.dino, n.device)[0]
                require(len(scores) == 15, 'Phase2 original metric batch shape differs')
                for candidate in selected:
                    snr, seed = int(latent.number(candidate, 'snr_db')), latent.noise_seed(candidate)
                    slot = budget.SNRS.index(snr)*3 + budget.SEEDS.index(seed)
                    image = slots[slot]; original = self.aliases[budget.identity(candidate)]
                    proof = latent.parity(latent.original_metrics(candidate), scores[slot])
                    proof.update(replay_parity_passed=True, original_row_sha256=budget.identity(original),
                        candidate_original_row_sha256=budget.identity(candidate), original_actual_event_verified=True,
                        native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                        frozen_policy_sha256=self.policy_sha256, selected_before_computation=True, **infos[slot])
                    output[budget.identity(original)] = (dict(original), image, target.copy(), proof)
            for original in self.expected_rows(index):
                yield output[budget.identity(original)]
            require(not n.models, 'unexpected learned enhancement model in selected digital replay')


def create_adapter(root, study, loaded=None, data=None):
    require(study in STUDIES, 'unregistered final-budget study')
    if study in STUDIES[:2]:
        return SelectedContinuous(root, study, loaded, data)
    if study in STUDIES[2:4]:
        return FinalDigital(root, study, loaded, data)
    return FinalPhase2(root, study, loaded, data)
