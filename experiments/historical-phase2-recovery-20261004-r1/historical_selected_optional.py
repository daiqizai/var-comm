"""User-selected historical analyses, after mandatory metrics jobs.

Only the selected outputs are inferred. Full original tables are read to bind
lineage; no old main, training, calibration, timing, or unselected image replay
is called. Cached aliases retain original Phase 2 rows and pixel provenance.
"""
from pathlib import Path
from copy import deepcopy
import csv
import hashlib
import json
import numpy as np

import historical_budget as budget
import historical_receiver as receiver
import historical_latent as latent
import historical_cached as cached
from historical_selected_budget import registered_quality_weight

STUDIES = ('OPTIONAL_RX_STEP2_A', 'OPTIONAL_RX_STEP2_B',
           'OPTIONAL_H6_13DB', 'OPTIONAL_PHASE2_MAIN')
H6 = ('H6-V_N4084_seed2026092304', 'H6-P_N4084_seed2026092304')
PHASE2 = ('original_1024_Dc', 'calibration_frozen_resource_lookup', 'pure_continuous')
SEEDS = (2001, 2002, 2003)
require = receiver.require


def snr(row):
    return float(row['snr_db'])


def seed(row):
    return int(row.get('noise_seed', row.get('seed')))


def index(row):
    return int(row.get('source_index', row.get('image_index')))


def key(row):
    return index(row), snr(row), seed(row)


def selected_rows(rows, methods, snrs, sources=100):
    """Select before inference and reject missing/duplicate requested keys."""
    result = [dict(r) for r in rows if r['method'] in methods and snr(r) in snrs]
    found = [(r['method'], *key(r)) for r in result]
    wanted = {(m, i, float(s), n) for m in methods for i in range(sources)
              for s in snrs for n in SEEDS}
    require(len(found) == len(set(found)) and set(found) == wanted,
            'selected historical subset is incomplete or duplicated')
    return result


def assert_alias(row, candidate):
    require(key(row) == key(candidate), 'cache alias frame identity differs')
    require(row.get('image_id', row.get('source_id')) ==
            candidate.get('image_id', candidate.get('source_id')), 'cache alias source differs')
    require(float(row['N']) == float(candidate['N']) == 4084 and
            float(row['E']) == float(candidate['E']) == 8168, 'cache alias resources differ')
    # This alias has been observed to be numerically exact. Do not relax it to
    # the native replay tolerance or use it for a merely similar model.
    for metric in receiver.METRICS:
        require(float(row[metric]) == float(candidate[metric]), 'cache alias old metric differs: ' + metric)


def policy_plan(level, base):
    """Only genuine CORRECT policies need a candidate; BYPASS never runs VAR."""
    out = [(base, None)]
    for prior in ('A1', 'A2', 'V'):
        choice = level['methods'][prior]
        require(choice['policy_action'] in ('BYPASS', 'CORRECT'), 'unknown frozen action')
        out.append((prior + '_policy', None if choice['policy_action'] == 'BYPASS'
                    else (prior, float(choice['raw_selected_lambda']))))
    return out


def validate_lookup_actions(policy):
    expected = {str(float(s)): ('m8_plus_latent_512_fold_N4084' if s == 1 else
                               'm8_plus_latent_1024') for s in latent.SNRS}
    require(policy['actions'] == expected, 'optional lookup actions differ from the exact selected scope')


def prepare_budget_quality_path(adapter):
    """Admit the installed LPIPS linear path only by original byte identity.

    The native BudgetAdapter checks absolute registered paths. A private
    runtime context adds the venv location after the shared strict SHA proof;
    the original registration object and files remain unchanged.
    """
    if adapter.ready:
        return
    import lpips
    linear = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    expected = registered_quality_weight(adapter, linear)
    adapter.context = deepcopy(adapter.context)
    adapter.context['bindings'][str(linear.resolve())] = expected


def original_quality_slots(rows):
    """Original per-method quality order: five SNRs, three seeds (8+7)."""
    grid = [(float(s), k) for s in latent.SNRS for k in SEEDS]
    require(bool(rows) and len({(r['method'], index(r)) for r in rows}) == 1,
            'one original method/source is required for quality slot replay')
    slots = []
    for row in rows:
        frame = snr(row), seed(row)
        require(frame in grid, 'quality frame escaped the original registered grid')
        slots.append(grid.index(frame))
    require(len(slots) == len(set(slots)), 'duplicate original quality slot')
    return slots


def selected_quality_15(target, images, rows, checker):
    """Preserve native metric batch shapes without inferring excluded outputs.

    Frozen evaluation models operate independently on each image. Source image
    placeholders retain the original batch position and size; their metrics
    are discarded and never become additional experiment rows.
    """
    slots = original_quality_slots(rows)
    require(len(images) == len(slots), 'selected quality image/row count differs')
    padded = [target] * 15
    for slot, image in zip(slots, images):
        padded[slot] = image
    result = checker(target, padded)
    require(len(result[0]) == 15, 'native original quality batch coverage differs')
    return ([result[0][slot] for slot in slots], *result[1:])


class _SelectedQuality:
    """Adapter-local proxy; the shared original quality module is unchanged."""
    def __init__(self, module, rows):
        self.module, self.rows = module, rows

    def __getattr__(self, name):
        return getattr(self.module, name)

    def quality_metrics(self, target, images, *args, **kwargs):
        result = selected_quality_15(target, images, self.rows,
            lambda source, padded: self.module.quality_metrics(source, padded, *args, **kwargs))
        # Native LatentAdapter currently consumes scores only. Preserve the
        # full documented return contract if its embedding output is used too.
        if len(result) >= 3:
            result = (result[0], result[1], result[2][original_quality_slots(self.rows)], *result[3:])
        return result


class _View:
    def _inventory(self, rows):
        self.rows = rows
        self.by_source = {i: [r for r in rows if index(r) == i] for i in range(100)}

    def expected_rows(self, source_index):
        return [dict(r) for r in self.by_source[int(source_index)]]

    @property
    def bindings(self):
        return self.native.bindings

    @property
    def records(self):
        return self.native.records

    def describe(self):
        return dict(study=self.study, rows=len(self.rows), sources=100,
                    methods=sorted({r['method'] for r in self.rows}),
                    snrs_db=sorted({snr(r) for r in self.rows}),
                    training_updates=0, policy_selection_updates=0,
                    optional_after_mandatory=True, synthetic=False,
                    excluded_outputs_are_not_inferred=True)

    def metadata(self, row):
        meta = self.native.metadata(row)
        meta.update(study=self.study, optional_after_mandatory=True,
                    training_updates=0, policy_selection_updates=0,
                    weight_path_relocations=getattr(self.native, 'weight_path_relocations', {}))
        return meta


class ReceiverView(_View):
    def __init__(self, root, study):
        self.study = study
        self.native = receiver.Step2Adapter(root, 'RX_STEP2_' + study[-1])
        n = self.native
        rows = selected_rows(n.rows, (n.base, 'A1_policy', 'A2_policy', 'V_policy'), (-5, -2))
        self._inventory(rows)
        if n.arm == 'B':
            require(all(plan[1] is None for s in (-5, -2)
                        for plan in policy_plan(n.policy['levels'][str(s)], n.base)),
                    'P_low selected scope was registered as all BYPASS')

    def setup(self):
        n = self.native
        n.setup()
        # Original float receive latent is sufficient; it avoids repeat TX,
        # channel and P receive. An original tensor receipt is mandatory.
        import torch
        path = n.out / 'development_observations.pt'
        receipt = n._json(Path(str(path) + '.json'))
        n._bind(path, receipt['sha256'])
        self.observations = torch.load(path, map_location='cpu', weights_only=True)
        identity = dict(config_sha256=receiver.sha(n.result / 'config.json'), role='development',
                        seeds=list(SEEDS), source_ids=[r['image_id'] for r in n.records],
                        preprocessing_ids=[r['preprocessing_id'] for r in n.records])
        require(self.observations['identity'] == identity, 'original observation identity differs')
        self.obs_rows = {}
        for j, row in enumerate(self.observations['rows']):
            require(key(row) not in self.obs_rows, 'duplicate observation key')
            self.obs_rows[key(row)] = j
        wanted = {(i, float(s), k) for i in range(100) for s in n.snrs for k in SEEDS}
        require(set(self.obs_rows) == wanted, 'original receive-latent coverage differs')
        require(tuple(self.observations['Z'].shape) == (len(wanted), 32, 16, 16), 'receive latent shape differs')
        require(self.observations['Z'].dtype == torch.float32 and
                bool(torch.isfinite(self.observations['Z']).all()), 'invalid receive latent tensor')
        return self

    def iterate_source(self, source_index):
        import torch
        n, i = self.native, int(source_index)
        with n._context(), torch.no_grad():
            n._before_source(i)
            rec = n.records[i]
            target = rec['pixels'].astype(np.float32) / 255
            records = [r['native_record'] for r in n.records]
            f = n.models['vae'].quant_conv(n.models['vae'].encoder(
                torch.tensor(rec['pixels'][None], device='cuda', dtype=torch.float32) / 127.5 - 1))[0].cpu()
            results = {}
            for level_snr in (-5, -2):
                level = n.policy['levels'][str(level_snr)]
                positions = [self.obs_rows[i, float(level_snr), k] for k in SEEDS]
                z = self.observations['Z'][positions]
                ii = torch.tensor([i] * 3)
                estimates, rendered = {n.base: z}, {}
                for name, candidate in policy_plan(level, n.base):
                    if name != n.base:
                        prior = name.split('_')[0]
                        if candidate is None:
                            estimates[name] = z
                        else:
                            q = n.rx.infer(n.models['vae'], n.models['var'], z.cuda(), prior,
                                level['variance_by_scale'], n.static, lam=candidate[1], details=False)['fhat'].cpu()
                            a = torch.tensor(level['methods'][prior]['alpha'], dtype=torch.float32).reshape(1,32,1,1)
                            estimates[name] = n.rx.fusion(z, q, a)
                    same = next((k for k in rendered if torch.equal(estimates[k], estimates[name])), None)
                    rendered[name] = rendered[same] if same is not None else n.engine.images_metrics(
                        estimates[name], ii, records, n.models, n.references, n.permutation, save_images=True)
                for row in self.by_source[i]:
                    if snr(row) != level_snr:
                        continue
                    j = SEEDS.index(seed(row))
                    obs = self.observations['rows'][positions[j]]
                    require(int(self.observations['source_indices'][positions[j]]) == i, 'observation source index differs')
                    for field in ('source_id', 'preprocessing_id', 'waveform_sha256', 'observation_sha256', 'N', 'E'):
                        require(row[field] == obs[field], 'original receive observation field differs: ' + field)
                    values, images = rendered[row['method']]
                    result = n._result(row, images[j].numpy(), target, values[j],
                        replay_parity_passed=True, decoder_batch_size=3,
                        observation_hash_verified=True, original_receive_latent_reused=True,
                        true_bypass_verified=(row['policy_action'] == 'BYPASS' and torch.equal(estimates[row['method']], z)),
                        latent_actual=float((f.double()-estimates[row['method']][j].double()).square().sum()))
                    results[receiver.canonical_hash(row)] = result
            require(n.engine.dump_hashes(n.models) == n.model_identity['models'], 'frozen receiver weights changed')
            for row in self.by_source[i]:
                yield results[receiver.canonical_hash(row)]


class H6View(_View):
    def __init__(self, root):
        self.study = 'OPTIONAL_H6_13DB'
        self.native = budget.BudgetAdapter(root, 'C_SELECTED_GRID')
        n = self.native
        self._inventory(selected_rows(n.rows, H6, (13,)))
        n.methods = list(H6)
        n.model_metadata = {m: n.model_metadata[m] for m in H6}
        require(all(v['selected']['step'] == 30000 and v['training_seed'] == 2026092304
                    and v['N'] == 4084 for v in n.model_metadata.values()), 'H6 selected lineage differs')

    def setup(self):
        prepare_budget_quality_path(self.native)
        self.native.setup()
        return self

    def iterate_source(self, source_index):
        n, i = self.native, int(source_index)
        require(n.ready, 'setup required')
        record = n.records[i]
        target = budget.rgb(record['pixels'].astype(np.float32)/255)
        with n.torch.no_grad():
            for method in H6:
                rows = [r for r in self.by_source[i] if r['method'] == method]
                images, infos = [], []
                for row in rows:
                    n.safe.check()
                    image, info = n.native.execute(record, n.models[method], n.model_metadata[method],
                        13, seed(row), n.vae, n.var, n.decoder, n.device)
                    images.append(budget.rgb(image))
                    infos.append({**info['ledger'], **info['rx'], 'waveform_sha256': info['waveform_sha256'],
                                  'observation_sha256': info['observation_sha256']})
                values = selected_quality_15(target, images, rows,
                    lambda source, padded: n.native.quality_metrics(source, padded, n.lp, n.dino, n.device))[0]
                for row, image, info, value in zip(rows, images, infos, values):
                    proof = budget.parity_check(row, {**value, **info,
                        'mse': float(np.square(image-target, dtype=np.float64).mean())})
                    proof.update(original_location=n.metadata(row)['original_location'],
                                 native_quality_batch_sizes=[8, 7],
                                 native_quality_selected_slots=original_quality_slots(rows),
                                 native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                    yield dict(row), image, target, proof


class _FoldOnly(latent.LatentAdapter):
    def _load_specific(self):
        self.engine = self._module('latent_research.system_policy')
        require(self.state['decoder'] == self.policies['policy']['decoder_state_sha256'], 'lookup Dc differs')
        self.models['enhancement512'] = self._load_arm('enhancement512')
        require(self.selected['enhancement512'] == self.policies['policy']['selected_arms']['enhancement512'],
                'folded selected arm differs')

    def iterate_source(self, source_index):
        rows = self.expected_rows(int(source_index))
        slots = original_quality_slots(rows)
        require(slots == [0, 1, 2], 'folded optional scope must be only the original 1dB repeats')
        original = self.q
        self.q = _SelectedQuality(original, rows)
        try:
            for row, image, target, proof in super().iterate_source(source_index):
                proof = dict(proof, native_quality_batch_sizes=[8, 7],
                             native_quality_selected_slots=slots)
                yield row, image, target, proof
        finally:
            self.q = original


class Phase2View(_View):
    def __init__(self, root):
        self.study = 'OPTIONAL_PHASE2_MAIN'
        self.root = Path(root).resolve()
        self.native = budget.BudgetAdapter(root, 'CONTINUOUS_GRID')
        n = self.native
        n.methods = ['P4084']
        n.model_metadata = {'P4084': n.model_metadata['P4084']}
        require(n.model_metadata['P4084']['step'] == 10000, 'Phase2 control must be legacy selected10k')
        self.cache = cached.create_adapter(root, 'LATENT_ORIGINAL')
        self.fold = _FoldOnly(root, 'PHASE2_SYSTEM_LOOKUP')
        validate_lookup_actions(self.fold.policies['policy'])
        self.fold.rows = selected_rows(self.fold.rows, ('calibration_frozen_resource_lookup',), (1,))
        self.fold._groups = {i: [r for r in self.fold.rows if index(r) == i] for i in range(100)}
        self.fold.methods = {'calibration_frozen_resource_lookup'}
        lineage_path = self.root/'results/review_20260923_phase2/artifact_lineage.json'
        n.bind(lineage_path)
        table = self.root/'results/review_20260923_phase2/comparison/per_frame.csv'
        lineage = json.loads(lineage_path.read_text(encoding='utf-8'))['artifacts']
        record = next(x for x in lineage if x['published_path'] == table.relative_to(self.root).as_posix())
        n.bind(table, record['sha256'])
        with table.open(newline='', encoding='utf-8') as f:
            self._inventory(selected_rows(list(csv.DictReader(f)), PHASE2, latent.SNRS))
        self.candidates = {}
        for i in range(100):
            for row in self.cache.expected_rows(i):
                if row['method'] == 'm8_plus_latent_1024':
                    require(key(row) not in self.candidates, 'duplicate original1024 cache key')
                    self.candidates[key(row)] = row
        require(len(self.candidates) == 1500, 'complete original1024 float cache required')
        self.fold_rows = {key(r): r for r in self.fold.rows}
        for row in self.rows:
            if row['method'] == 'original_1024_Dc' or (row['method'] == 'calibration_frozen_resource_lookup' and snr(row) != 1):
                assert_alias(row, self.candidates[key(row)])
            if row['method'] == 'pure_continuous':
                require(row['model_context_sha256'] == n.model_metadata['P4084']['checkpoint_sha256'],
                        'Phase2 pure checkpoint differs from legacy P4084')
        self._merge_bindings()

    def _merge_bindings(self):
        for adapter in (self.cache, self.fold):
            for p,h in adapter.bindings.items():
                previous = self.native.bindings.get(p)
                require(previous in (None, h), 'shared original binding conflict')
                self.native.bindings[p] = h

    def setup(self):
        n = self.native
        prepare_budget_quality_path(n)
        n.setup()
        self.fold.loaded = dict(vae=n.vae, var=n.var, decoder=n.decoder, scale=n.scale,
                                lpips=n.lp, dino=n.dino, bindings=dict(n.bindings))
        self.fold.setup()
        for i,(a,b,c) in enumerate(zip(n.records, self.fold.records, self.cache.records)):
            require(a['image_id'] == b['image_id'] == c['image_id'], 'Phase2 population differs')
            require(np.array_equal(a['pixels'], b['pixels']), 'Phase2 native source pixels differ')
        self._merge_bindings()
        return self

    def metadata(self, row):
        hybrid = row['method'] != 'pure_continuous'
        return dict(study=self.study, method_id=row['method'], decoder='Dc',
                    decoder_sha256=row['decoder_sha256'], N=4084,
                    model_id=row['model_context_sha256'], checkpoint_sha256=row['model_context_sha256'] if not hybrid else '',
                    label_conditioned=hybrid, true_class_sent=hybrid, class_condition='C' if hybrid else 'U',
                    classification_main_eligible=not hybrid, oracle=False, reference_only=False,
                    original_metric_columns={m:m for m in receiver.METRICS}, snr_field='snr_db',
                    weight_path_relocations=getattr(self.native, 'weight_path_relocations', {}),
                    selected_step=10000 if not hybrid else None, training_seed=None,
                    optional_after_mandatory=True, training_updates=0, policy_selection_updates=0)

    def iterate_source(self, source_index):
        n, i = self.native, int(source_index)
        require(n.ready, 'setup required')
        target = n.records[i]['pixels'].astype(np.float32)/255
        pure = {key(r): (img, ref, proof) for r,img,ref,proof in n.iterate_source(i)}
        folded = {key(r): (img, ref, proof) for r,img,ref,proof in self.fold.iterate_source(i)}
        cache, originals, pixels = {}, {}, []
        for k in [(i, s, k) for s in latent.SNRS for k in SEEDS]:
            image, reference, proof = self.cache.load_image(i, self.candidates[k], cache)
            require(np.array_equal(reference, target), 'cached source pixels differ from Phase2 target')
            originals[k] = (image, proof)
            pixels.append(image)
        with n.torch.no_grad():
            scores = n.native.quality_metrics(target, pixels, n.lp, n.dino, n.device)[0]
        score_by_key = dict(zip(originals, scores))
        for row in self.by_source[i]:
            k = key(row)
            if row['method'] == 'pure_continuous':
                image, reference, base_proof = pure[k]
                require(np.array_equal(reference, target), 'pure source pixels differ')
                actual = {m: float(n._main[i,'P4084',int(snr(row)),seed(row)][m]) +
                          base_proof['metric_deltas'][m] for m in receiver.METRICS}
                proof = receiver.parity(row, actual)
                proof.update(native_replay_proof=base_proof)
            elif row['method'] == 'calibration_frozen_resource_lookup' and snr(row) == 1:
                image, reference, base_proof = folded[k]
                require(np.array_equal(reference, target), 'folded source pixels differ')
                # Native replay has independently checked the original lookup
                # row; comparison rows must retain the exact same old metrics.
                assert_alias(row, self.fold_rows[k])
                proof = dict(base_proof)
            else:
                image, base_proof = originals[k]
                proof = receiver.parity(row, score_by_key[k])
                proof.update(cache_alias_proof=base_proof, alias_of='m8_plus_latent_1024')
            proof.update(replay_parity_passed=True, original_row_sha256=receiver.canonical_hash(row),
                         native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            yield dict(row), image, target, proof


def create_adapter(root, study, loaded=None, data=None):
    require(loaded is None and data is None, 'optional adapters require their bound original inputs')
    require(study in STUDIES, 'unknown optional selected study')
    if study in ('OPTIONAL_RX_STEP2_A', 'OPTIONAL_RX_STEP2_B'):
        return ReceiverView(root, study)
    return H6View(root) if study == 'OPTIONAL_H6_13DB' else Phase2View(root)
