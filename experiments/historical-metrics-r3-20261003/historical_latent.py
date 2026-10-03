"""Native inference and exact policy views of completed latent/Phase1/2 studies.

Original source files, frozen W/projection/selected records and all CSV strings
are preserved. No historical main, calibration, fit, optimizer, writer, timing
queue or checkpoint selection is executed. Resource admission belongs to caller.
"""
from contextlib import contextmanager
from pathlib import Path
import csv
import importlib
import json
import math
import os
import sys
import numpy as np

from historical_receiver import sha, canonical_hash, require, parity, PREFIX

SNRS = (1., 4., 7., 13., 19.)
SEEDS = (2001, 2002, 2003)
BASE = 'outputs/VAR-LATENT-ENHANCEMENT-20260917/'
EXP = 'experiments/var-latent-enhancement-20260917/'
SPECS = {
 'LATENT_FROZEN_POLICIES': ('results/latent_followup_runs/digital_policy_development_v1', 'per_frame.csv', 'FROZEN_DIGITAL_POLICY_DEVELOPMENT_COMPLETE'),
 'LATENT_DECODER_ADAPTATION': (BASE+'followup/adaptation_development_v1', 'per_frame.csv', 'ADAPTATION_DEVELOPMENT_COMPLETE'),
 'LATENT_FOLDED_512': (BASE+'followup/allocation_v1', 'per_frame.csv', 'ALLOCATION_FOLD_COMPLETE'),
 'LINEAR_REPAIRED': ('results/review_20260923_phase1/linear', 'per_frame.csv', 'LINEAR_MEASUREMENT_DEVELOPMENT_COMPLETE'),
 'PREDICTOR_REPAIRED': ('results/review_20260923_phase1/predictor', 'per_frame.csv', 'EXISTING_SELECTED_CHECKPOINTS_REEVALUATED'),
 'PHASE2_SELECTED_MODELS': ('results/review_20260923_phase2/evaluation', 'per_frame.csv', 'SELECTED_NEW_METHODS_DEVELOPMENT_AND_TIMING_COMPLETE'),
 'PHASE2_STRICT_DIGITAL': ('results/review_20260923_phase2/digital_strict', 'development.csv', 'N4084_DC_DIGITAL_STRICT_PRECISION_COMPLETE'),
 'PHASE2_PROJECTIONS': ('results/review_20260923_phase2/comparison', 'projection_per_frame.csv', None),
 'PHASE2_SYSTEM_LOOKUP': ('results/review_20260923_phase2/system_policy', 'per_frame.csv', 'STRICT_PRECISION_EXISTING_ENDPOINT_LOOKUP_EVALUATED'),
 'PHASE2_DIAGNOSTICS': ('results/review_20260923_phase2/diagnostics', 'per_frame.csv', 'FIXED_1024_DIAGNOSIS_COMPLETE'),
}
STUDIES = tuple(SPECS)


def number(row, *keys):
    for key in keys:
        if row.get(key) not in ('', None):
            return float(row[key])
    raise ValueError('missing numeric identity: ' + '/'.join(keys))


def source_index(row):
    return int(number(row, 'source_index', 'image_index'))


def noise_seed(row):
    return int(number(row, 'noise_seed', 'seed'))


def method_id(study, row):
    if study == 'LATENT_FROZEN_POLICIES':
        return f"{row['family']}_N{int(number(row,'N','budget'))}_{row['renderer']}_{row['policy']}"
    if study == 'PHASE2_STRICT_DIGITAL':
        return f"{row['family']}_N{int(number(row,'budget','N'))}_m{int(number(row,'mode','sent_mode'))}_{row['renderer']}"
    return row['method']


def original_metrics(row):
    out = dict(row)
    if 'lpips_alex' not in out and 'lpips' in out:
        out['lpips_alex'] = out['lpips']
    return out


def row_key(study, row):
    return (method_id(study, row), source_index(row), number(row, 'snr_db'), noise_seed(row))


def validate_grid(study, rows, expected_methods=None):
    keys = [row_key(study, r) for r in rows]
    methods = set(k[0] for k in keys)
    if expected_methods is not None:
        require(methods == set(expected_methods), 'historical method inventory differs')
    wanted = {(m, i, s, n) for m in methods for i in range(100) for s in SNRS for n in SEEDS}
    require(len(keys) == len(set(keys)) and set(keys) == wanted,
            'incomplete or duplicate original source/SNR/noise/method grid')
    return methods


def exact_alias_row(policy_row, candidate_rows):
    """Join original frozen-policy views with complete source/noise identities."""
    mode = int(number(policy_row, 'selected_mode', 'sent_mode'))
    renderer = policy_row['renderer']
    wanted = f"{policy_row['family']}_N{int(number(policy_row,'N','budget'))}_m{mode}_{renderer}"
    matches = [r for r in candidate_rows if r.get('method') == wanted
               and source_index(r) == source_index(policy_row)
               and number(r, 'snr_db') == number(policy_row, 'snr_db')
               and noise_seed(r) == noise_seed(policy_row)]
    require(len(matches) == 1, 'policy view must have one exact original candidate')
    candidate = matches[0]
    for key, value in candidate.items():
        require(policy_row.get(key) == value, 'policy changed candidate original field: ' + key)
    return candidate


class LatentAdapter:
    def __init__(self, root, study, loaded=None, data=None):
        require(study in STUDIES, 'unsupported completed latent study: ' + study)
        self.root, self.study = Path(root).resolve(), study
        self.loaded, self.data = loaded, data
        self.native_numeric_flags = dict(matmul_tf32=False,
            cudnn_tf32=study in ('LATENT_DECODER_ADAPTATION','LATENT_FOLDED_512'),
            cudnn_benchmark=False, cpu_threads=6, interop_threads=2)
        self.bindings, self.rows, self._groups = {}, [], {}
        self._records, self._ready, self.modules, self.aliases = None, False, {}, {}
        self.lineage, self.unconsumed_snapshot_drift = {}, {}
        for rel, field in (('results/review_20260923_phase1/lineage.json', 'path'),
                           ('results/review_20260923_phase2/artifact_lineage.json', 'published_path')):
            p = self.root / rel
            if p.exists():
                value = self._json(p)
                for rec in value if isinstance(value, list) else value['artifacts']:
                    self.lineage[rec[field]] = rec
        folder, table, status = SPECS[study]
        self.folder = self.root / folder
        self.done = None
        if status:
            self.done = self._json(self.folder / 'completion.json')
            require(self.done['status'] == status, 'original scientific stage is not complete')
            require(self.done.get('new_holdout_used') is False, 'original population scope differs')
        self.registration = None
        if (self.folder / 'registration.json').exists():
            self.registration = self._json(self.folder / 'registration.json')
            if self.done and 'registration_sha256' in self.done:
                require(sha(self.folder / 'registration.json') == self.done['registration_sha256'],
                        'historical registration SHA differs')
        self._read_table(self.folder / table)
        if study == 'PHASE2_SYSTEM_LOOKUP':
            self._read_table(self.folder / 'digital_adaptive_per_frame.csv')
            self._read_table(self.folder / 'folded_development.csv')
        self.methods = validate_grid(study, self.rows)
        expected_count = {'LATENT_FROZEN_POLICIES': 24, 'LATENT_DECODER_ADAPTATION': 5,
                          'LATENT_FOLDED_512': 1,
                          'LINEAR_REPAIRED': 2, 'PREDICTOR_REPAIRED': 2,
                          'PHASE2_SELECTED_MODELS': 3, 'PHASE2_STRICT_DIGITAL': 6,
                          'PHASE2_PROJECTIONS': 4, 'PHASE2_SYSTEM_LOOKUP': 4,
                          'PHASE2_DIAGNOSTICS': 5}[study]
        # Decoder adaptation's four canonical views and an explicitly published
        # m8_Dc_adapted alias are all kept; older complete four-view CSV is allowed
        # only when its actual completion and exact coverage both state four.
        if study == 'LATENT_DECODER_ADAPTATION' and len(self.methods) == 4:
            require(self.done['rows'] == 6000, 'adaptation alias unexpectedly missing')
        else:
            require(len(self.methods) == expected_count, 'cohort method count differs')
        for row in self.rows:
            self._groups.setdefault(source_index(row), []).append(row)
        self.policies = self._read_frozen_inputs()
        if self.registration:
            self._verify_consumed_sources(self.registration.get('source_snapshot',
                                          self.registration.get('source_bindings', {})))

    def path(self, path):
        text = str(path)
        if text.startswith(PREFIX):
            return self.root / text[len(PREFIX):]
        p = Path(text)
        return p if p.is_absolute() else self.root / p

    def _bind(self, path, expected=None):
        path = self.path(path)
        require(path.is_file(), 'required original input missing: ' + str(path))
        digest = sha(path)
        require(expected is None or digest == expected, 'historical input SHA differs: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest, 'bound input changed')
        self.bindings[str(path)] = digest
        return path

    def _resolve(self, path):
        path = self.path(path)
        rel = path.relative_to(self.root).as_posix()
        entry = self.lineage.get(rel)
        if not path.exists() and entry:
            path = self.root / entry['source']
        return self._bind(path, entry['sha256'] if entry else None)

    def _json(self, path):
        return json.loads(self._resolve(path).read_text(encoding='utf-8'))

    def _read_table(self, path):
        path = self._resolve(path)
        if self.done:
            key = 'development_sha256' if path.name == 'development.csv' else 'per_frame_sha256'
            if key in self.done:
                require(sha(path) == self.done[key], 'original per-frame completion SHA differs')
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        require(rows, 'empty original table')
        self.rows.extend(rows)

    def _read_frozen_inputs(self):
        s = self.study
        if s == 'LATENT_FROZEN_POLICIES':
            p = self.path(self.done['policies'])
            return dict(policy=self._json(p))
        if s in ('LINEAR_REPAIRED', 'PHASE2_PROJECTIONS'):
            folder = self.folder if s == 'LINEAR_REPAIRED' else self.root / 'results/review_20260923_phase2/pca_evaluation'
            rec = self._json(folder / 'projection.json')
            if s == 'PHASE2_PROJECTIONS':
                done = self._json(folder / 'completion.json')
                require(done['status'] == 'LINEAR_MEASUREMENT_DEVELOPMENT_COMPLETE', 'PCA stage incomplete')
                random = self._json(self.root / 'results/review_20260923_phase1/linear/projection.json')
                return dict(pca=rec, random=random)
            return dict(random=rec)
        if s == 'PHASE2_SELECTED_MODELS':
            return dict(training=self._json(self.root / 'results/review_20260923_phase2/training/completion.json'))
        if s == 'PHASE2_SYSTEM_LOOKUP':
            policy = self._json(self.folder / 'frozen_policy.json')
            require(policy['status'] == 'FROZEN_ON_CALIBRATION_ONLY', 'lookup not calibration frozen')
            require(sha(self._resolve(self.folder / 'frozen_policy.json')) == self.done['policy_sha256'], 'lookup seal differs')
            return dict(policy=policy, adaptive=self._json(self.folder / 'digital_adaptive_policies.json'))
        return {}

    def _verify_consumed_sources(self, registered):
        for path, digest in registered.items():
            p = self.path(path)
            if p.suffix in ('.py', '.json', '.yaml') and p.exists():
                # The predictor registration snapshotted every followup module.
                # A later strict-digital repair changed this unrelated module;
                # predictor inference imports no digital_policy_matrix symbols.
                # Preserve the drift proof and reject any later attempt to load it.
                if (self.study == 'PREDICTOR_REPAIRED' and
                    p.relative_to(self.root).as_posix() == EXP+'followup/src/latent_followup/digital_policy_matrix.py' and
                    sha(p) != digest):
                    self.unconsumed_snapshot_drift[str(p)] = dict(registered_sha256=digest,
                        current_sha256=sha(p), reason='unconsumed later strict-digital repair')
                    self._bind(p)
                    continue
                self._bind(p, digest)

    def expected_rows(self, index):
        return [dict(r) for r in self._groups[int(index)]]

    @property
    def records(self):
        require(self._ready, 'setup() must precede records')
        return self._records

    def describe(self):
        return dict(study=self.study, sources=100, rows=len(self.rows), methods=sorted(self.methods),
                    synthetic=False, training_updates=0, policy_selection_updates=0,
                    original_scope_preserved=True, native_main_never_called=True,
                    native_numeric_flags=dict(self.native_numeric_flags),
                    unconsumed_snapshot_drift=self.unconsumed_snapshot_drift)

    @contextmanager
    def _context(self):
        old = list(sys.path)
        e = self.root / EXP
        sys.path[:0] = [str(self.root / 'src'), *[str(e / x) for x in
            ('src', 'phase_b/src', 'evaluation/src', 'followup/src', 'research/src', 'mechanisms/src')]]
        prior = os.environ.get('VAR_COMM_DECODER_GATE')
        gate = self.root / (BASE + 'stage_B_v2_repaired_20260921/decoder_gate.json')
        if gate.exists():
            os.environ['VAR_COMM_DECODER_GATE'] = str(gate)
        try:
            yield
        finally:
            sys.path[:] = old
            if prior is None:
                os.environ.pop('VAR_COMM_DECODER_GATE', None)
            else:
                os.environ['VAR_COMM_DECODER_GATE'] = prior

    def _module(self, name):
        if name not in self.modules:
            module = importlib.import_module(name)
            require(str(Path(module.__file__).resolve()) not in self.unconsumed_snapshot_drift,
                    'Cannot consume a module with unqualified historical source drift')
            self._bind(module.__file__)
            self.modules[name] = module
        return self.modules[name]

    def _checkpoint(self, rec):
        require(isinstance(rec, dict) and 'checkpoint_sha256' in rec, 'unbound selected checkpoint')
        path = self._bind(rec['checkpoint'], rec['checkpoint_sha256'])
        import torch
        return torch.load(path, map_location='cpu', weights_only=True)

    def _load_arm(self, name):
        old = self._json(self.root / (BASE + f'stage_B_v1/training/selected_{name}.json'))
        self.selected[name] = old
        payload = self._checkpoint(old)
        arms = self.bmodel.build_arms((32, 16, 16), self.scale, self.runtime.settings()['stage_B']).to(self.device)
        arms.load_state_dict(payload['arms'], strict=True)
        return arms[name].eval().requires_grad_(False)

    def setup(self):
        if self._ready:
            return self
        with self._context():
            import torch
            self.device = torch.device('cuda:0')
            self.runtime = self._module('latent_enhancement.runtime')
            self.bcommon = self._module('latent_enhancement_b.common')
            self.bmodel = self._module('latent_enhancement_b.model')
            self.latent = self._module('latent_enhancement.latent')
            self.physical = self._module('latent_enhancement_eval.runner')
            self.q = self._module('var_comm.quality')
            self.prior = self._module('var_comm.next_scale_prior')
            self.study_module = self._module('var_comm.study')
            self.progressive = self._module('var_comm.progressive')
            # Match strict historical operators without reinitializing an active
            # process's interop pool. Loading an injected common model set is a
            # reuse of frozen weights; per-study selected arms are still checked.
            torch.set_num_threads(6)
            if torch.get_num_interop_threads() != 2:
                torch.set_num_interop_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            # These two early entry points did not call configure(); preserve
            # their original PyTorch cuDNN default, then require metric parity.
            torch.backends.cudnn.allow_tf32 = self.study in ('LATENT_DECODER_ADAPTATION', 'LATENT_FOLDED_512')
            torch.backends.cudnn.benchmark = False
            require(torch.backends.cuda.matmul.allow_tf32 == self.native_numeric_flags['matmul_tf32'] and
                    torch.backends.cudnn.allow_tf32 == self.native_numeric_flags['cudnn_tf32'], 'native precision initialization differs')
            self._bind(self.root / (EXP + 'configs/experiment.json'))
            self._bind(self.root / 'configs/next_scale_prior_diagnostic.yaml')
            self._bind(self.root / 'configs/progressive_channel.yaml')
            if self.loaded is None:
                paths = self.runtime.model_paths()
                self.vae, self.var = self.prior.load_models(paths, self.device)
                self.decoder = self.bcommon.load_decoder(self.vae, self.device)
                self.scale = self.bcommon.scale_statistics(self.device)
                import yaml
                quality = yaml.safe_load((self.root / 'configs/progressive_channel.yaml').read_text())['quality']
                self.lp, self.dino, linear = self.q.load_quality_models(quality, self.device)
                for k in ('vae_checkpoint', 'var_checkpoint'):
                    self._bind(paths[k], paths[k + '_sha256'])
                for k in ('dino_checkpoint', 'alexnet_checkpoint'):
                    self._bind(quality[k], quality[k + '_sha256'])
                self._bind(linear)
            else:
                self.vae, self.var, self.decoder, self.scale, self.lp, self.dino = (
                    self.loaded[k] for k in ('vae', 'var', 'decoder', 'scale', 'lpips', 'dino'))
                for p, h in self.loaded['bindings'].items():
                    self._bind(p, h)
            gate_path = self.bcommon.decoder_gate_path()
            gate = self._json(gate_path)
            self._bind(gate['selection']['checkpoint'], gate['selection']['checkpoint_sha256'])
            self._bind(self.bcommon.CACHE / 'training_statistics.json')
            self.state = {k: self.prior.state_sha256(v) for k, v in
                dict(vae=self.vae, var=self.var, decoder=self.decoder, lpips=self.lp, dino=self.dino).items()}
            expected_decoder = (self.registration or {}).get('decoder_state_sha256')
            if expected_decoder:
                require(self.state['decoder'] == expected_decoder, 'registered decoder differs')
            targets = self.physical.load_targets() if self.data is None else self.data['targets']
            require(len(targets) == 100, 'source grid differs')
            self.targets = targets
            self._records = []
            for i, target in enumerate(targets):
                require(target['index'] == i, 'target ordering differs')
                sid = target['target']['image_id']
                require(all(r.get('image_id', r.get('source_id', sid)) == sid for r in self._groups[i]), 'source id differs')
                self._records.append(dict(source_index=i, source_id=sid, image_id=sid,
                    preprocessing_id=target['rgb_sha256'], class_index=int(target['target']['class_index']),
                    true_class_index=int(target['target']['class_index']), pixels=target['pixels']))
                self._bind(self.physical.SOURCE_RECON_ROOT / f'images/{i:03d}/reconstructions.npz', target['source_npz_sha256'])
            self._bind(self.physical.TOKENS_PATH)
            self._bind(self.physical.DIGITAL_ROOT / 'population.json')
            self.selected, self.models = {}, {}
            self._load_specific()
            for model in (self.vae, self.var, self.decoder, self.lp, self.dino, *self.models.values()):
                model.eval().requires_grad_(False)
            self.specific_state = {k: self.prior.state_sha256(v) for k, v in self.models.items()}
            self._ready = True
        return self

    def _load_specific(self):
        import torch
        s = self.study
        if s == 'LATENT_FROZEN_POLICIES':
            import historical_cached
            self.aliases['cached'] = historical_cached.create_adapter(self.root, 'LATENT_ORIGINAL')
            for p, h in self.aliases['cached'].bindings.items():
                self._bind(p, h)
        elif s == 'LATENT_DECODER_ADAPTATION':
            self.engine = self._module('latent_followup.adaptation_eval')
            aliases = self.done.get('aliases', {})
            self.adaptation_aliases = aliases
            for name, key in (('communication_continuation_Dc_frozen', 'control'),
                              ('communication_plus_Dc_adaptation', 'adapt')):
                rec = self.done['selected_checkpoints'][name]
                payload = self._checkpoint(rec)
                arm = self.bmodel.build_arms((32, 16, 16), self.scale, self.runtime.settings()['stage_B']).to(self.device)['enhancement1024']
                arm.load_state_dict(payload[key], strict=True)
                self.models[key] = arm
                self.selected[key] = rec
                if key == 'adapt':
                    decoder = self.latent.ContinuousDecoder(self.vae).to(self.device)
                    decoder.load_state_dict(payload['decoder_adapt'], strict=True)
                    self.models['decoder_adapt'] = decoder
        elif s == 'LATENT_FOLDED_512':
            self.engine = self._module('latent_followup.allocation_eval')
            self.models['enhancement512'] = self._load_arm('enhancement512')
            config = self._json(self.root / (EXP + 'followup/config.json'))
            require(tuple(config['snrs_db']) == SNRS and
                    tuple(config['noise_seeds']['development']) == SEEDS, 'folded population differs')
        elif s in ('LINEAR_REPAIRED', 'PHASE2_PROJECTIONS'):
            self.engine = self._module('latent_mechanisms.linear_measurement')
            self.projections = {}
            for name, rec in self.policies.items():
                folder = (self.root / 'results/review_20260923_phase1/linear' if name == 'random' else
                          self.root / 'results/review_20260923_phase2/pca_evaluation')
                rel = (folder / 'projection.json').relative_to(self.root).as_posix()
                require(rel in self.lineage, 'missing original projection lineage')
                origin = self.root / self.lineage[rel]['source']
                path = self._bind(origin.with_name('A.pt'), rec['A_sha256'])
                a = torch.load(path, map_location='cpu', weights_only=True)
                require(tuple(a.shape) == (8192, 1984), 'projection coordinate budget differs')
                self.projections[name] = (a.to(self.device), rec)
        elif s == 'PREDICTOR_REPAIRED':
            self.engine = self._module('latent_mechanisms.predictor_innovation')
            self.predictor_cfg = self._json(self.root / (EXP + 'mechanisms/predictor_innovation_v2_config.json'))
            self.models['predictor'] = self._load_arm('receiver_only_refiner')
            for name, rec in self.done['selected'].items():
                arm = self.bmodel.build_arms((32, 16, 16), self.scale, self.runtime.settings()['stage_B']).to(self.device)['enhancement1024']
                arm.load_state_dict(self._checkpoint(rec)[rec['arm_key']], strict=True)
                self.models[name] = arm
                self.selected[name] = rec
        elif s == 'PHASE2_SELECTED_MODELS':
            self.engine = self._module('latent_research.evaluate')
            train = self._module('latent_research.train')
            parent = self._json(self.root / (BASE + 'stage_B_v1/training/selected_enhancement1024.json'))
            self.models = dict(train.make_models(self.scale, self.runtime.settings()['stage_B'], self._checkpoint(parent), self.device).items())
            self._bind(self.root / (EXP + 'research/config.json'))
            for name, rec in self.registration['selected'].items():
                require(rec == self.policies['training']['selected'][name], 'frozen training selection differs')
                payload = self._checkpoint(rec)
                require(payload['registration_sha256'] == rec['registration_sha256'], 'model checkpoint registration differs')
                self.models[name].load_state_dict({k[len(name)+1:]: v for k, v in payload['models'].items()
                                                 if k.startswith(name + '.')}, strict=True)
                self.selected[name] = rec
        elif s == 'PHASE2_STRICT_DIGITAL':
            self.engine = self._module('latent_followup.digital_policy_matrix')
            require(self.registration['precision'] == {'matmul_tf32': False, 'cudnn_tf32': False}, 'strict digital precision differs')
        elif s == 'PHASE2_DIAGNOSTICS':
            self.models['enhancement1024'] = self._load_arm('enhancement1024')
            require(self.selected['enhancement1024'] == self.done['checkpoint'], 'diagnostic checkpoint changed')
        elif s == 'PHASE2_SYSTEM_LOOKUP':
            self.engine = self._module('latent_research.system_policy')
            require(self.state['decoder'] == self.policies['policy']['decoder_state_sha256'], 'lookup decoder differs')
            self.models['enhancement512'] = self._load_arm('enhancement512')
            self.models['enhancement1024'] = self._load_arm('enhancement1024')
            for name in self.models:
                require(self.selected[name] == self.policies['policy']['selected_arms'][name], 'lookup selected arm changed')
            self.digital = self._module('latent_followup.digital_policy_matrix')

    def metadata(self, row):
        s, name = self.study, method_id(self.study, row)
        adapted = s == 'LATENT_DECODER_ADAPTATION' and name != 'm8_1024_control'
        oracle = name.startswith('oracle_TX') or name == 'Dc_F_representation_reference'
        diagnostic = s == 'PHASE2_DIAGNOSTICS'
        decoder = ('Dc_adapted' if adapted else 'D0' if row.get('renderer') == 'D0' else 'Dc')
        reference = decoder == 'D0' or name == 'Dc_F_representation_reference'
        label_conditioned = name not in ('pure_continuous', 'Dc_F_representation_reference')
        return dict(study=s, scope='historical_diagnostic' if diagnostic else 'historical_development',
                    method_id=name, decoder_id=decoder, model_identity=dict(common=getattr(self, 'state', {}),
                    selected=getattr(self, 'selected', {}), specific=getattr(self, 'specific_state', {})),
                    label_conditioned=label_conditioned, reference_only=reference,
                    N=row.get('N',row.get('budget','')), condition='C' if label_conditioned else 'U',
                    model_id=s+':'+name,
                    classification_main_eligible=not (label_conditioned or reference or diagnostic or oracle),
                    class_condition='paid_class_header' if label_conditioned else 'U',
                    oracle=oracle, zero_noise='zero_noise' in name, output_role='reference' if decoder == 'D0' else 'oracle' if oracle else 'diagnostic' if diagnostic else 'label_conditioned' if name != 'pure_continuous' else 'main',
                    is_main_conclusion=not (diagnostic or oracle or decoder == 'D0' or name != 'pure_continuous'),
                    original_metric_columns={'psnr_db': 'psnr_db', 'lpips_alex': 'lpips_alex' if 'lpips_alex' in row else 'lpips', 'dino_cosine': 'dino_cosine'},
                    snr_field='snr_db', source_index_field='source_index' if 'source_index' in row else 'image_index',
                    noise_seed_field='noise_seed' if 'noise_seed' in row else 'seed',
                    precision=dict(matmul_tf32=False, cudnn_tf32=s in ('LATENT_DECODER_ADAPTATION','LATENT_FOLDED_512')),
                    precision_evidence='inferred_from_original_entrypoint_defaults_then_required_native_metric_parity' if s in ('LATENT_DECODER_ADAPTATION','LATENT_FOLDED_512') else 'original_runtime_configure_strict_false',
                    unconsumed_snapshot_drift=self.unconsumed_snapshot_drift,
                    training_updates=0, policy_selection_updates=0)

    def _base_frame(self, record, snr, seed, source, tx, f):
        import torch
        key = (float(snr), int(seed))
        if key in self._base_frames:
            return self._base_frames[key]
        signal, ledger = self.physical.raw_transmit_budget(source, int(record['target']['class_index']), 8, 3060)
        y = signal + self.study_module.seeded_noise(record['target']['image_id'], seed, signal.shape) / np.sqrt(10 ** (snr / 10))
        phy = self.physical.raw_receive_budget(y, snr, 3060)
        ok = phy['label'] is not None
        rx = self.latent.complete_latent(self.vae, self.var, phy['prefix'], phy['label'], self.device) if ok else torch.zeros_like(tx)
        status = torch.tensor([[float(ok), float(phy['body_crc_accepted']), float(phy['mode'] or 0)]], device=self.device)
        noise = torch.as_tensor(self.latent.enhancement_noise(record['target']['image_id'], seed, 1024)[None], device=self.device, dtype=torch.float32)
        result = rx, status, noise, dict(F=f, Fb_TX=tx, Fb_RX=rx, rx_status=status,
                                      snr_db=torch.tensor([snr], device=self.device), standard_noise=noise)
        self._base_frames[key] = result
        return result

    def _digital_image(self, record, source, row, cache, decoder=None):
        from var_comm.whole_entropy import encode_prefixes
        family = row['family'] if 'family' in row else row['method'].split('_')[0]
        m = int(number(row, 'mode', 'sent_mode', 'selected_mode')) if any(k in row for k in ('mode','sent_mode','selected_mode')) else int(row['method'].split('_m')[1].split('_')[0])
        budget = int(number(row, 'budget', 'N'))
        label = int(record['target']['class_index'])
        key = ('TX', family, budget, m)
        if key not in cache:
            payload = encode_prefixes(self.vae, self.var, source, label, self.device, modes=(m,))[m] if family == 'arithmetic' else None
            cache[key] = (self.physical.raw_transmit_budget(source, label, m, budget) if family == 'raw' else
                          self.physical.arithmetic_transmit_budget(payload, label, m, budget))
        signal, ledger = cache[key]
        snr, seed = number(row, 'snr_db'), noise_seed(row)
        y = signal + self.study_module.seeded_noise(record['target']['image_id'], seed, signal.shape) / np.sqrt(10 ** (snr / 10))
        phy = self.physical.raw_receive_budget(y, snr, budget) if family == 'raw' else self.physical.arithmetic_receive_budget(y, snr, budget)
        renderer = row.get('renderer', 'Dc')
        module = getattr(self, 'digital', None) or self._module('latent_followup.digital_policy_matrix')
        image, item = module.clean_render(phy, family, renderer, self.vae, self.var, decoder or self.decoder,
                                          self.device, cache.setdefault('received_prefix_cache', {}))
        prefix = item.get('prefix', [])
        correct = bool(phy['header']['accepted'] and phy.get('body_crc_accepted',False) and
            phy['label']==label and phy['mode']==m and len(prefix)==m and
            all(np.array_equal(a,b) for a,b in zip(prefix,source[:m])))
        info = dict(header_ok=int(phy['label'] is not None), body_crc_ok=int(phy.get('body_crc_accepted', False)),
                    header_accepted=int(phy['header']['accepted']), body_crc_accepted=int(phy.get('body_crc_accepted', False)),
                    source_complete=int(item['source_complete']), accepted_correct=int(correct),
                    raw_payload_bits=ledger['payload_bits'], actual_payload_bits=ledger['actual_payload_bits'], N=budget,
                    E=float(np.square(signal, dtype=np.float64).sum()))
        return image, info

    def _verify_info(self, row, info):
        for key in ('header_ok', 'body_crc_ok', 'norm_control_ok', 'norm_code',
                    'header_accepted', 'body_crc_accepted', 'source_complete', 'accepted_correct',
                    'raw_payload_bits', 'actual_payload_bits', 'N', 'E', 'normalized_latent', 'enhancement_energy'):
            if key not in row or row[key] == '' or key not in info:
                continue
            if row[key] in ('not_applicable', 'not_a_wireless_method'):
                continue
            if key in ('normalized_latent', 'enhancement_energy', 'E'):
                require(math.isclose(float(row[key]), float(info[key]), abs_tol=1e-5, rel_tol=1e-6), 'native diagnostic differs: '+key)
            else:
                expected = {'True': 1., 'False': 0.}.get(str(row[key]))
                expected = float(row[key]) if expected is None else expected
                require(expected == float(info[key]), 'native actual receiver event differs: '+key)

    def iterate_source(self, index):
        import torch
        with self._context(), torch.no_grad():
            require(self._ready, 'setup() must precede replay')
            require(torch.backends.cuda.matmul.allow_tf32 == self.native_numeric_flags['matmul_tf32'] and
                    torch.backends.cudnn.allow_tf32 == self.native_numeric_flags['cudnn_tf32'] and
                    torch.backends.cudnn.benchmark == self.native_numeric_flags['cudnn_benchmark'],
                    'native inference precision changed between sources')
            self.runtime.require_available()
            i = int(index)
            record, rows = self.targets[i], self.expected_rows(i)
            target = record['pixels'].astype(np.float32) / 255.
            self._base_frames = {}
            image = torch.from_numpy(record['pixels'][None].astype(np.float32) / 127.5 - 1).to(self.device)
            f = self.vae.quant_conv(self.vae.encoder(image))
            source = self.progressive.split_prefix(record['tokens'], 10)
            label, sid = int(record['target']['class_index']), record['target']['image_id']
            tx = None
            if self.study not in ('LATENT_FROZEN_POLICIES', 'PHASE2_STRICT_DIGITAL', 'PHASE2_SELECTED_MODELS'):
                tx = self.latent.complete_latent(self.vae, self.var, source[:8], label, self.device)
            images, infos, cache = [], [], {}
            for row in rows:
                name, snr, seed = method_id(self.study, row), number(row, 'snr_db'), noise_seed(row)
                info = {}
                if self.study == 'LATENT_FROZEN_POLICIES':
                    alias = self.aliases['cached']
                    candidate = exact_alias_row(row, alias.expected_rows(i))
                    img, stored_target, proof = alias.load_image(i, candidate, cache.setdefault('RGB', {}))
                    require(np.array_equal(stored_target, target), 'policy cache source target differs')
                    mode = int(row['selected_mode'])
                    selected = self.policies['policy']['actions'][row['family']][str(int(row['N']))][row['renderer']][str(snr)][row['policy']]
                    require(mode == int(selected), 'frozen policy action differs')
                    info['alias_proof'] = proof
                elif self.study == 'PHASE2_SELECTED_MODELS':
                    img, info = self.engine.execute(record, name, self.models[name], snr, seed,
                        self.vae, self.var, self.decoder, self.device)
                    require(row['checkpoint_sha256'] == self.selected[name]['checkpoint_sha256'], 'row checkpoint differs')
                elif self.study == 'PHASE2_STRICT_DIGITAL':
                    img, info = self._digital_image(record, source, row, cache)
                elif self.study == 'LATENT_DECODER_ADAPTATION':
                    canonical = self.adaptation_aliases.get(name, name)
                    if canonical.startswith('raw_'):
                        img, info = self._digital_image(record, source, {**row, 'method': canonical}, cache, self.models['decoder_adapt'])
                    else:
                        rx, status, noise, batch = self._base_frame(record, snr, seed, source, tx, f)
                        key = 'control' if canonical == 'm8_1024_control' else 'adapt'
                        arm = self.models[key]
                        if ('wave', key) not in cache:
                            cache['wave', key] = arm.encoder(f - tx, tx)[0].cpu().numpy()
                        wave = cache['wave', key]
                        obs = torch.as_tensor(wave[None], device=self.device, dtype=torch.float32) + noise / np.sqrt(10 ** (snr / 10))
                        latent = arm.receiver(obs, rx, batch['snr_db'], status)
                        decoder = self.decoder if key == 'control' else self.models['decoder_adapt']
                        img = self.bmodel.render_received(decoder, latent, status)[0].cpu().numpy()
                        info = dict(header_ok=int(status[0,0]), body_crc_ok=int(status[0,1]))
                        require(row['checkpoint_sha256'] == self.selected[key]['checkpoint_sha256'], 'adaptation row checkpoint differs')
                elif self.study == 'LATENT_FOLDED_512':
                    img, info = self._original_folded_image(record, source, tx, f, row, cache)
                elif self.study in ('LINEAR_REPAIRED', 'PHASE2_PROJECTIONS'):
                    kind = 'pca' if name.startswith('PCA_') else 'random'
                    mode = name.split('_')[-1]
                    a, frozen = self.projections[kind]
                    protocol, w = frozen['protocol'], frozen['W']
                    rx, status, noise, batch = self._base_frame(record, snr, seed, source, tx, f)
                    ff, tt, rr = f[0].reshape(8192), tx[0].reshape(8192), rx[0].reshape(8192)
                    vec = ff if mode == 'source' else ff - tt
                    wave, observation, _ = self.engine.transmit_projection(vec[None] @ a, noise, snr, protocol=protocol)
                    received = self.engine.receive_projection(observation, snr, protocol=protocol)
                    delta = self.engine.correction_coordinates(received['measurement'], rr[None] @ a, received['control_ok'], mode)
                    latent = (rr + w[str(snr)][mode] * (delta @ a.T)).reshape(1,32,16,16)
                    img = self.decoder(latent)[0].cpu().numpy() if bool(status[0,0]) else np.full((3,256,256), .5, np.float32)
                    info = dict(header_ok=int(status[0,0]), body_crc_ok=int(status[0,1]), norm_control_ok=bool(received['control_ok'][0]),
                                norm_code=int(received['norm_code'][0]), enhancement_energy=float(wave.square().sum()))
                elif self.study == 'PREDICTOR_REPAIRED':
                    rx, status, noise, batch = self._base_frame(record, snr, seed, source, tx, f)
                    latent, _, _, _ = self.engine.matched_batch(self.models[name], self.models['predictor'], batch,
                        self.predictor_cfg['predictor_nominal_snr_db'], name == 'predicted_innovation_candidate')
                    img = self.bmodel.render_received(self.decoder, latent, status)[0].cpu().numpy()
                    info = dict(header_ok=int(status[0,0]), body_crc_ok=int(status[0,1]))
                    require(row['checkpoint_sha256'] == self.selected[name]['checkpoint_sha256'], 'predictor row checkpoint differs')
                elif self.study == 'PHASE2_DIAGNOSTICS':
                    rx, status, noise, batch = self._base_frame(record, snr, seed, source, tx, f)
                    if name == 'Dc_F_representation_reference':
                        img = self.decoder(f)[0].cpu().numpy()
                        info = dict(header_ok=int(status[0,0]), normalized_latent=0.)
                    else:
                        arm = self.models['enhancement1024']
                        if 'wave' not in cache:
                            cache['wave'] = arm.encoder(f - tx, tx)
                        base = tx if name.startswith('oracle_TX') else rx
                        multiplier = 0. if name.endswith('zero_noise') else 1.
                        latent = arm.receiver(cache['wave'] + noise * multiplier / np.sqrt(10 ** (snr / 10)),
                                              base, batch['snr_db'], status)
                        img = self.bmodel.render_received(self.decoder, latent, status)[0].cpu().numpy()
                        info = dict(header_ok=int(status[0,0]), normalized_latent=float(((latent-f)/self.scale[None,:,None,None]).square().mean()))
                else:
                    img, info = self._system_image(record, source, tx, f, row, cache)
                self._verify_info(row, info)
                images.append(np.asarray(img, dtype=np.float32));infos.append(info)
            # Keep original quality batches: strict candidates separately; PCA
            # and random projections were separate runs; adaptation aliases reuse
            # their original canonical score instead of changing the batch tail.
            actual = [None] * len(rows)
            groups = {}
            for j, row in enumerate(rows):
                name = method_id(self.study, row)
                if self.study == 'LATENT_DECODER_ADAPTATION' and name in self.adaptation_aliases:
                    continue
                key = name if self.study == 'PHASE2_STRICT_DIGITAL' else name.split('_')[0] if self.study == 'PHASE2_PROJECTIONS' else 'all'
                groups.setdefault(key, []).append(j)
            for inds in groups.values():
                scores = self.q.quality_metrics(target, [images[j] for j in inds], self.lp, self.dino, self.device)[0]
                for j, value in zip(inds, scores):actual[j] = value
            if self.study == 'LATENT_DECODER_ADAPTATION':
                lookup = {(r['method'], number(r,'snr_db'), noise_seed(r)): j for j,r in enumerate(rows)}
                for j,row in enumerate(rows):
                    if row['method'] in self.adaptation_aliases:
                        k = lookup[(self.adaptation_aliases[row['method']], number(row,'snr_db'), noise_seed(row))]
                        require(np.array_equal(images[j],images[k]), 'adapted decoder alias image differs')
                        actual[j] = actual[k]
            for row, img, info, value in zip(rows, images, infos, actual):
                evidence = parity(original_metrics(row), value)
                evidence.update(replay_parity_passed=True, original_row_sha256=canonical_hash(row),
                    native_rgb_sha256=__import__('hashlib').sha256(img.tobytes()).hexdigest(),
                    original_actual_event_verified=True, **info)
                yield dict(row), img, target, evidence
            require({k:self.prior.state_sha256(v) for k,v in self.models.items()} == self.specific_state, 'selected weights changed')

    def _original_folded_image(self, record, source, tx, f, row, cache):
        """Original allocation_v1 adds noise in NumPy before making the tensor."""
        import torch
        snr, seed = number(row, 'snr_db'), noise_seed(row)
        arm = self.models['enhancement512']
        if 'original_folded' not in cache:
            wave = arm.encoder(f-tx,tx)[0].cpu().numpy()
            base,_ = self.physical.raw_transmit_budget(source,int(record['target']['class_index']),8,3572)
            cache['original_folded'] = (base,wave)
        base,wave = cache['original_folded']
        observed = base + self.study_module.seeded_noise(record['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10))
        phy = self.physical.raw_receive_budget(observed,snr,3572)
        if phy['label'] is None:
            rx = torch.zeros_like(tx);status = torch.zeros((1,3),device=self.device)
        else:
            rx = self.latent.complete_latent(self.vae,self.var,phy['prefix'],phy['label'],self.device)
            status = torch.tensor([[1.,float(phy['body_crc_accepted']),float(phy['mode'])]],device=self.device)
        added = wave + self.latent.enhancement_noise(record['target']['image_id'],seed,512)/np.sqrt(10**(snr/10))
        z = arm.receiver(torch.as_tensor(added[None],device=self.device,dtype=torch.float32),rx,
                         torch.tensor([snr],device=self.device),status)
        image = self.bmodel.render_received(self.decoder,z,status)[0].cpu().numpy()
        received_mode = phy['mode'] if phy['mode'] is not None else ''
        accepted = int(phy['header']['accepted'] and phy['body_crc_accepted'] and
            phy['label']==int(record['target']['class_index']) and phy['mode']==8 and
            len(phy['prefix'])==8 and all(np.array_equal(a,b) for a,b in zip(phy['prefix'],source[:8])))
        require(str(row['received_mode']) == str(received_mode), 'folded received mode differs')
        require(int(row['accepted_correct']) == accepted, 'folded correctness event differs')
        return image,dict(header_ok=int(phy['header']['accepted']),body_crc_ok=int(phy['body_crc_accepted']))

    def _system_image(self, record, source, tx, f, row, cache):
        import torch
        name, snr, seed = row['method'], number(row,'snr_db'), noise_seed(row)
        if name == 'calibration_frozen_resource_lookup':
            selected = self.policies['policy']['actions'][str(snr)]
            require(row['selected_endpoint'] == selected, 'lookup policy selection differs')
        elif name.endswith('_adaptive_m789_Dc'):
            family = name.split('_')[0]
            mode = self.policies['adaptive']['actions'][family]['4084']['Dc'][str(snr)]['quality']
            require(int(row['selected_mode']) == int(mode), 'adaptive policy selection differs')
            selected = f'{family}_N4084_m{mode}_Dc'
        else:
            selected = name
        key = ('endpoint', selected, snr, seed)
        if key in cache:
            return cache[key]
        if selected.startswith(('raw_', 'arithmetic_')):
            result = self._digital_image(record, source, {**row, 'method': selected, 'N': '4084'}, cache)
        elif selected == 'm8_plus_latent_1024':
            rx,status,noise,batch = self._base_frame(record,snr,seed,source,tx,f)
            arm = self.models['enhancement1024']
            if 'wave1024' not in cache:cache['wave1024'] = arm.encoder(f-tx,tx)
            latent = arm.receiver(cache['wave1024']+noise/np.sqrt(10**(snr/10)),rx,batch['snr_db'],status)
            result = (self.bmodel.render_received(self.decoder,latent,status)[0].cpu().numpy(),
                      dict(header_ok=int(status[0,0]),body_crc_ok=int(status[0,1]),alias_endpoint=selected))
        elif selected == 'm8_plus_latent_512_fold_N4084':
            if 'folded' not in cache:
                base,_ = self.physical.raw_transmit_budget(source,int(record['target']['class_index']),8,3572)
                wave = self.models['enhancement512'].encoder(f-tx,tx)
                cache['folded'] = (base,wave)
            base,wave = cache['folded']
            latent,status = self.engine.folded_receive(base,wave,tx,record['target']['image_id'],snr,seed,
                self.vae,self.var,self.models['enhancement512'],self.device,cache.setdefault('folded_prefixes',{}))
            result = (self.bmodel.render_received(self.decoder,latent,status)[0].cpu().numpy(),
                      dict(header_ok=int(status[0,0]),body_crc_ok=int(status[0,1]),alias_endpoint=selected))
        else:
            raise ValueError('unsupported frozen lookup endpoint '+selected)
        cache[key] = result
        return result


def create_adapter(root, study, loaded=None, data=None):
    return LatentAdapter(root, study, loaded, data)
