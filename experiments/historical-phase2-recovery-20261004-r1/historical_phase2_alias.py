"""R1: explicit Phase2-to-continuous-grid numerical-implementation alias.

The original R3 sources, scalar values, 81 checkpoints and float caches are
read-only. New metrics always consume the actual grid RGB; no metric receives
an offset. The registered old-table difference is audit evidence only.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import historical_selected_optional as original
import historical_receiver as receiver
import historical_budget as budget

STUDY = 'OPTIONAL_PHASE2_MAIN'
REVISION = 'PHASE2-HISTORICAL-ALIAS-20261004-R1'
TABLE_SHA = {
    'results/review_20260923_phase2/comparison/per_frame.csv':
        '07e029db51882233cb28a1cf6171b000f2dcf62f7e9478ab01ec11ea8ae96f81',
    'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/continuous_grid_v1/per_frame.csv':
        '2464ad86354e09418f8ba26a1c26d6c67188c2c0c24783e5971294c07b525915'}
CHECKPOINT = 'b6bd1f0bb01bbd734eeb6a63afac1b8f7c57998085c616011f37554035434e1a'
DECODER = 'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'
METRICS = receiver.METRICS
LIMITS = dict(receiver.TOLERANCES)
require = receiver.require


def values(row):
    answer = {name: float(row[name]) for name in METRICS}
    require(all(math.isfinite(v) for v in answer.values()), 'Finite legacy measurements required')
    return answer


def pair_record(phase2, grid):
    """Identity audit. A numerical difference never admits an unrelated pair."""
    require(phase2['method'] == 'pure_continuous' and grid['method'] == 'P4084', 'Only registered pure/grid alias')
    require(original.key(phase2) == original.key(grid), 'Alias source/SNR/noise key differs')
    require(phase2['image_id'] == grid['source_id'], 'Alias image identity differs')
    require(float(phase2['N']) == float(grid['N']) == 4084 and
            abs(float(phase2['E']) - 8168) < .02 and abs(float(grid['E']) - 8168) < .02, 'Alias resource differs')
    require(phase2['model_context_sha256'] == grid['checkpoint_sha256'] == CHECKPOINT, 'Alias checkpoint differs')
    require(phase2['decoder_sha256'] == DECODER, 'Alias frozen decoder differs')
    require(grid['preprocessing_id'] and len(grid['preprocessing_id']) == 64, 'Grid source pixels require SHA identity')
    require(grid['noise_namespace'] == 'VAR-CONTINUOUS-4084', 'Alias noise namespace differs')
    a, b = values(phase2), values(grid)
    return dict(source_index=original.index(phase2), snr_db=original.snr(phase2), noise_seed=original.seed(phase2),
        source_id=phase2['image_id'], grid_preprocessing_id=grid['preprocessing_id'],
        phase2_original_row_sha256=receiver.canonical_hash(phase2), grid_original_row_sha256=budget.identity(grid),
        phase2_original_metrics=a, grid_original_metrics=b,
        historical_grid_minus_phase2={m: b[m] - a[m] for m in METRICS},
        checkpoint_sha256=CHECKPOINT, decoder_sha256=DECODER,
        grid_waveform_sha256=grid['waveform_sha256'], grid_observation_sha256=grid['observation_sha256'])


def observed_alias_proof(phase2, grid, actual, base_proof, image, target, table_bindings):
    """Strict measured grid replay plus separately recorded historical delta."""
    pair = pair_record(phase2, grid)
    checked = receiver.parity(grid, actual)  # Existing strict limits, unchanged.
    require(base_proof.get('passed') is True and base_proof.get('replay_parity_passed') is True
            and base_proof.get('synthetic') is False, 'Native strict grid proof required')
    require(base_proof.get('original_row_sha256') == budget.identity(grid), 'Native grid row proof differs')
    require(base_proof.get('rgb_sha256') == hashlib.sha256(image.tobytes()).hexdigest(), 'Actual image differs from strict grid replay')
    require(base_proof.get('target_sha256') == hashlib.sha256(target.tobytes()).hexdigest(), 'Actual target differs from strict grid replay')
    require({'waveform_sha256','observation_sha256'} <= set(base_proof.get('exact_fields_checked', [])),
            'Exact physical waveform/observation checks required')
    measured = values(actual)
    residual = {m: measured[m] - pair['grid_original_metrics'][m] for m in METRICS}
    direct = {m: measured[m] - pair['phase2_original_metrics'][m] for m in METRICS}
    for name in METRICS:
        require(abs(residual[name]) <= LIMITS[name], 'New grid metric changed beyond original strict limit')
        require(abs(direct[name] - (pair['historical_grid_minus_phase2'][name] + residual[name])) <= 1e-12,
                'Historical and fresh deltas do not close')
    return dict(replay_parity_passed=True, synthetic=False, training_updates=0, policy_selection_updates=0,
        parity_basis='strict native grid replay; explicitly bound Phase2 historical numerical alias',
        parity_revision=REVISION, original_row_sha256=receiver.canonical_hash(phase2),
        native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        native_target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
        native_grid_replay_proof=base_proof, direct_fresh_grid_metric_parity=checked,
        historical_alias=pair, measured_native_grid_metrics=measured,
        measured_grid_minus_original_grid=residual, measured_grid_minus_original_phase2=direct,
        original_phase2_scalar_values_preserved=True, new_metric_offset_applied=False,
        numerical_implementation='token_efficiency.execution continuous grid',
        original_phase2_pixel_identity_claimed=False, original_phase2_scalar_parity_claimed=False,
        historical_table_bindings=table_bindings)


class Phase2AliasAdapter(original.Phase2View):
    def __init__(self, root):
        super().__init__(root)
        self.alias_table_bindings = {}
        for relative, expected in TABLE_SHA.items():
            path = self.root / relative
            self.native.bind(path, expected)
            self.alias_table_bindings[str(path)] = expected
        require(self.native.context['decoder_sha256'] == DECODER, 'Loaded grid decoder provenance differs')
        self.native.bind(Path(__file__).resolve())
        # Original Phase2 implementation evidence is separately bound: it used
        # CPU normalization and float64 AWGN, versus GPU/float32 in the grid.
        self.phase2_registration_path = self.root/'results/review_20260923_phase2/evaluation/registration.json'
        self.native.bind(self.phase2_registration_path)
        registration = budget.read(self.phase2_registration_path)
        require(registration['selected']['pure_continuous']['checkpoint_sha256'] == CHECKPOINT,
                'Original Phase2 selected model differs')
        for path, expected in registration['source_snapshot'].items():
            self.native.bind(path, expected)
        self.aliases = []
        for row in self.rows:
            if row['method'] == 'pure_continuous':
                i, snr, seed = original.key(row)
                grid = self.native._main[i, 'P4084', int(snr), seed]
                self.aliases.append(pair_record(row, grid))
        require(len(self.aliases) == 1500, 'Complete frozen old-table pair inventory required')
        outside = [r for r in self.aliases if abs(r['historical_grid_minus_phase2']['psnr_db']) > 1e-5]
        require(len(outside) == 1 and (outside[0]['source_index'],outside[0]['snr_db'],outside[0]['noise_seed']) == (81,1.,2003)
                and outside[0]['historical_grid_minus_phase2']['psnr_db'] == -1.52587890625e-05,
                'Frozen historical incident inventory differs')

    def historical_alias_manifest(self):
        return dict(status='FROZEN_HISTORICAL_PHASE2_GRID_ALIAS_AUDIT', revision=REVISION,
            sources=100, rows=1500, all_sources_disclosed=True, aliases=self.aliases,
            historical_table_bindings=self.alias_table_bindings,
            checkpoint_sha256=CHECKPOINT, decoder_sha256=DECODER,
            original_phase2='CPU NumPy float32 source normalization; float64 AWGN before RX float32 cast; quality interleaves three methods',
            replay_grid='GPU float32 source normalization; float32 AWGN multiply/add; quality batches 15 grid frames',
            same_pixels_or_waveforms_between_historical_implementations_claimed=False,
            strict_grid_parity_tolerances={m: LIMITS[m] for m in METRICS}, new_metric_offset_applied=False,
            actual_new_metric_image='unchanged actual continuous-grid receiver RGB',
            inherited_sources_require_original_registration_and_parity_hashes=True)

    def metadata(self, row):
        meta = super().metadata(row)
        if row['method'] == 'pure_continuous':
            meta.update(historical_numerical_alias_revision=REVISION,
                reconstructed_implementation='token_efficiency.execution continuous grid',
                phase2_exact_pixel_alias=False, new_metric_offset_applied=False)
        return meta

    def iterate_source(self, source_index):
        n, i = self.native, int(source_index)
        require(n.ready, 'setup required')
        target = n.records[i]['pixels'].astype(np.float32)/255
        pure = {original.key(r):(r,img,ref,proof) for r,img,ref,proof in n.iterate_source(i)}
        pure_keys = [(i,float(s),seed) for s in original.latent.SNRS for seed in original.SEEDS]
        require(set(pure) == set(pure_keys), 'Complete exact grid replay required')
        with n.torch.no_grad():
            fresh = n.native.quality_metrics(target, [pure[k][1] for k in pure_keys], n.lp, n.dino, n.device)[0]
        fresh_grid = dict(zip(pure_keys,fresh))
        folded = {original.key(r):(img,ref,proof) for r,img,ref,proof in self.fold.iterate_source(i)}
        cache, originals, images = {}, {}, []
        for key in pure_keys:
            image, reference, proof = self.cache.load_image(i,self.candidates[key],cache)
            require(np.array_equal(reference,target), 'Cached source pixels differ from Phase2 target')
            originals[key] = image,proof; images.append(image)
        with n.torch.no_grad():
            scores = n.native.quality_metrics(target,images,n.lp,n.dino,n.device)[0]
        score_by_key = dict(zip(pure_keys,scores))
        for row in self.by_source[i]:
            key = original.key(row)
            if row['method'] == 'pure_continuous':
                grid,image,reference,base_proof = pure[key]
                require(np.array_equal(reference,target), 'Pure target differs')
                proof = observed_alias_proof(row,grid,fresh_grid[key],base_proof,image,target,self.alias_table_bindings)
            elif row['method'] == 'calibration_frozen_resource_lookup' and original.snr(row) == 1:
                image,reference,base_proof = folded[key]
                require(np.array_equal(reference,target), 'Folded target differs')
                original.assert_alias(row,self.fold_rows[key]); proof=dict(base_proof)
            else:
                image,base_proof = originals[key]
                proof=receiver.parity(row,score_by_key[key])
                proof.update(cache_alias_proof=base_proof,alias_of='m8_plus_latent_1024')
            proof.update(replay_parity_passed=True,original_row_sha256=receiver.canonical_hash(row),
                native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            yield dict(row),image,target,proof


def create_adapter(root, study, loaded=None, data=None):
    require(study == STUDY and loaded is None and data is None, 'Only the bound Phase2 continuation is admitted')
    return Phase2AliasAdapter(root)
