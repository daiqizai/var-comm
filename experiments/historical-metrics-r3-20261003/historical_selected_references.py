"""Selected no-channel references on the original 100 development images.

Seventeen existing views are replayed against their exact original scalar rows.
Dc_U_m8 is a newly registered deterministic reference, with an independent
repeat check, not a fabricated historical scalar comparison. No channel noise,
training, calibration, policy selection, original main, or old cache writer runs.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from history_common import identity, read, sha, write, python_source_in_repo

STUDY = 'SELECTED_NOISELESS_REFERENCES'
MODEL_STATES = {
    'vae': '6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
    'var': 'b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
    'decoder': 'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1',
    'lpips': '973be7f57af4d36c6aeecee198cb98ae315192054653634e3ba5d1a421a5023b',
    'dino': '5ba70ff8cbcc944b73eb1d49f71e1ea54b009ad866b34d0cb8b84ed143aa56eb',
}
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')
TOLERANCES = {'psnr_db': 1e-5, 'lpips_alex': 2e-6, 'dino_cosine': 2e-6}
FULL = ('Dc_F', 'Dc_Fq', 'D0_Fq')
METHODS = FULL + tuple(f'Dc_{kind}_m{m}' for m in range(4, 9) for kind in ('prefix', 'U', 'C'))
REMOTE_ROOT = '/home/liulu/projects/VAR_COMM/'


class ReferenceMismatch(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ReferenceMismatch(message)


def csv_text(value):
    return '' if value is None else str(value)


def output_spec(method):
    if method not in METHODS:
        raise ReferenceMismatch('Unselected no-channel reference: ' + str(method))
    if method in FULL:
        decoder, latent = method.split('_')
        return dict(method=method, decoder=decoder, kind=latent, m=10, label_conditioned=False)
    decoder, kind, scale = method.split('_')
    return dict(method=method, decoder=decoder, kind=kind, m=int(scale[1:]), label_conditioned=kind == 'C')


def scalar_parity(original, actual):
    deltas = {}
    for name in METRICS:
        require(name in original and original[name] != '' and name in actual, 'Missing old scalar ' + name)
        expected, measured = float(original[name]), float(actual[name])
        require(math.isfinite(expected) and math.isfinite(measured), 'Nonfinite reference scalar')
        require(abs(measured - expected) <= TOLERANCES[name], 'No-channel scalar replay differs: ' + name)
        deltas[name] = measured - expected
    if original.get('latent_sq_err_final', '') != '':
        expected, measured = float(original['latent_sq_err_final']), float(actual['latent_sq_err_final'])
        require(math.isfinite(expected) and math.isfinite(measured)
                and abs(measured-expected) <= 1e-5+1e-7*abs(expected), 'Original F recovery error differs')
        deltas['latent_sq_err_final'] = measured-expected
    return dict(replay_parity_passed=True, synthetic=False, original_scalar_parity_available=True,
                parity_origin='original_scalar_replay', metric_deltas=deltas,
                native_metrics={name: float(actual[name]) for name in METRICS})


def deterministic_parity(first, second, actual):
    first, second = valid_rgb(first), valid_rgb(second)
    require(np.array_equal(first, second), 'New deterministic reference failed independent repeat')
    require(all(name in actual and math.isfinite(float(actual[name])) for name in METRICS),
            'New reference native metrics missing')
    return dict(replay_parity_passed=True, synthetic=False, original_scalar_parity_available=False,
                parity_origin='new_deterministic_reference', historical_scalar_comparison='NOT_AVAILABLE',
                independent_repeat_equal=True, independent_repeat_rgb_sha256=rgb_sha(second),
                native_metrics={name: float(actual[name]) for name in METRICS})


def valid_rgb(value):
    a = np.asarray(value)
    require(a.dtype == np.float32 and a.shape == (3, 256, 256), 'float32 CHW256 reference RGB required')
    require(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, 'Invalid reference RGB')
    return np.ascontiguousarray(a)


def rgb_sha(image):
    return hashlib.sha256(valid_rgb(image).tobytes()).hexdigest()


class SelectedReferences:
    def __init__(self, root, study, loaded=None, data=None):
        require(study == STUDY, 'Only the selected no-channel study is supported')
        require(loaded is None and data is None, 'Unqualified model/data injection is forbidden')
        self.root, self.study = Path(root).resolve(), study
        self.bindings, self.records, self.rows, self.locations, self.row_methods = {}, [], [], {}, {}
        self.ready = False
        self.source_folder = self.root/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/source/development'
        self.digital_folder = self.root/'outputs/EXTREME-BW-20261001-R1-N1024/digital'
        self.ref_folder = self.digital_folder/'references/development'
        self.cache_folder = self.root/'outputs/HISTORICAL-METRICS-R3-20261003'/STUDY/'native_reference_reconstructions'
        self._read_originals()

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        require(expected is None or value == expected, 'Original hash differs: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == value,
                'Input changed while registering')
        self.bindings[str(path)] = value
        return value

    def _original_path(self, path):
        text = str(path)
        return self.root/text[len(REMOTE_ROOT):] if text.startswith(REMOTE_ROOT) else Path(text)

    def _json(self, path, expected=None):
        self.bind(path, expected)
        return read(path)

    def _table(self, path, expected):
        self.bind(path, expected)
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))

    def _read_originals(self):
        self.reference_identity = self._json(self.ref_folder/'identity.json')
        ref_done = self._json(self.ref_folder/'completion.json')
        require(ref_done.get('status') == 'SOURCE_REFERENCE_COMPLETE' and ref_done.get('sources') == 100
                and ref_done.get('rows') == 2400 and ref_done.get('reference_only') is True,
                'Completed 100-source m4-m7 references required')
        require(self.reference_identity['assets']['models'] == MODEL_STATES,
                'Frozen visual or original quality model differs')
        require(self.reference_identity['role'] == 'development', 'Development references required')
        ids, prep = self.reference_identity['sources'], self.reference_identity['preprocessing_ids']
        require(len(ids) == len(prep) == 100 and len(set(ids)) == 100, 'Original 100-source identity differs')
        self.records = [dict(source_index=i, image_id=sid, preprocessing_id=prep[i]) for i, sid in enumerate(ids)]
        source_done = self._json(self.source_folder/'completion.json')
        self.source_registration = self._json(self.source_folder/'registration.json', source_done['registration_sha256'])
        require(source_done.get('status') == 'REAL_SOURCE_CODEC_COMPLETE' and source_done.get('sources') == 100
                and source_done.get('source_rows') == 2500 and source_done.get('synthetic') is False,
                'Completed real Source A required')
        require(source_done['decoder_sha256'] == self.source_registration['decoder_sha256'] == MODEL_STATES['decoder'],
                'Source A and current frozen decoder differ')
        source_table = self.source_folder/'source_codec_per_image.csv'
        ref_table = self.digital_folder/'source_references_development.csv'
        arows = self._table(source_table, source_done['files'][source_table.name])
        rrows = self._table(ref_table, ref_done['sha256'])
        require(len(arows) == 2500 and len(rrows) == 2400, 'Original table sizes differ')
        aindex, rindex = {}, {}
        for rows, target in ((arows, aindex), (rrows, rindex)):
            for number, row in enumerate(rows):
                key = (int(row['source_index']), row['method'])
                require(key not in target, 'Duplicate original reference')
                require(row['population'] == 'development' and row['source_id'] == ids[key[0]]
                        and row['preprocessing_id'] == prep[key[0]], 'Original reference source differs')
                require(row.get('noise_seed', '') == '' and row.get('snr_db', '') == '',
                        'No-channel reference contains noise/SNR repetition')
                target[key] = (row, number)
        for i, record in enumerate(self.records):
            acell_path = self.source_folder/'images'/f'{i:04d}.json'
            acell = self._json(acell_path, source_done['source_receipts'][acell_path.name])
            rcell_path = self.ref_folder/f'{i:04d}.json'
            rcell = self._json(rcell_path)
            require(acell['registration_sha256'] == source_done['registration_sha256']
                    and acell['source_id'] == record['image_id'], 'Original Source A cell differs')
            require(rcell['identity_sha256'] == sha(self.ref_folder/'identity.json'), 'Original reference cell differs')
            cells = ({row['method']: row for row in acell['rows']}, {row['method']: row for row in rcell['rows']})
            selected = []
            for method in METHODS:
                spec = output_spec(method)
                if method == 'Dc_U_m8':
                    row = dict(population='development', source_id=record['image_id'], source_index=i,
                        preprocessing_id=record['preprocessing_id'], method=method, action_m=8, source_bits=3060,
                        N='', E='', decoder_id='Dc', class_condition='U', noise_seed='', snr_db='',
                        reference_only=True, finite_channel_result=False,
                        reference_origin='new_deterministic_reference_20261003', historical_scalar_reference=False)
                    location = dict(origin='new_registered_reference', original_scalar_parity_available=False)
                else:
                    from_a = method in FULL or spec['m'] == 8
                    old_method = 'Dc_VAR_m8' if method == 'Dc_C_m8' else method
                    table, index, cell, cell_path = ((source_table, aindex, cells[0], acell_path) if from_a else
                                                   (ref_table, rindex, cells[1], rcell_path))
                    row, number = index[i, old_method]
                    typed = cell[old_method]
                    require(all(csv_text(typed.get(k)) == v for k, v in row.items()), 'Original cell/table disagree')
                    require(all(row.get(name, '') != '' for name in METRICS), 'Old reference metrics missing')
                    location = dict(path=str(table), sha256=self.bindings[str(table.resolve())], csv_row=number,
                        cell_path=str(cell_path), cell_sha256=self.bindings[str(cell_path.resolve())], original_method=old_method,
                        original_scalar_parity_available=True,
                        origin='source_A' if from_a or row.get('reference_origin') == 'compatible_Source_A' else 'extreme_N1024_reference')
                row = dict(row)
                self.locations[identity(row)] = location
                self.row_methods[identity(row)] = method
                selected.append(row)
            self.rows.append(selected)
        require(sum(map(len, self.rows)) == 1800, 'Selected no-channel scope differs')

    def expected_rows(self, source_index):
        require(type(source_index) is int and 0 <= source_index < 100, 'Invalid source index')
        return [dict(row) for row in self.rows[source_index]]

    def describe(self):
        return dict(study=self.study, sources=100, rows=1800, methods=list(METHODS),
                    original_scalar_rows=1700, new_deterministic_rows=100, noise_seeds=[], snrs_db=[],
                    channel='none', reference_only=True, population='development', training_updates=0,
                    policy_updates=0, new_holdout=False, old_metric_tolerances=TOLERANCES)

    def metadata(self, row):
        method = self.row_methods[identity(row)]
        spec = output_spec(method)
        fresh = method == 'Dc_U_m8'
        return dict(study=self.study, method_id=method, scope='source_only_reference',
            decoder=spec['decoder'], model_identity=MODEL_STATES, N='', E='', phy='no_channel',
            snr_db='', snr_definition='not_applicable_no_channel', noise_seed_role='not_applicable',
            condition=spec['kind'], prefix_scales=spec['m'], label_conditioned=spec['label_conditioned'],
            class_condition='C' if spec['label_conditioned'] else 'U' if spec['kind'] == 'U' else 'none',
            class_embedding_index=1000 if spec['kind'] == 'U' else 'source_class' if spec['kind'] == 'C' else None,
            reference_only=True, output_role='source_reference', oracle=False,
            classification_main_eligible=False, classification_caveat='True class supplied to no-channel C reference'
                if spec['label_conditioned'] else 'No-channel reference; not a communication accuracy claim',
            original_location=self.locations[identity(row)], original_scalar_parity_available=not fresh,
            native_reference_kind='new_deterministic_reference' if fresh else 'original_scalar_replay',
            original_metric_columns={k: 'new_native_'+k if fresh else k for k in METRICS},
            completion='deterministic_argmax_no_CFG_no_sampling', channel_uses_not_applicable=True,
            training_updates=0, policy_selection_updates=0)

    def setup(self):
        if self.ready:
            return self
        here = self.root/'experiments/extreme-bandwidth-20261001-N1024'
        for name in ('assets', 'digital', 'digital_protocol'):
            old = sys.modules.get(name)
            require(old is None or Path(old.__file__).resolve().parent == here,
                    'Conflicting original module already loaded: ' + name)
        sys.path.insert(0, str(here))
        self.assets = importlib.import_module('assets')
        self.native = importlib.import_module('digital')
        require(self.assets.ROOT.resolve() == self.root, 'Original checkout differs')
        for paths in (self.reference_identity['source_bindings'], self.source_registration['bindings']):
            for path, expected in paths.items():
                self.bind(self._original_path(path), expected)
        self.loaded = self.assets.setup()
        require(self.loaded['identity'] == self.reference_identity['assets'], 'Original frozen assets changed')
        self.torch, self.device = self.native.torch, self.loaded['device']
        from latent_enhancement_eval.runner import load_targets, TOKENS_PATH, DIGITAL_ROOT
        from latent_enhancement_b.common import load_gate, decoder_gate_path
        from latent_enhancement.runtime import model_paths
        from var_comm.quality import quality_metrics
        self.source_quality = quality_metrics
        self.bind(TOKENS_PATH)
        self.bind(DIGITAL_ROOT/'population.json')
        paths = model_paths()
        for name in ('vae_checkpoint', 'var_checkpoint'):
            self.bind(paths[name], paths[name+'_sha256'])
        gate = load_gate()
        self.bind(decoder_gate_path(), self.loaded['identity']['decoder_gate_sha256'])
        self.bind(gate['selection']['checkpoint'], gate['selection']['checkpoint_sha256'])
        for group in ('stage_A_bindings', 'frozen_stage_A_sources'):
            for path, expected in gate[group].items():
                self.bind(path, expected)
        targets = load_targets()
        require(len(targets) == 100, 'Native original population differs')
        for i, target in enumerate(targets):
            record = self.records[i]
            pixels = target['pixels']
            require(record['image_id'] == target['target']['image_id'] and pixels.dtype == np.uint8
                    and pixels.shape == (3, 256, 256)
                    and hashlib.sha256(pixels.tobytes()).hexdigest() == record['preprocessing_id'],
                    'Native source/pixel identity differs')
            record.update(class_index=int(target['target']['class_index']), pixels=pixels,
                          tokens=target['tokens'])
            self.bind(self.root/'outputs/VAR-PROGRESSIVE-CHANNEL-001/images'/f'{i:03d}'/'reconstructions.npz',
                      target['source_npz_sha256'])
        for module in tuple(sys.modules.values()):
            path = python_source_in_repo(module, self.root)
            if path is not None:
                self.bind(path)
        self.cache_binding = identity(dict(input_bindings=self.bindings, methods=METHODS, models=MODEL_STATES,
                                          protocol='selected_no_channel_replay_v1'))
        self.ready = True
        return self

    def _cache(self, index):
        path = self.cache_folder/f'{index:04d}.npz'
        receipt = self.cache_folder/f'{index:04d}.json'
        return path, receipt

    def _read_cache(self, index):
        path, receipt = self._cache(index)
        if not receipt.exists():
            require(not path.exists(), 'Orphan reference RGB cache needs explicit repair')
            return None
        proof = read(receipt)
        require(proof.get('payload_sha256') == identity({k:v for k,v in proof.items() if k != 'payload_sha256'}),
                'Reference cache receipt checksum differs')
        require(proof['binding'] == self.cache_binding and proof['rgb_archive_sha256'] == sha(path)
                and proof['original_row_ids'] == [identity(r) for r in self.rows[index]], 'Reference cache identity differs')
        with np.load(path, allow_pickle=False) as archive:
            require(archive['methods'].tolist() == list(METHODS), 'Reference cached methods differ')
            images, target = archive['images'].copy(), valid_rgb(archive['target'].copy())
        require(images.shape == (18, 3, 256, 256) and len(proof['parities']) == 18, 'Reference RGB cache incomplete')
        require(rgb_sha(target) == proof['target_sha256'], 'Cached target differs')
        for row, image, parity in zip(self.rows[index], images, proof['parities']):
            valid_rgb(image)
            require(parity['replay_parity_passed'] is True and parity['synthetic'] is False
                    and parity['rgb_sha256'] == rgb_sha(image)
                    and parity['original_row_sha256'] == identity(row), 'Cached parity differs')
        return images, target, proof['parities']

    def iterate_source(self, index):
        require(self.ready, 'Setup is required before native inference')
        rows = self.expected_rows(index)
        cached = self._read_cache(index)
        if cached is None:
            rec = self.records[index]
            target = valid_rgb(rec['pixels'].astype(np.float32)/255.)
            with self.torch.no_grad():
                self.assets.check()
                vae, var, decoder = (self.loaded[k] for k in ('vae', 'var', 'decoder'))
                f = vae.quant_conv(vae.encoder(self.torch.as_tensor(rec['pixels'][None],
                    dtype=self.torch.float32, device=self.device)/127.5-1))
                scales = self.native.split_tokens(rec['tokens'])
                fresh = vae.quantize.f_to_idxBl_or_fhat(f, to_fhat=False)
                require(all(np.array_equal(x[0].cpu().numpy(), y) for x, y in zip(fresh, scales)),
                        'Original token/encoder identity differs')
                latents = {'F': f.cpu(), 'Fq': self.native.prefix_latent(vae, scales).cpu()}
                outputs = {}
                for method in METHODS:
                    spec = output_spec(method)
                    if spec['kind'] in ('F', 'Fq'):
                        z = latents[spec['kind']]
                    elif spec['kind'] == 'prefix':
                        z = self.native.prefix_latent(vae, scales[:spec['m']]).cpu()
                    else:
                        z = self.native.complete_latent(vae, var, scales[:spec['m']],
                            1000 if spec['kind'] == 'U' else rec['class_index'], self.device).cpu()
                    outputs[method] = (z, spec['decoder'])
                computed = self.native.quality_outputs(self.loaded, rec, outputs, images=True)
                images = [valid_rgb(computed[method][1]) for method in METHODS]
                source_indices = [j for j, row in enumerate(rows) if self.locations[identity(row)]['origin'] == 'source_A']
                source_metrics, *_ = self.source_quality(target, [images[j] for j in source_indices],
                    self.loaded['lpips'], self.loaded['dino'], self.device)
                source_actual = dict(zip(source_indices, source_metrics))
                # This new view has no earlier scalar value. Run its generation
                # again without ReceivedCache, then require exact float RGB.
                repeat_z = self.native.complete_latent(vae, var, scales[:8], 1000, self.device)
                repeat_rgb = decoder(repeat_z)[0].cpu().numpy()
                parities = []
                for j, (row, method, image) in enumerate(zip(rows, METHODS, images)):
                    actual = dict(source_actual.get(j, computed[method][0]))
                    actual['latent_sq_err_final'] = float((f[0].cpu().double()-outputs[method][0][0].double()).square().sum())
                    parity = deterministic_parity(image, repeat_rgb, actual) if method == 'Dc_U_m8' else scalar_parity(row, actual)
                    parity.update(original_row_sha256=identity(row), original_location=self.locations[identity(row)],
                                  rgb_sha256=rgb_sha(image), target_sha256=rgb_sha(target),
                                  no_channel=True, noise_repetitions=0)
                    parities.append(parity)
                self.assets.assert_frozen(self.loaded)
            self.cache_folder.mkdir(parents=True, exist_ok=True)
            path, receipt = self._cache(index)
            temp = path.with_name(path.name+'.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, methods=np.asarray(METHODS), images=np.stack(images), target=target)
            os.replace(temp, path)
            proof = dict(status='REAL_SELECTED_REFERENCE_RGB_CACHE', binding=self.cache_binding,
                source_index=index, source_id=rec['image_id'], rgb_archive_sha256=sha(path),
                original_row_ids=[identity(row) for row in rows], parities=parities,
                target_sha256=rgb_sha(target), new_deterministic_rows=1, original_scalar_rows=17,
                training_updates=0, policy_selection_updates=0)
            proof['payload_sha256'] = identity(proof)
            write(receipt, proof)
            cached = (images, target, parities)
        for row, image, parity in zip(rows, cached[0], cached[2]):
            yield row, image.copy(), cached[1].copy(), dict(parity)


def create_adapter(root, study, loaded=None, data=None):
    return SelectedReferences(root, study, loaded, data)
