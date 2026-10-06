"""Local prepared H18 metric adapter; callers supply already loaded frozen models.

No model constructor, network, GPU admission, subprocess, packet decoder, policy
selection, statistics or runnable entry point is implemented here. A future
registered metric driver must first prove the normal render owner/worker exits.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

COUNT, SLOTS, SEEDS = 100, 18, (6201, 6202, 6203)
REPLAY_SHA = '9582ce2f89c430369b7b19fd7810d1f90e56240f6d7b9845506fc57ded00bcbd'
CONVNEXT_SHA = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
FINAL = {'RAW_SOURCE_DECODED', 'ARITHMETIC_SOURCE_DECODED', 'WIRE_REJECT_GRAY', 'ARITHMETIC_SOURCE_INVALID_GRAY'}
REQUIRED = ('lpips_alex', 'dino_cosine', 'dinov2_vitl14_cosine', 'clip_image_cosine', 'dists',
    'dreamsim', 'ms_ssim', 'resnet50_prediction', 'resnet50_source_prediction', 'resnet50_top1_label',
    'resnet50_top1_source_prediction', 'resnet50_source_top1_label', 'resnet50_top1_probability',
    'semantic_error', 'confidently_wrong', 'convnext_prediction', 'convnext_source_prediction',
    'convnext_top1_label', 'convnext_top1_source_prediction', 'convnext_source_prediction_agreement',
    'convnext_source_correct_to_wrong', 'convnext_source_wrong_to_correct', 'dino_mismatched', 'dino_specificity')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(bindings):
    for path, checksum in bindings.items(): require(sha(path) == checksum, 'Bound input changed: ' + str(path))


def pixels(value):
    a = np.asarray(value)
    require(a.dtype == np.float32 and a.shape == (3, 256, 256) and np.isfinite(a).all()
            and a.min() >= 0 and a.max() <= 1, 'Only original float32 CHW RGB01 may be scored')
    return np.ascontiguousarray(a)


def rgb_sha(value):
    a = pixels(value)
    # This is h64_phy.array_sha's existing image domain, not the old A2 hash.
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def frozen_mismatch(replay_module, source_ids):
    """Use the actual original static function; no new permutation is selected."""
    path = Path(replay_module.__file__)
    require(sha(path) == REPLAY_SHA and len(source_ids) == len(set(source_ids)) == COUNT,
            'Exact original replay source and development100 required')
    values = replay_module.ReplayEngine._derangement()
    require(len(values) == COUNT and set(values) == set(range(COUNT))
            and all(type(j) is int and i != j for i, j in enumerate(values)), 'Invalid original derangement')
    return dict(source_ids=list(source_ids), permutation=values, seed=20260930,
                replay_source_sha256=REPLAY_SHA, selection=False)


def validate_source_rows(rows, index, sid, schedule):
    require(type(index) is int and 0 <= index < COUNT and len(rows) == 54 and len(schedule) == SLOTS,
            'One complete H development source required')
    byslot = {r['development_slot']: r for r in schedule}
    require(set(byslot) == set(range(SLOTS)), 'Reserved MAIN or duplicated/missing H slot')
    for slot, declaration in byslot.items():
        role = 'H_WHOLE_SYSTEM' if slot < 8 else 'H_RAW_PARTIAL_SYSTEM' if slot < 10 else 'H_FIXED_M7_ATTRIBUTION'
        require(declaration['status'] == 'FROZEN_POLICY_READY' and declaration['phase'] == 'development'
                and declaration['role'] == role, 'Original eighteen ready H declarations required')
    seen = set()
    for row in rows:
        key = row['development_slot'], row['noise_seed']
        require(key in {(s, n) for s in range(SLOTS) for n in SEEDS} and key not in seen,
                'Missing/duplicate slot or old noise seed')
        seen.add(key); expected = byslot[key[0]]; c = expected['candidate']
        require(row['source_index'] == index and row['source_id'] == sid and row['phase'] == 'development'
                and row['snr_db'] == c['snr_db'] and row['snr_db'] in (13, 19)
                and row['candidate_id'] == c['candidate_id'] and row['role'] == expected['role']
                and row['arm'] == c['arm'] and row['source_status'] in FINAL
                and row['rx_summary']['source_decode_complete'] is True,
                'Rendered result is not its frozen actual-RX declaration')
    return rows


def declared_points(schedule):
    """Stable public names for all eighteen declarations, without quality ranking."""
    require(len(schedule) == SLOTS and {r['development_slot'] for r in schedule} == set(range(SLOTS)),
            'All frozen eighteen declarations required')
    return {f'H18_SLOT_{r["development_slot"]:02d}': dict(snr_db=r['candidate']['snr_db'], role=r['role'],
            candidate_id=r['candidate']['candidate_id']) for r in sorted(schedule, key=lambda r:r['development_slot'])}


def attach_wire_accounting(scored_rows, source_trace):
    """Post-score attribution only: TX fallback never supplies reconstruction.

    The caller verifies the original CPU trace SHA in its completed receipt.
    TX source statistics count each source once; accepted-RX distributions count
    all three noisy receptions and retain legally accepted wrong content.
    """
    require(source_trace['status'] == 'H_DEVELOPMENT_CPU_SOURCE_TRACES' and len(scored_rows) == len(source_trace['frames']) == 54,
            'One complete CPU trace and scored H source required')
    frames = {f['event_id']: f for f in source_trace['frames']}
    require(len(frames) == 54 and {r['event_id'] for r in scored_rows} == set(frames), 'Actual event pairing differs')
    output = []; source_choices = {}
    for row in scored_rows:
        frame = frames[row['event_id']]
        require(all(row[k] == frame[k] for k in ('source_id','source_index','development_slot','noise_seed','snr_db','candidate_id','role','arm'))
                and frame['stage'] == 'development' and frame['total_symbols'] == 1024
                and frame['frame_normalized'] is False, 'Wire/score identity or paid resource differs')
        tx = frame['tx']; choice = (tx['profile_id'], tx['target_m'], tx['actual_m'], tx['K'], tx['mode'], tx['fell_back'])
        require(type(tx['actual_m']) is int and 1 <= tx['actual_m'] <= 10 and type(tx['fell_back']) is bool,
                'Explicit transmitted source scale required')
        previous = source_choices.setdefault(row['development_slot'], choice)
        require(previous == choice, 'Sender source choice must not depend on receiver noise')
        if row['role'] == 'H_FIXED_M7_ATTRIBUTION':
            require((tx['target_m'], tx['actual_m'], tx['K'], tx['fell_back']) == (7,7,0,False)
                    and row['nominal_rate'] == '1/2', 'Shared fixed-m7 attribution changed')
        accepted = row['source_status'] in ('RAW_SOURCE_DECODED', 'ARITHMETIC_SOURCE_DECODED')
        require(accepted != row['gray'], 'Final actual RX status/gray differs')
        for field in ('header_energy', 'body_energy', 'total_energy'):
            require(math.isfinite(frame[field]) and frame[field] >= 0, 'Invalid actual transmitted energy')
        output.append(dict(row, tx_target_m=tx['target_m'], tx_actual_m=tx['actual_m'], tx_actual_K=tx['K'],
            tx_source_mode=tx['mode'], tx_fell_back=tx['fell_back'], tx_profile_id=tx['profile_id'],
            rx_accepted_m=row['received_m'] if accepted else None, rx_accepted_K=row['received_K'] if accepted else None,
            rx_declared_profile_id=row['received_profile_id'], rx_canonical_accepted=accepted,
            header_energy=frame['header_energy'], body_energy=frame['body_energy'], total_energy=frame['total_energy'],
            total_symbols=1024, frame_normalized=False, accounting_scope='POST_SCORE_ONLY_NO_RX_INPUT',
            tx_scale_denominator='one source per declared point', rx_scale_denominator='all three received frames per source',
            online_latency_status='NOT_MEASURED_BY_CACHED_RENDER_OR_METRIC_BATCH'))
    return output


def load_source(index, population, asset_manifest, asset_completion, render_completion, render_out, schedule):
    """Read only the already sealed source pixels and actual float RGB.

    The caller binds all five input documents and proves normal owner closure.
    This function additionally verifies every per-source archive/checkpoint hash.
    It never extracts source tokens or receiver bit arrays.
    """
    require(population['stage'] == population['calibration_or_development'] == 'm1_development'
            and len(population['source_ids']) == len(set(population['source_ids'])) == COUNT,
            'Original development100 population required')
    require(render_completion['status'] == 'H_DEVELOPMENT_RX_COMPLETE'
            and render_completion['source_count'] == COUNT and render_completion['frame_count'] == 5400
            and render_completion['source_ids'] == population['source_ids']
            and render_completion['source_decode_complete'] is True and render_completion['MAIN_frames'] == 0,
            'Actual complete H18 float-RGB render required')
    require(type(index) is int and 0 <= index < COUNT, 'Wrong source index')
    sid = population['source_ids'][index]; pre = population['preprocessing_ids'][index]
    require(asset_manifest['source_ids'] == population['source_ids'] and len(asset_manifest['records']) == COUNT,
            'Original asset list differs')
    require(asset_manifest['status'] == asset_completion['status'] == 'H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
            and asset_manifest['source_count'] == asset_completion['source_count'] == COUNT, 'Original source-only asset receipt required')
    entry = asset_manifest['records'][index]; ap = Path(entry['checkpoint'])
    require(asset_completion['outputs'].get(str(ap)) == entry['checkpoint_sha256'] == sha(ap), 'Asset checkpoint unsealed')
    asset = read(ap)
    for item in (entry, asset):
        require(item['source_index'] == index and item['source_id'] == sid and item['preprocessing_id'] == pre
                and item['original_development_data_binding'] == population['data_bindings'][index],
                'Original source identity changed')
    label = asset['evaluation_class_index']
    require(type(label) is int and 0 <= label < 1000 and entry['evaluation_class_index'] == label
            and entry['archive'] == asset['archive'], 'Original evaluation label/archive differs')
    verify(asset['outputs'])
    require(all(asset_completion['outputs'].get(p) == s for p, s in asset['outputs'].items()), 'Target archive unsealed')
    with np.load(asset['archive'], allow_pickle=False) as z: original = z['pixels'].copy()
    require(original.dtype == np.uint8 and original.shape == (3, 256, 256)
            and hashlib.sha256(original.tobytes()).hexdigest() == pre, 'Original preprocessing hash differs')
    target = np.ascontiguousarray(original.astype(np.float32) / 255.)
    base = Path(render_out); cp_path = base/'source_checkpoints'/f'{index:04d}.json'
    rows_path = base/'sources'/f'{index:04d}.json'; archive = base/'images'/f'{index:04d}.npz'
    for p in (cp_path, rows_path, archive):
        require(render_completion['outputs'].get(str(p)) == sha(p), 'Rendered source output unsealed')
    cp = read(cp_path)
    require(cp['status'] == 'H_DEVELOPMENT_RX_SOURCE_COMPLETE' and cp['source_index'] == index
            and cp['source_id'] == sid and cp['frame_count'] == 54
            and cp['registration_sha256'] == render_completion['registration_sha256']
            and cp['source_decode_complete'] is True and cp['outputs'] == {str(rows_path): sha(rows_path), str(archive): sha(archive)},
            'Rendered source checkpoint differs')
    rows = validate_source_rows(read(rows_path), index, sid, schedule)
    with np.load(archive, allow_pickle=False) as z:
        require(set(z.files) == {r['image_key'] for r in rows}, 'Unreferenced/missing float images')
        images = {key: pixels(z[key]).copy() for key in z.files}
    for row in rows:
        require(row['image_archive'] == str(archive) and row['target_preprocessing_sha256'] == pre
                and rgb_sha(images[row['image_key']]) == row['image_sha256'] == row['rx_summary']['image_sha256'],
                'Actual reconstruction RGB or target hash differs')
    bindings = {str(p): sha(p) for p in (ap, cp_path, rows_path, archive)}
    bindings.update(asset['outputs'])
    return dict(source_index=index, source_id=sid, class_index=label, target=target, target_preprocessing_sha256=pre,
                reference_sha256=rgb_sha(target), rows=rows, images=images, input_bindings=bindings)


class SuiteBackend:
    """Batch-one bridge to injected load_suite outputs and ConvNeXtValidation."""
    def __init__(self, suite, classifier, torch, numeric_flags):
        self.evaluator, self.metric, self.native, self.lpips, self.dino, self.bindings, self.flags = suite
        self.classifier, self.torch, self.numeric_flags = classifier, torch, numeric_flags
        require(classifier.identity['weights_sha256'] == CONVNEXT_SHA
                and classifier.identity['classification_batch_size'] == 1
                and classifier.identity['used_for_selection'] is False, 'Frozen independent ConvNeXt identity required')
        require(all(v['status'] == 'READY' for v in self.evaluator.metadata['metrics'].values()), 'Full existing metric suite required')
        self.metadata = dict(metric_evaluator=copy.deepcopy(self.evaluator.metadata), independent=copy.deepcopy(classifier.identity),
            numerical_flags=copy.deepcopy(self.flags), metric_batch_size=1, legacy_DINO='DINOv2 ViT-S/14',
            legacy_quality_psnr='float32 Torch mean retained separately; H main PSNR remains original float64 mean',
            model_bindings=dict(self.bindings), no_new_model_loading=True, used_for_selection=False)
        self.identity = identity(self.metadata)
        self.evaluator_identity = self.evaluator.identity()
        self.check()

    def check(self):
        require(self.numeric_flags(self.torch) == self.flags, 'Original metric numerical flags changed')
        require(self.evaluator.identity() == self.evaluator_identity == identity(self.evaluator.metadata)
                and self.classifier.identity == self.metadata['independent'], 'Evaluator identity changed')

    def prepare(self, reference):
        self.check(); t = self.torch; rgb = pixels(reference); x = t.from_numpy(rgb[None])
        with t.inference_mode(), self.metric.no_network(), t.autocast(device_type=self.evaluator.device.type, enabled=False):
            prepared = self.evaluator.prepare_reference(x)
            embedding = self.native.dino_features(self.dino, x.to(self.evaluator.device))[0].cpu().numpy()
            prediction = self.classifier.predict(rgb)
        self.check()
        return dict(reference_tensor=x, prepared=prepared, dino_feature=np.asarray(embedding).copy(),
                    convnext_prediction=prediction, resnet50_prediction=int(prepared['resnet50'][0]))

    def score(self, reference, reconstruction, label, prepared, negative_feature):
        self.check(); t = self.torch; image = pixels(reconstruction); reference = pixels(reference)
        with t.inference_mode(), self.metric.no_network(), t.autocast(device_type=self.evaluator.device.type, enabled=False):
            native_rows, source_feature, reconstructed_feature = self.native.quality_metrics(
                reference, [image], self.lpips, self.dino, self.evaluator.device)
            require(len(native_rows) == 1, 'Legacy batch must remain one')
            require(np.array_equal(np.asarray(source_feature), prepared['dino_feature']), 'Original DINO feature changed at batch one')
            newer = self.evaluator.score(prepared['reference_tensor'], t.from_numpy(image[None]), [label],
                label_conditioned=False, prepared=prepared['prepared'])
            require(len(newer) == 1, 'Unified batch must remain one')
            prob, pred = self.evaluator.models['resnet50'](self.metric.preprocess_resnet50(
                t.from_numpy(image[None]).to(self.evaluator.device))).float().softmax(-1).max(-1)
            prob, pred = float(prob.cpu().tolist()[0]), int(pred.cpu().tolist()[0])
            independent = self.classifier.score(image, true_label=label, source_prediction=prepared['convnext_prediction'])
            mismatch = t.nn.functional.cosine_similarity(t.from_numpy(np.asarray(reconstructed_feature)).to(self.evaluator.device),
                t.from_numpy(np.asarray(negative_feature)[None]).to(self.evaluator.device), dim=1).cpu().tolist()[0]
        require(pred == newer[0]['resnet50_prediction'] and newer[0]['resnet50_source_prediction'] == prepared['resnet50_prediction'],
                'Classifier confidence/prediction identity differs')
        values = dict(newer[0], **independent)
        values.update(lpips_alex=float(native_rows[0]['lpips_alex']), dino_cosine=float(native_rows[0]['dino_cosine']),
            legacy_quality_psnr_db=float(native_rows[0]['psnr_db']), resnet50_top1_probability=prob,
            semantic_error=int(pred != prepared['resnet50_prediction']), confidently_wrong=int(prob >= .5 and pred != prepared['resnet50_prediction']),
            dino_mismatched=float(mismatch), dino_specificity=float(native_rows[0]['dino_cosine']) - float(mismatch),
            convnext_model_id='torchvision_ConvNeXt_Tiny_IMAGENET1K_V1', label_conditioned=False)
        self.check(); return values


def reference_table(sources, backend, mismatch):
    """Prepare each already admitted original once; no additional image population."""
    require(len(sources) == COUNT and [s['source_id'] for s in sources] == mismatch['source_ids']
            and mismatch['replay_source_sha256'] == REPLAY_SHA, 'Wrong mismatch/source population')
    records = []
    for index, source in enumerate(sources):
        require(source['source_index'] == index and source['reference_sha256'] == rgb_sha(source['target']), 'Reference input differs')
        prepared = backend.prepare(source['target']); f = np.asarray(prepared['dino_feature'])
        require(f.dtype == np.float32 and f.ndim == 1 and np.isfinite(f).all(), 'Invalid original DINO embedding')
        records.append(dict(source_id=source['source_id'], reference_sha256=source['reference_sha256'], prepared=prepared,
            feature_sha256=hashlib.sha256(f.tobytes()).hexdigest(), evaluator_identity=backend.identity))
    return records


def score_source(source, backend, references, mismatch, schedule):
    """Keep all54 frames; only exact same-source paired RGB scores are reused."""
    index, sid = source['source_index'], source['source_id']
    require(len(references) == COUNT and mismatch['source_ids'][index] == sid
            and mismatch['replay_source_sha256'] == REPLAY_SHA, 'Original mismatch proof absent')
    target, label = pixels(source['target']), source['class_index']
    require(type(label) is int and 0 <= label < 1000, 'Original evaluation label required')
    ref, negative = references[index], references[mismatch['permutation'][index]]
    require(ref['source_id'] == sid and negative['source_id'] == mismatch['source_ids'][mismatch['permutation'][index]]
            and ref['reference_sha256'] == source['reference_sha256'] == rgb_sha(target)
            and ref['evaluator_identity'] == negative['evaluator_identity'] == backend.identity, 'Reference feature provenance changed')
    for feature in (ref, negative):
        require(hashlib.sha256(feature['prepared']['dino_feature'].tobytes()).hexdigest() == feature['feature_sha256'],
                'Original feature bytes changed')
    rows = validate_source_rows(source['rows'], index, sid, schedule); result, cache = [], {}
    for row in rows:
        image = pixels(source['images'][row['image_key']]); image_hash = rgb_sha(image)
        require(image_hash == row['image_sha256'], 'Actual RGB changed before metric scoring')
        key = identity(dict(source_id=sid, reference_sha256=source['reference_sha256'], image_sha256=image_hash,
            label=label, evaluator=backend.identity, mismatch_source_id=negative['source_id'],
            mismatch_reference_sha256=negative['reference_sha256'], mismatch_feature_sha256=negative['feature_sha256']))
        if key not in cache:
            values = backend.score(target, image, label, ref['prepared'], negative['prepared']['dino_feature'])
            require(all(k in values and values[k] is not None and math.isfinite(float(values[k])) for k in REQUIRED),
                    'Incomplete actual metrics; unavailable values cannot be replaced with zero')
            require(values['label_conditioned'] is False and values['convnext_used_for_selection'] is False,
                    'Metric selection/label-conditioned claim differs')
            for field in ('resnet50_prediction', 'resnet50_source_prediction', 'convnext_prediction', 'convnext_source_prediction'):
                require(type(values[field]) is int and 0 <= values[field] < 1000, 'Invalid classifier index: ' + field)
            for field in ('resnet50_top1_label', 'resnet50_top1_source_prediction', 'resnet50_source_top1_label',
                          'convnext_top1_label', 'convnext_top1_source_prediction', 'convnext_source_correct_to_wrong',
                          'convnext_source_wrong_to_correct'):
                require(type(values[field]) is bool, 'Classifier booleans must remain booleans: ' + field)
            require(0 <= values['resnet50_top1_probability'] <= 1, 'Invalid classifier confidence')
            cache[key] = copy.deepcopy(values)
        values = copy.deepcopy(cache[key])
        mse = float(np.mean(np.square(image.astype(np.float64) - target.astype(np.float64))))
        require(mse > 0 and math.isfinite(mse), 'Unexpected zero/nonfinite H reconstruction error')
        psnr = float(-10 * np.log10(mse))
        require(mse == row['mse'] and psnr == row['psnr_db'], 'Original H float64 MSE/PSNR changed')
        legacy = float(values['legacy_quality_psnr_db'])
        require(math.isfinite(legacy), 'Invalid legacy quality PSNR diagnostic')
        result.append(dict(row, **values, class_index=label, reference_sha256=source['reference_sha256'],
            point_id=f'H18_SLOT_{row["development_slot"]:02d}', N=1024, true_class=label, used_for_selection=False, holdout_used=False,
            mse=mse, psnr_db=psnr, legacy_minus_H_psnr_db=legacy-psnr, mismatch_source_id=negative['source_id'],
            mismatch_reference_sha256=negative['reference_sha256'], metric_evaluator_identity=backend.identity,
            metric_batch_size=1, metric_cache_key=key, F_recovery_error=None, F_recovery_error_status='NOT_MEASURED_NO_REGISTERED_LATENT_RECOVERY_METRIC',
            FID_status='DEFERRED_HOLDOUT', KID_status='DEFERRED_HOLDOUT', metric_selection=False))
    return result, dict(frame_count=54, unique_metric_pairs=len(cache), exact_duplicate_metric_reuses=54-len(cache),
                        metric_batch_size=1, statistical_aggregation_run=False, new_packet_decodes=0)
