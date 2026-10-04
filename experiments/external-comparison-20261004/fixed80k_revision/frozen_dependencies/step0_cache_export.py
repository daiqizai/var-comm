"""CPU-only export of already scored, inheritance-verified R2 float RGB."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image

FIXED = (0, 25, 50, 75)
ALL_FIXED = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
SNRS = (7, 13)
BUDGETS = (2048, 3060, 4084)
STUDIES = ('FINAL_P2048_P3060', 'FINAL_P4084_SELECTED_SEEDS',
           'FINAL_DIGITAL_QPSK', 'LEGACY_N3060_FINAL')
DOMAIN = b'float32:3,256,256:RGB\0'
EVAL_FIELDS = ('modelmanifest_sha256', 'evaluator_identity', 'numerical_runtime',
               'metric_batch_size', 'batch_qualification_sha256', 'metric_qualified_batch_sizes')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Existing Step0 artifact differs: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def pixels(image):
    image = np.asarray(image)
    require(image.dtype == np.float32 and image.shape == (3, 256, 256), 'Expected original CHW float32 256 RGB')
    require(np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1, 'Invalid float RGB')
    return np.ascontiguousarray(image)


def pixel_hash(image, domain=True):
    return hashlib.sha256((DOMAIN if domain else b'') + pixels(image).tobytes()).hexdigest()


def verified_arrays(checkpoint, registration, expected_source, study, cache_path):
    """Validate both historical hash definitions without editing old receipts."""
    cp = checkpoint
    require(cp['payload_sha256'] == identity({k:v for k,v in cp.items() if k != 'payload_sha256'}), 'Checkpoint seal differs')
    require(cp['binding'] == identity(registration), 'Checkpoint registration differs')
    index = expected_source['source_index']
    require(cp['source_index'] == index and cp['evaluation_identity'] == {k:registration[k] for k in EVAL_FIELDS}, 'Evaluator/source differs')
    source = registration['source_identity'][index]
    require(source['image_id'] == expected_source['image_id'] and source['class_index'] == expected_source['class_index'],
            'Frozen source identity differs')
    require(source['preprocessing_id'] == expected_source['preprocessing_id'] or
            (study == 'LEGACY_N3060_FINAL' and source['preprocessing_id'] == 'original_rounded_uint8_div255_RGB'),
            'Unregistered source preprocessing identity differs')
    rows, proof = cp['rows'], cp['float_reconstructions']
    require(proof['path'] == str(cache_path) and proof['sha256'] == sha(cache_path), 'Float cache path/container differs')
    require(proof['dtype'] == 'float32' and proof['layout'] == 'CHW' and proof['rows'] == len(rows), 'Float cache schema differs')
    ids = [r['history_row_id'] for r in rows]
    require(ids == proof['row_ids'] and len(ids) == len(set(ids)), 'Duplicate or reordered row identity')
    require(len(cp['parity']) == len(rows), 'Missing original replay parity')
    used = cp['metric_batch_sizes_used']
    require(used and all(type(n) is int and n in registration['metric_qualified_batch_sizes'] for n in used)
            and sum(used) == cp['unique_images'], 'Metric batch qualification differs')
    with np.load(cache_path, allow_pickle=False) as archive:
        require(set(archive.files) == {'images', 'source_rgb', 'row_ids', 'image_slots'}, 'Unexpected float cache members')
        images, target = archive['images'], pixels(archive['source_rgb'])
        slots = proof['image_slots']
        require(archive['row_ids'].tolist() == ids and archive['image_slots'].tolist() == slots, 'Float cache row slots differ')
    require(type(proof['unique_images']) is int and len(images) == proof['unique_images'] and len(slots) == len(rows)
            and all(type(s) is int and 0 <= s < len(images) for s in slots), 'Invalid float cache slots')
    require([pixel_hash(x, False) for x in images] == proof['image_sha256']
            and pixel_hash(target, False) == proof['reference_sha256'], 'Raw float receipt hash differs')
    target_hash = pixel_hash(target)
    require(hashlib.sha256(np.rint(target * 255).astype(np.uint8).tobytes()).hexdigest() == expected_source['preprocessing_id'],
            'Cached source pixels differ from preregistered input')
    scientific = [pixel_hash(x) for x in images]
    for row, parity, slot in zip(rows, cp['parity'], slots):
        require(row['history_study'] == study and row['history_source_index'] == index
                and row['history_source_id'] == source['image_id']
                and row['history_preprocessing_id'] == source['preprocessing_id']
                and row['history_true_class_index'] == source['class_index'], 'Scientific source metadata differs')
        require(row['history_image_sha256'] == scientific[slot] and row['history_reference_sha256'] == target_hash,
                'Scientific row RGB hash differs')
        require(row['history_original_row_sha256'] == identity({k:v for k,v in row.items() if not k.startswith(('new_', 'history_'))}),
                'Original scientific row changed')
        require(row['history_replay_parity_passed'] is True and parity.get('replay_parity_passed') is True
                and parity.get('synthetic') is False and parity.get('history_row_id') == row['history_row_id'], 'Unqualified replay row')
    return target, images, dict(zip(ids, slots))


def selected_row(row, study):
    meta = json.loads(row['history_metadata_json'])
    snr = int(float(row.get(meta.get('snr_field', 'snr_db'), row.get('snr_db'))))
    noise = int(row.get(meta.get('noise_seed_field', 'noise_seed'), row.get('seed', -1)))
    if snr not in SNRS or noise != 2001:
        return None
    method = meta.get('method_id', meta.get('method', row.get('method')))
    n = int(meta.get('N', row.get('N', row.get('complex_uses', -1))))
    decoder = meta.get('decoder_id', meta.get('decoder'))
    if study == STUDIES[0] and method in ('P2048', 'P3060'):
        group = 'P'
    elif study == STUDIES[1] and method == 'P4084_N4084_seed2026092304':
        group = 'P'
    elif study == STUDIES[2] and method in {f'final_{f}_QPSK_N{n}_Dc' for f in ('raw', 'arithmetic')} and n in BUDGETS:
        group = 'Digital raw / Dc' if method.startswith('final_raw_') else 'Digital arithmetic / Dc'
    elif study == STUDIES[3] and row.get('method') in ('raw_adaptive', 'arithmetic_adaptive'):
        group = 'Legacy raw / D0' if row['method'] == 'raw_adaptive' else 'Legacy arithmetic / D0'
    else:
        return None
    require(n in BUDGETS and decoder == ('D0' if group.startswith('Legacy') else 'Dc'), 'Selected protocol/decoder differs')
    label = group != 'P'
    require(meta.get('label_conditioned') is label, 'Selected true-class conditioning differs')
    if group == 'P':
        require(meta.get('training_seed') == 2026092304, 'Preselected continuous training seed differs')
    columns = meta.get('original_metric_columns', {'psnr_db':'psnr_db', 'lpips_alex':'lpips_alex'})
    metrics = {key:float(row[columns[key]]) for key in ('psnr_db', 'lpips_alex')}
    metrics['dinov2_vitl14_cosine'] = float(row['new_dinov2_vitl14_cosine'])
    require(all(math.isfinite(v) for v in metrics.values()), 'Missing/nonfinite measured metrics')
    prediction = int(row['new_resnet50_prediction'])
    original_prediction = int(row['new_resnet50_source_prediction'])
    require(int(row['new_resnet50_top1_source_prediction']) == int(prediction == original_prediction), 'Classifier agreement differs')
    return dict(group=group, method=method, N=n, snr_db=snr, noise_seed=noise, decoder=decoder,
                true_class_paid=label, classification_main_eligible=not label,
                legacy_protocol=group != 'P', protocol=meta.get('original_protocol', row.get('protocol', row.get('run_id', study))),
                training_seed=meta.get('training_seed', ''), metrics=metrics,
                resnet50_prediction=prediction, resnet50_source_prediction=original_prediction,
                semantic_error=int(prediction != original_prediction), metadata=meta)


def export(root, fixed_path, out, *, all_examples=False):
    root, out, fixed_path = Path(root).resolve(), Path(out).resolve(), Path(fixed_path).resolve()
    new_output = root / 'outputs/EXTERNAL-COMPARISON-20261004'
    require(out != root and (root not in out.parents or out == new_output or new_output in out.parents),
            'Step0 output must be separate from historical inputs')
    fixed = read(fixed_path)
    require(fixed['status'] == 'FROZEN_FIXED_EXAMPLES' and fixed['source_indices'] == list(ALL_FIXED), 'Fixed examples differ')
    chosen = fixed['source_indices'] if all_examples else list(FIXED)
    records = {r['source_index']:r for r in fixed['records']}
    require(len(records) == len(fixed['records']) == len(ALL_FIXED) and set(records) == set(ALL_FIXED), 'Fixed source records differ')
    queue_path = root/'outputs/HISTORICAL-METRICS-R3-20261003/queue_registration.json'
    queue = read(queue_path)
    inputs = {str(queue_path):sha(queue_path), str(fixed_path):sha(fixed_path)}
    out.mkdir(parents=True, exist_ok=True)
    frames, sources, targets = [], [], {}
    for study in STUDIES:
        item = queue['inherited_studies'][study]
        folder = root/'outputs/HISTORICAL-METRICS-R2-20261003'/study
        result = root/'results/historical_metrics_r2_20261003'/study
        receipt_path = root/'outputs/HISTORICAL-METRICS-R3-20261003/inheritance'/(study+'.json')
        require(item['out'] == str(folder) and item['result'] == str(result) and item['receipt_path'] == str(receipt_path), 'Inheritance path substitution')
        require(sha(receipt_path) == item['receipt_sha256'] == queue['input_proof_bindings'][str(receipt_path)], 'Inheritance receipt not registered')
        receipt = read(receipt_path)
        require(receipt['status'] == 'VERIFIED_COMPLETE_R2_STUDY' and receipt['study'] == study
                and receipt['source_checkpoints_verified'] == receipt['float_caches_verified'] == 100
                and receipt['original_identities_preserved'] is True and receipt['metrics_recomputed'] is False, 'Incomplete inheritance admission')
        inputs[str(receipt_path)] = sha(receipt_path)
        for origin in ('original_queue', 'source_publication'):
            p = receipt[origin]['path']
            require(sha(p) == receipt[origin]['sha256'] == receipt['bindings'][p], 'Original provenance differs')
            inputs[p] = sha(p)
        regpath = result/'registration.json'
        require(sha(regpath) == receipt['bindings'][str(regpath)], 'Score registration differs')
        registration = read(regpath)
        inputs[str(regpath)] = sha(regpath)
        for index in chosen:
            checkpoint_path = folder/'source_checkpoints'/f'{index:04d}.json'
            cache_path = folder/'reconstructions'/f'{index:04d}.npz'
            for p in (checkpoint_path, cache_path):
                require(sha(p) == receipt['bindings'][str(p)], 'Selected inherited input differs: '+str(p))
                inputs[str(p)] = sha(p)
            checkpoint = read(checkpoint_path)
            target, images, slots = verified_arrays(checkpoint, registration, records[index], study, cache_path)
            digest = pixel_hash(target)
            raw_digest = hashlib.sha256(np.rint(target * 255).astype(np.uint8).tobytes()).hexdigest()
            if index not in targets:
                targets[index] = raw_digest
                name = f'source_{index:03d}.png'
                Image.fromarray(np.rint(target.transpose(1,2,0)*255).astype(np.uint8)).save(out/name)
                sources.append(dict(**records[index], file=name, reference_sha256=digest, raw_uint8_sha256=raw_digest,
                                    png_sha256=sha(out/name)))
            require(targets[index] == raw_digest, 'Original uint8 source pixels differ across historical methods')
            for row in checkpoint['rows']:
                selected = selected_row(row, study)
                if selected is None:
                    continue
                image = images[slots[row['history_row_id']]]
                name = f'src{index:03d}_snr{selected["snr_db"]}_{row["history_row_id"][:16]}.png'
                Image.fromarray(np.rint(image.transpose(1,2,0)*255).astype(np.uint8)).save(out/name)
                frames.append(dict(**selected, source_index=index, source_id=records[index]['image_id'],
                    file=name, png_sha256=sha(out/name), float_rgb_sha256=pixel_hash(image),
                    original_reference_float_sha256=digest,original_preprocessing_id=registration['source_identity'][index]['preprocessing_id'],
                    study=study, history_row_id=row['history_row_id'], exact_completed_rgb_match=True,
                    checkpoint=str(checkpoint_path), measured_row=row))
    expected = {(i,s,g,n) for i in chosen for s in SNRS for g in ('P','Digital raw / Dc','Digital arithmetic / Dc') for n in BUDGETS}
    expected |= {(i,s,g,3060) for i in chosen for s in SNRS for g in ('Legacy raw / D0','Legacy arithmetic / D0')}
    keys = [(r['source_index'],r['snr_db'],r['group'],r['N']) for r in frames]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Selected Step0 scope is incomplete or duplicated')
    for p,h in inputs.items():
        require(sha(p) == h, 'Step0 source changed during export')
    manifest = dict(status='EXACT_COMPLETED_FLOAT_RGB_CPU_EXPORT_PASS', sources=sources, frames=frames,
        source_indices=chosen, snrs_db=list(SNRS), noise_seed=2001, budgets=list(BUDGETS), input_bindings=inputs,
        own_source_bindings={str(Path(__file__).resolve()):sha(__file__)}, gpu_inference=False,
        metric_inference=False, policy_selection_updates=0, training_updates=0,
        display_conversion='float32 CHW to RGB uint8 using round-to-nearest; scientific metrics use original float pixels',
        scope_note='Existing P and paid true-class digital protocols only; no unconditional N2048 digital point is asserted.',
        interpretation_note='Legacy digital uses true class in a charged header. D0 and Dc are separate rows. Qualitative resource comparison; protocol changes confound causal attribution.')
    manifest['source_preprocessing_note'] = ('Legacy N3060 used uint8/255 float RGB; P and later digital use the VAR normalization roundtrip. '
        'Original uint8 pixels match exactly. Every row retains its original float reference hash and original measured metrics.')
    seal(out/'step0_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fixed-examples', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all-examples', action='store_true')
    args = parser.parse_args()
    value = export(args.root, args.fixed_examples, args.out, all_examples=args.all_examples)
    print(value['status'], len(value['frames']), flush=True)
