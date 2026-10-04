"""Read-only, CPU admission of the three registered own methods at both budgets.

This module never loads a reconstruction or metric model. Native historical
scores remain provenance; the score runner measures the common float target.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import own_controls_common as c
from external_eval_common import sha, read, identity, verify, pixels, rgb_sha

SOURCES = 100
ROWS = 5400
SNRS = (1, 7, 13)
SEEDS = (2001, 2002, 2003)
METHODS = {1024: ('P1024', 'D_U_QPSK', 'M1_entropy_N1024_frozen'),
           2048: ('P2048', 'D_U_whole_N2048_v1', 'M1_entropy_N2048_v1')}
STUDY = 'FINAL_P2048_P3060'
VERSION = 'OWN-CONTROLS-COMMON-REFERENCE-METRICS-20261004-R1'
require = c.require


def key(row):
    return int(row['N']), int(row['snr_db']), int(row['noise_seed']), row['method']


def expected_keys():
    return [(n, s, seed, m) for n in METHODS for s in SNRS for seed in SEEDS for m in METHODS[n]]


def row_id(index, item):
    return identity(dict(study=VERSION, source_index=index, N=item[0], snr_db=item[1],
                         noise_seed=item[2], method=item[3]))


def expected_ids(index):
    return [row_id(index, item) for item in expected_keys()]


def selected_p2048(row):
    """Exact old selected point; never substitute another seed or class policy."""
    meta = json.loads(row['history_metadata_json'])
    method = meta.get('method_id', meta.get('method', row.get('method')))
    if method != 'P2048':
        return None
    snr = int(float(row.get(meta.get('snr_field', 'snr_db'), row.get('snr_db'))))
    seed = int(row.get(meta.get('noise_seed_field', 'noise_seed'), row.get('seed', -1)))
    if snr not in SNRS or seed not in SEEDS:
        return None
    require(int(meta.get('N', row.get('N', -1))) == 2048
            and meta.get('decoder_id', meta.get('decoder')) == 'Dc'
            and meta.get('label_conditioned') is False
            and meta.get('training_seed') == 2026092304,
            'P2048 decoder, class access or preselected training seed differs')
    return (2048, snr, seed, 'P2048'), meta


def _bound(bindings, path):
    path = Path(path)
    require(bindings.get(str(path)) == sha(path), 'Unbound or changed completed input: ' + str(path))
    return str(path)


def _population(reg, records):
    pop = reg['original_population']
    require(pop['source_ids'] == [r['image_id'] for r in records]
            and pop['preprocessing_ids'] == [r['preprocessing_id'] for r in records]
            and pop['source_classes'] == [r['class_index'] for r in records]
            and reg.get('development_used_for_selection') is False
            and reg.get('training_updates') == 0 and reg.get('holdout_access') is False,
            'Own reconstruction registration population/selection differs')


def admit_inputs(root):
    """Freeze all upstream completion proofs before any new metric is observed."""
    from step0_reference_prepare import admitted, targets_from_completed
    root = Path(root).resolve(); paths = c.locations(root); out = paths['out']
    current = c.source_gate(root, c.HERE/'own_controls_protocol.json')
    bindings = dict(current); completions = {}
    for stage in c.STAGES:
        require(not (out/(stage+'_failure.json')).exists(), 'Upstream failure requires review: '+stage)
        path = out/(stage+'_completion.json'); done = read(path)
        require(done.get('status') == 'OWN_CONTROL_STAGE_COMPLETE' and done.get('stage') == stage
                and done.get('source_bindings') == current and done.get('synthetic') is False
                and done.get('training_updates') == 0 and done.get('holdout_access') is False,
                'Real registered control stage is incomplete: '+stage)
        verify(done['inputs']); verify(done['outputs'])
        bindings.update(done['inputs']); bindings.update(done['outputs']); bindings[str(path)] = sha(path)
        completions[stage] = done
    require(completions['development'].get('sources') == 100 and completions['development'].get('rows') == 1800
            and completions['development'].get('policy_selection_updates') == 0
            and completions['export-n1024'].get('sources') == 100 and completions['export-n1024'].get('rows') == 2700
            and completions['export-n1024'].get('policy_selection_updates') == 0
            and completions['export-n1024'].get('exact_previous_rgb') is True,
            'Full new N2048 and exact old N1024 outputs are required')
    policy_path = paths['result']/'N2048_policy.json'; _bound(bindings, policy_path)
    policy = read(policy_path)
    require(policy.get('source_count') == 1000 and policy.get('noise_seeds') == list(c.CAL_SEEDS)
            and policy.get('snrs') == list(SNRS) and policy.get('development_read') is False
            and policy.get('training_updates') == 0 and policy.get('original_N1024_policy_changed') is False
            and sha(policy_path) == completions['development']['policy_sha256']
            == completions['calibrate']['policy_sha256'], 'Final policy is not the complete calibration winner')
    cells = {(r['snr_db'], r['method']):r for r in policy['cells']}
    require(len(cells) == len(policy['cells']) == 6
            and set(cells) == {(s,m) for s in SNRS for m in c.METHODS}, 'Calibrated policy cells differ')
    targets, target_bindings = targets_from_completed(root); bindings.update(target_bindings)
    records = [{k:v for k,v in item.items() if k != 'rgb'} for item in targets]
    require(len(records) == 100 and [r['source_index'] for r in records] == list(range(100)), 'Expected 100 original targets')
    stage_regs = {}
    for stage in ('development','export-n1024'):
        path = out/(stage+'_registration.json'); _bound(bindings,path)
        reg = read(path); _population(reg,records); stage_regs[stage] = reg
    _, p_receipt, p_bindings = admitted(root, STUDY); bindings.update(p_bindings)
    p_result = root/'results/historical_metrics_r2_20261003'/STUDY
    p_out = root/'outputs/HISTORICAL-METRICS-R2-20261003'/STUDY
    p_reg_path = p_result/'registration.json'; _bound(p_receipt['bindings'], p_reg_path)
    bindings[str(p_reg_path)] = sha(p_reg_path)
    cache_inventory = []
    for index, (record,target) in enumerate(zip(records,targets)):
        item = dict(source_index=index, record=record, reference_sha256=rgb_sha(target['rgb']), stages={})
        for stage in ('development','export-n1024'):
            cp_path = out/stage/'source_checkpoints'/f'{index:04d}.json'
            cache_path = out/stage/'float_reconstructions'/f'{index:04d}.npz'
            _bound(completions[stage]['outputs'],cp_path); _bound(completions[stage]['outputs'],cache_path)
            cp = c.checkpoint(cp_path, identity(stage_regs[stage]), index)
            require(cp.get('synthetic') is False and cp['float_reconstructions']['path'] == str(cache_path)
                    and cp['float_reconstructions']['sha256'] == sha(cache_path), 'Completed float cache scope differs')
            item['stages'][stage] = dict(checkpoint=str(cp_path), checkpoint_sha256=sha(cp_path),
                registration=str(out/(stage+'_registration.json')), registration_sha256=sha(out/(stage+'_registration.json')),
                archive=str(cache_path), archive_sha256=sha(cache_path))
        cp_path = p_out/'source_checkpoints'/f'{index:04d}.json'; cache_path = p_out/'reconstructions'/f'{index:04d}.npz'
        for path in (cp_path,cache_path):
            _bound(p_receipt['bindings'],path); bindings[str(path)] = sha(path)
        item['stages']['P2048'] = dict(checkpoint=str(cp_path),checkpoint_sha256=sha(cp_path),
            registration=str(p_reg_path),registration_sha256=sha(p_reg_path),archive=str(cache_path),archive_sha256=sha(cache_path))
        cache_inventory.append(item)
    return dict(version=VERSION, source_identity=records, source_float_sha256=[r['reference_sha256'] for r in cache_inventory],
        cache_inventory=cache_inventory, input_bindings=bindings, policy=policy, policy_sha256=sha(policy_path),
        completion_sha256={stage:sha(out/(stage+'_completion.json')) for stage in c.STAGES})


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def normalized(original, index, record, item_key, image, target, native_target, entry, slot, *, metadata=None):
    n,snr,seed,method = item_key
    require(item_key in expected_keys(), 'Unregistered own method/SNR/seed/budget')
    if n == 1024:
        native = original['original_scientific_row']
        require(original.get('original_scalar_parity',{}).get('status') == 'PASS'
                and original.get('original_scientific_row_sha256') == identity(native)
                and original['original_completed_metrics']['image_sha256'] == rgb_sha(image)
                and original['original_completed_metrics']['reference_sha256'] == rgb_sha(native_target),
                'N1024 original native/metric provenance differs')
        values = {name:float(native[name]) for name in ('psnr_db','lpips_alex','dino_cosine')}
        origin = 'exact_old_selected_N1024_rgb'
    elif method == 'P2048':
        native = original
        columns = metadata.get('original_metric_columns', {m:m for m in ('psnr_db','lpips_alex','dino_cosine')})
        values = {name:float(native[columns[name]]) for name in ('psnr_db','lpips_alex','dino_cosine')}
        origin = 'admitted_original_P2048_float_cache'
    else:
        native = original
        values = {name:float(native['native_'+name]) for name in ('psnr_db','lpips_alex','dino_cosine')}
        origin = 'new_registered_N2048_float_cache'
    require(all(np.isfinite(v) for v in values.values()), 'Missing finite original/native scores')
    row = dict(experiment='OWN_CONTROLS_20261004', scope='own_matched_total_budget', N=n, snr_db=snr,
        noise_seed=seed, method=method, source_index=index, source_id=record['image_id'],
        preprocessing_id=record['preprocessing_id'], true_class_index=int(record['class_index']),
        projection='',control='',phy_family='continuous_AWGN' if method.startswith('P') else 'QPSK',
        output_role='main',decoder_id='Dc',label_conditioned=False,is_main_conclusion=True,
        semantic_side_information_used=False,receiver_class_embedding=None if method.startswith('P') else 1000,
        class_bits_protocol_note=('paid_class_field_but_unused_by_VAR_generation' if method=='D_U_QPSK'
                                  else 'no_semantic_class_condition'),
        E=2*n,reference_sha256=rgb_sha(target),image_sha256=rgb_sha(image),
        native_reference_sha256=rgb_sha(native_target),source_origin=origin,
        replay_row_id=row_id(index,item_key),replay_parity_passed=True,synthetic=False,
        image_archive=entry['archive'],image_archive_sha256=entry['archive_sha256'],image_array_key='images',image_slot=slot,
        original_checkpoint=entry['checkpoint'],original_checkpoint_sha256=entry['checkpoint_sha256'],
        original_row_sha256=identity(original),original_row_json=_canonical_json(original),
        native_metric_reference='original_native_float_target',metric_reference='common_FINAL_P2048_P3060_float_RGB',
        nominal_noise_pairing='same_source_and_seed_other_methods_have_distinct_physical_noise_namespaces',
        selection_uses_development=False,policy_selection_updates=0,
        dino_model_id='DINOv2_ViT-S/14',lpips_model_id='LPIPS_Alex',dinov2_vitl14_model_id='DINOv2_ViT-L/14',
        training_seed=(metadata.get('training_seed') if metadata else native.get('training_seed')),
        **{'native_'+name:value for name,value in values.items()})
    # Supplementary native transport/feature/receiver-cost observations are
    # copied by name and remain explicit provenance, never new measurements.
    for name in ('m','q','order','E','header_ok','body_crc_ok','N_header','N_data','N_continuous',
                 'F_mse','F_nmse','f_mse','f_nmse','feature_mse','receiver_seconds','RX_seconds','TX_seconds'):
        if name in native:
            row['original_'+name] = native[name]
    return row


def load_source_payload(root, index, score_registration=None):
    """Return (target, normalized rows, keyed RGB, verified file bindings)."""
    from step0_cache_export import verified_arrays
    require(type(index) is int and 0 <= index < SOURCES, 'Unregistered source index')
    root = Path(root).resolve()
    reg = score_registration if score_registration is not None else read(c.locations(root)['result']/'metrics_registration.json')
    inventory = reg['cache_inventory']; entry = inventory[index]
    require(len(inventory) == 100 and entry['source_index'] == index, 'Cache inventory source differs')
    record = entry['record']; bindings = {}; stages = entry['stages']; loaded = {}
    for stage,spec in stages.items():
        for what in ('checkpoint','registration','archive'):
            path = spec[what]; require(sha(path) == spec[what+'_sha256'], 'Registered own cache input changed: '+path)
            bindings[path] = spec[what+'_sha256']
        registration = read(spec['registration']); cp = c.checkpoint(spec['checkpoint'],identity(registration),index)
        if stage == 'P2048':
            target, images, slots = verified_arrays(cp,registration,record,STUDY,Path(spec['archive']))
            loaded[stage] = (cp,images,target,slots)
        else:
            c.verify_float_cache(cp['float_reconstructions'],cp['rows'])
            require(cp.get('synthetic') is False and cp['float_reconstructions']['path'] == spec['archive'], 'Synthetic/substituted own cache')
            with np.load(spec['archive'],allow_pickle=False) as archive:
                images = archive['images'].copy(); native = pixels(archive['native_reference']).copy()
                target = pixels(archive['comparison_reference']).copy()
            require(hashlib.sha256(np.rint(native*255).astype(np.uint8).tobytes()).hexdigest() == record['preprocessing_id'],
                    'Native target uint8 source changed')
            if stage == 'development':
                cells = {(r['snr_db'],r['method']):r for r in reg['policy']['cells']}
                c.validate_development_rows(cp['rows'],index,record,cells,reg['policy_sha256'])
            else:
                require(len(cp['rows']) == 27 and all(r['source_id'] == record['image_id']
                    and r['source_index'] == index and r['source_class_index'] == record['class_index']
                    and r['N'] == 1024 and r['policy_selection_updates'] == 0 for r in cp['rows']), 'Old export source/scope differs')
            loaded[stage] = (cp,images,target,native)
        require(rgb_sha(target) == entry['reference_sha256'], 'Different common reference target')
    require(set(loaded) == {'development','export-n1024','P2048'}, 'Missing own cache family')
    target = loaded['P2048'][2]; rows, mapping = {}, {}
    for stage in ('export-n1024','development','P2048'):
        cp,images,_,aux = loaded[stage]; spec = stages[stage]
        for old in cp['rows']:
            metadata = None
            if stage == 'P2048':
                selected = selected_p2048(old)
                if selected is None: continue
                item_key,metadata = selected; slot = aux[old['history_row_id']]; native = target
            else:
                item_key = key(old); slot = old['image_slot']; native = aux
            require(item_key not in rows, 'Duplicate own scientific frame')
            image = pixels(images[slot]).copy()
            rows[item_key] = normalized(old,index,record,item_key,image,target,native,spec,slot,metadata=metadata)
            mapping[item_key] = image
    require(set(rows) == set(expected_keys()) and len(rows) == 54, 'Incomplete own source frame grid')
    return target.copy(), [rows[k] for k in expected_keys()], mapping, bindings


def load_source_images(root, index, score_registration=None):
    """Public CPU-only figure API: target, {(N,SNR,seed,method): RGB}, bindings."""
    target,_,images,bindings = load_source_payload(root,index,score_registration)
    return target,images,bindings


def deduplicate(rows, mapping):
    images, slots, lookup = [], [], {}
    for row in rows:
        image = mapping[key(row)]; digest = rgb_sha(image)
        require(digest == row['image_sha256'], 'Metric input differs from registered reconstruction')
        if digest not in lookup:
            lookup[digest] = len(images); images.append(image)
        slots.append(lookup[digest])
    return np.stack(images),slots


def validate_scored_checkpoint(value, binding, index, input_bindings, numerical_identity):
    require(value.get('binding') == binding and value.get('source_index') == index
            and value.get('payload_sha256') == identity({k:v for k,v in value.items() if k != 'payload_sha256'})
            and value.get('input_bindings') == input_bindings and value.get('evaluation_identity') == numerical_identity,
            'Metric resume input, numerical identity or payload changed')
    rows = value.get('rows',[])
    require([r['replay_row_id'] for r in rows] == expected_ids(index)
            and all(r.get('replay_parity_passed') is True for r in rows), 'Metric source lacks exact verified 54 rows')
    used = value.get('metric_batch_sizes_used',[])
    require(used and all(type(x) is int and x in numerical_identity['qualified_batch_sizes'] for x in used)
            and sum(used) == value['unique_images'], 'Unqualified metric execution batch')
    baseline = value['baseline']
    for row in rows:
        require(row['source_index'] == index and row['source_id'] == baseline['source_id']
                and row['preprocessing_id'] == baseline['preprocessing_id']
                and row['reference_sha256'] == baseline['reference_sha256']
                and row['resnet50_source_prediction'] == baseline['resnet50_source_prediction']
                and row['resnet50_top1_label'] == int(row['resnet50_prediction'] == baseline['true_class_index'])
                and row['resnet50_top1_source_prediction'] == int(row['resnet50_prediction'] == baseline['resnet50_source_prediction'])
                and row['semantic_error'] == int(row['resnet50_prediction'] != baseline['resnet50_source_prediction'])
                and row['confidently_wrong'] == int(row['resnet50_top1_probability'] >= .5 and row['semantic_error']),
                'Scored source/classification identity changed')
    return value
