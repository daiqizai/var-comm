"""R1 actual-image three-metric scoring and authenticated calibration selection.

Import and pure CPU tests construct no model and execute no channel. The score
entry requires a reviewed master registration and closed actual visual stage.
Source pixels are available only in this metric process, never sent to RX/VAR.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = HERE.parents[2]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import contract
import policy_selection
import r1_phy

METRICS = policy_selection.R1_METRICS
FLAGS = dict(matmul_tf32=False, cudnn_tf32=False, precision='highest',
             cudnn_benchmark=False, cudnn_deterministic=True,
             deterministic=True, threads=6, interop_threads=2)
SCHEMA = 'LAYERED_RAW_R1_ACTUAL_THREE_METRICS_V1'
JOIN_FIELDS = ('source_id', 'source_index', 'selection_slot', 'snr_hat',
               'snr_true', 'noise_seed', 'candidate_id', 'physical_key',
               'receiver_rule', 'view_id', 'observation_key', 'actual_rx_key',
               'receiver_state_sha256', 'model_identity')
require = contract.require
digest = contract.digest
sha = r1_phy.sha
pin = r1_phy.pin
checkpin = r1_phy.checkpin
readpin = r1_phy.readpin
save = r1_phy.save


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def validate_image(image):
    require(isinstance(image, np.ndarray) and image.dtype == np.float32
            and image.shape == (3, 256, 256) and np.isfinite(image).all()
            and image.min() >= 0 and image.max() <= 1,
            'Actual original float32 RGB crop required')


def source_pixels(record):
    """Metric-only original crop. Never read tokens or return them to renderer."""
    with np.load(checkpin(record['archive']), allow_pickle=False) as z:
        pixels = z['pixels'].copy()
    require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256)
            and hashlib.sha256(pixels.tobytes()).hexdigest() == record['preprocessing_id'],
            'Original calibration source pixels changed')
    image = pixels.astype(np.float32) / 255
    validate_image(image)
    return image


def reconstructed_pixels(row):
    reconstruction = row['reconstruction']
    with np.load(checkpin(reconstruction['image_archive']), allow_pickle=False) as z:
        require(z.files == ['image'], 'Unexpected reconstruction archive members')
        image = z['image'].copy()
    validate_image(image)
    require(array_sha(image) == reconstruction['image_sha256'], 'Actual reconstruction bytes changed')
    return image


def logical_key(row):
    return tuple(row[k] for k in ('source_id', 'snr_hat', 'snr_true',
                                  'noise_seed', 'candidate_id', 'receiver_rule'))


def expected_grid(sources, nominations):
    contract.validate_shortlist(nominations)
    require(len(sources) == 200 and [r['selection_slot'] for r in sources] == list(range(200))
            and len({r['source_id'] for r in sources}) == 200,
            'The fixed ordered 200 sources are required')
    choices = {}
    for n in nominations:
        for rule in n['receiver_variants']:
            key = n['snr_hat'], n['candidate_id'], rule
            require(choices.setdefault(key, n['physical_key']) == n['physical_key'],
                    'Ambiguous canonical candidate or physical aliases')
    return {(s['source_id'], hat, true, noise, cid, rule): physical
            for s in sources for (hat, cid, rule), physical in choices.items()
            for true in contract.true_snrs(hat) for noise in contract.CAL_SEEDS}


def validate_visual_rows(rows, sources, nominations):
    expected = expected_grid(sources, nominations)
    by_source = {r['source_id']: r for r in sources}
    seen = set(); views = set(); events = {}
    for row in rows:
        key = logical_key(row)
        require(key in expected and key not in seen and row['physical_key'] == expected[key],
                'Foreign, duplicated or mislabelled visual logical row')
        seen.add(key)
        source = by_source[row['source_id']]
        require(all(row[k] == source[k] for k in ('source_index', 'selection_slot')),
                'Visual source identity changed')
        require(row['status'] == 'COMPLETE' and row['view_id'] not in views
                and row.get('source_truth_supplied_to_RX') is False,
                'Only closed isolated received-state images qualify')
        views.add(row['view_id'])
        for field in ('observation_key', 'actual_rx_key', 'receiver_state_sha256', 'model_identity'):
            require(isinstance(row[field], str) and len(row[field]) == 64,
                    'Missing actual identity: ' + field)
        physical = key[:4] + (row['physical_key'],)
        actual = row['observation_key'], row['actual_rx_key']
        require(events.setdefault(physical, actual) == actual,
                'KEEP/DROP must refer to one actual physical event')
        reconstruction = row['reconstruction']
        require(isinstance(reconstruction['image_sha256'], str)
                and len(reconstruction['image_sha256']) == 64
                and isinstance(reconstruction['actual_input_key'], str)
                and len(reconstruction['actual_input_key']) == 64,
                'Actual image and model-input hashes required')
    require(seen == set(expected) and len(rows) <= contract.R1_CAPS['logical_var_images'],
            'Visual calibration grid is not complete')
    return dict(logical_rows=len(rows), physical_events=len(events), sources=200, noises=3)


def pair_key(row, source, metric_identity):
    """Exact shared event/model input may reuse scoring across KEEP/DROP aliases."""
    return digest(dict(source_id=source['source_id'], pixels=source['preprocessing_id'],
        source_archive=source['archive'], protocol=contract.PLAN_ID,
        snr_hat=row['snr_hat'], snr_true=row['snr_true'], noise_seed=row['noise_seed'],
        candidate_id=row['candidate_id'], physical_key=row['physical_key'],
        observation_key=row['observation_key'], actual_rx_key=row['actual_rx_key'],
        model_identity=row['model_identity'],
        actual_model_input=row['reconstruction']['actual_input_key'],
        image=row['reconstruction']['image_sha256'], metric_identity=metric_identity))


def score_caps(logical_count):
    require(type(logical_count) is int and 0 < logical_count <= 46800,
            'Registered logical score budget required')
    return dict(model_constructions=2, reference_preparations=200,
        dinov2_vitl14_reference=200, image_scores=logical_count,
        dinov2_vitl14_reconstruction=logical_count, lpips_pair=logical_count,
        lpips_alexnet_backbone_forward=2 * logical_count)


def check_score_counts(counts, logical_count, unique_pairs):
    require(counts['caps'] == score_caps(logical_count) and counts['unresolved'] == 0
            and counts['reserved'] == counts['completed'], 'Unclosed metric ledger')
    expected = dict(model_constructions=2, reference_preparations=200,
        dinov2_vitl14_reference=200, image_scores=unique_pairs,
        dinov2_vitl14_reconstruction=unique_pairs, lpips_pair=unique_pairs,
        lpips_alexnet_backbone_forward=2 * unique_pairs)
    require(counts['completed'] == expected and 0 < unique_pairs <= logical_count,
            'Unexpected metric model, reference, pair or backbone calls')


def metric_library():
    directory = ROOT / 'experiments/entropy-partial-mismatch-20261010/scripts'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import h800_ep_pilot_metrics_v1
    return h800_ep_pilot_metrics_v1


class ThreeMetrics:
    """Unmodified original PSNR/LPIPS expressions and original DINO-L methods."""
    def __init__(self, config, ledger, boundary, device, cpus):
        self.ledger, self.boundary = ledger, boundary
        metric = metric_library()
        assets = metric.bound_assets(config)
        import torch
        import lpips
        self.torch = torch
        torch.set_num_threads(6); torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.flags = metric.flags_function(config['sources']['step0_reference_metrics.py']['path'])
        require(self.flags(torch) == FLAGS, 'Original metric numerical settings differ')
        device.check_cuda_environment(); torch.cuda.set_device(0)
        require(torch.cuda.device_count() == 1, 'One registered visible GPU required')
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction((16 << 30) / props.total_memory)
        import leo_whole_math_v1 as original_math
        gpu = device.validate_cuda_device_identity(str(props.uuid), original_math.cuda_driver_pci_bus_id())
        runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        require(runtime == device.EXPECTED_RUNTIME and sorted(os.sched_getaffinity(0)) == sorted(cpus),
                'Qualified numerical runtime or CPU affinity changed')
        self.psnr, psnr_proof = metric.psnr_projection(config['sources']['score_bpg_cached_holdout_r2.py']['path'])
        load_lpips, self.lpips_pair, lpips_proof = metric.lpips_projection(
            config['sources']['quality.py']['path'], dict(Path=Path, lpips=lpips, torch=torch))
        linear = Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'
        require(sha(linear) == metric.WEIGHT_SHA['lpips'], 'LPIPS linear weights changed')
        original = metric.load_module(config['sources']['metric_models.py'], '_r1_original_DINO_L')
        boundary()
        with original.no_network():
            self.perceptual = ledger.call('model_constructions', lambda: load_lpips(
                dict(alexnet_checkpoint=assets['weights']['alexnet']['path']), 'cuda:0'), model='LPIPS_ALEX')
            self.evaluator = ledger.call('model_constructions', lambda: original.MetricEvaluator(
                assets['dinov2_manifest'], device='cuda:0', requested=('dinov2_vitl14',)), model='DINOv2_ViTL14')
        require(set(self.evaluator.models) == {'dinov2_vitl14'}, 'Unexpected metric model')
        self.evaluator.metadata['metric_selection_usage'] = 'Registered R1 original-calibration200 DINO objective only'
        feature = original.dinov2_vitl14_features
        self.context = {}
        def counted_feature(model, image):
            return ledger.call(self.context['feature_kind'], lambda: feature(model, image),
                               source_id=self.context['source_id'], pair_key=self.context.get('pair_key'))
        original.dinov2_vitl14_features = counted_feature
        lpips_forward = self.perceptual.net.forward
        def counted_forward(*args, **kwargs):
            return ledger.call('lpips_alexnet_backbone_forward', lambda: lpips_forward(*args, **kwargs),
                               pair_key=self.context['pair_key'])
        self.perceptual.net.forward = counted_forward
        self.metadata = dict(schema=SCHEMA, requested_metrics=list(METRICS), source_population='original_calibration200',
            metric_assets=assets, original_sources=config['sources'], numeric_flags=FLAGS,
            numerical_runtime=runtime, GPU=gpu, CPU_slots=list(cpus),
            original_math_proof=dict(PSNR=psnr_proof, LPIPS=lpips_proof),
            DINO_L_metadata=self.evaluator.metadata, original_prepare_and_score_unchanged=True,
            source_and_reconstruction_preprocessing_identical=True, batch_size=1, AMP=False,
            actual_loaded_lpips_linear=dict(path=str(linear), sha256=sha(linear)),
            source_reference_cap=200, ConvNeXt_constructed=False, bootstrap_calls=0,
            source_pixels_used_only_in_metric_process=True, received_truth_supplied_to_reconstruction=False)
        self.identity = digest(self.metadata)
        boundary()

    def tensor(self, image):
        validate_image(image); self.boundary()
        require(self.flags(self.torch) == FLAGS, 'Metric numerical settings changed')
        return self.torch.as_tensor(np.ascontiguousarray(image)[None], dtype=self.torch.float32, device='cuda:0')

    def prepare(self, target, source_id):
        self.context = dict(feature_kind='dinov2_vitl14_reference', source_id=source_id)
        def operation():
            x = self.tensor(target)
            with self.torch.inference_mode(), self.torch.autocast('cuda', enabled=False):
                prepared = self.evaluator.prepare_reference(x)
            return dict(target_sha256=array_sha(target), metric_identity=self.identity, dino=prepared)
        return self.ledger.call('reference_preparations', operation, source_id=source_id)

    def score(self, target, image, class_index, prepared, source_id, key):
        self.context = dict(feature_kind='dinov2_vitl14_reconstruction', source_id=source_id, pair_key=key)
        def operation():
            x, y = self.tensor(target), self.tensor(image)
            require(prepared['target_sha256'] == array_sha(target) and prepared['metric_identity'] == self.identity,
                    'Source preparation identity changed')
            with self.torch.inference_mode(), self.torch.autocast('cuda', enabled=False):
                perceptual = self.ledger.call('lpips_pair', lambda: self.lpips_pair(self.perceptual, y, x), pair_key=key)
                dino = self.evaluator.score(x, y, [class_index], label_conditioned=False, prepared=prepared['dino'])
            values = dict(psnr_db=self.psnr(target, image), lpips_alex=float(perceptual[0].item()),
                          dinov2_vitl14_cosine=float(dino[0]['dinov2_vitl14_cosine']))
            require(dino[0]['label_conditioned'] is False and all(math.isfinite(v) for v in values.values()),
                    'Missing/nonfinite metric or class-conditioned metric')
            return values
        return self.ledger.call('image_scores', operation, source_id=source_id, pair_key=key)


def score_rows(rows, sources, evaluator, out, guard):
    """Actual archive scoring; fake evaluator injection is used only by CPU tests."""
    out = Path(out); by_source = {s['source_id']: s for s in sources}
    cache = {}; refs = {}; scored = []; prepared_sources = []
    for index, row in enumerate(rows):
        guard(); source = by_source[row['source_id']]
        if source['source_id'] not in refs:
            target = source_pixels(source)
            prepared = evaluator.prepare(target, source['source_id'])
            refs[source['source_id']] = (target, prepared)
            prepared_sources.append(dict(source_id=source['source_id'], source_archive=source['archive'],
                source_preprocessing_id=source['preprocessing_id'], target_array_sha256=array_sha(target),
                metric_identity=evaluator.identity))
        target, prepared = refs[source['source_id']]
        image = reconstructed_pixels(row)
        key = pair_key(row, source, evaluator.identity)
        if key not in cache:
            values = evaluator.score(target, image, source['evaluation_class_index'], prepared, source['source_id'], key)
            require(set(values) == set(METRICS) and all(math.isfinite(float(v)) for v in values.values()),
                    'Exactly three real original metrics required')
            cache[key] = dict(values=values, first_view_id=row['view_id'])
            reuse = 'ACTUAL_NEW_PAIR'
        else:
            reuse = 'EXACT_ACTUAL_EVENT_PAIR_REUSE'
        result = {k: row[k] for k in JOIN_FIELDS}
        result.update(cache[key]['values'], execution_status='COMPLETE', source_truth_supplied_to_RX=False,
            source_preprocessing_id=source['preprocessing_id'], metric_identity=evaluator.identity,
            reconstruction_sha256=row['reconstruction']['image_sha256'], image_archive=row['reconstruction']['image_archive'],
            actual_input_key=row['reconstruction']['actual_input_key'], metric_pair_key=key,
            first_scored_view_id=cache[key]['first_view_id'], metric_reuse=reuse,
            actual_receiver_outcome=receiver_outcome(row), failed_logical_observation_retained=True)
        save(out / 'rows' / f'{index:05d}.json', result); scored.append(result)
    return scored, prepared_sources, len(cache)


def receiver_outcome(row):
    if 'actual_receiver_outcome' in row:
        return row['actual_receiver_outcome']
    return {k: row[k] for k in ('actual_received_kind', 'actual_received_m',
                                'actual_received_K', 'actual_failure_state') if k in row}


def same_pin(left, right):
    return (isinstance(left, dict) and isinstance(right, dict)
            and left.get('path') == right.get('path')
            and left.get('sha256') == right.get('sha256'))


def successful_wait(reg, registration, stage, process_probe=None):
    """Bind an actual wait to the owner's exact dispatched child, not an exit flag."""
    require(stage in ('projection', 'visual', 'score'), 'Unsupported score prerequisite stage')
    directory = Path(reg['runtime'][stage + '_directory'])
    root = Path(reg['runtime']['root'])
    path = directory / 'actual_wait.json'; value = read(path)
    require(value.get('actual_wait') is True and value.get('returncode') == 0
            and not any(value.get(k, False) for k in ('timeout', 'timed_out', 'stopped', 'terminated', 'killed'))
            and value.get('stage') == stage and type(value.get('pid')) is int and value['pid'] > 0,
            'Actual successful child wait required')
    require(not (directory / 'failure.json').exists()
            and read(root / (stage + '_actual_wait.json')) == value,
            'Failure scene or mismatched owner wait receipt')
    dispatch = read(root / (stage + '_launch.json'))
    owner = read(root / 'owner_launch.json')
    child = dispatch.get('process', {})
    require(dispatch.get('stage') == stage and same_pin(dispatch.get('registration'), registration)
            and same_pin(owner.get('registration'), registration)
            and type(child.get('pid')) is int and child['pid'] == value['pid']
            and type(child.get('start_ticks')) is int and child['start_ticks'] > 0,
            'Actual wait does not identify the registered dispatched child')
    if 'start_ticks' in value:
        require(value['start_ticks'] == child['start_ticks'], 'Wait start ticks differ from dispatch')
    argv = dispatch['argv']
    expected = reg['visual_registration'] if stage == 'visual' else registration
    script, action, sha_flag = (('score_selection.py', 'score', '--registration-sha256')
        if stage == 'score' else ('receiver_visual.py', 'visual' if stage == 'visual' else 'project', '--expected-registration'))
    require(isinstance(argv, list) and argv.count('--registration') == argv.count(sha_flag) == 1
            and argv[argv.index('--registration') + 1] == expected['path']
            and argv[argv.index(sha_flag) + 1] == expected['sha256']
            and str(HERE / script) in argv and argv[argv.index(str(HERE / script)) + 1] == action,
            'Dispatched program or registration differs')
    parent_flag = '--parent' if stage == 'score' else '--parent-pid'
    require(argv.count(parent_flag) == 1 and argv[argv.index(parent_flag) + 1] == str(owner['process']['pid']),
            'Dispatched parent differs from the actual owner')
    if stage != 'score':
        require(argv.count('--parent-ticks') == 1
                and argv[argv.index('--parent-ticks') + 1] == str(owner['process']['start_ticks']),
                'Dispatched owner start ticks differ')
    if process_probe is None:
        import shared
        process_probe = shared.process_identity
    current = process_probe(child['pid'])
    require(not current or current.get('state') == 'Z'
            or str(current.get('start_ticks')) != str(child['start_ticks']),
            'Dispatched child is still running despite a completion receipt')
    return pin(path)


def closed_visual(reg, registration):
    out = Path(reg['runtime']['visual_directory'])
    wait = successful_wait(reg, registration, 'visual'); completed = read(out / 'completion.json')
    require(completed['status'] == 'COMPLETE_R1_VISUAL' and same_pin(completed['registration'], reg['visual_registration'])
            and completed['source_truth_supplied_to_RX'] is False, 'Wrong or unclosed actual visual stage')
    projection_out = Path(reg['runtime']['projection_directory'])
    projection_wait = successful_wait(reg, registration, 'projection')
    projection = read(projection_out / 'completion.json')
    require(projection['status'] == 'COMPLETE_R1_RX_PROJECTION'
            and same_pin(projection['registration'], registration)
            and same_pin(completed['model_inputs'], projection['model_inputs'])
            and same_pin(completed['projection_completion'], projection['visual_input_completion'])
            and same_pin(completed['projection_actual_wait'], projection_wait),
            'Visual stage did not consume this authenticated projection')
    inputs = readpin(projection['model_inputs'])
    receive_only = readpin(projection['visual_input_completion'])
    require(receive_only['status'] == 'COMPLETE_RX_ONLY_PROJECTION'
            and same_pin(receive_only['model_inputs'], projection['model_inputs'])
            and receive_only['source_truth_supplied_to_RX'] is False
            and receive_only['logical_count'] == projection['logical_count'] == len(inputs)
            and receive_only['projection_digest'] == digest([
                projection['model_inputs']['sha256'], projection['logical_join']['sha256'], projection['audit']['sha256']]),
            'Received-state-only projection binding differs')
    checkpin(projection['audit'])
    image_rows = readpin(completed['image_rows'])
    joins = readpin(projection['logical_join'])
    rows = join_visual_rows(joins, image_rows)
    sources = readpin(reg['pins']['cal200_sources'])['records']
    nominations = readpin(reg['pins']['nominations'])['rows']
    grid = validate_visual_rows(rows, sources, nominations)
    require(completed['logical_rows'] == grid['logical_rows'] == reg['expected_logical_var_images'],
            'Actual visual grid does not match registration')
    import receiver_visual
    rendered_audit = receiver_visual.audit_visual(reg)
    require(rendered_audit['status'] == 'PASS_CPU_ACTUAL_VISUAL_CLOSURE'
            and same_pin(rendered_audit['completion'], pin(out / 'completion.json'))
            and same_pin(rendered_audit['model_inputs'], projection['model_inputs']),
            'Actual received-state rendering ledger did not authenticate')
    return rows, sources, nominations, dict(completion=pin(out / 'completion.json'), actual_wait=wait,
        image_rows=completed['image_rows'], projection_completion=pin(projection_out / 'completion.json'),
        projection_actual_wait=projection_wait, model_inputs=projection['model_inputs'],
        visual_input_completion=projection['visual_input_completion'], logical_join=projection['logical_join'],
        visual_dispatch=pin(Path(reg['runtime']['root']) / 'visual_launch.json'),
        projection_dispatch=pin(Path(reg['runtime']['root']) / 'projection_launch.json'),
        actual_rendering_audit=rendered_audit)


def join_visual_rows(joins, images):
    """Join only in the scoring process, never pass TX/source fields to visual."""
    index = {r['view_id']: r for r in images}
    require(len(index) == len(images) == len(joins) and len({r['view_id'] for r in joins}) == len(joins),
            'Duplicate/missing projected or visual view')
    require(set(index) == {r['view_id'] for r in joins}, 'Foreign visual view')
    rows = []
    for join in joins:
        image = index[join['view_id']]
        require(all(join[k] == image[k] for k in set(join) & set(image)), 'Projection/image join conflicts')
        rows.append(dict(join, **{k: v for k, v in image.items() if k not in join}))
    return rows


def score_worker(registration, parent):
    import shared
    reg = shared.validate_registration(registration)
    rows, sources, _, visual = closed_visual(reg, registration)
    cfg = reg['runtime']; out = Path(cfg['score_directory']); out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'launch.json').exists() and not (out / 'calls.sqlite').exists(), 'No implicit score retry')
    guard = shared.guard_for(reg, out, parent)
    device, g = shared.resource_device(reg, out)
    history = []; last = 0.; ledger = None
    def boundary():
        nonlocal last
        guard()
        if time.monotonic() - last >= 5:
            snapshot, _ = device.resource_snapshot(g.helper()); history.append(snapshot); last = time.monotonic()
    try:
        while True:
            guard(); admission = shared.admission(reg, out)
            if admission['status'] == 'READY': break
            require(admission['status'] == 'WAIT', 'Unknown resource admission result')
            time.sleep(30)
        ledger = shared.load_ledger(reg)(out / 'calls.sqlite', score_caps(len(rows)), guard)
        save(out / 'launch.json', dict(registration=registration, process=shared.process_identity(os.getpid()),
                                      pid=os.getpid(), parent_pid=parent, stage='score'))
        evaluator = ThreeMetrics(reg['metric_config'], ledger, boundary, device, cfg['cpu_affinity'])
        save(out / 'metric_identity.json', evaluator.metadata)
        values, references, unique = score_rows(rows, sources, evaluator, out, boundary)
        evaluator.torch.cuda.synchronize(); boundary()
        counts = ledger.summary(); check_score_counts(counts, len(rows), unique); ledger.close()
        values_pin = save(out / 'scored_rows.json', values)
        refs_pin = save(out / 'source_preparations.json', references)
        resource_pin = save(out / 'resource_history.json', history)
        return save(out / 'completion.json', dict(status='COMPLETE_R1_SCORE', registration=registration,
            visual_closure=visual, logical_rows=len(values), unique_actual_pairs=unique,
            metrics=list(METRICS), scored_rows=values_pin, source_preparations=refs_pin,
            metric_identity=pin(out / 'metric_identity.json'), ledger=pin(out / 'calls.sqlite'),
            counts=counts, resource_history=resource_pin, source_count=200, noise_count=3,
            bootstrap_calls=0, ConvNeXt_calls=0, training_updates=0, policy_selection='NOT_RUN'))
    except BaseException as error:
        save(out / 'failure.json', dict(status='STOPPED_NO_RETRY', error=repr(error), traceback=traceback.format_exc(),
            counts=None if ledger is None else ledger.summary(), resource_history=history)); raise


def audit_score_ledger(value, counts, rows, references):
    """Read-only audit of actual nested model/metric calls, not result recomputation."""
    path = checkpin(value); db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    bykind = collections.defaultdict(list)
    try:
        db.execute('PRAGMA query_only=ON')
        require(db.execute('PRAGMA quick_check').fetchone() == ('ok',), 'Corrupt metric ledger')
        for kind, metadata, state in db.execute('SELECT kind,metadata,state FROM calls ORDER BY id'):
            require(kind in counts['caps'] and state == 'COMPLETE', 'Unresolved/foreign metric call')
            bykind[kind].append(json.loads(metadata))
    finally:
        db.close()
    actual = {k: len(bykind[k]) for k in counts['caps']}
    require(actual == counts['completed'], 'Actual ledger counts disagree with receipt')
    sources = [r['source_id'] for r in references]
    require([r['source_id'] for r in bykind['reference_preparations']] == sources
            and [r['source_id'] for r in bykind['dinov2_vitl14_reference']] == sources,
            'Actual reference feature grid differs')
    unique = [r for r in rows if r['metric_reuse'] == 'ACTUAL_NEW_PAIR']
    pairs = [r['metric_pair_key'] for r in unique]
    require(len(pairs) == len(set(pairs)), 'Repeated actual metric pair')
    for kind, expected in (('image_scores', pairs), ('dinov2_vitl14_reconstruction', pairs),
                            ('lpips_pair', pairs), ('lpips_alexnet_backbone_forward', [k for p in pairs for k in (p, p)])):
        require([r['pair_key'] for r in bykind[kind]] == expected, 'Actual metric pair sequence differs: ' + kind)
    for kind in ('image_scores', 'dinov2_vitl14_reconstruction'):
        require([r['source_id'] for r in bykind[kind]] == [r['source_id'] for r in unique],
                'Actual metric pair source differs: ' + kind)
    require([r['model'] for r in bykind['model_constructions']] == ['LPIPS_ALEX', 'DINOv2_ViTL14'],
            'Actual model scope changed')
    checkpin(value)


def quality_summaries(means, sources, selected):
    """Descriptive source averages only; no interval, bootstrap or gate change."""
    cells = collections.defaultdict(dict)
    for (source, hat, true, candidate, rule, metric), value in means.items():
        cells[(hat, true, candidate, rule, metric)][source] = value
    expected_sources = set(sources); summary = []
    for key, values in sorted(cells.items()):
        require(set(values) == expected_sources, 'Incomplete source-level quality cell')
        summary.append(dict(snr_hat=key[0], snr_true=key[1], candidate_id=key[2],
            receiver_rule=key[3], metric=key[4], mean=math.fsum(values[s] for s in sources)/len(sources),
            source_count=200, noise_count=3, confidence_interval='NOT_COMPUTED_R1_SELECTION'))
    lookup = {(r['snr_hat'], r['snr_true'], r['candidate_id'], r['receiver_rule'], r['metric']):r['mean'] for r in summary}
    winners = {(r['method'],r['snr_hat']):r for r in selected['winners']}
    output=[]
    for (method,hat), winner in sorted(winners.items()):
        for true in contract.true_snrs(hat):
            for metric in METRICS:
                output.append(dict(method=method,snr_hat=hat,snr_true=true,metric=metric,
                    mean=lookup[(hat,true,winner['candidate_id'],winner['receiver_rule'],metric)],
                    source_count=200,noise_count=3,candidate_id=winner['candidate_id'],receiver_rule=winner['receiver_rule']))
    engineering=[]
    for hat in contract.HATS:
        u=winners[('U_two',hat)]; nominal=winners[('S_nom',hat)]
        def psnr(w,true):return lookup[(hat,true,w['candidate_id'],w['receiver_rule'],'psnr_db')]
        delta=psnr(u,hat)-psnr(nominal,hat); loss=psnr(u,hat)-psnr(u,hat-3)
        engineering.append(dict(snr_hat=hat,matched_U_minus_S_nom_psnr_db=delta,
            same_U_overestimate_psnr_loss_db=loss,matched_cost_target_pass=delta>=-.3,
            own_loss_target_pass=loss<=3.,used_for_GO=False))
    lowcost=all(r['matched_cost_target_pass'] and r['own_loss_target_pass'] for r in engineering)
    description=('NO_CALIBRATION_GO' if selected['gate']['status']!='GO_R2' else
                 ('LOW_COST_TARGETS_MET_ON_CALIBRATION' if lowcost else 'TRADEOFF_ONLY_ON_CALIBRATION'))
    return summary,output,dict(per_hat=engineering,classification=description,
        separate_from_frozen_DINO_GO=True,development_claim=False,
        matched_reference='S_nom at true=hat',overestimate_definition='Same U policy at true=hat versus true=hat-3')


def save_csv(path, rows):
    require(rows and all(set(r)==set(rows[0]) for r in rows),'Nonempty fixed CSV schema required')
    with Path(path).open('x',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return pin(path)


def authenticate_scored_rows(rows, visual_rows, sources, metric_identity):
    require(len(rows) == len(visual_rows), 'Incomplete scored logical grid')
    by_source = {s['source_id']: s for s in sources}; cache = {}
    for row, visual in zip(rows, visual_rows):
        source = by_source[visual['source_id']]
        require(all(row[k] == visual[k] for k in JOIN_FIELDS), 'Metric row does not match its actual receiver/image')
        require(row['execution_status'] == 'COMPLETE' and row['source_truth_supplied_to_RX'] is False
                and row['metric_identity'] == metric_identity
                and row['source_preprocessing_id'] == source['preprocessing_id']
                and row['image_archive'] == visual['reconstruction']['image_archive']
                and row['reconstruction_sha256'] == visual['reconstruction']['image_sha256']
                and row['actual_input_key'] == visual['reconstruction']['actual_input_key']
                and row['actual_receiver_outcome'] == receiver_outcome(visual)
                and row['failed_logical_observation_retained'] is True, 'Metric image/source identity differs')
        reconstructed_pixels(visual)
        key = pair_key(visual, source, metric_identity)
        require(row['metric_pair_key'] == key, 'Scoring cache identity changed')
        values = {k: row[k] for k in METRICS}
        require(all(math.isfinite(float(v)) for v in values.values()), 'Missing/nonfinite actual metric')
        if key in cache:
            require(row['metric_reuse'] == 'EXACT_ACTUAL_EVENT_PAIR_REUSE'
                    and row['first_scored_view_id'] == cache[key]['view_id'] and values == cache[key]['values'],
                    'Exact reuse changed actual values or repeated model call')
        else:
            require(row['metric_reuse'] == 'ACTUAL_NEW_PAIR' and row['first_scored_view_id'] == row['view_id'],
                    'Unproven historical metric reuse')
            cache[key] = dict(values=values, view_id=row['view_id'])
    return len(cache)


def select(registration):
    import shared
    reg = shared.validate_registration(registration)
    visual_rows, sources, nominations, visual = closed_visual(reg, registration)
    # The projection auditor verifies the original complete PHY grid and both
    # receive domains; it never consults source truth to repair received states.
    import receiver_visual
    phy_projection = receiver_visual.audit_projection(reg)
    out = Path(reg['runtime']['score_directory']); score_wait = successful_wait(reg, registration, 'score')
    done = read(out / 'completion.json')
    require(done['status'] == 'COMPLETE_R1_SCORE' and same_pin(done['registration'], registration)
            and done['visual_closure'] == visual and done['metrics'] == list(METRICS)
            and done['source_count'] == 200 and done['noise_count'] == 3
            and done['bootstrap_calls'] == done['ConvNeXt_calls'] == done['training_updates'] == 0,
            'Actual score closure does not match the frozen R1 scope')
    rows = readpin(done['scored_rows']); metadata = readpin(done['metric_identity'])
    require(metadata['requested_metrics'] == list(METRICS) and metadata['numeric_flags'] == FLAGS
            and metadata['original_sources'] == reg['metric_config']['sources']
            and metadata['ConvNeXt_constructed'] is False
            and metadata['source_and_reconstruction_preprocessing_identical'] is True
            and metadata['received_truth_supplied_to_reconstruction'] is False,
            'Actual original metric identity differs')
    identity = digest(metadata)
    unique = authenticate_scored_rows(rows, visual_rows, sources, identity)
    require(unique == done['unique_actual_pairs'] and len(rows) == done['logical_rows'], 'Score count mismatch')
    references = readpin(done['source_preparations'])
    require([r['source_id'] for r in references] == [s['source_id'] for s in sources], 'Source order changed')
    for reference, source in zip(references, sources):
        target = source_pixels(source)
        require(reference == dict(source_id=source['source_id'], source_archive=source['archive'],
            source_preprocessing_id=source['preprocessing_id'], target_array_sha256=array_sha(target),
            metric_identity=identity), 'Actual reference preprocessing differs')
    check_score_counts(done['counts'], len(rows), unique)
    audit_score_ledger(done['ledger'], done['counts'], rows, references)
    for index, row in enumerate(rows):
        require(read(out / 'rows' / f'{index:05d}.json') == row, 'Actual per-view score receipt differs')
    selected = policy_selection.select_from_complete_rows(rows, [s['source_id'] for s in sources], nominations)
    means, _, _ = policy_selection.validate_and_reduce(rows, [s['source_id'] for s in sources], nominations)
    source_rows = [dict(source_id=k[0], snr_hat=k[1], snr_true=k[2], candidate_id=k[3],
                        receiver_rule=k[4], metric=k[5], mean=v, noise_count=3) for k, v in sorted(means.items())]
    selected.update(status='COMPLETE_R1_CALIBRATION_SELECTION', authenticated_actual_closure=True,
        R2='NOT_REGISTERED_NOT_RUN', registration=registration,
        source_means_definition='Three noises averaged within each source before frozen matched/overestimate J',
        no_bootstrap=True, stress_true_snr_plus3_used_for_selection=False)
    summary,winner_summary,engineering=quality_summaries(means,[s['source_id'] for s in sources],selected)
    selected['separate_engineering_targets']=engineering
    selected['scope_notes']=dict(single_receive_domain='Full original raw433; KEEP and DROP share the same actual receive event',
        grouped_receive_domain='Full mixed689; legal wrongly received raw433 IDs are decoded under fixed DROP',
        equal_U_and_EEP_winner_allowed=True,screening_proxy_is_not_a_quality_bound=True,
        R0_historical_clean_values_are_not_actual_R1_scores=True,
        true_hat_plus3_is_pressure_reporting_only=True,
        no_go_or_one_point_action='Deliver and stop; no extra samples, retuning or automatic R2',
        calibration_scope='Actual H800 calibration200 scores; no development or holdout selection')
    destination = Path(reg['runtime']['selection_directory'])
    require(not destination.exists(), 'No implicit selection overwrite/retry')
    destination.mkdir(parents=True)
    source_pin = save(destination / 'source_means.json', source_rows)
    source_csv=save_csv(destination/'source_means.csv',source_rows)
    summary_csv=save_csv(destination/'candidate_summary.csv',summary)
    winner_csv=save_csv(destination/'frozen_winner_summary.csv',winner_summary)
    result_pin = save(destination / 'selection.json', selected)
    return save(destination / 'completion.json', dict(status='COMPLETE_R1_CALIBRATION_SELECTION',
        registration=registration, selection=result_pin, source_means=source_pin,source_means_csv=source_csv,
        candidate_summary_csv=summary_csv,frozen_winner_summary_csv=winner_csv,phy_projection_audit=phy_projection,
        visual_closure=visual, score_completion=pin(out / 'completion.json'), score_actual_wait=score_wait,
        scientific_calls=0, bootstrap_calls=0, R2='NOT_REGISTERED_NOT_RUN'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('score', 'select'))
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--parent', type=int)
    args = parser.parse_args()
    registration = dict(path=str(Path(args.registration).absolute()), sha256=args.registration_sha256)
    if args.action == 'score':
        require(args.parent is not None, 'Actual owner PID required')
        result = score_worker(registration, args.parent)
    else:
        result = select(registration)
    print(contract.canonical(result))
