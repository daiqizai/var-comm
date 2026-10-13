"""Finite Kodak24 scoring of the two final frozen baseline reconstructions.

Only metric models are loaded. Original Kodak pixels, metric mathematics and
weights are retained; this H800 process has its own explicit runtime identity.
Every model/reference/score is reserved durably before invocation. No retry.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'))
METHODS = ('P100K', 'SWIN200K')
SNRS = (4, 10, 19)
SEEDS = (2001, 2002, 2003)
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine',
           'convnext_top1_source_prediction')
ORIGINAL_IDENTITY = '8313f062d2435be361add61aceb9848ccb8844c1f3c9c4743f662480057489e2'
ORIGINAL_BACKEND_SHA = 'd8e9f85c83f63a288cf0500e53857bc6e81c205cec93897e4d46f9f177c79d8e'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(value):
    require(sha(value['path']) == value['sha256'], 'Pinned input changed: '+value['path'])
    return Path(value['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def rgb(np, image):
    require(isinstance(image, np.ndarray) and image.dtype == np.float32
            and image.shape == (3, 256, 256) and np.isfinite(image).all()
            and image.min() >= 0 and image.max() <= 1, 'Untouched CHW float32 RGB01 required')
    return np.ascontiguousarray(image)


def ash(np, image):
    a = np.ascontiguousarray(image)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def assignment(tree, name):
    found = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, 'Unique original Kodak expression required: '+name)
    return found[0]


def ast_equal(actual, expected):
    return ast.dump(actual, include_attributes=False) == ast.dump(ast.parse(expected, mode='eval').body,
                                                                  include_attributes=False)


def original_math_proof(reg, metric):
    """Bind the successful original Kodak evaluator to unchanged projected math.

    The portable loader itself verifies AST equality for the LPIPS loader/pair,
    ConvNeXt class and float64 PSNR, and exact original DINO implementation and
    all four weight hashes. This adds the Kodak-specific expression checks.
    """
    old_done = read(pin(reg['original_score_completion']))
    require(old_done['original_metric_backend_identity'] == reg['original_metric_identity']
            == ORIGINAL_IDENTITY and old_done['metrics'] == list(METRICS)
            and old_done['source_count'] == 24 and old_done['noise_count'] == 3,
            'Actual original Kodak metric identity/scope differs')
    old_rows = pin(reg['original_rows'])
    row_bindings = [digest for path, digest in old_done['outputs'].items()
                    if Path(path).name == 'rows.csv']
    require(row_bindings == [reg['original_rows']['sha256']],
            'Original scoring completion does not bind old rows')
    original_request = read(ROOT/'results/generalization_kodak_20261009/evidence/materials/score_request.json')
    require(original_request['backend_module']['sha256'] == ORIGINAL_BACKEND_SHA
            == reg['metric_config']['sources']['score_bpg_cached_holdout_r2.py']['sha256'],
            'Portable PSNR/LPIPS source is not the actual old backend source')
    original = ROOT/'experiments/generalization_kodak_20261009/score.py'
    require(reg['code_bindings'][str(original)] == sha(original), 'Original Kodak scoring code not bound')
    tree = ast.parse(original.read_text(encoding='utf8'))
    expected = {
        'lpips': 'float(backend.lpips(y*2-1,x*2-1).reshape(-1)[0].item())',
        'cosine': 'float(torch.nn.functional.cosine_similarity(source_feature,feature,dim=-1)[0].item())',
        'mse': 'float(np.square(image.astype(np.float64)-target.astype(np.float64)).mean())',
        'source_feature': "backend.metric.dinov2_vitl14_features(backend.evaluator.models['dinov2_vitl14'],x)",
        'feature': "backend.metric.dinov2_vitl14_features(backend.evaluator.models['dinov2_vitl14'],y)",
        'source_prediction': 'backend.classifier.predict(target)',
        'reconstruction_prediction': 'backend.classifier.predict(image)',
    }
    proofs = {}
    for name, expression in expected.items():
        node = assignment(tree, name)
        require(ast_equal(node, expression), 'Original Kodak expression changed: '+name)
        proofs[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    # Full original source files are byte-bound by the portable loader. Its
    # DINO-only prepare/score use the very same feature and cosine functions.
    assets = metric.bound_assets(reg['metric_config'])
    mtree = ast.parse(Path(reg['metric_config']['sources']['metric_models.py']['path']).read_text())
    cls = next(n for n in mtree.body if isinstance(n, ast.ClassDef) and n.name == 'MetricEvaluator')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    calls = lambda node: [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    require(any(isinstance(n.func, ast.Name) and n.func.id == 'dinov2_vitl14_features'
                for n in calls(methods['prepare_reference']))
            and any(isinstance(n.func, ast.Name) and n.func.id == 'dinov2_vitl14_features'
                    for n in calls(methods['score']))
            and any(isinstance(n.func, ast.Attribute) and n.func.attr == 'cosine_similarity'
                    for n in calls(methods['score'])), 'Original DINO feature/cosine calls differ')
    return dict(status='PASS_ORIGINAL_KODAK_MATH_AND_WEIGHT_BINDINGS',
                original_Kodak_evaluator_identity=ORIGINAL_IDENTITY,
                original_backend_source_sha256=ORIGINAL_BACKEND_SHA,
                Kodak_expression_AST_sha256=proofs,
                original_computation_source_bindings=reg['metric_config']['sources'],
                actual_frozen_weights=assets['weights'],
                original_manifest=assets['original_manifest'],
                original_registration=assets['original_registration'],
                DINO_feature_and_preprocess_source_unchanged=True,
                original_FullSuite_identity_reused_as_current_runtime=False,
                old_host_bitwise_equivalence_claimed=False,
                policy_admission_metadata_is_portable_scope_only=True)


class Ledger:
    CAPS = dict(model_constructions=3, reference_preparations=24, image_scores=432,
                dinov2_vitl14_reference=24, convnext_reference=24,
                dinov2_vitl14_reconstruction=432, convnext_reconstruction=432,
                lpips_pair=432, lpips_alexnet_backbone_forward=864)

    def __init__(self, out, guard):
        self.out = out; self.guard = guard; self.counts = collections.Counter()
        self.sequence = 0; self.current_case = {'stage': 'metric_model_load'}

    def call(self, name, function, **metadata):
        self.guard()
        require(self.counts[name] < self.CAPS[name], 'Metric call cap: '+name)
        self.counts[name] += 1; index = self.sequence; self.sequence += 1
        event = dict(index=index, operation=name, case=self.current_case,
                     operation_count=self.counts[name], cap=self.CAPS[name], **metadata)
        write(self.out/'calls'/f'{index:05d}_reserved.json', dict(event, status='RESERVED'))
        try:
            value = function()
            write(self.out/'calls'/f'{index:05d}_completed.json', dict(event, status='COMPLETE'))
            return value
        except BaseException:
            write(self.out/'calls'/f'{index:05d}_failed.json', dict(event, status='FAILED'))
            raise


def run(registration, frames_path, out):
    import numpy as np
    from PIL import Image
    import h800_ep_pilot_metrics_v1 as metric
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'score.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (out/'launch.json').exists() and not (out/'failure.json').exists(),
            'Already attempted; preserve scene and do not retry')
    reg = read(registration)
    for path, digest in reg['code_bindings'].items():
        require(sha(path) == digest, 'Registered code changed: '+path)
    require(reg['code_bindings'][str(Path(__file__).resolve())] == sha(__file__), 'Scoring child unbound')
    sources = reg['sources']; frames = read(frames_path)
    require(len(sources) == 24 and [r['source_index'] for r in sources] == list(range(24))
            and [r['source_id'] for r in sources] == [f'kodak/kodim{i+1:02d}' for i in range(24)],
            'Original fixed24 source order required')
    expected = {(m, i, s, n) for m in METHODS for i in range(24) for s in SNRS for n in SEEDS}
    seen = set()
    for frame in frames:
        key = (frame['method'], frame['source_index'], frame['snr_db'], frame['noise_seed'])
        require(key in expected and key not in seen and frame['source_id'] == sources[frame['source_index']]['source_id']
                and frame.get('N', 1024) == 1024 and frame['status']
                and frame['reconstruction'].get('key', 'rgb') == 'rgb', 'Exact new frame identity/failure grid')
        seen.add(key)
    require(seen == expected and len(frames) == 432, 'Full432 final-baseline frames required')
    done = read(Path(frames_path).parent/'inference_completion.json')
    actual = read(Path(frames_path).parent/'actual_wait_inference.json')
    require(actual['returncode'] == 0 and done.get('frames', done.get('frame_count')) == 432,
            'Inference must be actually complete before scoring')
    started = time.monotonic(); rows = []; ledger = None

    def guard():
        require(time.time() < reg['deadline_unix'] and time.monotonic()-started < reg.get('score_max_seconds', 7200)
                and not any(Path(p).exists() for p in reg.get('stop_files', []))
                and not (out/'STOP').exists(), 'STOP/deadline: preserve partial metric evidence')

    try:
        guard()
        proof = original_math_proof(reg, metric)
        write(out/'metric_math_identity_proof.json', proof)
        write(out/'launch.json', dict(pid=os.getpid(), started_unix=time.time(),
                                     registration_sha256=sha(registration), frames_sha256=sha(frames_path)))
        ledger = Ledger(out, guard)
        import torch
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
                and torch.cuda.device_count() == 1, 'One visible GPU and original numerical environment required')
        torch.cuda.set_per_process_memory_fraction((16 << 30)/torch.cuda.get_device_properties(0).total_memory)
        config = dict(reg['metric_config'])
        config['runtime_identity'] = dict(purpose='Kodak24 final frozen baseline supplement',
                                         physical_gpu=os.environ.get('BASELINE_GPU_ID'),
                                         torch=torch.__version__, cuda=torch.version.cuda,
                                         old_host_numeric_identity_claimed=False)
        backend = metric.FourMetrics(config, ledger, guard)
        backend.evaluator.metadata['metric_selection_usage'] = 'Frozen Kodak evaluation only; no selection/training'
        backend.metadata.update(population='Kodak24 final frozen baseline supplement', selection_objective=None,
                                policy_selection=False, DINO_L_metadata=backend.evaluator.metadata)
        backend.identity = metric.core.digest(backend.metadata)
        # Capture the integer produced by the existing score's single classifier
        # call. Do not invoke a second classifier forward merely for reporting.
        classifier_predictions = []
        original_predict = backend.classifier.predict

        def captured_predict(image):
            prediction = original_predict(image)
            require(type(prediction) is int and 0 <= prediction < 1000, 'Original classifier integer domain')
            classifier_predictions.append(prediction)
            return prediction

        backend.classifier.predict = captured_predict
        write(out/'metric_identity.json', dict(identity=backend.identity, metadata=backend.metadata,
                                               original_Kodak_identity=ORIGINAL_IDENTITY,
                                               equality_scope='source mathematics, preprocessing and weights; runtime identity distinct'))
        for record in sources:
            guard(); i = record['source_index']
            require(record['preprocessing_id'] == 'kodak_rgb_center_crop_256_v1', 'No new image preprocessing')
            with Image.open(pin(record['png'])) as image:
                require(image.mode == 'RGB' and image.size == (256, 256), 'Frozen RGB center crop required')
                pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == record['pixels_chw_uint8_sha256'], 'Original Kodak pixels changed')
            target = rgb(np, pixels.astype(np.float32)/np.float32(255))
            ledger.current_case = dict(stage='reference', source_index=i, source_id=record['source_id'])
            with torch.inference_mode():
                prepared = backend.prepare(target)
            require(classifier_predictions == [int(prepared['convnext_prediction'])],
                    'One actual source classifier prediction per reference')
            classifier_predictions.clear()
            own = []
            for frame in [f for f in frames if f['source_index'] == i]:
                ledger.current_case = {k: frame[k] for k in ('method', 'source_index', 'source_id', 'snr_db', 'noise_seed', 'status')}
                archive = pin(frame['reconstruction'])
                with np.load(archive, allow_pickle=False) as z:
                    image = rgb(np, z['rgb'].copy())
                with torch.inference_mode():
                    values = backend.score(target, image, 0, prepared)
                require(len(classifier_predictions) == 1, 'One actual reconstruction classifier call per score')
                reconstruction_prediction = classifier_predictions.pop()
                require(values['convnext_top1_source_prediction']
                        == int(reconstruction_prediction == int(prepared['convnext_prediction'])),
                        'Reported agreement must match captured actual classifier prediction')
                require(all(math.isfinite(float(values[m])) for m in METRICS), 'Finite frozen scores required')
                common = dict(ledger.current_case, N=1024, population='Kodak24',
                              preprocessing_id=record['preprocessing_id'], reference_sha256=ash(np, target),
                              reconstruction_sha256=ash(np, image), reconstruction_archive_sha256=frame['reconstruction']['sha256'],
                              original_metric_backend_identity=ORIGINAL_IDENTITY, metric_backend_identity=backend.identity,
                              source_prediction=int(prepared['convnext_prediction']),
                              reconstruction_prediction=reconstruction_prediction,
                              true_class_label_available=False, dummy_label_unused_for_requested_metrics=True,
                              frame_level_noise_pairing_claimed=False)
                own.extend(dict(common, metric=m, value=values[m]) for m in METRICS)
            require(len(own) == 72, 'Every source has18 new frames and72 metric rows')
            rows.extend(own)
            write(out/'sources'/f'{i:04d}.json', dict(status='SOURCE_COMPLETE', source_index=i,
                source_id=record['source_id'], source_prediction=int(prepared['convnext_prediction']), rows=own,
                metric_backend_identity=backend.identity))
            print('SOURCE_COMPLETE', i, len(rows), flush=True)
        require(len(rows) == 1728 and ledger.counts['image_scores'] == 432
                and ledger.counts['reference_preparations'] == 24, 'Full432 x4 metric output')
        with (out/'rows.csv').open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        for path, digest in reg['code_bindings'].items():
            require(sha(path) == digest, 'Registered source changed during scoring')
        outputs = {str(p): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'score.lock'}
        proof_path = out/'metric_math_identity_proof.json'
        write(out/'completion.json', dict(status='KODAK24_FINAL_BASELINES_FOUR_METRICS_COMPLETE_V1', N=1024, source_count=24,
              noise_count=3, frame_count=432, metric_rows=1728, methods=list(METHODS), metrics=list(METRICS),
              source_ids=[r['source_id'] for r in sources],
              snr_db=list(SNRS), counts=dict(ledger.counts), call_caps=ledger.CAPS, outputs=outputs,
              metric_math_identity_proof=dict(path=str(proof_path), sha256=sha(proof_path)),
              original_metric_identity_proof_status=proof['status'],
              metric_backend_identity=backend.identity, original_metric_backend_identity=ORIGINAL_IDENTITY,
              old_host_bitwise_equivalence_claimed=False, source_and_reconstruction_preprocessing_identical=True,
              failures_retained=True, new_transmissions=0, new_reconstructions=0, training_updates=0,
              bootstrap_calls=0, policy_selection=False, seconds=time.monotonic()-started))
    except BaseException:
        write(out/'failure.json', dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY', completed_metric_rows=len(rows),
              counts=dict(ledger.counts) if ledger is not None else {}, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(); run(args.registration, args.frames, args.out)
