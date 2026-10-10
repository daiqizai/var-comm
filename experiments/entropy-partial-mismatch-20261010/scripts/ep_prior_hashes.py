"""Bounded old1900 canonical-hash completion, separate prepare/run commands.

Prepare reads only the declared audit and source code. Run uses the pinned
original preprocess function on old images; it loads no models and runs no PHY.
Run requires a fresh CPU-only process and makes no retries or replacements.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback

EXPECTED_PREPROCESS_SHA = '568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562'
PIXEL_DOMAIN = 'sha256_uint8_CHW_3x256x256_bytes'
IMAGE_ROOT = '/home/liulu/projects/VAR-MAP-GATE0/data/imagenet/val'
COUNT = 1900
SHA = re.compile(r'[0-9a-f]{64}')
SID = re.compile(r'n\d{8}/ILSVRC2012_val_(\d{8})_n\d{8}')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def pin(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def save_new(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    # Every science-related record is write-once; a partial failed write is kept.
    with path.open('xb') as handle:
        handle.write(raw)
    return pin(path)


def load_bound(desc):
    path = Path(desc['path'])
    require(path.is_file() and not path.is_symlink(), 'Regular pinned input required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == desc['sha256'], 'Pinned input changed')
    return json.loads(raw.decode('utf-8-sig'))


def checked_rows(audit):
    require(audit['schema'] == 'EP_SOURCE_POPULATION_READONLY_AUDIT_V1', 'Wrong source audit')
    require(audit['pixel_hash_missing_count'] == COUNT, 'This finite job is only the audited old1900')
    rows = audit['pixel_hash_missing_records']
    require(len(rows) == COUNT, 'Exactly1900 old input rows required')
    ids, paths = set(), set()
    for row in rows:
        sid = row['source_id']
        match = SID.fullmatch(sid)
        require(match and sid.split('/')[0] == sid.rsplit('_', 1)[-1], 'Invalid old source identity')
        cid = 'imagenet-val:' + match.group(1)
        require(cid == row['canonical_source_id'] and cid not in ids, 'Old source identity duplicate/differs')
        expected = IMAGE_ROOT + '/' + sid + '.JPEG'
        require(row['path'] == expected and '..' not in PurePosixPath(expected).parts and
                expected not in paths and SHA.fullmatch(row['original_file_sha256']),
                'Old input escaped fixed pool, duplicated, or lacks original SHA')
        ids.add(cid)
        paths.add(expected)
    return rows


def prepare(audit_path, preprocess_module, out, python, stop_files, deadline_unix):
    require(0 < deadline_unix - time.time() <= 86400, 'Explicit future deadline within24h required')
    audit_pin = pin(audit_path)
    audit = load_bound(audit_pin)
    rows = checked_rows(audit)
    pre_pin = pin(preprocess_module)
    require(pre_pin['sha256'] == EXPECTED_PREPROCESS_SHA == audit['original_preprocess']['sha256'],
            'Only the frozen original preprocess module is admitted')
    interpreter = Path(python).absolute()  # Keep the venv spelling; do not resolve its executable symlink.
    require(interpreter.is_file(), 'Explicit existing CPU environment interpreter required')
    stop_files = [str(Path(p).absolute()) for p in stop_files]
    require(stop_files and len(set(stop_files)) == len(stop_files), 'Explicit distinct STOP paths required')
    out = Path(out).absolute()
    require(not out.exists(), 'Fresh output directory required')
    request = dict(schema='EP_OLD1900_HASH_REQUEST_V1', status='EP_OLD1900_HASH_PREPARED_NO_IMAGE_ACCESS',
        source_audit=audit_pin, source_count=COUNT, records=rows, preprocess_module=pre_pin,
        script=pin(__file__), python=str(interpreter), out=str(out),
        stop_files=stop_files + [str(out/'STOP')], deadline_unix=deadline_unix,
        max_seconds=86400, threads=2, pixel_hash_domain=PIXEL_DOMAIN,
        image_reads=0, Encoder_calls=0, VAR_calls=0, PHY_calls=0, metric_calls=0,
        source_reselection=False, automatic_retry=False, automatic_successor=False,
        numerical_preprocess_replacement=False, scope='Only previously registered1900; never new100')
    out.mkdir(parents=True)
    proof = save_new(out/'request.json', request)
    return dict(status=request['status'], request=proof, source_count=COUNT, image_reads=0)


def guard(request, now=time.time):
    require(now() < request['deadline_unix'], 'Registered old1900 deadline reached')
    require(not any(Path(p).exists() for p in request['stop_files']), 'Registered STOP requested')


def image_file_matches(row):
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink() and not path.parent.is_symlink(),
            'Expected regular old JPEG (not symlink) required')
    require(path.resolve().is_relative_to(Path(IMAGE_ROOT).resolve()), 'Old image escaped original pool')
    require(sha(path) == row['original_file_sha256'], 'Original old JPEG SHA differs')
    return path


def process_one(row, preprocess, np, file_checker=image_file_matches):
    path = file_checker(row)
    tensor, expected_pixels_sha = preprocess(path)
    require(str(tensor.device) == 'cpu', 'Original preprocessing must stay on CPU')
    pixels = np.ascontiguousarray(np.rint((tensor.detach().cpu().numpy() + 1) * 127.5), dtype=np.uint8)
    require(pixels.shape == (3, 256, 256), 'Original canonical256 shape differs')
    forward = hashlib.sha256(pixels.tobytes()).hexdigest()
    require(forward == expected_pixels_sha, 'Original preprocessing/tensor round-trip hash differs')
    flip = hashlib.sha256(np.ascontiguousarray(pixels[:, :, ::-1]).tobytes()).hexdigest()
    file_checker(row)  # Detect a changed input during image decode/preprocessing.
    return dict(source_id=row['source_id'], canonical_source_id=row['canonical_source_id'], path=str(path),
        original_file_sha256=[row['original_file_sha256']], pixel_sha256=sorted({forward, flip}),
        roles=['historical_registered3000_old1900_completed_hashes'],
        original_preprocess_pixel_sha256=forward, horizontal_flip_pixel_sha256=flip)


def run(request_path):
    request_path = Path(request_path).absolute()
    request_pin = pin(request_path)
    request = load_bound(request_pin)
    require(request['schema'] == 'EP_OLD1900_HASH_REQUEST_V1' and request['source_count'] == COUNT,
            'Wrong finite request')
    require(request['script'] == pin(__file__), 'Hash tool changed after preparation')
    require(request_path == Path(request['out'])/'request.json', 'Request/output identity differs')
    require(os.path.abspath(sys.executable) == request['python'], 'Run the exact registered venv interpreter')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Launch a dedicated CPU-only process with CUDA_VISIBLE_DEVICES empty')
    audit = load_bound(request['source_audit'])
    require(checked_rows(audit) == request['records'], 'Source rows changed after prepare')
    require(sha(request['preprocess_module']['path']) == request['preprocess_module']['sha256'] == EXPECTED_PREPROCESS_SHA,
            'Original preprocessing module changed')
    guard(request)
    out = Path(request['out'])
    # Exclusive one-shot claim. A failed/interrupted owner is preserved; same-path rerun is rejected.
    save_new(out/'owner_started.json', dict(request=request_pin, pid=os.getpid(), started_unix=time.time(),
                                          no_models_or_scientific_evaluation=True))
    completed, input_hashes, outputs = [], {}, {}
    started = time.time()
    try:
        import numpy as np
        import torch
        import torchvision
        import PIL
        require(not torch.cuda.is_initialized(), 'CUDA was initialized in CPU-only owner')
        torch.set_num_threads(request['threads'])
        module_spec = importlib.util.spec_from_file_location('_ep_exact_prior_preprocess', request['preprocess_module']['path'])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        runtime = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                       torchvision=torchvision.__version__, pillow=PIL.__version__, cuda_initialized=False)
        proof = save_new(out/'runtime.json', runtime)
        outputs[proof['path']] = proof['sha256']
        (out/'records').mkdir()
        for i, row in enumerate(request['records']):
            guard(request)
            record = process_one(row, module.preprocess, np)
            guard(request)
            require(not torch.cuda.is_initialized(), 'Preprocessing unexpectedly initialized CUDA')
            proof = save_new(out/'records'/f'{i:04d}.json', dict(request=request_pin, record=record))
            outputs[proof['path']] = proof['sha256']
            input_hashes[row['path']] = row['original_file_sha256']
            completed.append(record)
            if (i+1) % 100 == 0:
                print(json.dumps(dict(old_source_hashes_complete=i+1, total=COUNT)), flush=True)
        require(len(completed) == COUNT, 'Old hash population incomplete')
        require(sha(request['preprocess_module']['path']) == EXPECTED_PREPROCESS_SHA and pin(request_path) == request_pin,
                'Request or original preprocess source changed while running')
        guard(request)
        normalized = dict(schema='EP_NORMALIZED_PRIOR_SOURCES_V1', source_count=COUNT,
            pixel_hash_domain=PIXEL_DOMAIN, records=completed, request=request_pin,
            actual_original_JPEG_inputs=input_hashes, original_preprocess=request['preprocess_module'],
            new100_images_read=0, Encoder_calls=0, VAR_calls=0, PHY_calls=0, metric_calls=0)
        proof = save_new(out/'normalized_records.json', normalized)
        outputs[proof['path']] = proof['sha256']
        completion = dict(status='EP_OLD1900_CANONICAL_HASH_COMPLETION', source_count=COUNT, request=request_pin,
            normalized_records=proof, outputs=outputs, actual_original_JPEG_inputs=input_hashes,
            elapsed_seconds=time.time()-started, image_preprocessing_calls=COUNT,
            new100_images_read=0, Encoder_calls=0, VAR_calls=0, PHY_calls=0, metric_calls=0,
            execution_only='Parent actual-wait receipt must be bound separately', automatic_successor=False)
        return dict(status=completion['status'], completion=save_new(out/'completion.json', completion))
    except BaseException:
        save_new(out/'failure.json', dict(status='EP_OLD1900_HASH_STOPPED_NO_RETRY', request=request_pin,
            completed_source_count=len(completed), elapsed_seconds=time.time()-started,
            traceback=traceback.format_exc(), existing_records_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    for name in ('audit', 'preprocess-module', 'python', 'out'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--stop-file', action='append', required=True)
    prep.add_argument('--deadline-unix', required=True, type=float)
    execute = subs.add_parser('run')
    execute.add_argument('--request', required=True)
    a = parser.parse_args()
    if a.command == 'prepare':
        result = prepare(a.audit, a.preprocess_module, a.out, a.python, a.stop_file, a.deadline_unix)
    else:
        result = run(a.request)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
