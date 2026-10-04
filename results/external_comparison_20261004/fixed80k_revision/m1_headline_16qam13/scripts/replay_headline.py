"""Replay the requested N1024 / 16QAM / 13 dB fixed16 headline pixels.

Uses the frozen replay and original batching without changing any old file.
The original PSNR/LPIPS/DINO calculation is a strict replay check, not a new
metric evaluation. All reported metrics are copied from the completed rows.
No training, calibration, policy choice, N2048, or full queue is started.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

FIXED = (0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
MISSING = (4,21,24,29,33,41,52,60,64,87,92,95)
SNRS = (13,)
STUDIES = ('N1024','M1')
PARENT_CONFIG_SHA = 'cf1dcdb627989fe26854a182b84248b7206957ce7ad808bc9f613290de853d7d'
VERSION = 'M1-HEADLINE-16QAM13-EXACT-EXPORT-20261004-R1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selected(study, row):
    if int(row['noise_seed']) != 2001 or int(row['snr_db']) not in SNRS:
        return False
    if study == 'N1024':
        return row['method'] == 'P1024'
    return (study == 'M1' and int(row['N']) == 1024 and
            row['phy_family'] == '16QAM' and row['method'] == 'entropy_policy')


def wanted_rows(engine, index):
    result = {}
    for study in STUDIES:
        rows = [row for row in engine.by_source[study][index] if selected(study,row)]
        require(len(rows) == 1 and {int(r['snr_db']) for r in rows} == set(SNRS),
                'Expected exactly the previously scored headline row per method')
        require(all(int(row['source_index']) == index for row in rows), 'Source index differs')
        result[study] = rows
    return result


def admit_runtime(root):
    base = root/'outputs/EXTERNAL-COMPARISON-20261004'
    config_path = base/'fixed80k_revision/delivery/config.json'
    require(sha(config_path) == PARENT_CONFIG_SHA, 'Frozen parent configuration differs')
    config = read(config_path); runtime = base/'runtime'
    require(Path(config['root']).resolve() == root and Path(config['runtime']) == runtime,
            'Frozen runtime/root path differs')
    # Use absolute executable spelling, not its symlink target: venvs can share
    # a binary while supplying different scientific packages.
    require(os.path.abspath(sys.executable) == os.path.abspath(config['native_python']),
            'Run with the already qualified native/metric Python environment')
    for key, value in config['native_environment'].items():
        if key in os.environ:
            require(os.environ[key] == value, 'Native environment differs: '+key)
        os.environ[key] = value
    for path in config['native_environment']['PYTHONPATH'].split(os.pathsep):
        if path not in sys.path:
            sys.path.insert(0,path)
    bindings = {str(config_path):sha(config_path), str(Path(__file__).resolve()):sha(__file__)}
    names = ('own_controls_native.py','own_controls_common.py','own_controls_phy.py',
             'external_eval_common.py')
    for name in names:
        path = runtime/name
        require(config['bindings'].get(str(path)) == sha(path), 'Frozen runtime dependency differs: '+name)
        bindings[str(path)] = sha(path)
    sys.path.insert(0,str(runtime))
    import own_controls_common as common
    from own_controls_native import Native
    for module_name in ('own_controls_common','own_controls_native','external_eval_common'):
        require(Path(sys.modules[module_name].__file__).resolve() == runtime/(module_name+'.py'),
                'A different runtime module was already imported')
    original = common.source_gate(root,runtime/'own_controls_protocol.json')
    for name, expected in original.items():
        if Path(name).parent == runtime:
            require(config['bindings'].get(name) == expected, 'Unbound own-controls support file: '+name)
    bindings.update(original)
    return common, Native, bindings


def verify_source(value, binding, index, expected, measured, target, replay, common):
    require(value.get('binding') == binding and value.get('source_index') == index
            and value.get('payload_sha256') == common.identity({k:v for k,v in value.items() if k != 'payload_sha256'}),
            'Saved export source identity differs')
    rows = value['rows']; ids = [r['replay_row_id'] for r in rows]
    require(len(rows) == 2 and len(set(ids)) == 2 and set(ids) == set(expected), 'Saved export coverage differs')
    proof = value['float_reconstructions']; require(sha(proof['path']) == proof['sha256'], 'Float NPZ changed')
    with np.load(proof['path'],allow_pickle=False) as data:
        require(set(data.files) == {'images','source_rgb','row_ids','image_slots'}, 'Float NPZ schema differs')
        images, source = data['images'], data['source_rgb']
        require(images.shape == (2,3,256,256) and images.dtype == np.float32
                and np.array_equal(source,target) and source.dtype == np.float32
                and data['row_ids'].tolist() == ids and data['image_slots'].tolist() == list(range(2)),
                'Float NPZ population, shape, precision or reference differs')
        for slot,row in enumerate(rows):
            rid = row['replay_row_id']; old = measured[rid]
            require(row['image_slot'] == slot and row['original_row'] == expected[rid]
                    and row['completed_metrics'] == old and row['original_scalar_parity']['status'] == 'PASS',
                    'Original scientific row, copied metrics or parity differs')
            require(row['float_rgb_sha256'] == old['image_sha256'] == replay.rgb_fingerprint(images[slot])
                    and row['reference_sha256'] == old['reference_sha256'] == replay.rgb_fingerprint(source),
                    'Replayed pixels differ from completed float RGB fingerprints')
    return value


def run(root, scope, launch_id):
    import fcntl
    common, Native, inputs = admit_runtime(root)
    require(scope == 'all16', 'The headline requires all 16 registered sources')
    indices = FIXED
    out = root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/m1_headline_16qam13'/('replay_'+scope)
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping = False; started = time.monotonic(); count = 0
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def status(state, **kwargs):
        common.write(out/'status.json',dict(status=state,pid=os.getpid(),launch_id=launch_id,
            safe_pause_handler_installed=True,completed_images=count,expected_images=2*len(indices),
            source_indices=list(indices),elapsed_seconds=time.monotonic()-started,**kwargs))
    try:
        require(not (out/'failure.json').exists(), 'A previous failed export requires review before retry')
        status('LOADING_FROZEN_MODELS')
        inventory = common.old_source_inventory(root); inputs.update(inventory['bindings'])
        native = Native(root,stop); engine = native.replay_engine(); replay = native.replay
        registration = dict(version=VERSION,scope=scope,source_indices=list(indices),N=1024,
            methods=['P1024','M1_entropy_16QAM'],snrs=list(SNRS),noise_seed=2001,
            source_bindings=inputs,replay_manifest=engine.manifest(),numerical_runtime=native.flags,
            published_replay_compatibility=native.compatibility,synthetic=False,
            training_updates=0,policy_selection_updates=0,calibration_executed=False,
            new_metrics_computed=False,historical_scalar_recomputed_for_verification=True,
            output_scope='Only exact existing P1024/M1 rows; original batching and scalar parity are retained')
        common.seal(out/'registration.json',registration); binding = common.identity(registration)
        frames, sources, outputs = [], [], {}
        for index in indices:
            if stopping:
                status('PAUSED'); return 75
            source = engine.records[index]; target = engine.target(index)
            oldpath = root/f'outputs/UNIFIED-METRICS-20261002/source_checkpoints/{index:03d}.json'
            measured = common.original_checkpoint(oldpath,inventory,index,source)
            measured = {r['replay_row_id']:r for r in measured['rows']}
            wanted = wanted_rows(engine,index)
            expected = {replay.row_id(study,r):r for study,rows in wanted.items() for r in rows}
            require(len(expected) == 2 and set(expected).issubset(measured), 'Original metric rows missing')
            cp = out/'source_checkpoints'/f'{index:03d}.json'
            if cp.exists():
                saved = read(cp)
            else:
                rows, images = [], []
                for study in STUDIES:
                    seen = set(); selected_ids = {replay.row_id(study,r) for r in wanted[study]}
                    # Preserve the original iterator, including continuous
                    # three-noise rendering batches. Do not switch kernels or
                    # truncate its internal plugin work to optimize this export.
                    with native.torch.no_grad():
                        for row,image,computed in engine.adapters[study].iterate_source(index,wanted[study]):
                            rid = replay.row_id(study,row)
                            require(rid in selected_ids and rid not in seen and row == expected[rid], 'Frozen replay row differs')
                            parity = replay.parity_check(row,computed,required_quality=True)
                            rgb = replay.rgb_array(image); fingerprint = replay.rgb_fingerprint(rgb); old = measured[rid]
                            require(fingerprint == old['image_sha256'] and replay.rgb_fingerprint(target) == old['reference_sha256'],
                                    'Reconstruction or reference differs from completed float RGB')
                            rows.append(dict(study=study,source_index=index,source_id=source['image_id'],N=1024,
                                snr_db=int(row['snr_db']),noise_seed=2001,method='P' if study == 'N1024' else 'M1',
                                image_slot=len(images),replay_row_id=rid,float_rgb_sha256=fingerprint,
                                reference_sha256=old['reference_sha256'],exact_completed_rgb_match=True,
                                original_row=row,completed_metrics=old,original_scalar_parity=parity))
                            images.append(rgb); seen.add(rid)
                    require(seen == selected_ids, 'Frozen replay omitted requested rows')
                npz = out/'float_reconstructions'/f'{index:03d}.npz'
                arrays = dict(images=np.stack(images),source_rgb=target,
                    row_ids=np.asarray([r['replay_row_id'] for r in rows]),image_slots=np.arange(2,dtype=np.int64))
                if npz.exists():
                    with np.load(npz,allow_pickle=False) as cached:
                        require(set(cached.files) == set(arrays) and all(np.array_equal(cached[k],v) for k,v in arrays.items()),
                                'Uncheckpointed existing float NPZ differs; preserve for review')
                else:
                    common.atomic_npz(npz,**arrays)
                saved = common.sealed_checkpoint(cp,dict(binding=binding,source_index=index,rows=rows,
                    original_completed_checkpoint_sha256=sha(oldpath),synthetic=False,
                    float_reconstructions=dict(path=str(npz),sha256=sha(npz),dtype='float32',layout='CHW256',lossless=True)))
            require(saved['original_completed_checkpoint_sha256'] == sha(oldpath), 'Completed source checkpoint changed')
            verify_source(saved,binding,index,expected,measured,target,replay,common)
            inputs[str(oldpath)] = sha(oldpath); outputs[str(cp)] = sha(cp)
            proof = saved['float_reconstructions']; outputs[proof['path']] = proof['sha256']
            sources.append(dict(source_index=index,source_id=source['image_id'],preprocessing_id=source['preprocessing_id'],
                class_index=int(source['class_index']),reference_sha256=replay.rgb_fingerprint(target),
                checkpoint=str(cp),float_reconstructions=proof))
            frames.extend(saved['rows']); count = len(frames); status('EXPORTING',source_index=index)
        manifest = engine.verify_frozen(); native.frozen(); common.verify(inputs)
        common.seal(out/'manifest.json',dict(status='EXACT_COMPLETED_RGB_EXPORT_PASS',version=VERSION,
            sources=sources,frames=frames,source_indices=list(indices),images=count,N=1024,snrs=list(SNRS),noise_seed=2001,
            synthetic=False,new_reconstruction_export=True,new_metric_evaluation=False,new_training=False,new_calibration=False,
            historical_scalar_recomputed_for_verification=True,training_updates=0,policy_updates=0,
            scientific_selection_changed=False,exact_completed_rgb_match=True,full_queue_resumed=False,
            registration_sha256=sha(out/'registration.json'),input_bindings=inputs,outputs=outputs,replay_manifest=manifest))
        status('COMPLETE'); print('EXACT_COMPLETED_RGB_EXPORT_PASS',count,flush=True); return 0
    except BaseException as error:
        common.write(out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False))
        status('FAILED_REQUIRES_REVIEW',error=repr(error)); raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--scope',choices=('all16',),default='all16')
    parser.add_argument('--launch-id',required=True)
    args = parser.parse_args()
    return run(Path(args.root).resolve(),args.scope,args.launch_id)


if __name__ == '__main__':
    raise SystemExit(main())
