"""CPU provenance and cache contracts for an isolated N2048 control extension."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import re
import sys
import numpy as np
from external_eval_common import (sha, read, write, seal, identity, verify, pixels,
                                 rgb_sha, atomic_npz, PauseRequested, write_csv)

HERE = Path(__file__).resolve().parent
VERSION = 'OWN-CONTROLS-N2048-20261004-R1'
PHY_VERSION = 'own-controls-raw-partial-N2048-20261004-v1'
ORIGINAL_PHY_SHA = 'd5b26b35b1ce5ba8c147998e2ec6aa88615a2c84a0ab13d59026e27094b7d756'
SNRS = (1, 7, 13)
CAL_SEEDS = (4101, 4102, 4103)
DEV_SEEDS = (2001, 2002, 2003)
METHODS = ('D_U_whole_N2048_v1', 'M1_entropy_N2048_v1')
STAGES = ('qualify', 'screen', 'calibrate', 'development', 'export-n1024')
OWN_SOURCES = ('own_controls.py', 'own_controls_common.py', 'own_controls_native.py',
               'own_controls_phy.py', 'own_controls_tests.py', 'own_controls_phy_tests.py',
               'own_controls_protocol.json', 'own_controls_README.md')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_protocol(protocol):
    expected = read(HERE/'own_controls_protocol.json')
    require(protocol == expected and protocol.get('version') == VERSION,
            'Only the complete registered control protocol is allowed')
    require(protocol['snrs'] == list(SNRS) and protocol['calibration_seeds'] == list(CAL_SEEDS)
            and protocol['development_seeds'] == list(DEV_SEEDS)
            and protocol['methods'] == list(METHODS) and protocol['N'] == 2048
            and protocol['phy'] == 'QPSK' and protocol['holdout_access'] is False,
            'Control population, budget or noise scope differs')
    return protocol


def locations(root):
    root = Path(root).resolve()
    return dict(root=root, out=root/'outputs/EXTERNAL-COMPARISON-20261004/own_controls',
                result=root/'results/external_comparison_20261004/own_controls',
                source=root/'experiments/scale-causal-partial-residual-20261002',
                original_out=root/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002',
                original_result=root/'results/scale_causal_partial_residual_20261002')


def published(value):
    require(value.get('status') == 'PUSHED' and value.get('checks') == 'PASS'
            and re.fullmatch('[0-9a-f]{40}', str(value.get('commit', '')))
            and value.get('remote_commit') == value['commit'], 'Checked original publication is required')


def derived_phy_bytes(raw):
    require(__import__('hashlib').sha256(raw).hexdigest() == ORIGINAL_PHY_SHA,
            'Original M1 PHY bytes changed')
    for old, new, count in ((b'raw-partial-token-20261002-v1', PHY_VERSION.encode(), 1),
                           (b'N not in (512, 1024)', b'N not in (2048,)', 2),
                           (b'budgets=[512, 1024]', b'budgets=[2048]', 1)):
        require(raw.count(old) == count, 'Unexpected PHY derivation occurrence count')
        raw = raw.replace(old, new)
    return raw


def source_gate(root, protocol_path):
    p = locations(root); old = p['source']/'partial_phy.py'
    publication_path = p['original_result']/'provenance/m1_publication.json'
    publication = read(publication_path); published(publication)
    require(publication['source_bindings'].get(str(old)) == ORIGINAL_PHY_SHA,
            'Original PHY lacks its published source identity')
    require((HERE/'own_controls_phy.py').read_bytes() == derived_phy_bytes(old.read_bytes()),
            'N2048 PHY contains changes beyond its explicit budget/protocol extension')
    # Bind the exact scientific primitives reused through private module aliases.
    expected = publication['source_bindings']
    for name in ('partial_phy.py', 'partial_receiver.py', 'common.py', 'm1_runner.py'):
        path = p['source']/name
        require(expected.get(str(path)) == sha(path), 'Original scientific primitive changed: '+name)
    files = [publication_path, protocol_path, *[HERE/name for name in OWN_SOURCES],
             HERE/'external_eval_common.py', HERE/'step0_reference_prepare.py', HERE/'step0_reference_protocol.py',
             HERE/'step0_cache_export.py']
    bindings = {str(Path(path).resolve()):sha(path) for path in files}
    original, registration_bindings = original_registration(root, 'calibration')
    bindings.update(registration_bindings)
    verify(original['source_bindings'])
    bindings.update(original['source_bindings'])
    paths = [Path(original['identity']['decoder_gate'])]
    # model_paths and decoder loader verify the original checkpoint file hashes;
    # the registered model state hashes below are checked again after real load.
    bindings.update({str(path):sha(path) for path in paths})
    require(original.get('training_updates') == 0 and len(original.get('source_ids', [])) == 1000,
            'Original complete calibration registration is absent')
    return bindings


def original_registration(root, role):
    require(role in ('calibration', 'development'), 'Only original calibration/development are admitted')
    p = locations(root); name = 'm1_'+role+'_registration.json'
    publication_path = p['original_result']/'provenance/m1_publication.json'
    publication = read(publication_path); published(publication)
    copy = p['original_result']/'provenance'/name; original = p['original_out']/name
    require(publication['published_files'].get(str(copy)) == sha(copy) == sha(original),
            'Original population registration differs from its published bytes')
    return read(original), {str(path):sha(path) for path in (publication_path, copy, original)}


def data_proof(data, original, role):
    count = 1000 if role == 'calibration' else 100
    records = data['records']
    ids = [r['image_id'] for r in records]; preprocessing = [r['preprocessing_id'] for r in records]
    require(len(records) == len(set(ids)) == count and original['source_ids'] == ids
            and original['preprocessing_ids'] == preprocessing and original['data_bindings'] == data['bindings'],
            'Original '+role+' population or cached encoder/F/tokens changed')
    return dict(source_ids=ids, preprocessing_ids=preprocessing, data_bindings=data['bindings'],
                source_classes=[int(r['class_index']) for r in records])


def action_id(a):
    return f'N{a.N}/{a.phy}/m{a.m}/K{a.q}/{a.order}'


def action_dict(a):
    return dict(N=a.N, phy=a.phy, m=a.m, q=a.q, order=a.order)


def expected_keys(index, choices, seeds):
    return [(index, int(snr), int(seed), action_id(a))
            for snr, actions in choices.items() for a in actions for seed in seeds]


def row_key(row):
    return (int(row['source_index']), int(row['snr_db']), int(row['noise_seed']), row['action_id'])


def validate_rows(rows, index, source, choices, seeds):
    wanted = expected_keys(index, choices, seeds)
    require([row_key(r) for r in rows] == wanted, 'Calibration row order/count/source/noise/action changed')
    for row in rows:
        require(row.get('source_id') == source['image_id'] and row.get('preprocessing_id') == source['preprocessing_id']
                and row.get('N') == 2048 and row.get('phy_family') == 'QPSK' and row.get('E') == 4096,
                'Control source, modulation or actual energy differs')
        require(all(math.isfinite(float(row[k])) for k in ('psnr_db', 'lpips_alex', 'dino_cosine')),
                'Nonfinite calibration metric')


def summarize(rows):
    # Match the old runner's source-first sequential accumulation and ranking.
    sums = {}; first = {}
    for row in rows:
        key = (row['snr_db'], row['action_id']); first[key] = row
        value = sums.setdefault(key, dict(n=0, psnr_db=0., lpips_alex=0., dino_cosine=0., E=0., failure_fraction=0.))
        value['n'] += 1
        for metric in ('psnr_db', 'lpips_alex', 'dino_cosine', 'E'):
            value[metric] += float(row[metric])
        value['failure_fraction'] += float(not row['header_ok'] or not row['body_crc_ok'])
    summary = []
    for key, value in sorted(sums.items()):
        row = first[key]
        item = {k:row[k] for k in ('N', 'phy_family', 'snr_db', 'action_id', 'm', 'q', 'order')}
        item.update({k:v/value['n'] for k,v in value.items() if k != 'n'}); item['rows'] = value['n']
        summary.append(item)
    return summary


def sealed_checkpoint(path, value):
    value = dict(value); value['payload_sha256'] = identity(value); seal(path, value)
    return value


def checkpoint(path, binding, index):
    value = read(path)
    require(value.get('payload_sha256') == identity({k:v for k,v in value.items() if k != 'payload_sha256'})
            and value.get('binding') == binding and value.get('source_index') == index,
            'Saved control checkpoint identity changed')
    return value


def save_float_cache(path, images, source, target):
    images = [pixels(x) for x in images]; source, target = pixels(source), pixels(target)
    arrays = dict(images=np.stack(images), native_reference=source, comparison_reference=target)
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays)
                    and all(np.array_equal(saved[key], value) for key,value in arrays.items()),
                    'Existing isolated float cache differs; preserve it for review')
    else:
        atomic_npz(path, **arrays)
    return dict(path=str(path), sha256=sha(path), image_sha256=[rgb_sha(a) for a in images],
                native_reference_sha256=rgb_sha(source), comparison_reference_sha256=rgb_sha(target),
                dtype='float32', layout='CHW256', lossless=True)


def verify_float_cache(receipt, rows, native_reference=None, comparison_reference=None):
    path = Path(receipt['path']); require(sha(path) == receipt['sha256'], 'Float cache container changed')
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {'images', 'native_reference', 'comparison_reference'}, 'Float cache schema differs')
        images = data['images']; source = pixels(data['native_reference']); target = pixels(data['comparison_reference'])
        require([rgb_sha(image) for image in images] == receipt['image_sha256']
                and rgb_sha(source) == receipt['native_reference_sha256']
                and rgb_sha(target) == receipt['comparison_reference_sha256'], 'Float cache pixels changed')
        if native_reference is not None:
            require(np.array_equal(source, pixels(native_reference)), 'Native reference pixels differ from original source')
        if comparison_reference is not None:
            require(np.array_equal(target, pixels(comparison_reference)), 'Comparison reference pixels differ from admitted source')
        for row in rows:
            slot = row['image_slot']
            require(type(slot) is int and 0 <= slot < len(images)
                    and row['image_sha256'] == receipt['image_sha256'][slot]
                    and row['reference_sha256'] == receipt['comparison_reference_sha256'],
                    'Float image slot or row/reference association changed')


def old_source_inventory(root):
    root = Path(root); out = root/'outputs/UNIFIED-METRICS-20261002'
    result = root/'results/unified_metrics_20261002'
    path = out/'completion.json'; value = read(path)
    require(value.get('status') == 'UNIFIED_METRICS_COMPLETE', 'Original complete metrics are required')
    published(value['publication'])
    copy = result/'provenance/scoring_completion.json'; scoring = out/'scoring_completion.json'
    require(value['publication']['published_files'].get(str(copy)) == sha(copy) == sha(scoring),
            'Scoring completion differs from its checked published bytes')
    scored = read(scoring)
    require(scored.get('status') == 'COMPLETE' and scored.get('synthetic') is False
            and scored.get('parity_passed') is True and scored.get('training_updates') == 0
            and scored.get('policy_selection_updates') == 0, 'Original real scoring is incomplete')
    inventory_path = result/'scoring_inventory.json'; registration_path = result/'metrics_registration.json'
    for item in (inventory_path, registration_path):
        require(scored['outputs'].get(str(item)) == sha(item), 'Published original scoring evidence differs')
    inventory = read(inventory_path); registration = read(registration_path)
    cp = inventory['source_checkpoint_sha256']
    require(inventory.get('sources') == 100 and inventory.get('parity_passed') is True
            and set(cp) == {str(out/'source_checkpoints'/f'{i:03d}.json') for i in range(100)},
            'Original complete 100-source checkpoint inventory differs')
    bindings = {str(item):sha(item) for item in (path, copy, scoring, inventory_path, registration_path)}
    return dict(checkpoints=cp, registration_binding=identity(registration), bindings=bindings)


def original_checkpoint(path, inventory, index, source):
    require(inventory['checkpoints'].get(str(path)) == sha(path), 'Original checkpoint differs from published inventory')
    value = checkpoint(path, inventory['registration_binding'], index)
    baseline = value['baseline']
    require(baseline.get('source_index') == index and baseline.get('source_id') == source['image_id']
            and baseline.get('preprocessing_id') == source['preprocessing_id']
            and baseline.get('true_class_index') == int(source['class_index']), 'Original checkpoint source identity differs')
    require(len({r['replay_row_id'] for r in value['rows']}) == len(value['rows']), 'Original checkpoint has duplicate metric rows')
    return value


def validate_development_rows(rows, index, source, cells, policy_sha):
    wanted = [(s, seed, method) for s in SNRS for seed in DEV_SEEDS for method in METHODS]
    require([(r['snr_db'], r['noise_seed'], r['method']) for r in rows] == wanted,
            'New development method/SNR/noise scope changed')
    for row in rows:
        action = cells[row['snr_db'], row['method']]['action']
        require(row['source_id'] == source['image_id'] and row['source_index'] == index
                and row['preprocessing_id'] == source['preprocessing_id']
                and row['source_class_index'] == int(source['class_index'])
                and row['N'] == 2048 and row['E'] == 4096 and row['phy_family'] == 'QPSK'
                and all(row[k] == action[k] for k in ('m', 'q', 'order'))
                and row['policy_sha256'] == policy_sha and row['policy_selected_on_development'] is False
                and row['semantic_side_information'] is False and row['receiver_class_embedding'] == 1000,
                'New development selected action, source or policy differs')


def validate_export_rows(rows, index, source, originals, measured):
    require(len(rows) == len(originals) == 27 and {r['original_replay_row_id'] for r in rows} == set(originals),
            'N1024 selected replay identity coverage differs')
    for row in rows:
        rid = row['original_replay_row_id']; original = originals[rid]
        expected_method = original['method'] if original['method'] in ('P1024', 'D_U_QPSK') else 'M1_entropy_N1024_frozen'
        require(row['original_scientific_row'] == original and row['original_completed_metrics'] == measured[rid]
                and row['original_scientific_row_sha256'] == identity(original)
                and row['source_index'] == index and row['source_id'] == source['image_id']
                and row['source_class_index'] == int(source['class_index']) and row['N'] == 1024
                and row['method'] == expected_method and row['snr_db'] == int(original['snr_db'])
                and row['noise_seed'] == int(original['noise_seed'])
                and row['image_sha256'] == measured[rid]['image_sha256']
                and row['native_reference_sha256'] == measured[rid]['reference_sha256']
                and row['original_scalar_parity'].get('status') == 'PASS'
                and row['original_scalar_parity'].get('missing_fields') == []
                and row['policy_selection_updates'] == 0,
                'N1024 saved scientific row/pixel/parity association changed')
