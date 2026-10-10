"""Metadata-only source exclusion and selection; never opens image/model assets.

An explicit, pinned study-scope closure is required before selecting IDs. A
separate complete canonical-pixel hash gate is required before any encoding.
Missing historical hashes are never silently treated as successful deduplication.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

PIXEL_DOMAIN = 'sha256_uint8_CHW_3x256x256_bytes'
SELECTION_SEED = 'VAR_COMM_EP_NEW100_20261010_v1'
NOISE_SEEDS = [9301, 9302, 9303]
METHODS = ['RAW_WHOLE', 'RAW_PARTIAL', 'EC_VAR_WHOLE', 'EC_VAR_PARTIAL']
SHA = re.compile(r'[0-9a-f]{64}')
VAL = re.compile(r'ILSVRC2012_val_(\d{8})(?:_|\.|/|$)')
SID = re.compile(r'n\d{8}/ILSVRC2012_val_\d{8}_n\d{8}')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_id(value):
    require(isinstance(value, str) and value, 'Nonempty source ID required')
    match = VAL.search(value)
    if match:
        return 'imagenet-val:' + match.group(1)
    if value.startswith('imagenet-val:'):
        require(re.fullmatch(r'imagenet-val:\d{8}', value), 'Invalid canonical val ID')
    return value


def read_json(path, expected=None):
    path = Path(path)
    require(path.suffix.lower() == '.json' and path.is_file() and not path.is_symlink(),
            'Explicit regular JSON metadata input required')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Metadata changed while reading')
    pin = {'path': str(path.absolute()), 'sha256': digest(raw)}
    if expected is not None:
        require(SHA.fullmatch(expected) and pin['sha256'] == expected, 'Metadata SHA mismatch')
    return json.loads(raw.decode('utf-8-sig')), pin


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), 'Fresh output required; existing registrations are immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as handle:
        handle.write(raw)
    return {'path': str(path.absolute()), 'sha256': digest(raw)}


def valid_hashes(values):
    require(isinstance(values, list) and all(isinstance(x, str) and SHA.fullmatch(x) for x in values),
            'Hash list must contain only full lowercase SHA256 strings')
    return set(values)


def imported_rows(value, fmt):
    """Yield (source ID, raw hashes, canonical pixel hashes, roles, source path)."""
    if fmt == 't6_reference_inventory':
        require(value['status'] == 'T6_AUDITED_OLD_CONTENT_HASH_INVENTORY_COMPLETE', 'Wrong T6 inventory')
        for row in value['records']:
            yield (row['source_id'], row['raw_JPEG_sha256'], row['preprocessing_sha256'], row['roles'], None)
    elif fmt == 'legacy_exposure_union':
        require(value['status'] == 'PROVEN_LOCAL_EXPOSURE_ID_UNION_WITH_UNRESOLVED_INVENTORY',
                'Wrong historical exposure audit')
        for role, ids in value['primary_roles'].items():
            for sid in ids:
                yield sid, [], [], [role], None
        for field in ('source_ids', 'conservative_pending_exclusions', 'conservative_exclusion_union'):
            for sid in value.get(field, []):
                yield sid, [], [], ['historical_conservative_exclusion'], None
    elif fmt == 'legacy_holdout_manifest':
        require(value['role'] == 'independent_holdout_after_method_freeze', 'Wrong old holdout manifest')
        require(value['source_images'] == len(value['images']), 'Old holdout cardinality mismatch')
        for row in value['images']:
            yield (row['image_id'], [row['file_sha256']], [row['preprocessed_rgb_sha256']],
                   ['historically_evaluated_holdout1000'], row['path'])
    elif fmt == 'legacy_source_integrity':
        # Earlier selection could inspect a rejected source. Keep these IDs too;
        # do not silently equate a rejected candidate with an unopened candidate.
        require('inspected' in value and 'rejections' in value, 'Source integrity receipt required')
        for field in ('inspected', 'rejections'):
            for row in value[field]:
                sid = row.get('image_id') or row.get('source_id') or row.get('path')
                yield (sid, [row['file_sha256']] if row.get('file_sha256') else [],
                       [row['preprocessed_rgb_sha256']] if row.get('preprocessed_rgb_sha256') else [],
                       ['historical_content_' + field], row.get('path'))
    elif fmt == 'legacy_catalog_inventory':
        require(value['status'] == 'METADATA_ONLY_HOLDOUT_CANDIDATE_CATALOG_NOT_TEST_DATA_ACCESSED',
                'Original prior-use catalog inventory required')
        # Only already-used IDs; never import its 50k unread candidate pool.
        for index in value['used_val_indices']:
            require(re.fullmatch(r'\d{8}', index), 'Invalid historical val index')
            yield 'imagenet-val:' + index, [], [], ['historical_catalog_prior_use'], None
    elif fmt == 't6_content_check':
        require(value['status'] == 'T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS', 'Wrong T6 check')
        require(value['source_count'] == len(value['actual_content']), 'T6 cardinality mismatch')
        for row in value['actual_content']:
            yield (row['source_id'], [row['original_JPEG_sha256']],
                   [row['preprocessing_sha256'], row['horizontal_flip_preprocessing_sha256']],
                   ['T6_confirmation100'], None)
    elif fmt == 'kodak_dataset_manifest':
        require(value['schema'] == 'KODAK24_CENTER256_DATASET_V1', 'Wrong Kodak manifest')
        require(value['source_count'] == len(value['records']), 'Kodak cardinality mismatch')
        for row in value['records']:
            yield (row['source_id'], [row['original_png']['sha256']], [row['pixels_chw_uint8_sha256']],
                   ['Kodak_generalization24'], row['original_png']['path'])
    elif fmt == 'normalized_records':
        require(value['schema'] == 'EP_NORMALIZED_PRIOR_SOURCES_V1', 'Wrong normalized schema')
        require(value['pixel_hash_domain'] == PIXEL_DOMAIN, 'Incompatible pixel hash domain')
        require(value['source_count'] == len(value['records']), 'Normalized cardinality mismatch')
        for row in value['records']:
            yield (row['source_id'], row.get('original_file_sha256', []), row.get('pixel_sha256', []),
                   row['roles'], row.get('path'))
    else:
        raise ValueError('Unsupported explicit metadata adapter: ' + fmt)


def build_registry(spec, spec_pin, loader=read_json):
    require(spec['schema'] == 'EP_SOURCE_REGISTRY_INPUT_SPEC_V1', 'Wrong input spec')
    require(spec.get('inputs'), 'At least one pinned manifest required')
    by_id, inputs, seen_inputs = {}, [], set()
    for desc in spec['inputs']:
        identity = (desc['path'], desc['sha256'])
        require(identity not in seen_inputs, 'Duplicate declared manifest')
        seen_inputs.add(identity)
        value, pin = loader(desc['path'], desc['sha256'])
        inputs.append(dict(pin, format=desc['format'], role=desc['role']))
        for sid, raw, pixels, roles, path in imported_rows(value, desc['format']):
            cid = canonical_id(sid)
            row = by_id.setdefault(cid, dict(canonical_source_id=cid, source_ids=set(), roles=set(),
                original_file_sha256=set(), pixel_sha256=set(), source_paths=set(), manifest_sha256=set()))
            require(isinstance(roles, list) and roles and all(isinstance(r, str) and r for r in roles),
                    'Explicit historical source roles required')
            row['source_ids'].add(sid)
            row['roles'].update(roles + [desc['role']])
            row['original_file_sha256'].update(valid_hashes(raw))
            row['pixel_sha256'].update(valid_hashes(pixels))
            row['manifest_sha256'].add(pin['sha256'])
            if path:
                row['source_paths'].add(path)
    rows = [{k: sorted(v) if isinstance(v, set) else v for k, v in row.items()}
            for _, row in sorted(by_id.items())]
    gaps = [r['canonical_source_id'] for r in rows if not r['pixel_sha256']]
    raw_gaps = [r['canonical_source_id'] for r in rows if not r['original_file_sha256']]
    scope_closed, closure_pin = False, None
    if spec.get('study_scope_closure'):
        desc = spec['study_scope_closure']
        closure, closure_pin = loader(desc['path'], desc['sha256'])
        require(closure['schema'] == 'EP_STUDY_SOURCE_SCOPE_CLOSURE_V1' and
                closure['status'] == 'EP_STUDY_SOURCE_REGISTRIES_CLOSED', 'Incomplete study registry scope')
        require(closure['unresolved_registry_paths'] == [] and closure['unresolved_source_roles'] == [] and
                closure['source_content_access_cutoff'] and closure['scanned_project_scopes'],
                'Unresolved study source-use inventory')
        require(set(closure['manifest_sha256']) == {p['sha256'] for p in inputs},
                'Scope closure must cover exactly the supplied source registries')
        require(closure['unique_source_count'] == len(rows), 'Scope closure population differs')
        scope_closed = True
    return dict(schema='EP_STUDY_SOURCE_EXCLUSION_REGISTRY_V1',
        status='EP_SOURCE_EXCLUSION_METADATA_COMPILED', spec=spec_pin, input_manifests=inputs,
        study_scope_closure=closure_pin, study_scope_closed=scope_closed,
        pixel_hash_domain=PIXEL_DOMAIN, source_count=len(rows), records=rows,
        canonical_source_ids=[r['canonical_source_id'] for r in rows],
        pixel_hash_missing_count=len(gaps), pixel_hash_missing_source_ids=gaps,
        original_file_hash_missing_count=len(raw_gaps), original_file_hash_missing_source_ids=raw_gaps,
        full_canonical_content_dedup_ready=scope_closed and not gaps,
        raw_file_dedup_scope='Only comparable original-file hashes; parquet sources need canonical pixel hashes',
        full_resolution_or_perceptual_duplicate_exclusion_claimed=False,
        input_image_reads=0, model_calls=0, packet_calls=0,
        limitations=['Pinned supplied registries do not themselves establish complete study coverage.',
            'Study-scope closure must include previously inspected/rejected source candidates and donors.',
            'Canonical pixel identity is exact equality, including horizontal flips at candidate checking; '
            'it is not a perceptual near-duplicate or pretrained-model dataset-overlap guarantee.'])


def select_population(registry, registry_pin, pool, pool_pin, seed=SELECTION_SEED):
    require(registry['schema'] == 'EP_STUDY_SOURCE_EXCLUSION_REGISTRY_V1' and
            registry['study_scope_closed'], 'Close study source-use registry scope before selecting IDs')
    require(isinstance(seed, str) and seed and '\0' not in seed, 'Explicit deterministic selection seed required')
    used = set(registry['canonical_source_ids'])
    require(len(used) == registry['source_count'], 'Duplicate exclusion IDs')
    source_ids, eligible = set(), []
    for row in pool['records']:
        sid = row['source_id']
        require(SID.fullmatch(sid), 'Only explicit ImageNet val class/stem metadata supported')
        require(sid.split('/')[0] == sid.rsplit('_', 1)[-1], 'Class/stem identity differs')
        cid = canonical_id(sid)
        require(cid not in source_ids, 'Pool source aliases/duplicates are forbidden')
        source_ids.add(cid)
        require(row.get('canonical_source_id', cid) == cid, 'Pool canonical identity differs')
        path = PurePosixPath(row['path'])
        require(path.is_absolute() and '..' not in path.parts and '\\' not in row['path'] and
                path.as_posix().endswith('/' + sid + '.JPEG'), 'Invalid pinned original image path')
        if cid not in used:
            eligible.append(dict(source_id=sid, canonical_source_id=cid, path=row['path'],
                original_bytes=row.get('original_bytes'), selection_key=digest((seed + '\0' + cid).encode())))
    require(len(eligible) >= 100, 'Fewer than 100 eligible distinct sources')
    ordered = sorted(eligible, key=lambda r: (r['selection_key'], r['canonical_source_id']))
    selected = [dict(row, source_index=i) for i, row in enumerate(ordered[:100])]
    return dict(schema='EP_NEW100_METADATA_SELECTION_V1', status='EP_SOURCE_IDS_FIXED_NO_PIXEL_ACCESS',
        registry=registry_pin, pool_inventory=pool_pin, source_count=100, records=selected,
        source_ids=[r['source_id'] for r in selected], pool_source_count=len(source_ids),
        eligible_source_count=len(eligible), seed=seed, selection_rule='SHA256(seed+NUL+canonical_source_id), first100',
        source_reselection_allowed=False, source_count_reduction_allowed=False,
        N=1024, SNRs=[4, 10, 19], noise_seeds=NOISE_SEEDS, methods=METHODS,
        frame_count=3600, max_packet_calls=7200, qualification_calls_included=False,
        policy_selection_uses_new100=False, Encoder_calls_allowed=False,
        content_gate='Require full registry canonical hashes and check-content PASS before any Encoder/VAR call',
        pixel_reads=0, model_calls=0, packet_calls=0)


def check_content(registry, registry_pin, selection, selection_pin, content, content_pin):
    require(registry['full_canonical_content_dedup_ready'] and registry['study_scope_closed'] and
            registry['pixel_hash_missing_count'] == 0, 'Full study canonical hash coverage is required')
    require(selection['registry'] == registry_pin, 'Selection registry pin differs')
    require(content['schema'] == 'EP_CANDIDATE100_CONTENT_RECEIPT_V1' and
            content['selection'] == selection_pin and content['pixel_hash_domain'] == PIXEL_DOMAIN,
            'Candidate content receipt identity differs')
    require(content['Encoder_calls_before_check'] == 0 and content['VAR_calls_before_check'] == 0 and
            content['metric_calls_before_check'] == 0, 'No source/model/quality calls before dedup gate')
    require(len(content['records']) == 100 and selection['source_count'] == 100,
            'Exactly 100 registered candidate records required')
    raw_old = {h for r in registry['records'] for h in r['original_file_sha256']}
    pix_old = {h for r in registry['records'] for h in r['pixel_sha256']}
    ids_old = set(registry['canonical_source_ids'])
    raw_seen, pix_seen, conflicts = set(), set(), []
    for expected, actual in zip(selection['records'], content['records']):
        require(actual['source_id'] == expected['source_id'] and actual['source_index'] == expected['source_index'],
                'Candidate order/identity differs from metadata selection')
        raw = actual['original_file_sha256']
        pre, flip = actual['pixel_sha256'], actual['horizontal_flip_pixel_sha256']
        valid_hashes([raw, pre, flip])
        reasons = []
        if canonical_id(actual['source_id']) in ids_old:
            reasons.append('prior_source_id')
        if raw in raw_old or raw in raw_seen:
            reasons.append('exact_original_file')
        if {pre, flip} & (pix_old | pix_seen):
            reasons.append('exact_canonical_pixels_or_horizontal_flip')
        if reasons:
            conflicts.append(dict(source_index=actual['source_index'], source_id=actual['source_id'], reasons=reasons))
        raw_seen.add(raw)
        pix_seen.update((pre, flip))
    return dict(schema='EP_NEW100_CONTENT_GATE_V1',
        status='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS' if not conflicts else 'EP_NEW100_CONTENT_DUPLICATE_STOP',
        registry=registry_pin, selection=selection_pin, content_receipt=content_pin,
        source_count=100, prior_source_count=registry['source_count'], conflicts=conflicts,
        Encoder_calls_allowed=not conflicts, source_reselection_allowed=False,
        exact_canonical_content_scope=True, full_resolution_or_perceptual_duplicate_exclusion_claimed=False,
        scope='All study registries covered by the pinned explicit scope closure; exact source IDs and canonical pixels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build-registry')
    build.add_argument('--spec', required=True)
    build.add_argument('--out', required=True)
    select = commands.add_parser('select')
    select.add_argument('--registry', required=True)
    select.add_argument('--pool', required=True)
    select.add_argument('--seed', default=SELECTION_SEED)
    select.add_argument('--out', required=True)
    check = commands.add_parser('check-content')
    check.add_argument('--registry', required=True)
    check.add_argument('--selection', required=True)
    check.add_argument('--content', required=True)
    check.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'build-registry':
        value, pin = read_json(args.spec)
        result = build_registry(value, pin)
    elif args.command == 'select':
        registry, rp = read_json(args.registry)
        pool, pp = read_json(args.pool)
        result = select_population(registry, rp, pool, pp, args.seed)
    else:
        registry, rp = read_json(args.registry)
        selection, sp = read_json(args.selection)
        content, cp = read_json(args.content)
        result = check_content(registry, rp, selection, sp, content, cp)
    output = write_new(args.out, result)
    print(json.dumps(dict(status=result['status'], output=output, science_calls=0)))
    if result['status'] == 'EP_NEW100_CONTENT_DUPLICATE_STOP':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
