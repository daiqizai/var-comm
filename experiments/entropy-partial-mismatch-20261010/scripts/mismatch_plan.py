#!/usr/bin/env python3
"""Read-only metadata plan for the raw one-bin configuration mismatch diagnostic.

No NumPy, channel, PHY, model, metric, statistical or process calls. Creating this
plan neither admits historical reuse nor authorizes a scientific execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCHEMA = 'RAW_CONFIGURATION_MISMATCH_METADATA_PLAN_V1'
SNRS = (1, 4, 7, 10, 13, 19)
ACTUAL_SNRS = (4, 7, 10)
LOOKUPS = {4: (1, 4, 7), 7: (4, 7, 10), 10: (7, 10, 13)}
FAMILIES = ('WHOLE', 'PARTIAL')
SEEDS = (6201, 6202, 6203)
SOURCE_COUNT = 100
ORIGINAL_SOURCE_COUNT = 500
PROTOCOL = 'CONTENT-REAL-64QAM-H-20261006-V1'
ORIGINAL_PLAN_SHA = '1ab63b0e0bdde865d677467232c956ebbae746239ac300dd7723e22dbe8c7bce'
SOURCE_MANIFEST_SHA = 'b27128fb8eedee7f2cc74bed25c63e39e8add8c938c76d449f85c4b0d5a51e09'
ORIGINAL_NOISE_CODE_SHA = '1acc294032f46166ff2b714b6d47de1ac7aa8bb37619c8b75729f801ac249fec'
SOURCE_KEYS = ('source_index', 'source_id', 'evaluation_class_index', 'preprocessing_id',
               'tokens_sha256', 'archive', 'archive_sha256', 'checkpoint', 'checkpoint_sha256')
POLICY_KEYS = ('candidate_id', 'profile_id', 'wire_key', 'm', 'K', 'modulation')
EXPECTED_PROFILES = {1: (183, 141), 4: (152, 181), 7: (264, 264),
                     10: (274, 231), 13: (387, 243), 19: (270, 378)}
NOISE_SPEC = dict(protocol=PROTOCOL, stage='holdout', noise_seeds=list(SEEDS),
    seed_rule='SHA256(canonical[protocol,stage,source_id,snr_db,noise_seed])[:16] little-endian',
    generator='numpy.Generator(numpy.PCG64(seed)).standard_normal((1024,2))',
    dtype='float64', candidate_or_family_in_seed=False,
    counter_rule='(SNRindex*500+source_index)*3+noise_index', SNRs=list(SNRS), source_count=500)

# The receiver evidence adapter must construct both maps independently from
# authenticated actual artifacts. Matching maps alone is NOT admission.
PHY_IDENTITY_KEYS = (
    'source_index', 'source_id', 'source_archive_sha256', 'preprocessing_id', 'tokens_sha256',
    'actual_snr_db', 'noise_seed', 'public_counter', 'noise_seed_material_sha256',
    'standard_noise_sha256', 'tx_profile_id', 'candidate_id', 'wire_key',
    'payload_sha256', 'waveform_sha256', 'received_sha256', 'effective_catalogue_sha256',
    'aliases_sha256', 'receiver_source_bindings_sha256', 'phy_runtime_identity_sha256',
    'receiver_rule', 'receiver_phase', 'body_session', 'body_group',
    'noise_algorithm_identity_sha256')
IMAGE_IDENTITY_KEYS = ('source_index', 'source_id', 'actual_state_sha256',
    'receiver_view_sha256', 'model_identity_sha256', 'numerical_runtime_sha256',
    'render_source_bindings_sha256', 'image_sha256', 'image_dtype', 'image_shape')
METRIC_IDENTITY_KEYS = ('source_index', 'source_id', 'preprocessing_id', 'image_sha256',
    'metric_models_sha256', 'metric_preprocessing_sha256', 'metric_runtime_sha256',
    'metric_source_bindings_sha256')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def pinned_json(path, expected):
    """Hash and parse the same metadata bytes, without opening source archives."""
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Expected regular metadata file: ' + str(p))
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'Metadata SHA mismatch: ' + str(p))
    return json.loads(data)


def validate_inputs(plan, manifest):
    require(plan['schema'] == 'MAIN_RAW64_UNIFIED500_HOLDOUT_PLAN_V1', 'Wrong original plan')
    require(plan['source_count'] == 500 and manifest['source_count'] == 500, 'Original500 required')
    require(plan['source_ids'] == manifest['source_ids'] and len(set(plan['source_ids'])) == 500,
            'Original ordered source identity mismatch')
    require(plan['noise'] == NOISE_SPEC and plan['noise_seeds'] == list(SEEDS), 'Original noise changed')
    require(manifest['status'] == 'COMMON500_SOURCE_ASSETS_MANIFEST_V1', 'Wrong source asset manifest')
    records = manifest['records']
    require(len(records) == 500 and [r['source_index'] for r in records] == list(range(500)),
            'Never reorder or renumber the original500')
    require([r['source_id'] for r in records] == plan['source_ids'], 'Record/source ID mismatch')
    schedule = {}
    for row in plan['schedule']:
        require(row['snr_db'] in SNRS, 'Unknown policy SNR')
        for family in row['families']:
            key = (row['snr_db'], family)
            require(family in FAMILIES and key not in schedule, 'Duplicate/unknown frozen family point')
            schedule[key] = {k: row[k] for k in POLICY_KEYS}
    require(set(schedule) == {(s, f) for s in SNRS for f in FAMILIES}, 'Incomplete frozen schedule')
    require(all(schedule[s, f]['profile_id'] == EXPECTED_PROFILES[s][FAMILIES.index(f)]
                for s in SNRS for f in FAMILIES), 'Frozen profile schedule changed')
    require(schedule[7, 'WHOLE'] == schedule[7, 'PARTIAL'], 'Original equal policy at7 changed')
    return [{k: r[k] for k in SOURCE_KEYS} for r in records[:100]], schedule


def channel_identity(source, actual_snr_db, noise_seed):
    require(type(source['source_index']) is int and 0 <= source['source_index'] < SOURCE_COUNT,
            'Only original source indices0..99')
    require(type(actual_snr_db) is int and actual_snr_db in ACTUAL_SNRS, 'True SNR outside fixed grid')
    require(type(noise_seed) is int and noise_seed in SEEDS, 'Original seed required')
    material = [PROTOCOL, 'holdout', source['source_id'], actual_snr_db, noise_seed]
    raw_sha = hashlib.sha256(canonical(material).encode('utf-8')).digest()
    return dict(actual_snr_db=actual_snr_db, receiver_snr_db=actual_snr_db,
        noise_seed=noise_seed,
        public_counter=(SNRS.index(actual_snr_db)*ORIGINAL_SOURCE_COUNT+source['source_index'])*3+SEEDS.index(noise_seed),
        noise_seed_material_sha256=raw_sha.hex(),
        pcg64_seed=int.from_bytes(raw_sha[:16], 'little'),
        standard_noise_shape=[1024, 2], standard_noise_dtype='float64',
        sigma_expression='10**(-actual_snr_db/20)',
        receiver_phase='raw_holdout', body_session='actual-body', body_group=0)


def frame(source, actual_snr_db, config_snr_db, family, schedule):
    require(actual_snr_db in ACTUAL_SNRS and config_snr_db in LOOKUPS[actual_snr_db],
            'Lookup outside frozen one-bin grid')
    require(family in FAMILIES, 'Only raw WHOLE/PARTIAL')
    return dict(source=source, actual_snr_db=actual_snr_db, config_snr_db=config_snr_db,
                family=family, **schedule[config_snr_db, family])


def logical_frames(sources, schedule):
    require(len(sources) == 100 and [r['source_index'] for r in sources] == list(range(100)),
            'Exactly original first100 in original order')
    for actual in ACTUAL_SNRS:
        for lookup in LOOKUPS[actual]:
            for family in FAMILIES:
                for source in sources:
                    base = frame(source, actual, lookup, family, schedule)
                    for seed in SEEDS:
                        yield dict(base, channel=channel_identity(source, actual, seed))


def physical_key(row):
    """Scheduled dedup key; runtime must additionally prove actual waveform/RX bytes."""
    src, c = row['source'], row['channel']
    require(c['actual_snr_db'] == row['actual_snr_db'] == c['receiver_snr_db'], 'RX must use true SNR')
    return digest(dict(source_index=src['source_index'], source_id=src['source_id'],
        archive_sha256=src['archive_sha256'], tokens_sha256=src['tokens_sha256'],
        channel=c, policy={k: row[k] for k in POLICY_KEYS}))


def check_identity_bindings(expected, observed, stage):
    """A diagnostic only: cannot authenticate files or claim completed/reused work."""
    groups = {'phy': PHY_IDENTITY_KEYS, 'image': IMAGE_IDENTITY_KEYS, 'metrics': METRIC_IDENTITY_KEYS}
    require(stage in groups, 'Unknown evidence stage')
    keys = groups[stage]
    missing = sorted(k for k in keys if k not in expected or k not in observed)
    mismatch = sorted(k for k in keys if k in expected and k in observed and expected[k] != observed[k])
    return dict(status='METADATA_IDENTITY_MATCH_ONLY' if not missing and not mismatch else 'IDENTITY_REJECTED',
                stage=stage, missing=missing, mismatched=mismatch, scientific_reuse_admitted=False)


def validate_receiver_summary(row):
    """Validate normalized metadata while preserving original KEEP semantics.

    Header-rejected legacy rows report body_crc_accept=False despite no body call.
    The explicit body_attempted field prevents miscounting these as body failures.
    TX profile is intentionally never a constraint on an accepted RX profile.
    """
    for key in ('header_ok', 'body_attempted', 'gray', 'source_decode_complete'):
        require(type(row[key]) is bool, 'Explicit receiver boolean required: ' + key)
    require(row['source_decode_complete'] and row['rule'] == 'KEEP', 'KEEP source presentation required')
    require(row['execution_status'] == 'complete', 'Unresolved execution cannot be a channel failure')
    if not row['header_ok']:
        require(not row['body_attempted'] and row['rx_profile_id'] is None and row['gray']
                and row['source_status'] == 'HEADER_REJECT_GRAY'
                and row['body_crc_accept'] is False and row['packet_call_count'] == 1,
                'Header reject requires gray, no body call; preserve legacy CRC field')
    else:
        require(row['body_attempted'] and type(row['rx_profile_id']) is int and row['rx_profile_id'] >= 0
                and type(row['body_crc_accept']) is bool and not row['gray']
                and row['source_status'] == 'KEEP_ACTUAL_HARD_TOKENS'
                and row['packet_call_count'] == 2,
                'Accepted header must decode actual RX profile and keep hard tokens even on body CRC fail')
        require(row['rx_profile_legal'] is True, 'RX profile must be admitted by full original catalogue')
    return True


def call_accounting(records):
    """Review a normalized completed physical-frame register, without running it.

    A reused frame needs an external authenticated admission descriptor; this
    function only counts that declared evidence, not validates historical reuse.
    """
    keys, fresh, reused, calls = set(), 0, 0, 0
    for row in records:
        require(row['physical_key'] not in keys, 'Duplicate physical execution/reservation')
        keys.add(row['physical_key'])
        require(row['state'] == 'complete', 'Unresolved reservation blocks automatic continuation')
        validate_receiver_summary(row['receiver'])
        if row['mode'] == 'new':
            require(row['actual_new_packet_calls'] == row['receiver']['packet_call_count'], 'Actual call count differs')
            fresh += 1
        elif row['mode'] == 'historical_reuse':
            descriptor = row.get('reuse_admission', {})
            require(set(descriptor) == {'path', 'sha256'} and bool(descriptor['path'])
                    and len(descriptor['sha256']) == 64 and row['actual_new_packet_calls'] == 0,
                    'Separate authenticated reuse admission required; never infer reuse from cached path')
            reused += 1
        else:
            raise ValueError('Unknown execution mode')
        calls += row['actual_new_packet_calls']
    require(len(keys) <= 4500 and calls <= 9000, 'Physical/call ceiling exceeded')
    return dict(fresh_physical_frames=fresh, declared_historical_reuse_frames=reused,
                actual_new_packet_calls=calls, independent_admission_check_required=True)


def make_plan(original, manifest):
    sources, schedule = validate_inputs(original, manifest)
    all_rows = list(logical_frames(sources, schedule))
    keys = {physical_key(row) for row in all_rows}
    matched_keys = {physical_key(r) for r in all_rows if r['actual_snr_db'] == r['config_snr_db']}
    require(len(all_rows) == 5400 and len(keys) == 4500 and len(matched_keys) == 1500, 'Changed exact accounting')
    cells = []
    for actual in ACTUAL_SNRS:
        for lookup in LOOKUPS[actual]:
            for family in FAMILIES:
                cells.append(dict(actual_snr_db=actual, config_snr_db=lookup, family=family,
                                  **schedule[lookup, family], logical_frames=300))
    return dict(schema=SCHEMA, status='METADATA_ONLY_NOT_SCIENTIFIC_REGISTRATION', N=1024,
        population_role='posthoc_first100_subset_of_original_common500_holdout',
        source_count=100, noise_count=3, source_ids=[s['source_id'] for s in sources], sources=sources,
        original_source_count=500, original_source_indices=list(range(100)),
        original_noise=NOISE_SPEC, actual_snrs_db=list(ACTUAL_SNRS),
        configuration_lookup_snrs={str(k): list(v) for k, v in LOOKUPS.items()}, cells=cells,
        source_subset_sha256=digest(sources), original_schedule_sha256=digest(original['schedule']),
        data_pins=dict(original_plan=ORIGINAL_PLAN_SHA, source_manifest=SOURCE_MANIFEST_SHA,
                       original_noise_source=ORIGINAL_NOISE_CODE_SHA),
        accounting=dict(logical_frames=5400, unique_physical_frames=4500,
            matched_logical_frames=1800, matched_unique_physical_frames=1500,
            off_diagonal_logical_frames=3600, off_diagonal_unique_physical_frames=3000,
            maximum_new_packet_calls_without_reuse=9000,
            maximum_new_packet_calls_if_all_matched_physical_frames_admitted=6000,
            historical_reuse_actually_admitted=0, packet_calls_per_fresh_frame='1 + int(header_ok)',
            fresh_render_attempt_upper_bound_without_reuse=4500,
            qualification_calls_included=0, qualification_not_authorized_by_plan=True),
        reuse_requirements=dict(phy_identity_keys=list(PHY_IDENTITY_KEYS), image_identity_keys=list(IMAGE_IDENTITY_KEYS),
            metric_identity_keys=list(METRIC_IDENTITY_KEYS),
            independent_required_evidence=['file bytes checked against original completed output seals',
                'normal CPU closure and actual parent exit/wait receipt',
                'original header/body paid ledger request and result, reparse actual hard bits',
                'full433 receiver catalogue, aliases, source/runtime bindings, not TX profile truth',
                'image closure/wait, receiver state and image hashes for model reuse',
                'per-frame metric identity and values; no500 aggregate reuse for100subset']),
        output_contract=dict(required_fields=[
            'logical_frame_id', 'physical_frame_id', 'source_index', 'source_id',
            'actual_snr_db', 'config_snr_db', 'family', 'noise_seed', 'public_counter',
            'tx_profile_id', 'tx_candidate_id', 'tx_m', 'tx_K', 'rx_profile_id', 'rx_m', 'rx_K',
            'header_crc_accept', 'header_fields_legal', 'header_ok', 'body_attempted', 'body_crc_accept',
            'source_status', 'execution_status', 'gray', 'actual_state_sha256',
            'noise_sha256', 'waveform_sha256', 'received_sha256', 'E_frame', 'rho',
            'paid_event_ids', 'actual_new_packet_calls', 'reuse_admission',
            'image_path', 'image_key', 'image_slot', 'image_sha256'],
            metrics=['psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction'],
            body_crc_denominator='Only body_attempted frames; header reject kept separately',
            accepted_bad_body='KEEP actual hard prefix/raster partial tokens; no CRC-drop',
            all_failures_and_zero_differences_retained=True),
        new_scientific_calls=0, policy_selection=False, new_source_selection=False,
        old_files_modified=False, automatic_successor_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-plan', required=True)
    p.add_argument('--source-manifest', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    plan = make_plan(pinned_json(args.original_plan, ORIGINAL_PLAN_SHA),
                     pinned_json(args.source_manifest, SOURCE_MANIFEST_SHA))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(canonical(dict(status=plan['status'], output=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                         accounting=plan['accounting'], new_scientific_calls=0)))


if __name__ == '__main__':
    main()
