"""Frozen common500 metadata and raw holdout schedule; no source content access.

Existing calibration-selected policies are validated by the original scheduler.
Only stage/population/noise/counter bookkeeping is new. No policy is selected.
"""
import hashlib
import json

SNRS = (1, 4, 7, 10, 13, 19)
FAMILIES = ('WHOLE', 'PARTIAL')
SEEDS = (6201, 6202, 6203)
PROTOCOL = 'CONTENT-REAL-64QAM-H-20261006-V1'
CANDIDATE_SHA = 'ea70992206125362d36c58d9797f93e361e1b525d03e1d8668c1c75fd917f04d'
POLICIES_SHA = '7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c'
SCIENCE_SHA = 'fddc600ff68ad4caa9afd62c7e0fe2c23356897cf1d91d05e5ba61af5b2174f2'
NOISE_SPEC = dict(protocol=PROTOCOL, stage='holdout', noise_seeds=list(SEEDS),
                  seed_rule='SHA256(canonical[protocol,stage,source_id,snr_db,noise_seed])[:16] little-endian',
                  generator='numpy.Generator(numpy.PCG64(seed)).standard_normal((1024,2))',
                  dtype='float64', candidate_or_family_in_seed=False,
                  counter_rule='(SNRindex*500+source_index)*3+noise_index',
                  SNRs=list(SNRS), source_count=500)


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def population(candidate, exclusions):
    require(candidate['status'] == 'HOLDOUT500_METADATA_CANDIDATE_DRAFT_NOT_FINAL_FREEZE'
            and candidate['holdout_opened'] is False and candidate['image_token_feature_metric_access'] is False,
            'Original unopened metadata candidate required')
    ids = candidate['source_ids']
    require(len(ids) == len(set(ids)) == 500 and len({v.split('/')[0] for v in ids}) == 500,
            'Original ordered500, one per distinct class required')
    require(all(isinstance(v, str) and v for v in ids) and not set(ids).intersection(exclusions),
            'Holdout/source exposure overlap')
    return list(ids)


def frozen_section(final_freeze, candidate, exclusions, split_admission):
    """Called only on an authenticated actual FINAL_FREEZE JSON by the owner."""
    ids = population(candidate, exclusions)
    pop = final_freeze['population']
    require(final_freeze['schema'] == 'VAR_COMM_FINAL_FREEZE_V1'
            and final_freeze['status'] == 'CONFIGURATIONS_AND_METRICS_FROZEN_BEFORE_UNIFIED_HOLDOUT'
            and final_freeze['science_registration']['sha256'] == SCIENCE_SHA
            and pop['source_ids'] == ids and pop['role'] == 'holdout'
            and pop['source_count'] == 500 and pop['candidate_draft']['sha256'] == CANDIDATE_SHA,
            'Actual same500 FINAL_FREEZE required')
    section = final_freeze['raw_holdout']
    require(section['frozen_policy_sha256'] == POLICIES_SHA and section['noise'] == NOISE_SPEC
            and section['phase'] == 'raw_holdout' and section['phase_cap'] == 36000
            and section['source_count'] == 500 and section['SNRs'] == list(SNRS)
            and section['families'] == list(FAMILIES) and section['no_reselection_after_open'] is True,
            'Frozen raw policy/noise/grid/cap changed')
    require(final_freeze['primary_methods'] == ['WHOLE', 'PARTIAL', 'P1024', 'matched SwinJSCC80k']
            and final_freeze['same_RX_outputs'] == ['VAR_completion', 'direct_Dc_missing_residual_zero']
            and final_freeze['holdout_used_at_freeze'] is False and final_freeze['policy_selection_after_freeze'] is False,
            'Unified primary methods or same-RX arms changed')
    require(split_admission['schema'] == 'VAR_COMM_COMMON500_SPLIT_ADMISSION_V1'
            and split_admission['status'] == 'COMMON500_METADATA_ADMITTED_BEFORE_CONTENT'
            and split_admission['candidate_sha256'] == CANDIDATE_SHA
            and split_admission['source_count'] == 500 and split_admission['candidate_overlap'] == 0
            and split_admission['root_post_cutoff_attestation']['status'] == 'NO_UNPLANNED_EXPOSURE_AFTER_CUTOFF'
            and split_admission['root_post_cutoff_attestation']['exposure_cutoff_unix'] == split_admission['exposure_cutoff_unix']
            and split_admission['root_post_cutoff_attestation']['unplanned_source_ids'] == []
            and split_admission['root_post_cutoff_attestation']['candidate_content_opened'] is False,
            'Actual exposure/scope/cutoff admission required')
    return section, ids


def plan(candidate, exclusions, selected, catalogue, shortlist, legacy_refs, original_schedule):
    ids = population(candidate, exclusions)
    original = original_schedule(selected, catalogue, shortlist, legacy_refs)
    points = []
    for point in original:
        # Keep every scientifically consumed wire field; old counter/reuse flags
        # apply only to development and must never authorize a holdout reuse.
        points.append({**{k:point[k] for k in ('snr_db','candidate_id','profile_id','wire_key','m','K','modulation','families')},
                       'holdout_slot':2*SNRS.index(point['snr_db'])+FAMILIES.index(point['families'][0]),
                       'historical_reception_reuse_admitted':False})
    require(len(points) == 11 and sum(len(p['families']) for p in points) == 12,
            'Exact frozen eleven unique/twelve family points required')
    require(len({(p['snr_db'],p['candidate_id']) for p in points}) == 11
            and len({p['holdout_slot'] for p in points}) == 11, 'Duplicate physical point')
    return dict(schema='MAIN_RAW64_UNIFIED500_HOLDOUT_PLAN_V1', status='FROZEN_REQUIRES_STAGE_REGISTRATION',
                source_ids=ids, source_count=500, population_role='holdout', phase='raw_holdout', phase_cap=36000,
                noise=NOISE_SPEC, noise_seeds=list(SEEDS), schedule=points,
                actual_unique_frames=16500, logical_family_frames=18000, maximum_new_packet_calls=33000,
                frozen_policy_digest=digest(selected), catalogue_digest=digest(catalogue), shortlist_digest=digest(shortlist),
                candidate_metadata_digest=digest(candidate), source_level_paired=True,
                same_RX_outputs=['VAR_completion','direct_Dc_missing_residual_zero'],
                new_packet_calls_for_same_RX_ablation=0, historical_reused_frames=0,
                policy_selection=False, new_model_or_candidate_selection=False)


def frame_counter(source_index, snr_db, seed):
    require(type(source_index) is int and 0 <= source_index < 500
            and type(snr_db) is int and snr_db in SNRS and type(seed) is int and seed in SEEDS,
            'Registered holdout source/SNR/noise required')
    return (SNRS.index(snr_db)*500 + source_index)*3 + SEEDS.index(seed)


def standard_noise(source_id, snr_db, seed):
    require(isinstance(source_id,str) and source_id and type(snr_db) is int and snr_db in SNRS
            and type(seed) is int and seed in SEEDS, 'Frozen holdout noise identity required')
    import numpy as np
    state = int.from_bytes(hashlib.sha256(canonical([PROTOCOL,'holdout',source_id,snr_db,seed]).encode()).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(state)).standard_normal((1024,2))
