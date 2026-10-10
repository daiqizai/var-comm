"""Finite metadata for entropy partial-scale extension; zero scientific calls.

This is registration preparation, not a completed experiment or a launch gate.
The original entropy whole candidates and final winners remain eligible.
"""
from __future__ import annotations
from fractions import Fraction

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
SNRS = (4, 10, 19)
RATES = ('1/2', '2/3', '3/4', '5/6')
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine',
           'convnext_top1_source_prediction')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ks(m):
    require(type(m) is int and 4 <= m <= 9, 'Registered m4..m9 only')
    return tuple(SIZES[m] ** 2 * j // 4 for j in (1, 2, 3))


def token_count(m, K):
    require(type(K) is int and K in (0,) + ks(m), 'Registered K only')
    return sum(x*x for x in SIZES[:m]) + K


def fallback_endpoints(target_m, target_K):
    """Whole-only actions stay identical; new actions allow all lower endpoints.

    A source encoder may select only the first actual stream that fits, never
    consult reconstruction scores, and must stop on missing stream evidence.
    """
    maximum = token_count(target_m, target_K)
    if target_K == 0:
        return tuple((m, 0) for m in range(target_m, 3, -1))
    points = [(m, k) for m in range(4, target_m + 1)
              for k in (0,) + ks(m) if token_count(m, k) <= maximum]
    return tuple(sorted(points, key=lambda p: token_count(*p), reverse=True))


def select_stream(lengths, candidate):
    """Length-only source decision; lengths must come from real flushed bits."""
    attempts = []
    for m, k in fallback_endpoints(candidate['target_m'], candidate['target_K']):
        key = (m, k)
        require(key in lengths, f'Missing actual encoded stream: m{m} K{k}')
        length = lengths[key]
        require(type(length) is int and length >= 2, 'Real nonempty bit count required')
        fits = length <= candidate['source_capacity_bits']
        attempts.append(dict(m=m, K=k, bits=length, fits=fits))
        if fits:
            return dict(m=m, K=k, bits=length, attempts=attempts)
    raise ValueError('No actual encoded endpoint fits; cannot fabricate channel failure')


def candidates():
    rows = []
    for m in (7, 8, 9):
        for k in (0,) + ks(m):
            for q in (2, 4, 6):
                for rate in RATES:
                    n = 956*q
                    fraction = Fraction(rate)
                    information = n*fraction.numerator//fraction.denominator
                    # Original whole candidate IDs remain unchanged.
                    cid = f'm{m}' + (f'_K{k}' if k else '') + f'_q{q}_r{rate.replace("/", "-")}'
                    rows.append(dict(candidate_id=cid, N=1024, target_m=m, target_K=k,
                        q=q, nominal_rate=rate, k=information, n=n,
                        source_capacity_bits=information-29,
                        header_symbols=68, body_symbols=956,
                        length_bits=13, crc_bits=16,
                        family='EC_VAR_WHOLE' if k == 0 else 'EC_VAR_PARTIAL'))
    require(len(rows) == len({r['candidate_id'] for r in rows}) == 144, 'Fixed grid')
    return rows


def full_shortlist(pilot_ranked_ids, whole_winner_id):
    """Preserve the original whole final winner even if absent from pilot top3."""
    legal = {r['candidate_id']: r for r in candidates()}
    require(len(pilot_ranked_ids) == 144 and set(pilot_ranked_ids) == set(legal),
            'Complete finite pilot ranking required')
    require(whole_winner_id in legal and legal[whole_winner_id]['target_K'] == 0,
            'Original calibrated whole anchor required')
    return tuple(dict.fromkeys(pilot_ranked_ids[:3] + [whole_winner_id]))


def budgets():
    return dict(
        status='METADATA_PREPARED_NOT_EXECUTED',
        source_qualification=dict(sources=32, TX_provider_passes=32,
            independent_partial_RX_decodes=32*18, missing_whole_RX_cap=32*2),
        real_PHY_qualification=dict(body_decodes=96, header_decodes=361,
            actual_packet_cap=457),
        real_source_link_gate=dict(original_calibration_sources=4, snrs=list(SNRS),
            seed=4101, actions_per_snr=4, logical_frames=48, packet_cap=96,
            independent_RX_VAR_cap=48, reconstruction_cap=48,
            quality_scoring=False),
        pilot=dict(sources=100, snrs=list(SNRS), seeds=[4101], candidates=144,
            logical_frames=43200, packet_cap=86400),
        full_calibration=dict(sources=1000, snrs=list(SNRS), seeds=[4101,4102,4103],
            finalists_per_snr_max=4, logical_frames_cap=36000, packet_cap=72000),
        confirmation=dict(new_sources=100, snrs=list(SNRS), noise_count=3,
            noise_seeds=[9301,9302,9303], methods=4, logical_frames=3600,
            packet_cap=7200, Encoder_cap=100, TX_VAR_provider_cap=100,
            independent_RX_VAR_cap=600),
        mismatch=dict(existing_fixed_sources=100, actual_snrs=[4,7,10],
            lookup_cases=3, noise_count=3, methods=2, logical_frames=5400,
            unique_physical_frames=4500, matched_logical_frames=1800,
            matched_unique_physical_frames=1500,
            packet_cap_before_exact_reuse=9000,
            packet_cap_if_all_matched_reuse_is_verified=6000),
        new_partial_TX_timing=dict(development_sources=16, snrs=list(SNRS),
            warmup_repeats=3, measured_repeats=3, fresh_TX_calls=288,
            actual_packet_decodes=0, exclusive_window_required=True,
            disabled_prepared_source_cache=True),
        old_ledger_mutation=False, actual_new_scientific_calls=0)
