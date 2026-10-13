"""Frozen finite design and selection arithmetic; no model or PHY imports."""
from __future__ import annotations
import hashlib
import json
import math

PLAN_ID = 'NESTCOM_LAYERED_RAW_SNR_MISMATCH_V1'
NAMESPACE = 'UEP_MISMATCH_RAW_V1'
CAL_DOMAIN = 'UEP_MISMATCH_CAL200_V1'
HATS = (7, 10)
ERRORS = (3, 0, -3)
CAL_SEEDS = (4101, 4102, 4103)
DEV_SEEDS = (2001, 2002, 2003)
P_SENSITIVITY = (0., .1, .25, .5, 1.)
METHODS = ('S_nom', 'S_back3', 'S_robust', 'E_two', 'U_two')
SELECTION_METRIC = 'dinov2_vitl14_cosine'
METRICS = ('psnr_db', 'lpips_alex', SELECTION_METRIC,
           'convnext_top1_source_prediction')
BOOTSTRAP_SEED = 2026101307
BOOTSTRAP_REPLICATES = 10000
R1_CAPS = dict(sources=200, physical_frames=36000,
               logical_var_images=46800, packet_decodes=108000)
R2_CAPS = dict(sources=100, main_logical_frames=9000,
               payload_physical_frames=1800, physical_frames=10800,
               logical_images=14400, packet_decodes=32400)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def select_cal200(source_records):
    """Rank original1000 source IDs, never image quality or development rows."""
    records = list(source_records)
    ids = [str(r['source_id']) for r in records]
    require(len(records) == 1000 and len(set(ids)) == 1000,
            'Exactly the authenticated original1000 list is required')
    ranked = sorted(records, key=lambda r: (
        digest([CAL_DOMAIN, str(r['source_id'])]), str(r['source_id'])))
    return [dict(r, selection_sha256=digest([CAL_DOMAIN, str(r['source_id'])]))
            for r in ranked[:200]]


def channel_key(source_id, hat, seed, split):
    require(hat in HATS, 'Unknown estimate')
    require(split in ('cal200', 'development100'), 'Unknown population')
    require(seed in (CAL_SEEDS if split == 'cal200' else DEV_SEEDS),
            'Unregistered noise seed')
    # True SNR and method/configuration deliberately cannot affect this key.
    return [NAMESPACE, split, str(source_id), int(hat), int(seed)]


def standard_noise(source_id, hat, seed, split):
    import numpy as np
    key = channel_key(source_id, hat, seed, split)
    integer_seed = int.from_bytes(hashlib.sha256(canonical(key).encode()).digest(), 'big')
    return np.random.Generator(np.random.PCG64(integer_seed)).standard_normal((1024, 2))


def public_counter(source_slot, hat, seed, split):
    """Counter is shared by configurations and true-SNR realizations."""
    count = R1_CAPS['sources'] if split == 'cal200' else R2_CAPS['sources']
    channel_key('counter_validation', hat, seed, split)
    require(isinstance(source_slot, int) and 0 <= source_slot < count,
            'Source slot outside the fixed population')
    seeds = CAL_SEEDS if split == 'cal200' else DEV_SEEDS
    return (HATS.index(hat) * count + source_slot) * 3 + seeds.index(seed)


def true_snrs(hat):
    require(hat in HATS, 'Unknown estimate')
    return tuple(hat - error for error in ERRORS)


def validate_shortlist(rows):
    """Budgets count physical candidates, KEEP/DROP reuse the same observation."""
    counts = {}
    for hat in HATS:
        selected = [r for r in rows if r['snr_hat'] == hat]
        identity_to_physical = {}; physical_to_identity = {}; seen = set()
        for row in selected:
            require(row['arm'] in ('U', 'S', 'EEP', 'S_nom', 'S_back3'), 'Unknown shortlist arm')
            cid, physical = str(row['candidate_id']), str(row['physical_key'])
            require(cid and physical, 'Missing candidate identity')
            require((row['arm'], cid) not in seen, 'Duplicate nomination within an arm')
            seen.add((row['arm'], cid))
            require(identity_to_physical.setdefault(cid, physical) == physical,
                    'Candidate identity maps to conflicting physical keys')
            require(physical_to_identity.setdefault(physical, cid) == cid,
                    'Physical aliases must share one canonical candidate identity')
        by_arm = {arm: [r for r in selected if r['arm'] == arm]
                  for arm in ('U', 'S', 'EEP', 'S_nom', 'S_back3')}
        for arm, cap in dict(U=3, S=3, EEP=2, S_nom=1, S_back3=1).items():
            require(1 <= len(by_arm[arm]) <= cap,
                    f'Missing or oversized {arm} nomination at hat={hat}')
        physical = {r['physical_key'] for r in selected}
        require(len(physical) <= 10, 'Per-hat physical shortlist exceeds ten')
        for row in selected:
            allowed = ('KEEP', 'DROP') if row['arm'] == 'S' else (
                ('KEEP',) if row['arm'] in ('S_nom', 'S_back3') else ('DROP',))
            require(tuple(row['receiver_variants']) == allowed,
                    'Receiver permissions differ from the registered comparison')
        counts[str(hat)] = len(physical)
    require(all(r['snr_hat'] in HATS for r in rows), 'Unknown shortlist estimate')
    return dict(physical_candidates_per_hat=counts,
                physical_frame_upper_bound=sum(counts.values()) * 200 * 3 * 3,
                logical_var_output_upper_bound=46800)


def design_score(matched_dino, overestimate_dino, p=.25):
    require(p in P_SENSITIVITY, 'Unregistered sensitivity mixture')
    require(math.isfinite(matched_dino) and math.isfinite(overestimate_dino),
            'Missing/nonfinite real quality cannot enter selection')
    return (1. - p) * matched_dino + p * overestimate_dino


def calibration_gate(j_by_method_hat):
    """Only authenticated real cal200 DINO can be supplied by the caller."""
    comparisons = {}
    for reference in ('S_robust', 'S_back3'):
        deltas = [float(j_by_method_hat['U_two'][hat]) -
                  float(j_by_method_hat[reference][hat]) for hat in HATS]
        require(all(math.isfinite(x) for x in deltas), 'Nonfinite gate input')
        comparisons[reference] = dict(per_hat=dict(zip(HATS, deltas)),
            mean=sum(deltas) / len(HATS), both_hats_positive=all(x > 0 for x in deltas))
    go = all(x['mean'] >= .01 and x['both_hats_positive'] for x in comparisons.values())
    one = any(any(d > 0 for d in c['per_hat'].values()) and
              not c['both_hats_positive'] for c in comparisons.values())
    return dict(status='GO_R2' if go else ('ONE_POINT_PROMISING' if one else 'NO_GO_STOP'),
                r2_authorized=go, comparisons=comparisons,
                selection_metric=SELECTION_METRIC, selection_p=.25,
                real_cal200_receipts_must_be_verified=True,
                engineering_targets_are_separate=True)


def interaction(u_var, s_var, u_direct, s_direct, metric):
    require(metric in METRICS, 'Unknown metric')
    raw = (u_var - s_var) - (u_direct - s_direct)
    return -raw if metric == 'lpips_alex' else raw
