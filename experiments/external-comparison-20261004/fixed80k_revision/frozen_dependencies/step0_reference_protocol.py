"""Register metric reference conditions before any reference metric inference."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import random
from step0_cache_export import read, require, seal, sha

PAIR_SEED = 2026100402
NOISE_SEED = 2026100403


def unrelated_pairs(records):
    """Uniform random permutation, rejecting identity or same-class matches."""
    require(len(records) == 100 and len({r['image_id'] for r in records}) == 100, 'Frozen 100 distinct sources required')
    rng = random.Random(PAIR_SEED)
    for attempt in range(100000):
        indices = list(range(100))
        rng.shuffle(indices)
        if all(i != j and records[i]['class_index'] != records[j]['class_index'] for i,j in enumerate(indices)):
            return [dict(source_index=i,source_id=records[i]['image_id'],donor_source_index=j,
                         donor_id=records[j]['image_id'],different_class=True) for i,j in enumerate(indices)], attempt+1
    raise RuntimeError('No different-class derangement found under frozen algorithm')


def select_same_class(records, donor_manifest):
    """Select metadata-only donors; no image, model, or metric influences choice."""
    require(donor_manifest.get('status') == 'VERIFIED_TRAIN_CALIBRATION_MEMBERSHIP', 'Donor pool membership not verified')
    require(donor_manifest.get('holdout_access') is False and donor_manifest.get('verified_before_metrics') is True,
            'Donor pool must be verified without holdout before metrics')
    require(donor_manifest.get('membership_bindings'), 'Donor membership requires original manifest hashes')
    donors = donor_manifest['records']
    dev_ids = {r['image_id'] for r in records}
    dev_pixels = {r.get('source_pixels_sha256',r.get('preprocessing_id')) for r in records}
    require(len({r['image_id'] for r in donors}) == len(donors), 'Duplicate donor IDs')
    for r in donors:
        require(r['population'] in ('train','calibration') and r.get('membership_verified') is True,
                'Donor is not in verified original train/calibration membership')
        require(r['image_id'] not in dev_ids and r['preprocessing_id'] not in dev_pixels,
                'Development image/pixels cannot be a same-class donor')
        require(r.get('path') and len(r.get('file_sha256','')) == 64, 'Donor pixels need an exact file identity')
    selected, missing = [], []
    for i,r in enumerate(records):
        candidates = [d for d in donors if d['class_index'] == r['class_index']]
        if not candidates:
            missing.append(dict(source_index=i,source_id=r['image_id'],class_index=r['class_index']))
            continue
        rank = lambda d: hashlib.sha256(f'{PAIR_SEED}|{r["image_id"]}|{d["image_id"]}'.encode()).hexdigest()
        donor = min(candidates,key=lambda d:(rank(d),d['image_id']))
        selected.append(dict(source_index=i,source_id=r['image_id'],donor=donor,selection_hash=rank(donor)))
    return selected,missing


def register(development_manifest, fixed_examples, output, donor_manifest=None):
    records = read(development_manifest)
    require(isinstance(records,list) and [r.get('image_index',r.get('source_index')) for r in records] == list(range(100)),
            'Original ordered development manifest required')
    fixed = read(fixed_examples)
    require(fixed['status'] == 'FROZEN_FIXED_EXAMPLES', 'Fixed example registration missing')
    for r in fixed['records']:
        original = records[r['source_index']]
        require(original['image_id'] == r['image_id'] and original['class_index'] == r['class_index']
                and original.get('source_pixels_sha256',original.get('preprocessing_id')) == r['preprocessing_id'],
                'Fixed sample differs from original population')
    pairs,attempts = unrelated_pairs(records)
    plan = dict(status='REFERENCE_CONDITIONS_PREREGISTERED_DONOR_POOL_PENDING',
        development_manifest_sha256=sha(development_manifest),fixed_examples_sha256=sha(fixed_examples),
        source_count=100,statistical_unit='source_image',wireless_experiment=False,
        channel_uses=None,snr_db=None,noise_triplication=False,holdout_access=False,
        selection_uses_metrics=False,models_or_conditions_tuned=False,
        unrelated=dict(seed=PAIR_SEED,algorithm='Python random.Random.shuffle; reject fixed points and same-class pairs',
                       draws=attempts,pairs=pairs,donor_population='same frozen 100 development originals'),
        same_class=dict(status='PENDING_ORIGINAL_TRAIN_CALIBRATION_DONOR_MEMBERSHIP',
            allowed_populations=['train','calibration'],exclude_all_development_ids_and_pixels=True,
            selection='Minimum SHA256(seed|source_id|donor_id), then donor_id; no visual or metric screening',
            seed=PAIR_SEED,missing_class_policy='Mark unavailable; never use holdout or substitute a different class'),
        mild_conditions=[
            dict(name='gaussian_blur_sigma1',operation='independent Gaussian blur on original float RGB',
                 sigma_pixels=1.,kernel_size=7,padding='reflect',implementation='separable float64 Gaussian kernel; cast output float32'),
            dict(name='gaussian_noise_sigma2_255',operation='independent additive noise then clip [0,1]',
                 sigma=2/255,seed=NOISE_SEED,namespace='SHA256(seed|image_id) first 8 bytes little-endian; NumPy PCG64',
                 implementation='NumPy default_rng.standard_normal, float64 addition then clip and cast float32'),
            dict(name='jpeg_quality90',operation='independent JPEG roundtrip of source uint8 RGB',quality=90,
                 subsampling=0,optimize=False,progressive=False,implementation='Pillow; freeze Pillow/libjpeg version before inference')],
        metrics=['PSNR','LPIPS-Alex','DISTS','DreamSim','CLIP-ViT-L/14','DINOv2-ViT-L/14','MS-SSIM',
                 'ResNet50-ImageNet1K-V2-label-accuracy','ResNet50-source-prediction-agreement','semantic_error','confidently_wrong'],
        evaluator_identity='Reuse frozen unified modelmanifest and preprocessing; register exact SHA before inference',
        interpretation='Reference distributions show metric scale; they are not acceptance thresholds, tuned baselines, or channel results.',
        uncertainty=dict(unit='100 source images',replicates=10000,seed=20261002,confidence=.95),
        next_gate='Verify original donor membership manifests and all pixel hashes, seal donor selection, then score each reference once.')
    if donor_manifest:
        donor = read(donor_manifest)
        for p,h in donor['membership_bindings'].items():
            require(sha(p) == h,'Original donor membership manifest changed')
        selected,missing = select_same_class(records,donor)
        plan['same_class'].update(status='METADATA_DONORS_SELECTED_PENDING_PIXEL_VERIFICATION',selected=selected,missing=missing,
                                  donor_manifest_sha256=sha(donor_manifest))
        plan['status'] = 'REFERENCE_CONDITIONS_AND_DONOR_IDS_PREREGISTERED'
    seal(output,plan)
    return plan


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--development-manifest',type=Path,required=True)
    parser.add_argument('--fixed-examples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--donor-manifest',type=Path)
    args=parser.parse_args()
    print(register(args.development_manifest,args.fixed_examples,args.output,args.donor_manifest)['status'],flush=True)
