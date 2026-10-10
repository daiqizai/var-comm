"""Pre-register unseen-to-this-run source IDs without reading image pixels.

No model, channel, quality score, entropy-family selection, or candidate tuning.
Historical exclusion coverage is explicitly limited to the supplied manifests.
"""
import argparse, hashlib, json, re
from pathlib import Path


def canonical_id(value):
    match = re.search(r'ILSVRC2012_val_(\d{8})(?:_|\.|/|$)', str(value))
    return 'imagenet-val:' + match.group(1) if match else str(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--excluded', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    excluded = json.loads(Path(a.excluded).read_text())
    assert excluded['schema'] == 'WCL_KNOWN_PRIOR_SOURCE_EXCLUSIONS_V1'
    used = set(excluded['canonical_source_ids'])
    assert len(used) == excluded['unique_source_count']
    root = Path(a.image_root).resolve()
    out = Path(a.out).resolve()
    assert root.is_dir() and not out.exists()
    rows = []
    seed = 'WCL_N2048_CONFIRMATION100_20261009_v1'
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        for image in sorted(directory.iterdir()):
            if image.suffix.lower() not in ('.jpeg', '.jpg', '.png'):
                continue
            sid = directory.name + '/' + image.stem
            key = canonical_id(sid)
            rows.append(dict(source_id=sid, canonical_source_id=key, path=str(image),
                original_bytes=image.stat().st_size, excluded=key in used,
                selection_key=hashlib.sha256((seed+'\0'+sid).encode()).hexdigest()))
    assert len({r['canonical_source_id'] for r in rows}) == len(rows)
    eligible = sorted((r for r in rows if not r['excluded']), key=lambda r:(r['selection_key'],r['source_id']))
    assert len(eligible) >= 100
    selected = [dict(r, confirmation_index=i) for i,r in enumerate(eligible[:100])]
    out.mkdir(parents=True)
    pool = out/'pool_inventory.json'
    pool.write_text(json.dumps(dict(records=rows), sort_keys=True)+'\n')
    result = dict(schema='WCL_N2048_CONFIRMATION100_METADATA_V1',
        status='SOURCE_IDS_FIXED_BEFORE_ANY_NEW_SOURCE_IMAGE_OR_SCORE_ACCESS',
        source_count=100, records=selected, SNRs=[4,10,19], noise_seeds=[9201,9202,9203],
        excluded_manifest=dict(path=str(Path(a.excluded).resolve()),sha256=sha(a.excluded)),
        pool_inventory=dict(path=str(pool),sha256=sha(pool)),image_root=str(root),
        pool_source_count=len(rows), eligible_source_count=len(eligible),
        selection_rule='First100 lexical SHA256(seed + NUL + canonical class/stem source_id); excludes all supplied prior source IDs',
        selection_seed=seed, pixel_reads=0, new_model_calls=0, new_packet_decodes=0,
        entropy_family_selected=False, historical_exclusions_complete_beyond_supplied_manifests=False,
        content_duplicate_check='PENDING; no content hash or preprocessing claims at metadata registration',
        scope='Fresh registered confirmation outside audited project source-use manifests; no claim about unrecorded use elsewhere',
        script_sha256=sha(__file__))
    (out/'registration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_count','pool_source_count','eligible_source_count','pixel_reads')}))


if __name__ == '__main__':
    main()
