"""Join the immutable legacy gray DINO observations to the exact Pilot50 roster."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from audit_pilot import blob, check, writecsv

GRAY = 'results/layered_entropy_le2_20261014_v1'

def pinned(path, oid):
    data = path.read_bytes()
    if blob(data) != oid:
        data = data.replace(b'\r\n', b'\n')
    check(blob(data) == oid, str(path))
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = pinned(a.repo / GRAY / 'gray_quality.csv', '9d75f240ca284013a08818e3838e63908104e784')
    reg = json.loads(pinned(a.repo / GRAY / 'gray_registration.json', '506282a5e3db5cb9e8addbde7fa4ebe5dfe9f75c'))
    audit = json.loads((a.out / 'pilot50_audit.json').read_bytes())
    check(reg['pilot50']['sha256'] == audit['manifest_sha256'], 'gray pilot identity')
    with (a.out / 'pilot50_asset_inventory.csv').open(encoding='utf8', newline='') as f:
        roster = list(csv.DictReader(f))
    archive = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    check(len(roster) == len(archive) == 50, 'gray rows')
    by_slot = {int(r['pilot_slot']): r for r in archive}
    check(set(by_slot) == set(range(50)), 'incomplete or duplicate gray rows')
    out = []
    for i, source in enumerate(roster):
        r = by_slot[i]
        check(int(source['pilot_slot']) == i and source['source_id'] == r['source_id'] and source['source_pixel_sha256'] == r['source_preprocessing_id'], 'source mismatch')
        check(r['gray_array_sha256'] == reg['gray_array_sha256'], 'gray image mismatch')
        check(math.isfinite(float(r['dinov2_vitl14_cosine'])), 'missing DINO score')
        out.append(dict(pilot_slot=i, source_id=r['source_id'], source_pixel_sha256=r['source_preprocessing_id'], gray_array_sha256=r['gray_array_sha256'], dinov2_cosine=r['dinov2_vitl14_cosine'], dists='', dreamsim='', evidence='LE2 gray DINO only; no new metric calls'))
    writecsv(a.out / 'sp0_gray_quality_evidence.csv', out)
    print('50 registered gray DINO pairs verified; DISTS and DreamSim absent in this source.')

if __name__ == '__main__':
    main()
