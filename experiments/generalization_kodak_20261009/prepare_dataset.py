"""Download the complete Kodak24 suite and make predetermined center crops.

No model, metric, label inference, or adaptive crop selection is performed.
Dataset pixels are runtime assets and must remain outside Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

SIZES = [736501,617995,502888,637432,785610,618959,566322,788470,
         582899,593463,621023,531024,822712,692201,612582,534247,
         602078,780947,671476,492462,637051,701970,557596,706397]
BASE = 'https://r0k.us/graphics/kodak/kodak/'
PREPROCESSING = 'kodak_rgb_center_crop_256_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, runtime_root):
    from PIL import Image
    import numpy as np
    out = Path(out)
    (out/'original').mkdir(parents=True, exist_ok=True)
    (out/'crops').mkdir(exist_ok=True)

    def one(index):
        name = f'kodim{index+1:02d}.png'
        original = out/'original'/name
        url = BASE+name
        if not original.exists():
            request = urllib.request.Request(url, headers={'User-Agent':'VAR-COMM-research/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                blob = response.read()
            assert len(blob) == SIZES[index], (name, len(blob))
            with original.open('xb') as stream:
                stream.write(blob)
        assert original.stat().st_size == SIZES[index]
        with Image.open(original) as image:
            assert image.format == 'PNG' and image.mode == 'RGB'
            assert image.size in [(768,512),(512,768)]
            width,height = image.size
            left,top = (width-256)//2,(height-256)//2
            box = [left,top,left+256,top+256]
            crop = image.crop(box)
            destination = out/'crops'/name
            if not destination.exists():
                crop.save(destination, format='PNG', compress_level=9)
            with Image.open(destination) as saved:
                assert saved.mode == 'RGB' and saved.size == (256,256)
                assert saved.tobytes() == crop.tobytes()
            pixels = np.asarray(crop,dtype=np.uint8).transpose(2,0,1).copy()
        return dict(source_index=index,source_id=f'kodak/kodim{index+1:02d}',
                    preprocessing_id=PREPROCESSING,
                    original_png=dict(path=runtime_root+'/original/'+name,sha256=sha(original),
                                      bytes=original.stat().st_size,url=url,width=width,height=height),
                    crop_box_xyxy=box,
                    png=dict(path=runtime_root+'/crops/'+name,sha256=sha(destination)),
                    pixels_chw_uint8_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one,range(24)))
    manifest = dict(schema='KODAK24_CENTER256_DATASET_V1',
                    dataset='Kodak Lossless True Color Image Suite',
                    source_page='https://r0k.us/graphics/kodak/',
                    source_count=24,preprocessing=PREPROCESSING,
                    transform='RGB; center crop 256x256 with floor integer offsets; no resizing',
                    source_selection='All kodim01..kodim24 in numeric order, fixed before inference',
                    metric_label_policy='No ground-truth ImageNet labels; source prediction agreement only',
                    full_resolution_transmission_claimed=False,
                    pretrained_model_dataset_overlap_exclusion_claimed=False,
                    records=records)
    encoded = json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    path=out/'dataset_manifest.json'
    if path.exists():
        assert path.read_text(encoding='utf-8')==encoded,'Preserve a different existing manifest'
    else:
        path.write_text(encoded,encoding='utf-8')
    print(json.dumps(dict(status='DATASET_PREPARED_NOT_INFERRED',source_count=24,
                         manifest=str(path.resolve()),sha256=sha(path)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--runtime-root',required=True)
    args=parser.parse_args()
    prepare(args.out,args.runtime_root.rstrip('/'))
