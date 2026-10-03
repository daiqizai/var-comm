"""Render fixed historical resource comparisons from verified Step0 PNGs."""
import argparse
import io
from pathlib import Path
import textwrap
import time
from PIL import Image, ImageDraw, ImageFont
from step0_cache_export import read, require, seal, sha

GROUPS = ('P', 'Digital raw / Dc', 'Digital arithmetic / Dc', 'Legacy raw / D0', 'Legacy arithmetic / D0')


def render(folder, output, class_names=None):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    manifest_path = folder/'step0_manifest.json'
    manifest = read(manifest_path)
    require(manifest['status'] == 'EXACT_COMPLETED_FLOAT_RGB_CPU_EXPORT_PASS', 'No verified float export')
    for record in manifest['sources'] + manifest['frames']:
        p = folder/record['file']
        require(p.parent == folder and sha(p) == record['png_sha256'], 'Export PNG identity differs')
    labels = read(class_names) if class_names else None
    require(labels is None or isinstance(labels, list) and len(labels) == 1000 and all(isinstance(x,str) for x in labels),
            'ImageNet class names must be an ordered 1000-element list')
    receipt_path = output/'step0_figures.json'
    if receipt_path.exists():
        previous = read(receipt_path)
        require(previous['source_manifest_sha256'] == sha(manifest_path)
                and previous['class_names_sha256'] == (sha(class_names) if class_names else None)
                and previous['source_bindings'] == {str(Path(__file__).resolve()):sha(__file__)},
                'Existing figure input/labels/source differ; use a new output directory')
    def label(index):
        return labels[index] if labels else 'class '+str(index)
    def font(size, bold=False):
        names = (['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf'] if bold else
                 ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'])
        for name in names:
            if Path(name).is_file():
                return ImageFont.truetype(name,size=size)
        return ImageFont.load_default(size=size)
    output.mkdir(parents=True, exist_ok=True)
    sources = {r['source_index']:r for r in manifest['sources']}
    figures = []
    for index in manifest['source_indices']:
        source = sources[index]
        original = Image.open(folder/source['file']).convert('RGB')
        for snr in manifest['snrs_db']:
            rows = [r for r in manifest['frames'] if r['source_index'] == index and r['snr_db'] == snr]
            by_key = {(r['group'],r['N']):r for r in rows}
            require(len(by_key) == len(rows) == 11, 'Unexpected figure row coverage')
            canvas = Image.new('RGB',(1640,2410),'#fafafa')
            draw = ImageDraw.Draw(canvas)
            def text(x,y,message,size=18,bold=False,color='#222222'):
                draw.multiline_text((x,y),message,font=font(size,bold),fill=color,anchor='ma',align='center',spacing=5)
            def tile(column,row,picture):
                x,y=42+column*400,158+row*416
                canvas.paste(picture.resize((352,352),Image.Resampling.LANCZOS),(x,y))
                return x+176,y+358
            text(820,22,f'Existing reconstructions across bandwidth | source {index:03d} | {snr} dB',28,True)
            text(820,65,source['image_id']+' | noise seed 2001 | fixed examples selected before rendering',17)
            tile(0,0,original)
            original_prediction = rows[0]['resnet50_source_prediction']
            require(all(r['resnet50_source_prediction'] == original_prediction for r in rows), 'Source prediction differs across rows')
            text(218,108,'Original',23,True)
            text(218,516,'Ground truth: '+label(source['class_index'])+'\nR50: '+label(original_prediction),15)
            for gi, group in enumerate(GROUPS):
                if gi:
                    title = group+'\nPaid true class\n'+('Older protocol' if group.startswith('Legacy') else 'Legacy protocol')
                    text(218,240+gi*416,title,23,True)
                    text(218,357+gi*416,'Class agreement is conditioned\non a transmitted true label.',17,color='#704522')
                for ni,n in enumerate(manifest['budgets'],1):
                    if gi == 0:
                        text(218+ni*400,106,f'N = {n} complex uses',23,True)
                        text(218+ni*400,136,'P / Dc; train seed 2026092304',16)
                    row = by_key.get((group,n))
                    if row is None:
                        text(218+ni*400,300+gi*416,'No measured point',22,color='#777777')
                        continue
                    x,y=tile(ni,gi,Image.open(folder/row['file']).convert('RGB'))
                    m = row['metrics']
                    agreement = 'same as original' if not row['semantic_error'] else 'differs from original'
                    prediction = textwrap.shorten(label(row['resnet50_prediction']),width=32,placeholder='...')
                    caption = f"PSNR {m['psnr_db']:.2f} dB | LPIPS {m['lpips_alex']:.3f} | DINOv2-L {m['dinov2_vitl14_cosine']:.3f}\nR50: {prediction} ({agreement})"
                    text(x,y,caption,13,color='#1e3a38' if not row['semantic_error'] else '#94372f')
            text(820,2280,'Digital rows use paid true class; they are not the current unconditional digital chain.\n'
                 'D0 and Dc use different decoders. Protocol changes limit causal conclusions about bandwidth.\n'
                 'Display PNGs are resized; all shown metrics were measured on original float RGB. Missing points remain blank.',17)
            stem = f'step0_source{index:03d}_snr{snr}'
            for suffix in ('.png','.pdf'):
                p = output/(stem+suffix)
                stream=io.BytesIO()
                if suffix == '.pdf':
                    canvas.save(stream,format='PDF',resolution=150.,title=stem,
                                creationDate=time.gmtime(0),modDate=time.gmtime(0))
                else:canvas.save(stream,format='PNG',dpi=(150,150))
                data=stream.getvalue()
                if p.exists():
                    require(p.read_bytes() == data,'Existing figure bytes differ; original figure preserved')
                else:p.write_bytes(data)
                figures.append(dict(file=p.name,sha256=sha(p),source_index=index,snr_db=snr))
    receipt = dict(status='STEP0_FIXED_RESOURCE_FIGURES_COMPLETE', figures=figures,
                   source_manifest_sha256=sha(manifest_path), source_manifest=str(manifest_path),
                   class_names_sha256=sha(class_names) if class_names else None,
                   rendered_class_labels='ImageNet names' if class_names else 'ImageNet class indices',
                   source_bindings={str(Path(__file__).resolve()):sha(__file__)}, inference=False)
    seal(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--class-names',type=Path)
    args = p.parse_args()
    print(render(args.folder,args.output,args.class_names)['status'],flush=True)
