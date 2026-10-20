"""CPU-only saved-output closure, lossless export and deterministic candidate layout."""
import ast
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
import execute as x

F = x.R/'paper/figures/selected_examples_10db_20261020_v1'
NAMES = dict(NEST='NeST-Com',P100000='Latent JSCC',Swin200000='SwinJSCC',
             BPG='BPG + LDPC',ARPC='ARPC + LDPC')
ORDER = ['P100000','Swin200000','BPG','ARPC','NEST']
SUBJECTS = ['Dog','Broccoli','Pastries','Shopping cart']


def gone(ident):
    cur = x.process(ident['pid'])
    return cur is None or cur['start_ticks'] != ident['start_ticks'] or cur['state'] == 'Z'


def main():
    import numpy as np
    from PIL import Image
    import worker
    reg = x.read(x.O/'registration.json')
    x.validate(reg)
    assert x.read(x.T/'dispatch/actual_wait.json')['returncode'] == 0
    for p in [x.T/'dispatch/launch.json',x.T/'control/launch.json']:
        assert gone(x.read(p)['process'])
    for method in x.METHODS:
        assert x.read(x.T/'control'/method/'actual_wait.json')['returncode'] == 0
        assert gone(x.read(x.T/method/'launch.json')['process'])
        assert not (x.T/method/'failure.json').exists()
        assert x.read(x.T/'control'/method/'hash_review.json')['exact_timing_hash_matches'] == 4
    mwait = x.read(x.T/'metric_control/actual_wait.json')
    assert mwait['returncode'] == 1 and mwait['child_gone'] and not mwait['timed_out']
    assert gone(mwait['child'])
    failure = x.read(x.T/'metrics/failure.json')
    assert failure['completed'] == 20
    assert "assert len(results) == 20 and calls == dict(started=40,completed=40)" in failure['traceback']
    # Static diagnosis only; no torch/LPIPS import and no model forward.
    lpips = x.RT/'assets/frozen_namespace_v1/external/home/liulu/.local/lib/python3.10/site-packages/lpips/lpips.py'
    tree = ast.parse(lpips.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LPIPS')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    direct = [n for n in ast.walk(fn) if isinstance(n,ast.Call) and
              ast.unparse(n.func)=='self.net.forward']
    assert len(direct) == 2
    assert not any(n in sys.modules for n in ('torch','lpips','tensorflow'))
    pairs = list((x.T/'metrics/pairs').glob('*/complete.json'))
    reservations = list((x.T/'metrics/pairs').glob('*/reserved.json'))
    assert len(pairs) == len(reservations) == 20
    assert not list((x.T/'metrics/pairs').glob('*/failed.json'))
    rows = [x.read(p) for p in pairs]
    assert {(v['method'],v['source_index']) for v in rows} == {
        (m,i) for m in x.METHODS for i in x.INDICES}
    out = x.O/'delivery'
    out.mkdir()
    F.mkdir()
    individual = F/'individual'
    individual.mkdir()
    originals = []
    metadata = []
    for rank,(source,subject) in enumerate(zip(reg['sources'],SUBJECTS),1):
        index = source['source_index']
        p = Path(source['original_png'])
        assert x.sha(p) == source['original_png_sha256']
        target = individual/f'{rank:02d}_original.png'
        shutil.copyfile(p,target)
        with np.load(x.checked(source['archive']),allow_pickle=False) as z:
            pixels = z['pixels'].copy()
        assert np.array_equal(np.asarray(Image.open(target)).transpose(2,0,1),pixels)
        originals.append(dict(source_index=index,subject=subject,source_id=source['source_id'],
            original=x.pin(target),preprocessing_id=source['preprocessing_id']))
        for method in ORDER:
            c = next(v for v in reg['cases'] if v['source_index']==index and v['method']==method)
            row = next(v for v in rows if v['source_index']==index and v['method']==method)
            assert row['source_id']==source['source_id'] and row['key']==c['key'] and row['snr_db']==10
            assert row['image_sha256']==c['expected_image_sha256']
            assert row['status']==c['expected_status']
            assert math.isfinite(row['psnr_db']) and math.isfinite(row['lpips_alex'])
            with np.load(x.checked(row['image']),allow_pickle=False) as z:
                image=z['image'].copy()
            assert worker.array_sha(image)==c['expected_image_sha256']
            rgb=np.rint(image.transpose(1,2,0)*np.float32(255)).astype(np.uint8)
            png=individual/f'{rank:02d}_{method}.png'
            Image.fromarray(rgb).save(png)
            assert np.array_equal(np.asarray(Image.open(png)),rgb)
            actual=None
            if c['received_input']['path'].endswith('.json'):
                actual=x.read(x.checked(c['received_input']))
            versions=dict(NEST='Current entropy-coded PARTIAL, frozen 10 dB policy',
                P100000='Final 100k checkpoint',Swin200000='Final adapted 200k checkpoint',
                BPG='Frozen adaptive source codec + LDPC',ARPC='Frozen official adapted ARPC + LDPC')
            metadata.append(dict(row_order=rank,subject=subject,source_id=source['source_id'],
                source_index=index,method=method,display_name=NAMES[method],version=versions[method],
                N=1024,snr_db=10,noise_seed=c['noise_seed'],noise_namespace=c['noise_namespace'],
                timing_case=c['key'],counter=c['counter'],receive_status=row['status'],
                actual_profile=json.dumps(actual.get('profile') if actual else None,sort_keys=True),
                image_path=png.relative_to(x.R).as_posix(),image_png_sha256=x.sha(png),
                float_image_path=row['image']['path'],float_image_file_sha256=row['image']['sha256'],
                timing_float_array_sha256=row['image_sha256'],
                psnr_db=row['psnr_db'],lpips_alex=row['lpips_alex']))
    for p in [out/'display_manifest.csv',F/'display_manifest.csv']:
        with p.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
    x.save(out/'originals.json',originals)
    diagnosis=dict(status='SAVED_SCORES_VALIDATED_WITH_DISCLOSED_POSTCHECK_FAILURE',
        original_metric_exit=mwait,original_failure=x.pin(x.T/'metrics/failure.json'),
        original_scorer=x.pin(x.E/'score.py'),lpips_implementation=x.pin(lpips),
        cause='LPIPS calls self.net.forward directly twice, bypassing Module __call__ hooks. Final hook-count assertion failed after all twenty saved pairs.',
        verified_saved_pairs=20,reservations=20,missing_pairs=0,neural_recomputation=0,
        runtime_backbone_hook_count='not persisted; cannot claim it as measured telemetry',
        static_backbone_calls_per_pair=2,model_constructions=1,
        score_files=[x.pin(p) for p in sorted(pairs)],metric_identity=x.pin(x.T/'metrics/metric_identity.json'))
    x.save(out/'score_closure_review.json',diagnosis)
    receipts=out/'receipts'
    receipts.mkdir()
    selected=[]
    for root in ['initial_control','dispatch','control','metric_control']:
        for p in (x.T/root).rglob('*'):
            if p.is_file() and p.suffix=='.json':selected.append(p)
    selected += [x.T/'metrics'/n for n in ['launch.json','model_reserved.json','model_complete.json',
        'metric_identity.json','failure.json']]
    for m in x.METHODS:
        selected += [x.T/m/n for n in ['launch.json','runtime.json','resource_before.json','completion.json']]
        if (x.T/m/'native_identity.json').exists():selected.append(x.T/m/'native_identity.json')
    copies=[]
    for p in sorted(set(selected)):
        q=receipts/p.relative_to(x.T);q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);assert x.sha(p)==x.sha(q)
        copies.append(dict(source=x.pin(p),copy=x.pin(q)))
    x.save(receipts/'copy_manifest.json',dict(files=copies))
    caption = (
        'Qualitative comparison at N = 1024 complex symbols and SNR = 10 dB on four '
        'previously selected development examples (dog, broccoli, pastries, shopping cart). '
        'Columns: original, Latent JSCC (100k), SwinJSCC (adapted, 200k), adaptive BPG + LDPC, '
        'ARPC + LDPC (adapted), and entropy-coded NeST-Com. Each cell uses the first fixed '
        'formal timing receive realization; all twenty native float outputs exactly match '
        'the recorded timing hashes. Method-specific noise namespaces are retained. '
        'Labels give single-reconstruction PSNR (dB) and LPIPS-Alex, not three-noise means. '
        'Metrics use the native float reconstruction; PNGs use uniform nearest-integer RGB8 '
        'display quantization. All images retain the bound full 256 x 256 crop. These selected '
        'examples are not a random common500 sample or a population-level quality summary.\n')
    (F/'caption_en.txt').write_text(caption)
    (out/'REPORT.md').write_text(
        '# Current five-method selected examples\n\n'
        'All 20 receiver outputs exactly match the first formal timing output hashes. '
        'All 20 are decoded outcomes; no rejected outcome was replaced. Four original crops '
        'are pixel-identical to the prior selection.\n\n'
        'Additional work: 20 RX reconstructions, 20 PSNR and 20 LPIPS pairs. '
        'Reused: 20 actual received inputs and 4 original PNGs. '
        'No TX/source encoding, channel events, training, timing grid, bootstrap or common500 evaluation.\n\n'
        'The original metric worker exited 1 only at its final hook-count assertion, after '
        'all 20 pair records were saved. The LPIPS implementation directly invokes its backbone '
        'forward function and bypasses those hooks. The original failure is retained; a separate '
        'CPU-only saved-output audit validates the complete records without rerunning any metric. '
        'Do not report the original scorer as exit 0 or its hook counter as measured telemetry.\n\n'
        'Metrics are defined on exact native float outputs, matching the main evaluation; '
        'PNG materials are uniformly quantized display copies, with no crop or enhancement.\n')
    render(F,metadata,originals)
    files=[x.pin(p) for p in sorted(F.rglob('*')) if p.is_file()]
    x.save(out/'manifest.json',dict(files=files,original_count=4,reconstruction_count=20))
    x.save(out/'completion.json',dict(status='DELIVERED_PENDING_VISUAL_REVIEW',exact_hash_matches=20,
        rx_reconstructions=20,received_inputs_reused=20,original_pngs_reused=4,PSNR_pairs=20,LPIPS_pairs=20,
        tx=0,channel_events=0,training=0,bootstrap=0,metric_original_returncode=1,
        score_closure_review=x.pin(out/'score_closure_review.json'),manifest=x.pin(out/'manifest.json')))
    print(json.dumps(dict(status='CPU_DELIVERY_COMPLETE',figure=str(F),rows=20)),flush=True)


def render(folder,rows,originals):
    """Pure rendering; can be rerun separately without scientific calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    mm=1/25.4
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
        'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(178*mm,156*mm),facecolor='white')
    left=3.0;gap=1.5;tile=(178-2*left-5*gap)/6
    headings=['Original']+[NAMES[m] for m in ORDER]
    for col,title in enumerate(headings):
        cx=(left+col*(tile+gap)+tile/2)/178
        fig.text(cx,.976,title,ha='center',va='top',fontsize=8.5,
            color='#3775BA' if col==5 else '#202020',
            fontweight='bold' if col==5 else 'normal')
    for row,original in enumerate(originals):
        ytop=145.0-row*35.3
        for col,method in enumerate([None]+ORDER):
            xleft=left+col*(tile+gap)
            ax=fig.add_axes([xleft/178,(ytop-tile)/156,tile/178,tile/156])
            if method is None:
                path=Path(original['original']['path'])
                label=f"{original['subject']}\nsource {original['source_index']}"
            else:
                record=next(v for v in rows if v['method']==method and v['source_index']==original['source_index'])
                path=x.R/record['image_path']
                label=f"PSNR {record['psnr_db']:.2f} dB\nLPIPS {record['lpips_alex']:.3f}"
            ax.imshow(np.asarray(Image.open(path)),interpolation='nearest')
            ax.set_axis_off()
            fig.text((xleft+tile/2)/178,(ytop-tile-1.1)/156,label,
                ha='center',va='top',fontsize=8.0,linespacing=1.15,color='#222222')
    fig.savefig(folder/'candidate_montage.png',dpi=600)
    fig.savefig(folder/'candidate_montage_preview.png',dpi=200)
    fig.savefig(folder/'candidate_montage.pdf')
    fig.savefig(folder/'candidate_montage.svg')
    plt.close(fig)
    x.save(folder/'layout.json',dict(width_mm=178,height_mm=156,png_dpi=600,
        heading_font_pt=8.5,metric_font_pt=8.0,photos='Embedded actual RGB8 pixels, nearest display interpolation',
        native_image_pixels=[256,256],vector_claim='Text/layout are vector; photographs remain raster',
        order=['Original']+[NAMES[m] for m in ORDER]))


if __name__=='__main__':
    main()
