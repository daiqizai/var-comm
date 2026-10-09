"""Export preselected qualitative panels from immutable common500 image caches."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

COMMIT = '252176e041758ecb2d3e81b6fde5b587e7e17bb7'
SALT = 'VAR_COMM_PAPER_EXAMPLES_20261008_V1'
SNRS = [13, 19]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p, j):
    Path(p).write_text(json.dumps(j, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def raw_array_sha(value):
    """Exact same_rx_renderer.array_sha / H metric image domain."""
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def baseline_rgb_sha(value):
    """Exact common500_baseline.rgb_sha domain; distinct from RAW."""
    a = np.ascontiguousarray(value)
    return hashlib.sha256(b'float32:3,256,256:RGB\0'+a.tobytes()).hexdigest()


def finite_rgb(value):
    assert value.dtype == np.float32 and value.shape == (3,256,256)
    assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
    return np.ascontiguousarray(value)


def diagnostic_description(source, shown_snrs):
    """Only recorded Proposed PARTIAL VAR outcomes receive failure labels."""
    events = [r for r in source.get('failure_evidence', []) if r['snr_db'] in shown_snrs]
    if not source.get('failure_evidence'):
        return ''
    if not events:
        return 'Diagnostic source'
    snrs = '/'.join(str(s) for s in sorted({r['snr_db'] for r in events}))
    if 'correct_to_wrong' in source['selection_role']:
        return 'Proposed prediction mismatch\n'+snrs+' dB'
    return 'Proposed link failure\n'+snrs+' dB'


def diagnostic_caption(plan):
    source = plan['samples'][-1]
    events = source['failure_evidence']
    snrs = sorted({r['snr_db'] for r in events})
    displayed_snrs = ', '.join(str(s) for s in snrs)+' dB'
    if 'correct_to_wrong' in source['selection_role']:
        assert all(r['convnext_source_correct_to_wrong'] is True for r in events)
        transitions = '; '.join(str(r['snr_db'])+' dB: '+r['convnext_source_prediction_name']
                               +' to '+r['convnext_prediction_name'] for r in events)
        detail = ('No PARTIAL header/body CRC failure was recorded at the two requested SNRs '
                  'for the fixed first noise realization. The fourth source was therefore '
                  'selected as the first source index with a recorded correct-source to '
                  'incorrect-reconstruction ConvNeXt transition in Proposed PARTIAL VAR completion. '
                  'The recorded prediction mismatch occurs at '+displayed_snrs+' ('+transitions+'). '
                  'A classifier mismatch is not proof that the visual semantic content is necessarily wrong. ')
    else:
        assert source['selection_role'] == 'first_index_recorded_link_failure'
        assert all(not r['header_ok'] or not r['body_crc_accept'] for r in events)
        detail = ('The fourth source was selected as the first source index with a recorded '
                  'PARTIAL header/body CRC failure at '+displayed_snrs+' for the fixed first '
                  'noise realization. It is labelled as a Proposed link failure, not a '
                  'classifier or visual-semantic failure. ')
    other = sorted(set(plan['snr_db'])-set(snrs))
    if other:
        detail += ('The same source at '+', '.join(str(s) for s in other)
                   +' dB is shown as a diagnostic source, without assigning that recorded failure to it. ')
    return detail


def prepare(root, out):
    """Choose without opening any image pixels or ranking image quality."""
    planpath = out/'example_selection.json'
    if planpath.exists():
        raise RuntimeError('Selection already frozen; use --render to reuse it')
    n = root/'outputs/MAIN-RAW64-20261007'
    mp = n/'common500_source_assets_v1/manifest.json'
    manifest = read(mp)
    records = manifest['records']
    assert len(records) == 500
    random_order = sorted(records, key=lambda r: hashlib.sha256((SALT+'|'+r['source_id']).encode()).hexdigest())
    fixed = [dict(r, selection_role='hash_selected_without_quality_ranking') for r in random_order[:3]]
    failure = None
    examined = []
    # A predeclared diagnostic criterion, not post-render visual cherry-picking.
    for source in sorted(records, key=lambda r:r['source_index']):
        p = n/f"unified500_same_rx_parallel_images_r2/sources/{source['source_index']:04d}.json"
        rows = read(p)
        examined.append({'path':str(p), 'sha256':sha(p)})
        bad = [r for r in rows if r['snr_db'] in SNRS and r['noise_seed']==6201
               and 'PARTIAL' in r['families'] and (not r['header_ok'] or not r['body_crc_accept'])]
        if bad:
            failure = dict(source, selection_role='first_index_recorded_link_failure',
                           failure_evidence=[{k:r[k] for k in ('snr_db','noise_seed','header_ok','body_crc_accept','gray')} for r in bad])
            break
    if failure is None:
        # No physical failure at these high-SNR first-noise points. Preserve that
        # finding and use a separately labelled semantic failure, never a fake CRC failure.
        for source in sorted(records, key=lambda r:r['source_index']):
            p=n/f"unified500_metrics_r6/sources/{source['source_index']:04d}.json"
            rows=read(p); examined.append({'path':str(p),'sha256':sha(p)})
            bad=[r for r in rows if r['snr_db'] in SNRS and r['noise_seed']==6201
                 and 'PARTIAL' in r['family_memberships'] and r['decoder_arm']=='VAR_completion'
                 and r['convnext_source_correct_to_wrong']]
            if bad:
                failure=dict(source,selection_role='first_index_recorded_source_correct_to_wrong',
                    failure_evidence=[{k:r[k] for k in ('snr_db','noise_seed','convnext_prediction_name','convnext_source_prediction_name','convnext_source_correct_to_wrong','dinov2_vitl14_cosine')} for r in bad])
                break
    if failure is None:
        raise RuntimeError('No eligible recorded failure; stop instead of fabricating one')
    fixed = [r for r in fixed if r['source_index'] != failure['source_index']]
    if len(fixed)<3:
        fixed.append(dict(next(r for r in random_order if r['source_index'] not in {q['source_index'] for q in fixed+[failure]}),selection_role='hash_selected_without_quality_ranking'))
    fixed.append(failure)
    bindings = {}
    for r in fixed:
        i = r['source_index']
        paths = [Path(r['archive']), Path(r['checkpoint']),
                 n/f'unified500_same_rx_parallel_images_r2/sources/{i:04d}.json',
                 n/f'unified500_same_rx_parallel_images_r2/images/{i:04d}.npz']
        for method in ('P1024','SwinJSCC80k'):
            cp = n/f'unified500_baseline_r2/{method}/source_checkpoints/{i:04d}.json'
            paths += [cp, Path(read(cp)['float_reconstructions']['path'])]
        for p in paths: bindings[str(p)] = sha(p)
    dump(planpath, dict(schema='PAPER_HOLDOUT_EXAMPLE_SELECTION_V1', created_unix=time.time(),
         data_commit=COMMIT, snr_db=SNRS, source_manifest=dict(path=str(mp),sha256=sha(mp)),
         samples=fixed, rule='Three source-ID SHA256-ranked examples plus first-index recorded PARTIAL header/body CRC failure at requested SNRs and fixed first noise; if none, first-index source-correct to reconstruction-wrong ConvNeXt transition. Frozen before opening reconstruction pixels.',
         noise_seeds=dict(Proposed=6201,P1024=2001,SwinJSCC80k=2001),
         identical_noise_across_methods=False, failure_diagnostic_not_representative=True,
         source_bindings=bindings, failure_search_metadata=examined,
         new_inference=0,new_channel_simulations=0))
    print(json.dumps({'selection_frozen':str(planpath),'indices':[r['source_index'] for r in fixed], 'failure':failure['failure_evidence']}))


def render(root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    n = root/'outputs/MAIN-RAW64-20261007'
    planpath = out/'example_selection.json'
    plan = read(planpath)
    assert plan['snr_db'] == SNRS and len(plan['samples']) == 4
    assert len({r['source_index'] for r in plan['samples']}) == 4
    assert sha(plan['source_manifest']['path']) == plan['source_manifest']['sha256']
    assert all(sha(p)==h for p,h in plan['source_bindings'].items()), 'Frozen cache bytes changed'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    images, evidence = [], []
    for source in plan['samples']:
        i = source['source_index']
        assert sha(source['archive']) == source['archive_sha256']
        assert sha(source['checkpoint']) == source['checkpoint_sha256']
        cp_source = read(source['checkpoint'])
        assert cp_source['source_index'] == i and cp_source['source_id'] == source['source_id']
        assert cp_source['archive'] == source['archive']
        with np.load(source['archive'],allow_pickle=False) as z: pixels=z['pixels']
        assert pixels.shape == (3,256,256) and pixels.dtype==np.uint8
        rowimages = [pixels.transpose(1,2,0)]
        rawrows = read(n/f'unified500_same_rx_parallel_images_r2/sources/{i:04d}.json')
        for snr in SNRS:
            r = [r for r in rawrows if r['snr_db']==snr and r['noise_seed']==6201 and 'PARTIAL' in r['families']]
            assert len(r)==1
            r=r[0]; ref=r['same_RX_images']['VAR_completion']
            assert r['source_index'] == i and r['source_id'] == source['source_id']
            assert ref['image_archive'] in plan['source_bindings']
            with np.load(ref['image_archive'],allow_pickle=False) as z:
                assert type(ref['image_slot']) is int and 0 <= ref['image_slot'] < len(z['images'])
                a=finite_rgb(z['images'][ref['image_slot']])
            assert raw_array_sha(a) == ref['image_sha256']
            assert raw_array_sha(a) == r['same_RX_visual_proof']['image_sha256']['VAR_completion']
            rowimages.append(np.rint(np.clip(a,0,1)*255).astype(np.uint8).transpose(1,2,0))
            evidence.append(dict(source_index=i,source_id=source['source_id'],method='Proposed',snr_db=snr,
                                 noise_seed=6201,image_slot=ref['image_slot'],archive=ref['image_archive'],
                                 frozen_image_sha256=ref['image_sha256'],header_ok=r['header_ok'],body_crc_accept=r['body_crc_accept']))
            for method in ('P1024','SwinJSCC80k'):
                cp=read(n/f'unified500_baseline_r2/{method}/source_checkpoints/{i:04d}.json')
                assert cp['source_index']==i and cp['source_id']==source['source_id'] and cp['method']==method
                rows=[r for r in cp['rows'] if r['snr_db']==snr and r['noise_seed']==2001]
                assert len(rows)==1
                r=rows[0]; archive=cp['float_reconstructions']['path']
                assert r['source_index']==i and r['source_id']==source['source_id'] and r['method']==method
                assert archive in plan['source_bindings']
                assert plan['source_bindings'][archive] == cp['float_reconstructions']['sha256'] == cp['outputs'][archive]
                with np.load(archive,allow_pickle=False) as z:
                    assert np.array_equal(z['source_rgb'],pixels.astype(np.float32)/255)
                    assert type(r['image_slot']) is int and 0 <= r['image_slot'] < len(z['images'])
                    a=finite_rgb(z['images'][r['image_slot']])
                assert baseline_rgb_sha(a) == r['image_sha256']
                assert baseline_rgb_sha(a) == cp['float_reconstructions']['image_sha256'][r['image_slot']]
                rowimages.append(np.rint(np.clip(a,0,1)*255).astype(np.uint8).transpose(1,2,0))
                evidence.append(dict(source_index=i,source_id=source['source_id'],method=method,snr_db=snr,
                                     noise_seed=2001,image_slot=r['image_slot'],archive=archive,
                                     frozen_image_sha256=r['image_sha256'],header_ok=r.get('header_accepted','NA'),body_crc_accept='NA'))
        images.append(rowimages)
        single=out/'individual'; single.mkdir(exist_ok=True)
        names=['source','proposed_13dB','p1024_13dB','swin80k_13dB','proposed_19dB','p1024_19dB','swin80k_19dB']
        for name,img in zip(names,rowimages): Image.fromarray(img).save(single/f'source{i:04d}_{name}.png')
    groups=[('fig05_qualitative_13_19',list(range(7)),(7.2,4.7)),
            ('fig05_qualitative_13dB',[0,1,2,3],(7,7.45)),
            ('fig05_qualitative_19dB',[0,4,5,6],(7,7.45))]
    titles=['Source','Proposed\n13 dB','P1024\n13 dB','Swin80k*\n13 dB','Proposed\n19 dB','P1024\n19 dB','Swin80k*\n19 dB']
    outputs=[]
    for stem,cols,size in groups:
        shown_snrs = [snr for snr,col in ((13,1),(19,4)) if col in cols]
        fig,axes=plt.subplots(4,len(cols),figsize=size,squeeze=False)
        fig.subplots_adjust(left=.058,right=.995,bottom=.02,top=.92,wspace=.035,hspace=.04)
        for ir,source in enumerate(plan['samples']):
            for jc,col in enumerate(cols):
                ax=axes[ir,jc]; ax.imshow(images[ir][col],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if ir==0: ax.set_title(titles[col],fontsize=8,pad=5)
                if jc==0:
                    label = diagnostic_description(source, shown_snrs) if ir==3 else ''
                    if ir==3 and stem=='fig05_qualitative_13_19':
                        label = 'Diagnostic source'
                    ax.set_ylabel(f"ID {source['source_index']:04d}"+ ('\n'+label if label else ''),fontsize=7)
                if ir==3 and col in (1,4):
                    snr = 13 if col==1 else 19
                    events = [r for r in source['failure_evidence'] if r['snr_db']==snr]
                    if events:
                        label = 'Prediction mismatch' if 'correct_to_wrong' in source['selection_role'] else 'Link failure'
                        ax.text(.5,.015,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=6,
                                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
        for ext in ('png','pdf','svg'):
            p=out/f'{stem}.{ext}';fig.savefig(p,dpi=600,facecolor='white');outputs.append(p)
        plt.close(fig)
    with (out/'example_provenance.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
    caption = ('Frozen common500 holdout reconstructions at 13 and 19 dB. Three sources were selected by a fixed source-ID hash. '
               +diagnostic_caption(plan)+
               'This is an outcome-selected diagnostic example, not a random or representative failure-rate estimate. '
               'Its image ID and both SNRs were sealed before any reconstruction was rendered. '
               'All images are exported from the original lossless float caches, with clipping and nearest-integer 8-bit conversion '
               'for display only; no model inference or channel simulation was repeated. The fixed first noise realization is used: '
               'seed 6201 for Proposed and 2001 for P1024 and SwinJSCC-80k (adapted). These are the same source images, not identical '
               'cross-method channel-noise draws. *Swin is the frozen adapted 80k baseline; 19 dB is outside its training and calibration range.')
    # Prediction labels may contain LaTeX metacharacters; escape data, not syntax.
    escaped = ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                       '~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in caption)
    (out/'caption.tex').write_text(r'\newcommand{\MainRawExamplesCaption}{'+escaped+'}\n',encoding='utf-8')
    assert all(sha(p)==h for p,h in plan['source_bindings'].items())
    dump(out/'completion.json',dict(status='COMPLETE_CACHED_RECONSTRUCTION_EXPORT',selection_sha256=sha(planpath),
         samples=4,reconstruction_cells=24,source_cells=4,new_inference=0,new_channel_simulations=0,
         failure_evidence=plan['samples'][-1]['failure_evidence'],array_image_hashes_verified=True,
         outputs={p.name:sha(p) for p in outputs}))
    print(json.dumps({'status':'COMPLETE','samples':4,'reconstructions':24,'exports':9}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    out=args.root/'paper/figures/mainraw64_supplement_20261008/qualitative';out.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(args.root,out)
    if args.render:render(args.root,out)
