"""Render cached, source-matched N1024/13 dB comparisons without model execution.

Internal experiment names occur only in provenance and this explicit mapping.
--collect reads immutable remote caches. --render uses the small portable bundle.
Missing HiFi holdout data is represented by text, never another source/SNR.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

COMMIT = '252176e041758ecb2d3e81b6fde5b587e7e17bb7'
REL_OUT = 'paper/figures/mainraw64_n1024_13db_comparisons'
MAPPING = {
    'source': 'Original image',
    'SwinJSCC80k': 'SwinJSCC\nimage transmission\n(adapted)',
    'P1024': 'Continuous latent-space\njoint source-channel\ncoding',
    'HiFiDiffCom': 'HiFi-DiffCom\ndiffusion-prior\nrecovery',
    'RAW64_PARTIAL_VAR_COMPLETION': 'Partial-scale digital\ntransmission +\nVAR completion (ours)',
    'RAW64_WHOLE_VAR_COMPLETION': 'Complete-scale transmission\n+ VAR completion',
    'RAW64_PARTIAL_DIRECT_DC': 'Partial-scale transmission\nwithout generative\ncompletion',
}
ARRAY_ORDER = ['source', 'SwinJSCC80k', 'P1024',
               'RAW64_PARTIAL_VAR_COMPLETION',
               'RAW64_WHOLE_VAR_COMPLETION', 'RAW64_PARTIAL_DIRECT_DC']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def raw_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def baseline_sha(a):
    return hashlib.sha256(b'float32:3,256,256:RGB\0'+np.ascontiguousarray(a).tobytes()).hexdigest()


def display(a):
    assert a.shape == (3, 256, 256) and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    return np.rint(a*255).astype(np.uint8).transpose(1, 2, 0)


def one(rows):
    assert len(rows) == 1, 'Missing or duplicate working point'
    return rows[0]


def collect(root, out):
    n = root/'outputs/MAIN-RAW64-20261007'
    oldpath = root/'paper/figures/mainraw64_supplement_20261008/qualitative/example_selection.json'
    old = read(oldpath)
    assert old['data_commit'] == COMMIT
    assert [s['source_index'] for s in old['samples']] == [418, 437, 271, 12]
    bindings = dict(old['source_bindings'])
    bindings[str(oldpath)] = sha(oldpath)
    mp = n/'common500_source_assets_v1/manifest.json'
    assert sha(mp) == old['source_manifest']['sha256']
    bindings[str(mp)] = sha(mp)
    assert all(sha(p) == h for p, h in bindings.items())
    population = read(mp)['records']
    assert len(population) == 500 and len({s['source_id'] for s in population}) == 500
    common_ids = {s['source_id'] for s in population}
    hdir = root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
    hifi = []
    for p in sorted((hdir/'source_checkpoints').glob('*.json')):
        v = read(p)
        rows = [r for r in v['rows'] if r['N'] == 1024 and r['snr_db'] == 13 and r['noise_seed'] == 2001]
        assert len(rows) == 2  # the historical Swin / HiFi pair
        hifi.append(dict(source_index=v['source_index'], source_id=v['source_id'],
                         checkpoint=str(p), checkpoint_sha256=sha(p),
                         methods=[r['method'] for r in rows], N=1024, snr_db=13, noise_seed=2001))
        bindings[str(p)] = sha(p)
    assert len(hifi) == 16
    intersection = sorted(common_ids & {s['source_id'] for s in hifi})
    assert intersection == [], 'HiFi coverage changed: review the intersection before rendering'
    rows_images, provenance, same_rx = [], [], []
    for s in old['samples']:
        i, sid = s['source_index'], s['source_id']
        assert sha(s['archive']) == s['archive_sha256']
        cp = read(s['checkpoint'])
        assert cp['source_index'] == i and cp['source_id'] == sid
        with np.load(s['archive'], allow_pickle=False) as z:
            original = z['pixels']
        assert original.dtype == np.uint8 and original.shape == (3, 256, 256)
        images = {'source': original.transpose(1, 2, 0)}
        common = dict(source_index=i, source_id=sid, N=1024, snr_db=13,
                      selection_role=s['selection_role'])
        provenance.append(dict(common, method='source', noise_seed='', archive=s['archive'],
                               image_slot='', image_sha256=raw_sha(original),
                               hash_domain='raw_array', received_state_sha256='', status='available'))
        for method in ('SwinJSCC80k', 'P1024'):
            p = n/f'unified500_baseline_r2/{method}/source_checkpoints/{i:04d}.json'
            cp = read(p)
            assert cp['method'] == method and cp['source_id'] == sid and cp['source_index'] == i
            r = one([r for r in cp['rows'] if r['snr_db'] == 13 and r['noise_seed'] == 2001])
            assert r['N'] == 1024 and r['source_id'] == sid and r['method'] == method
            archive = cp['float_reconstructions']['path']
            assert sha(archive) == cp['float_reconstructions']['sha256'] == bindings[archive]
            with np.load(archive, allow_pickle=False) as z:
                assert np.array_equal(z['source_rgb'], original.astype(np.float32)/255)
                a = z['images'][r['image_slot']]
            assert baseline_sha(a) == r['image_sha256']
            images[method] = display(a)
            provenance.append(dict(common, method=method, noise_seed=2001, archive=archive,
                                   image_slot=r['image_slot'], image_sha256=r['image_sha256'],
                                   hash_domain='baseline_float_rgb', received_state_sha256='', status='available'))
        raw = read(n/f'unified500_same_rx_parallel_images_r2/sources/{i:04d}.json')
        for method, family, arm in (
            ('RAW64_PARTIAL_VAR_COMPLETION', 'PARTIAL', 'VAR_completion'),
            ('RAW64_WHOLE_VAR_COMPLETION', 'WHOLE', 'VAR_completion'),
            ('RAW64_PARTIAL_DIRECT_DC', 'PARTIAL', 'direct_Dc_missing_residual_zero'),
        ):
            r = one([r for r in raw if r['snr_db'] == 13 and r['noise_seed'] == 6201 and family in r['families']])
            assert r['source_id'] == sid and r['source_index'] == i and r['transmission']['N'] == 1024
            assert r['reference_evidence']['archive_sha256'] == s['archive_sha256']
            proof, ref = r['same_RX_visual_proof'], r['same_RX_images'][arm]
            assert proof['same_received_information'] and not proof['target_used_for_generation']
            assert not proof['TX_truth_used_for_generation']
            assert sha(ref['image_archive']) == bindings[ref['image_archive']]
            with np.load(ref['image_archive'], allow_pickle=False) as z:
                a = z['images'][ref['image_slot']]
            assert raw_sha(a) == ref['image_sha256'] == proof['image_sha256'][arm]
            images[method] = display(a)
            provenance.append(dict(common, method=method, noise_seed=6201,
                                   archive=ref['image_archive'], image_slot=ref['image_slot'],
                                   image_sha256=ref['image_sha256'], hash_domain='raw_array',
                                   received_state_sha256=proof['actual_state_sha256'], status='available'))
            same_rx.append(dict(source_index=i, method=method, m=r['m'], K=r['K'],
                                actual_state_sha256=proof['actual_state_sha256'],
                                model_identity_sha256=proof['model_identity_sha256'],
                                received_tokens_sha256=proof['received_tokens_sha256'],
                                same_received_information=proof['same_received_information']))
        pair = [p for p in same_rx if p['source_index'] == i and 'PARTIAL' in p['method']]
        for key in ('actual_state_sha256', 'model_identity_sha256', 'received_tokens_sha256'):
            assert pair[0][key] == pair[1][key]
        provenance.append(dict(common, method='HiFiDiffCom', noise_seed='', archive='', image_slot='',
                               image_sha256='', hash_domain='', received_state_sha256='',
                               status='missing_same_source_N1024_SNR13'))
        rows_images.append(np.stack([images[k] for k in ARRAY_ORDER]))
    assert all(sha(p) == h for p, h in bindings.items()), 'Scientific input changed during export'
    np.savez_compressed(out/'display_images.npz', images=np.stack(rows_images))
    with (out/'provenance.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(provenance[0])); w.writeheader(); w.writerows(provenance)
    dump(out/'selection_and_integrity.json', dict(
        data_commit=COMMIT, N=1024, snr_db=13, samples=old['samples'],
        prior_selection_path=str(oldpath), prior_selection_sha256=sha(oldpath),
        array_order=ARRAY_ORDER, display_images_sha256=sha(out/'display_images.npz'),
        source_selection='Reuse previously frozen four holdout sources; no new quality ranking.',
        noise_selection='First registered draw only: continuous/Swin 2001; digital 6201. No best-noise selection.',
        identical_cross_method_noise=False, historical_hifi=hifi,
        hifi_common500_source_intersection=intersection, same_rx_proofs=same_rx,
        source_bindings=bindings, new_model_inference=0, new_channel_simulations=0,
        new_metric_calls=0, new_bootstrap=0))
    print(json.dumps({'collect':'complete','holdout_sources':4,'hifi_intersection':0,'mechanism_reconstructions':12}))


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plan = read(out/'selection_and_integrity.json')
    assert plan['N'] == 1024 and plan['snr_db'] == 13 and plan['array_order'] == ARRAY_ORDER
    assert sha(out/'display_images.npz') == plan['display_images_sha256']
    assert plan['hifi_common500_source_intersection'] == []
    with np.load(out/'display_images.npz', allow_pickle=False) as z:
        images = z['images']
    assert images.dtype == np.uint8 and images.shape == (4, 6, 256, 256, 3)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    groups = [
        ('comparison_N1024_13dB_HiFi_missing',
         ['source', 'SwinJSCC80k', 'P1024', 'HiFiDiffCom', 'RAW64_PARTIAL_VAR_COMPLETION']),
        ('mechanism_N1024_13dB',
         ['source', 'RAW64_WHOLE_VAR_COMPLETION', 'RAW64_PARTIAL_DIRECT_DC', 'RAW64_PARTIAL_VAR_COMPLETION'])]
    outputs = []
    for stem, columns in groups:
        count = len(columns)
        width, left, right, gap = 7.2, .30, .06, .06
        cell = (width-left-right-gap*(count-1))/count
        top, bottom, rowgap = .76, .12, .055
        height = top+bottom+4*cell+3*rowgap
        fig = plt.figure(figsize=(width, height), facecolor='white')
        fig.text(.52, 1-.17/height, r'$N = 1024$, SNR = 13 dB', ha='center', va='center', fontsize=9)
        for row in range(4):
            y = bottom+(3-row)*(cell+rowgap)
            fig.text(.11/width, (y+cell/2)/height, f'({chr(97+row)})', ha='center', va='center', fontsize=8)
            for col, method in enumerate(columns):
                x = left+col*(cell+gap)
                ax = fig.add_axes([x/width, y/height, cell/width, cell/height])
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if method == 'HiFiDiffCom':
                    ax.set_facecolor('white')
                    ax.text(.5, .5, 'Not available\nfor this holdout source', transform=ax.transAxes,
                            ha='center', va='center', fontsize=7.2, color='#555555', linespacing=1.5)
                    for spine in ax.spines.values():
                        spine.set_visible(True); spine.set_color('#BDBDBD'); spine.set_linestyle(':')
                else:
                    ax.imshow(images[row, ARRAY_ORDER.index(method)], interpolation='nearest')
                if row == 0:
                    fig.text((x+cell/2)/width, (y+cell+.055)/height, MAPPING[method],
                             ha='center', va='bottom', fontsize=8, linespacing=1.14)
        for ext in ('png', 'pdf', 'svg'):
            p = out/f'{stem}.{ext}'
            fig.savefig(p, dpi=600, facecolor='white'); outputs.append(p)
        plt.close(fig)
    native = out/'individual'; native.mkdir(exist_ok=True)
    public_names = ['original','swin_image_transmission','continuous_latent_jscc',
                    'partial_scale_with_var','full_scale_with_var','partial_scale_without_completion']
    for r in range(4):
        for c, name in enumerate(public_names):
            Image.fromarray(images[r, c]).save(native/f'example_{chr(97+r)}_{name}.png')
    selection = ('The same four previously frozen holdout sources are used in both figures. '
        'The first three were selected by a source-ID hash; the fourth was previously fixed as a '
        'diagnostic source using a classifier mismatch at 19 dB. That outcome is not assigned to its '
        '13 dB reconstruction here, and this selection is not a representative failure-rate sample. '
        'For every source and method, the first registered noise realization is retained without '
        'ranking the noise outcomes. The continuous and Swin systems use seed 2001, and the digital '
        'systems use seed 6201; these are not identical cross-method noise draws. ')
    external = ('Cached reconstructions at N = 1024 and SNR = 13 dB. Columns show the original image, '
        'adapted SwinJSCC image transmission, continuous latent-space joint source-channel coding, '
        'a missing-data placeholder for HiFi-DiffCom diffusion-prior recovery, and partial-scale '
        'digital transmission with VAR completion. The historical HiFi-DiffCom evaluation has '
        '16 source images at this working point, none of which belongs to the frozen 500-image holdout. '
        'Thus this is an incomplete five-column layout, not a completed five-method comparison. '
        'No other source, bandwidth or SNR is substituted into the missing column. SwinJSCC is the '
        'frozen adapted 80k-step baseline; no claim of an official optimal or fully converged model is made. '+selection)
    mechanism = ('Mechanism comparison at N = 1024 and SNR = 13 dB. Columns show the original image, '
        'complete-scale transmission with VAR completion, partial-scale transmission without generative '
        'completion, and partial-scale transmission with VAR completion. The last two columns use '
        'exactly the same actually received tokens and the same frozen decoder Dc. Without generative '
        'completion, untransmitted residual contributions are zero; unknown tokens are not replaced '
        'by codebook index zero, and no additional source-image ground truth is provided. Full-scale '
        'and partial-scale transmission have identical modulation and coding permissions, but use '
        'their own frozen selected configurations and actual receptions. Partial-scale selection '
        'allows fallback to a full-scale configuration; K need not be positive at every SNR. '
        'Complete-scale transmission here means the selected prefix of complete scales, not all source '
        'scales: at 13 dB it sends m = 8, K = 0 (255 tokens), whereas the partial-scale configuration '
        'sends m = 8, K = 9 (264 tokens). '+selection)
    (out/'captions.tex').write_text(
        '\\newcommand{\\CachedTransmissionComparisonCaption}{'+external+'}\n\n'
        '\\newcommand{\\CachedMechanismComparisonCaption}{'+mechanism+'}\n', encoding='utf-8')
    missing = dict(status='BLOCKED_NO_SHARED_HIFI_HOLDOUT_SOURCES', N=1024, snr_db=13,
                   historical_hifi_source_count=16, common500_intersection_count=0,
                   missing_cells=[dict(source_id=s['source_id'],source_index=s['source_index'],method='HiFiDiffCom')
                                  for s in plan['samples']], substitution=False, automatic_supplementary_inference=False)
    dump(out/'missing_data.json', missing)
    (out/'README.md').write_text(
        '# Cached N1024 / 13 dB visualizations\n\n'
        f'Data release: `{COMMIT}`. Read-only cache export; no model inference, channel simulation, '
        'metric evaluation, bootstrap or strategy changes. Original scientific files are unchanged.\n\n'
        '**The complete five-method holdout figure is blocked:** historical HiFi has zero shared '
        'source IDs with the frozen 500-image holdout. The five-column file explicitly shows four '
        'missing HiFi cells. The mechanism figure is complete. See `missing_data.json` for exact gaps.\n\n'
        +selection+'\n\nRows (a)–(d) correspond to holdout indices 418, 437, 271 and 12. '
        'Full source IDs, immutable cache hashes, image slots, noise seeds and same-reception proofs '
        'are in `provenance.csv` and `selection_and_integrity.json`. '
        'Internal experiment mappings are defined in the plotting script; visible labels use full names.\n\n'
        'All displayed images preserve the entire 256×256 source/reconstruction. Cached float RGB is '
        'rounded to 8-bit RGB for display only. No cropping, enhancement, resampling of the native '
        'images or per-method image selection is used. Native PNGs are in `individual/`. '
        'PDF/SVG embed these raster reconstructions with editable vector labels; PNG is 600 dpi.\n\n'
        'Re-render from the portable display bundle (NumPy, Pillow and matplotlib required):\n\n'
        '```sh\npython scripts/plot_paper_visual_comparisons.py --render\n```\n\n'
        'To verify and collect again on the original cache host, add `--collect --root /home/liulu/projects/VAR_COMM`.\n',
        encoding='utf-8')
    dump(out/'completion.json', dict(mechanism_status='COMPLETE',
         five_method_status='INCOMPLETE_HIFI_SOURCE_INTERSECTION_EMPTY',
         plotted_N=1024, plotted_snr_db=13, source_count=4, new_model_inference=0,
         new_channel_simulations=0, new_metric_calls=0, new_bootstrap=0,
         outputs={p.name:sha(p) for p in outputs}, portable_bundle_sha256=sha(out/'display_images.npz')))
    print(json.dumps({'render':'complete','exports':len(outputs),'hifi_missing_cells':4}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if not (args.collect or args.render):
        parser.error('Select --collect and/or --render')
    out = args.root/REL_OUT; out.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args.root, out)
    if args.render:
        render(out)
