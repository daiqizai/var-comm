"""Render the user's candidates06,15,16,04 from verified historical display bundles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plot_paper_figures4papers import apply_publication_style, finalize_figure
import matplotlib.pyplot as plt

ORDER = ['source', 'SwinJSCC80k', 'P1024', 'HiFiDiffCom',
         'RAW64_PARTIAL_VAR_COMPLETION', 'RAW64_WHOLE_VAR_COMPLETION', 'RAW64_PARTIAL_DIRECT_DC']
NAMES = ['original', 'swin', 'continuous_latent_jscc', 'hifi_diffcom',
         'partial_scale_with_var', 'complete_scales_with_var', 'partial_scale_without_completion']
SELECTION = [(1, 1, 6, 21), (2, 2, 15, 92), (2, 3, 16, 95), (0, 3, 4, 75)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group1', type=Path, default=root/'paper/figures/hifi_development_N1024_13dB')
    parser.add_argument('--group2', type=Path, default=root/'paper/figures/hifi_development_N1024_13dB_group02')
    parser.add_argument('--group4', type=Path, default=root/'paper/figures/additional_candidates_20261012/group04')
    args = parser.parse_args()
    out = root/'paper/figures/user_selected_examples_20261012'
    out.mkdir(parents=True, exist_ok=True)
    plans, arrays, provenance, bindings = [], [], [], {}
    for number, folder in zip((1, 2, 4), (args.group1, args.group2, args.group4)):
        plan = json.loads((folder/'selection_and_integrity.json').read_text())
        assert (plan['population'], plan['N'], plan['snr_db']) == ('development', 1024, 13)
        assert plan['array_order'] == ORDER
        assert plan['display_images_sha256'] == sha(folder/'display_images.npz')
        assert plan['noise_seeds'] == {'SwinJSCC80k':2001,'P1024':2001,'HiFiDiffCom':2001,'digital':6201}
        assert plan['swin_hifi_same_received_waveform']
        with np.load(folder/'display_images.npz', allow_pickle=False) as z:
            images = z['images']
        assert images.shape == (4, 7, 256, 256, 3) and images.dtype == np.uint8
        arrays.append(images)
        plans.append(plan)
        bindings[f'group{number}'] = {name: sha(folder/name) for name in
                                    ('display_images.npz', 'selection_and_integrity.json', 'provenance.csv')}
        with (folder/'provenance.csv').open(newline='', encoding='utf-8-sig') as f:
            provenance.extend(csv.DictReader(f))
    selected = np.stack([arrays[g][row] for g,row,_,_ in SELECTION])
    samples = [plans[g]['samples'][row] for g,row,_,_ in SELECTION]
    indices = [s['source_index'] for s in samples]
    assert indices == [s[3] for s in SELECTION] == [21, 92, 95, 75]
    proofs = [r for plan in plans for r in plan['mechanism_proofs'] if r['source_index'] in indices]
    for index in indices:
        pair = [r for r in proofs if r['source_index'] == index and 'PARTIAL' in r['method']]
        assert len(pair) == 2
        for key in ('actual_state_sha256', 'received_tokens_sha256', 'model_identity_sha256'):
            assert pair[0][key] == pair[1][key]
    rows = [r for r in provenance if int(r['source_index']) in indices]
    assert len(rows) == 28 and len({(r['source_index'],r['method']) for r in rows}) == 28
    rows.sort(key=lambda r: (indices.index(int(r['source_index'])), ORDER.index(r['method'])))
    with (out/'provenance.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    native = out/'individual'
    native.mkdir(exist_ok=True)
    for row, letter in enumerate('abcd'):
        for col, name in enumerate(NAMES):
            path = native/f'example_{letter}_{name}.png'
            Image.fromarray(selected[row,col]).save(path)
            with Image.open(path) as im:
                assert np.array_equal(np.asarray(im), selected[row,col])
    apply_publication_style()
    def render(cols, labels, stem, subtitle):
        n = len(cols)
        fig = plt.figure(figsize=(7, 7*4/n+.69))
        gs = fig.add_gridspec(4,n,left=.035,right=.997,bottom=.007,top=.90,wspace=.022,hspace=.022)
        for row, letter in enumerate('abcd'):
            for col, idx in enumerate(cols):
                ax = fig.add_subplot(gs[row,col])
                ax.imshow(selected[row,idx],interpolation='none'); ax.set_axis_off()
                if row == 0: ax.set_title(labels[col],fontsize=8,fontweight='bold',pad=5)
                if col == 0: ax.text(-.035,.5,f'({letter})',transform=ax.transAxes,ha='right',va='center',fontsize=8)
        fig.text(.52,.985,'N = 1024  |  SNR = 13 dB',ha='center',va='top',fontsize=9,fontweight='bold')
        fig.text(.52,.957,subtitle,ha='center',va='top',fontsize=7,color='#4D4D4D')
        finalize_figure(fig,out/stem)
    render([0,1,2,3,4],['Original','SwinJSCC\n(adapted, 80k)','Latent continuous\nJSCC (40k)',
                       'HiFi-DiffCom\n(adapted)','Partial digital + VAR\n(ours)'],
           'fig_qualitative_selected','Historical development examples · fixed first noise realization')
    render([0,5,6,4],['Original','Complete scales\n+ VAR','Partial scales\nwithout completion','Partial scales\n+ VAR (ours)'],
           'fig_mechanism_selected','Development · last two columns share received tokens and frozen decoder')
    shared = ('User-selected qualitative examples from the historical development set at N=1024 and SNR=13 dB. '
              'Rows show the white shaggy dog, broccoli, pastries, and shopping cart (source indices21,92,95,75; '
              'candidate numbers06,15,16,04). The user selected these illustrative examples after viewing the '
              '16-source gallery; this is not a preregistered or representative quantitative sample. '
              'All methods use the same source image and operating point. The fixed first noise is2001 for '
              'latent/Swin/HiFi and6201 for digital methods; cross-family noise draws differ. Swin and HiFi '
              'share the same received waveform. No per-method best-noise selection or new scientific calls were made. ')
    five = shared + ('The latent40k and adapted Swin80k checkpoints are historical baselines, not the final100k/200k '
                     'models of the common500 performance figure. HiFi uses the existing unconditional diffusion '
                     'prior and full posterior schedule. No official-optimal or convergence claim is made.')
    mech = shared + ('Complete-scale transmission uses the selected prefix(m=8,K=0), not every source scale. '
                     'Partial transmission uses m=8,K=9. Both configurations have the same modulation and coding '
                     'permissions. The last two columns share actual received tokens and the frozen decoder Dc. '
                     'Direct decoding sets missing residual contributions to zero; unknown tokens are not '
                     'codebook index0, and no additional source-image ground truth is supplied.')
    (out/'captions.tex').write_text('\\newcommand{\\SelectedComparisonCaption}{'+five+'}\n\n'
                                  '\\newcommand{\\SelectedMechanismCaption}{'+mech+'}\n',encoding='utf-8')
    record = dict(population='development',N=1024,snr_db=13,samples=samples,source_order=indices,
                  candidate_order=[6,15,16,4],selection='User selected after inspecting the16-source gallery.',
                  source_bindings=bindings,mechanism_proofs=proofs,all28_native_images_pixel_identical=True,
                  noise_seeds=plans[0]['noise_seeds'],all_methods_same_noise=False,new_scientific_calls=0,
                  individual_sha256={p.name:sha(p) for p in native.glob('*.png')},
                  outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.pdf','.svg')})
    (out/'selection_and_integrity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# User-selected reconstruction panels\n\n'+five+'\n\n'+mech+
        '\n\nBoth panels have PDF/SVG/600dpi PNG; image pixels are unchanged and text remains vector. '
        'The28 native PNGs are in the local delivery. Internal method mapping and hashes remain in the script '
        'and provenance. Figures4papers typography is reused.\n\nReproduce with the verified input bundles:\n\n'
        '```sh\npython scripts/plot_paper_user_selected_examples.py --group1 PATH_TO_GROUP1 '
        '--group2 PATH_TO_GROUP2 --group4 PATH_TO_GROUP4\n```\n',encoding='utf-8')
    print(json.dumps({'source_order':indices,'candidate_order':[6,15,16,4],'new_scientific_calls':0}))

if __name__ == '__main__':
    main()
