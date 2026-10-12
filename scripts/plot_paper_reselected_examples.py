"""Replace three display rows, retaining the user-selected second row; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plot_paper_figures4papers import apply_publication_style, finalize_figure
import matplotlib.pyplot as plt

ORDER = ['source','SwinJSCC80k','P1024','HiFiDiffCom','RAW64_PARTIAL_VAR_COMPLETION',
         'RAW64_WHOLE_VAR_COMPLETION','RAW64_PARTIAL_DIRECT_DC']
NAMES = ['original','swin','continuous_latent_jscc','hifi_diffcom','partial_scale_with_var',
         'complete_scales_with_var','partial_scale_without_completion']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--group1',type=Path,default=root/'paper/figures/hifi_development_N1024_13dB')
    p.add_argument('--group2',type=Path,default=root/'paper/figures/hifi_development_N1024_13dB_group02')
    a=p.parse_args();out=root/'paper/figures/reselected_examples_20261012';out.mkdir(parents=True,exist_ok=True)
    plans=[];arrays=[];bindings={};provenance=[]
    for folder in (a.group1,a.group2):
        plan=json.loads((folder/'selection_and_integrity.json').read_text())
        assert plan['population']=='development' and plan['N']==1024 and plan['snr_db']==13
        assert plan['array_order']==ORDER
        assert sha(folder/'display_images.npz')==plan['display_images_sha256']
        with np.load(folder/'display_images.npz',allow_pickle=False) as z:images=z['images']
        assert images.shape==(4,7,256,256,3) and images.dtype==np.uint8
        plans.append(plan);arrays.append(images)
        bindings[folder.name]={'selection_sha256':sha(folder/'selection_and_integrity.json'),'display_sha256':sha(folder/'display_images.npz')}
        with (folder/'provenance.csv').open(newline='',encoding='utf-8-sig') as f:provenance+=list(csv.DictReader(f))
    selection=[(0,0),(1,1),(0,1),(0,2)]
    selected=np.stack([arrays[g][i] for g,i in selection]);samples=[plans[g]['samples'][i] for g,i in selection]
    indices=[s['source_index'] for s in samples];assert indices==[0,21,25,50]
    assert np.array_equal(selected[1],arrays[1][1])
    proofs=[x for plan in plans for x in plan['mechanism_proofs'] if x['source_index'] in indices]
    for index in indices:
        pair=[x for x in proofs if x['source_index']==index and 'PARTIAL' in x['method']]
        assert len(pair)==2
        assert pair[0]['actual_state_sha256']==pair[1]['actual_state_sha256']
        assert pair[0]['received_tokens_sha256']==pair[1]['received_tokens_sha256']
    rows=[x for x in provenance if int(x['source_index']) in indices]
    assert len(rows)==28
    rows.sort(key=lambda x:(indices.index(int(x['source_index'])),ORDER.index(x['method'])))
    with (out/'provenance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    native=out/'individual';native.mkdir(exist_ok=True)
    for row,letter in enumerate('abcd'):
        for col,name in enumerate(NAMES):Image.fromarray(selected[row,col]).save(native/f'example_{letter}_{name}.png')
    apply_publication_style()
    def render(cols,labels,stem,subtitle):
        n=len(cols);fig=plt.figure(figsize=(7,7*4/n+.69))
        gs=fig.add_gridspec(4,n,left=.035,right=.997,bottom=.007,top=.90,wspace=.022,hspace=.022)
        for row,letter in enumerate('abcd'):
            for col,idx in enumerate(cols):
                ax=fig.add_subplot(gs[row,col]);ax.imshow(selected[row,idx],interpolation='none');ax.set_axis_off()
                if row==0:ax.set_title(labels[col],fontsize=8,fontweight='bold',pad=5)
                if col==0:ax.text(-.035,.5,f'({letter})',transform=ax.transAxes,ha='right',va='center',fontsize=8)
        fig.text(.52,.985,'N = 1024  |  SNR = 13 dB',ha='center',va='top',fontsize=9,fontweight='bold')
        fig.text(.52,.957,subtitle,ha='center',va='top',fontsize=7,color='#4D4D4D')
        finalize_figure(fig,out/stem)
    render([0,1,2,3,4],['Original','SwinJSCC\n(adapted, 80k)','Latent continuous\nJSCC (40k)','HiFi-DiffCom\n(adapted)','Partial digital + VAR\n(ours)'],
           'fig_qualitative_reselected','Historical development examples · fixed first noise realization')
    render([0,5,6,4],['Original','Complete scales\n+ VAR','Partial scales\nwithout completion','Partial scales\n+ VAR (ours)'],
           'fig_mechanism_reselected','Development · last two columns share received tokens and frozen decoder')
    shared=('Development examples at N=1024 and SNR=13 dB. Display rows use source indices0,21,25,50. '
        'Source21 is retained in the second row at the user request; the other rows take the first three entries '
        'of the other available, previously verified four-source display bundle. This is a user-requested display '
        'revision, not an independently preregistered qualitative sample. No reconstruction-quality ranking or '
        'best-noise selection is used. All methods use the same source pixels and operating point. The fixed '
        'first noise is2001 for latent/Swin/HiFi and6201 for digital methods; cross-family draws differ. '
        'Swin and HiFi use the same received waveform. No inference, channel simulation or statistics were repeated. ')
    five=shared+('The latent and Swin checkpoints are historical40k and80k adaptations, not the final100k/200k '
        'baselines of the common500 performance plot. HiFi uses the existing unconditional diffusion-prior '
        'recovery at13 dB. These examples are development illustrations, not common500 holdout results.')
    mech=shared+('Complete scales means the selected complete-scale prefix (m=8,K=0), not every source scale. '
        'Partial transmission uses m=8,K=9. The last two columns use identical actual received tokens and '
        'the same frozen decoder. Direct decoding sets absent residual contributions to zero, does not use '
        'codebook index0 as an unknown token, and receives no extra source truth.')
    (out/'captions.tex').write_text('\\newcommand{\\ReselectedComparisonCaption}{'+five+'}\n\n\\newcommand{\\ReselectedMechanismCaption}{'+mech+'}\n',encoding='utf-8')
    record={'population':'development','N':1024,'snr_db':13,'samples':samples,'display_order':indices,
        'selection':'User retained row b/source21. Replacement rows are first three entries in other verified cached group. Not a new preregistration.',
        'row_b_all_seven_images_pixel_identical':True,'source_bindings':bindings,'mechanism_proofs':proofs,
        'noise_seeds':plans[0]['noise_seeds'],'all_methods_same_noise':False,'new_scientific_calls':0,
        'individual_sha256':{f.name:sha(f) for f in native.glob('*.png')},
        'outputs':{f.name:sha(f) for f in out.iterdir() if f.suffix in ('.png','.pdf','.svg')}}
    (out/'selection_and_integrity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# Reselected reconstruction examples\n\n'+five+'\n\n'+mech+
        '\n\nStyle follows the installed figures4papers scientific-figure-making skill. Original panels are retained. '
        'Both panels have PDF/SVG and600dpi PNG. Photo pixels are unchanged; labels are vector. '
        'The28 individual PNGs are retained in the local delivery package.\n\nReproduce with the two original '
        'verified display bundles (no models):\n\n```sh\npython scripts/plot_paper_reselected_examples.py '
        '--group1 PATH_TO_FIRST_GROUP --group2 PATH_TO_SECOND_GROUP\n```\n',encoding='utf-8')
    print(json.dumps({'sources':indices,'retained_source21':True,'new_scientific_calls':0,'output':str(out)}))

if __name__=='__main__':main()
