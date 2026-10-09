"""A5: export existing fixed development receptions; no scientific execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from plot_paper_visual_comparisons import read, dump, sha, one, raw_sha, display, MAPPING

FIXED = [0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
METHODS = [('RAW64_WHOLE_VAR_COMPLETION','WHOLE','VAR_completion'),
           ('RAW64_PARTIAL_DIRECT_DC','PARTIAL','direct_Dc_missing_residual_zero'),
           ('RAW64_PARTIAL_VAR_COMPLETION','PARTIAL','VAR_completion')]
METRICS = ['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
REL = 'paper/figures/a5_mechanism_N1024_10_19dB_group02'

def csvout(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def collect(root,out):
    n=root/'outputs/MAIN-RAW64-20261007'
    directory=n/'development_fixed16_cached_display_v2'
    manifest=read(directory/'examples_manifest.json')
    assert manifest['source_indices']==FIXED
    metricsdir=n/'development_metrics_mixed_v1'
    md=read(metricsdir/'completion.json')
    bindings={}
    def bind(p, expected=None):
        p=Path(p);h=sha(p)
        assert expected is None or h==expected,str(p)
        bindings[str(p)]=h
    bind(directory/'examples_manifest.json');bind(metricsdir/'completion.json')
    finalpath=n/'unified500_same_rx_parallel_images_r2/sources/0000.json'
    bind(finalpath); final=read(finalpath)
    pages={p['source_index']:p for p in manifest['pages']}
    arrays=[];evidence=[];scores=[];states=[]
    selected=FIXED[4:8]
    for snr in [10,19]:
        imgs=[]
        for i in selected:
            page=pages[i];ref=page['reference'];sid=page['source_id']
            bind(ref['archive'],ref['archive_sha256'])
            with np.load(ref['archive'],allow_pickle=False) as z: original=z[ref['pixels_key']].copy()
            assert original.shape==(3,256,256) and original.dtype==np.uint8
            assert hashlib.sha256(original.tobytes()).hexdigest()==ref['preprocessing_id']
            rowimgs=[original.transpose(1,2,0)]
            rp=n/f'development_same_rx_images_r2/sources/{i:04d}.json'
            bind(rp,manifest['input_bindings'][str(rp)]); raw=read(rp)
            mp=metricsdir/f'sources/{i:04d}.json';bind(mp,md['outputs'][str(mp)]); mr=read(mp)
            common=dict(source_index=i,source_id=sid,population='development',N=1024,snr_db=snr)
            evidence.append(dict(common,method='source',noise_seed='',archive=ref['archive'],image_slot='',image_sha256=raw_sha(original)))
            partial=[]
            for method,family,arm in METHODS:
                r=one([r for r in raw if r['snr_db']==snr and r['noise_seed']==6201 and family in r['family_memberships']])
                f=one([r for r in final if r['snr_db']==snr and r['noise_seed']==6201 and family in r['family_memberships']])
                assert r['source_id']==sid and r['candidate_id']==f['candidate_id']
                proof=r['same_RX_visual_proof']; a=r['same_RX_images'][arm]
                assert proof['same_received_information'] and not proof['target_used_for_generation'] and not proof['TX_truth_used_for_generation']
                assert proof['model_identity_sha256']==f['same_RX_visual_proof']['model_identity_sha256']
                bind(a['image_archive'],manifest['input_bindings'][a['image_archive']])
                with np.load(a['image_archive'],allow_pickle=False) as z: image=z['images'][a['image_slot']].copy()
                assert raw_sha(image)==a['image_sha256']==proof['image_sha256'][arm]
                rowimgs.append(display(image))
                metric=one([x for x in mr if x['snr_db']==snr and x['noise_seed']==6201 and family in x['family_memberships'] and x['decoder_arm']==arm])
                assert metric['image_sha256']==a['image_sha256'] and metric['source_id']==sid
                scores.append(dict(common,method=method,noise_seed=6201,**{k:metric[k] for k in METRICS},
                    source_prediction=metric['convnext_source_prediction_name'],reconstruction_prediction=metric['convnext_prediction_name'],
                    agreement_is_accuracy=False,image_sha256=a['image_sha256']))
                evidence.append(dict(common,method=method,noise_seed=6201,archive=a['image_archive'],image_slot=a['image_slot'],image_sha256=a['image_sha256']))
                states.append(dict(common,method=method,m=r['m'],K=r['K'],candidate_id=r['candidate_id'],
                    actual_state_sha256=proof['actual_state_sha256'],model_identity_sha256=proof['model_identity_sha256']))
                if family=='PARTIAL':partial.append(proof['actual_state_sha256'])
            assert partial[0]==partial[1]
            imgs.append(np.stack(rowimgs))
        arrays.append(np.stack(imgs))
    np.savez_compressed(out/'display_images.npz',images=np.stack(arrays))
    csvout(out/'image_metrics.csv',scores);csvout(out/'provenance.csv',evidence);csvout(out/'reception_states.csv',states)
    refs=root/'results/main_raw64_20261007/final_common500_r6/development/reference_summary.json'
    bind(refs);rs=read(refs)
    assert len(rs)==126 and all(r['source_count']==100 for r in rs)
    csvout(out/'reference_summary_all_metrics.csv',rs)
    csvout(out/'reference_summary_core.csv',[r for r in rs if r['metric'] in METRICS])
    assert all(sha(p)==h for p,h in bindings.items())
    dump(out/'selection.json',dict(source_indices=selected,source_ids=[pages[i]['source_id'] for i in selected],
        snrs=[10,19],noise_seed=6201,method_order=['source']+[x[0] for x in METHODS],population='development',
        fixed_order=FIXED,selection='Entries5-8 of historical fixed16, matching the last requested second image group; no outcome-based selection',
        final_policy_match=True,input_bindings=bindings,display_sha256=sha(out/'display_images.npz'),
        new_inference=0,new_channel=0,new_metrics=0,new_bootstrap=0))
    print(json.dumps(dict(status='COLLECTED',sources=4,snrs=[10,19],reconstruction_cells=24)))

def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=read(out/'selection.json');assert sha(out/'display_images.npz')==p['display_sha256']
    with np.load(out/'display_images.npz',allow_pickle=False) as z: images=z['images']
    assert images.shape==(2,4,4,256,256,3)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    outputs=[]
    for si,snr in enumerate([10,19]):
        width,left,right,gap,top,bottom,rowgap=7.2,.3,.06,.055,.83,.12,.055
        cell=(width-left-right-3*gap)/4;height=top+bottom+4*cell+3*rowgap
        fig=plt.figure(figsize=(width,height),facecolor='white')
        fig.text(.52,1-.15/height,'Fixed development examples',ha='center',fontsize=8)
        fig.text(.52,1-.34/height,f'$N = 1024$, SNR = {snr} dB',ha='center',fontsize=9)
        for row in range(4):
            y=bottom+(3-row)*(cell+rowgap)
            fig.text(.11/width,(y+cell/2)/height,f'({chr(97+row)})',ha='center',va='center')
            for col,method in enumerate(p['method_order']):
                x=left+col*(cell+gap);ax=fig.add_axes([x/width,y/height,cell/width,cell/height])
                ax.imshow(images[si,row,col],interpolation='nearest');ax.set_axis_off()
                if row==0:fig.text((x+cell/2)/width,(y+cell+.055)/height,MAPPING[method],ha='center',va='bottom',fontsize=8,linespacing=1.12)
        for ext in ['png','pdf','svg']:
            path=out/f'mechanism_N1024_{snr}dB_development.{ext}';fig.savefig(path,dpi=600,facecolor='white');outputs.append(path)
        plt.close(fig)
    caption=('Fixed development examples at N = 1024 and SNR = 10 or 19 dB, using entries 5--8 of the historical fixed16 order. '
       'The same source images and the fixed noise seed 6201 are used for all digital arms; no best-noise selection is performed. '
       'Columns show the original, complete-scale transmission with VAR completion, partial-scale transmission without completion, '
       'and partial-scale transmission with VAR completion. The two partial-scale arms share actual received tokens and frozen decoder Dc. '
       'Direct decoding assigns zero contribution to absent residuals; it does not fill unknown tokens with codebook index zero or use extra source truth. '
       'Both transmission policies have the same modulation and coding permissions and match the final frozen policies. '
       'All images and per-image metrics are read from existing float reconstruction caches. These qualitative development samples are not an independent holdout evaluation.')
    (out/'captions.tex').write_text('\\newcommand{\\MechanismTenNineteenCaption}{'+caption+'}\n',encoding='utf-8')
    (out/'README.md').write_text('# Fixed mechanism examples\n\n'+caption+'\n\n'
       'Prediction names in image_metrics.csv are automatic ConvNeXt labels, not verified semantic error annotations. '
       'Agreement is stored in its original 0--1 units and is not classification accuracy. '
       'reference_summary_all_metrics.csv copies the existing six-condition, 100-source control statistics without new bootstrap. '
       'These reference levels do not calibrate DINO similarity into a percentage of semantic correctness.\n\n'
       'Reproduce the portable display: `python scripts/plot_paper_mechanism_10_19.py --render`. '
       'On the original cache host, add `--collect`. PDF/SVG use vector text over the source photographs.\n',encoding='utf-8')
    dump(out/'completion.json',dict(status='COMPLETE',new_inference=0,new_channel=0,new_metrics=0,new_bootstrap=0,
        exports=6,outputs={p.name:sha(p) for p in outputs}))
    print(json.dumps(dict(status='RENDERED',exports=6)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--collect',action='store_true');p.add_argument('--render',action='store_true');a=p.parse_args()
    if not (a.collect or a.render):p.error('Choose --collect and/or --render')
    out=a.root/REL;out.mkdir(parents=True,exist_ok=True)
    if a.collect:collect(a.root,out)
    if a.render:render(out)
