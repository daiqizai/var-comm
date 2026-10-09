"""Read-only, source-matched development examples including actual HiFi at 13 dB.

No holdout rows are merged into this qualitative development-only artifact.
"""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from plot_paper_visual_comparisons import read, dump, sha, one, raw_sha, baseline_sha, display, MAPPING

OUT = 'paper/figures/hifi_development_N1024_13dB'
FIXED = [0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
ORDER = ['source','SwinJSCC80k','P1024','HiFiDiffCom',
         'RAW64_PARTIAL_VAR_COMPLETION','RAW64_WHOLE_VAR_COMPLETION','RAW64_PARTIAL_DIRECT_DC']


def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect(root, out, group=1):
    selected = FIXED[(group-1)*4:group*4]
    assert len(selected)==4
    n=root/'outputs/MAIN-RAW64-20261007'
    devdir=n/'development_fixed16_cached_display_v2'
    manifest=read(devdir/'examples_manifest.json')
    done=read(devdir/'completion.json')
    assert done['status']=='RAW64_FIXED16_CACHED_RGB_DISPLAY_COMPLETE_V2' and not done['holdout_used']
    hdir=root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
    hd=read(hdir/'reconstruction_completion.json')
    assert hd['status']=='HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE'
    finalpath=n/'unified500_same_rx_parallel_images_r2/sources/0000.json'
    final=one([r for r in read(finalpath) if r['snr_db']==13 and r['noise_seed']==6201 and 'PARTIAL' in r['family_memberships']])
    bindings={str(finalpath):sha(finalpath)}

    def bind(path, expected=None):
        path=Path(path);h=sha(path)
        assert expected is None or h==expected, 'Changed scientific input: '+str(path)
        assert str(path) not in bindings or bindings[str(path)]==h
        bindings[str(path)]=h

    for p in (devdir/'examples_manifest.json',devdir/'completion.json',hdir/'reconstruction_completion.json'):
        bind(p)
    pd=root/'outputs/CONTENT-REAL-64QAM-20261006/H/p600_replay_v1'
    p_done=read(pd/'completion.json');bind(pd/'completion.json')
    assert p_done['status']=='P600_RECEIVED_LATENT_CONVNEXT_COMPLETE' and p_done['N']==1024
    assert manifest['source_indices']==FIXED
    pages={p['source_index']:p for p in manifest['pages']}
    availability=[]; evidence=[]; samples=[]; arrays=[]; mechanism_proofs=[]
    for i in FIXED:
        page=pages[i];sid=page['source_id']
        hp=hdir/f'source_checkpoints/{i:04d}.json';hc=read(hp)
        assert hc['source_id']==sid and hc['source_index']==i
        bind(hp,hd['outputs'][str(hp)])
        hr=[r for r in hc['rows'] if r['N']==1024 and r['snr_db']==13 and r['noise_seed']==2001]
        assert {r['method'] for r in hr}=={'SwinJSCC_new_shared','HiFiDiffCom_SwinJSCC'} and len(hr)==2
        for key in ('observed_sha256','transmitted_sha256','reference_sha256','selected_checkpoint_sha256'):
            assert hr[0][key]==hr[1][key]
        assert all(r['selected_step']==80000 and not r['label_conditioned'] for r in hr)
        pf=one([r for r in page['frames'] if r['column']=='P baseline' and r['snr_db']==13])
        rf=one([r for r in page['frames'] if r['column']=='PARTIAL VAR' and r['snr_db']==13])
        assert pf['noise_seed']==2001 and pf['image_key']=='P1024_snr13_noise2001'
        assert rf['noise_seed']==6201 and rf['candidate_id']==final['candidate_id']
        assert all(Path(p).exists() for p in (hc['float_reconstructions']['path'],pf['archive'],rf['archive']))
        availability.append(dict(source_index=i,source_id=sid,N=1024,snr_db=13,
                                 five_columns_available=True,selected_for_display=i in selected,
                                 pixel_hashes_verified=i in selected))
        if i not in selected:
            continue
        reference=page['reference'];bind(reference['archive'],reference['archive_sha256'])
        bind(reference['checkpoint'],reference['checkpoint_sha256'])
        with np.load(reference['archive'],allow_pickle=False) as z:
            original=z[reference['pixels_key']]
        assert original.dtype==np.uint8 and original.shape==(3,256,256)
        assert hashlib.sha256(original.tobytes()).hexdigest()==reference['preprocessing_id']
        assert all(r['preprocessing_id']==reference['preprocessing_id'] for r in hr)
        imgs={'source':original.transpose(1,2,0)}
        common=dict(source_index=i,source_id=sid,population='development',N=1024,snr_db=13)
        evidence.append(dict(common,method='source',noise_seed='',archive=reference['archive'],
                             key=reference['pixels_key'],image_slot='',image_sha256=raw_sha(original)))
        ha=hc['float_reconstructions']['path'];bind(ha,hc['float_reconstructions']['sha256'])
        assert hd['outputs'][ha]==bindings[ha]
        with np.load(ha,allow_pickle=False) as z:
            target=z['source_rgb'];cached=z['images'];slots=z['image_slots'].tolist()
            assert z['row_ids'].tolist()==[r['replay_row_id'] for r in hc['rows']]
            assert slots==hc['float_reconstructions']['image_slots']
            assert np.array_equal(np.rint(target*255).astype(np.uint8),original)
            for row,slot in zip(hc['rows'],slots):
                if row not in hr:
                    continue
                assert baseline_sha(target)==row['reference_sha256']
                assert baseline_sha(cached[slot])==row['image_sha256']
                method={'SwinJSCC_new_shared':'SwinJSCC80k','HiFiDiffCom_SwinJSCC':'HiFiDiffCom'}[row['method']]
                imgs[method]=display(cached[slot])
                evidence.append(dict(common,method=method,noise_seed=2001,archive=ha,key='images',
                                     image_slot=slot,image_sha256=row['image_sha256']))
        pcp=pd/f'source_checkpoints/{i:04d}.json';pc=read(pcp)
        assert pc['source_id']==sid and pc['status']=='P600_RECEIVED_REPLAY_SOURCE_COMPLETE'
        bind(pcp,p_done['outputs'][str(pcp)])
        ps=pd/f'sources/{i:04d}.json';bind(ps,pc['outputs'][str(ps)])
        pr=one([r for r in read(ps) if int(r['snr_db'])==13 and int(r['noise_seed'])==2001])
        assert int(pr['N'])==1024 and pr['source_id']==sid and pr['method']=='P1024'
        assert pr['model_id']=='P1024_step40000_12b979260ebf' and pr['p_replay_exact_RGB']
        assert pr['reference_sha256']==hr[0]['reference_sha256']
        bind(pf['archive'],pc['outputs'][pf['archive']])
        with np.load(pf['archive'],allow_pickle=False) as z:
            a=z[pf['image_key']]
        assert baseline_sha(a)==pf['image_sha256']==pr['image_sha256']
        imgs['P1024']=display(a)
        evidence.append(dict(common,method='P1024',noise_seed=2001,archive=pf['archive'],
                             key=pf['image_key'],image_slot='',image_sha256=pf['image_sha256']))
        rp=n/f'development_same_rx_images_r2/sources/{i:04d}.json'
        bind(rp,manifest['input_bindings'][str(rp)]);raw=read(rp)
        for method,family,arm in (
            ('RAW64_PARTIAL_VAR_COMPLETION','PARTIAL','VAR_completion'),
            ('RAW64_WHOLE_VAR_COMPLETION','WHOLE','VAR_completion'),
            ('RAW64_PARTIAL_DIRECT_DC','PARTIAL','direct_Dc_missing_residual_zero')):
            r=one([r for r in raw if r['snr_db']==13 and r['noise_seed']==6201 and family in r['family_memberships']])
            assert r['source_id']==sid and r['transmission']['N']==1024
            proof=r['same_RX_visual_proof'];ref=r['same_RX_images'][arm]
            assert proof['same_received_information'] and not proof['target_used_for_generation'] and not proof['TX_truth_used_for_generation']
            assert proof['model_identity_sha256']==final['same_RX_visual_proof']['model_identity_sha256']
            if family=='PARTIAL':
                for key in ('candidate_id','profile_id','m','K','modulation'):
                    assert r[key]==final[key]
            bind(ref['image_archive'],manifest['input_bindings'][ref['image_archive']])
            with np.load(ref['image_archive'],allow_pickle=False) as z:
                a=z['images'][ref['image_slot']]
            assert raw_sha(a)==ref['image_sha256']==proof['image_sha256'][arm]
            imgs[method]=display(a)
            evidence.append(dict(common,method=method,noise_seed=6201,archive=ref['image_archive'],
                                 key='images',image_slot=ref['image_slot'],image_sha256=ref['image_sha256']))
            mechanism_proofs.append(dict(source_index=i,method=method,m=r['m'],K=r['K'],
                                         candidate_id=r['candidate_id'],actual_state_sha256=proof['actual_state_sha256'],
                                         received_tokens_sha256=proof['received_tokens_sha256'],
                                         model_identity_sha256=proof['model_identity_sha256']))
        same=[r for r in mechanism_proofs if r['source_index']==i and 'PARTIAL' in r['method']]
        assert same[0]['actual_state_sha256']==same[1]['actual_state_sha256']
        samples.append(dict(source_index=i,source_id=sid,preprocessing_id=reference['preprocessing_id']))
        arrays.append(np.stack([imgs[k] for k in ORDER]))
    assert all(sha(p)==h for p,h in bindings.items())
    assert [r['source_index'] for r in samples]==selected
    np.savez_compressed(out/'display_images.npz',images=np.stack(arrays))
    csv_write(out/'availability_16.csv',availability);csv_write(out/'provenance.csv',evidence)
    dump(out/'selection_and_integrity.json',dict(population='development',N=1024,snr_db=13,
        group=group, selection=f'Group {group} of four consecutive entries in the historical fixed16 order, without image-quality or best-noise ranking.',
        samples=samples,all_fixed_indices=FIXED,array_order=ORDER,
        noise_seeds={'SwinJSCC80k':2001,'P1024':2001,'HiFiDiffCom':2001,'digital':6201},
        swin_hifi_same_received_waveform=True,all_methods_same_noise=False,
        final_partial_policy_and_visual_model_match=True,mechanism_proofs=mechanism_proofs,
        source_bindings=bindings,display_images_sha256=sha(out/'display_images.npz'),
        new_model_inference=0,new_channel_simulations=0,new_metric_calls=0,new_bootstrap=0))
    print(json.dumps({'matched_development_sources':16,'exported_sources':selected,'reconstruction_cells':24}))


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plan=read(out/'selection_and_integrity.json')
    group=plan.get('group',1)
    selected=[r['source_index'] for r in plan['samples']]
    assert selected==FIXED[(group-1)*4:group*4]
    assert plan['population']=='development' and plan['array_order']==ORDER
    assert sha(out/'display_images.npz')==plan['display_images_sha256']
    with np.load(out/'display_images.npz',allow_pickle=False) as z:
        images=z['images']
    assert images.shape==(4,7,256,256,3) and images.dtype==np.uint8
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    outputs=[]
    groups=[('five_methods_N1024_13dB_development',ORDER[:5]),
            ('mechanism_N1024_13dB_development',[ORDER[i] for i in (0,5,6,4)])]
    for stem,columns in groups:
        width,left,right,gap=7.2,.30,.06,.06
        cell=(width-left-right-gap*(len(columns)-1))/len(columns)
        top,bottom,rowgap=.88,.12,.055
        height=top+bottom+4*cell+3*rowgap
        fig=plt.figure(figsize=(width,height),facecolor='white')
        fig.text(.52,1-.15/height,'Development examples',ha='center',va='center',fontsize=8)
        fig.text(.52,1-.34/height,r'$N = 1024$, SNR = 13 dB',ha='center',va='center',fontsize=9)
        for row in range(4):
            y=bottom+(3-row)*(cell+rowgap)
            fig.text(.11/width,(y+cell/2)/height,f'({chr(97+row)})',ha='center',va='center',fontsize=8)
            for col,method in enumerate(columns):
                x=left+col*(cell+gap)
                ax=fig.add_axes([x/width,y/height,cell/width,cell/height])
                ax.imshow(images[row,ORDER.index(method)],interpolation='nearest')
                ax.set_axis_off()
                if row==0:
                    fig.text((x+cell/2)/width,(y+cell+.055)/height,MAPPING[method],ha='center',va='bottom',fontsize=8,linespacing=1.14)
        for ext in ('png','pdf','svg'):
            p=out/f'{stem}.{ext}';fig.savefig(p,dpi=600,facecolor='white');outputs.append(p)
        plt.close(fig)
    native=out/'individual';native.mkdir(exist_ok=True)
    names=['original','swin','continuous_latent_jscc','hifi_diffcom','partial_scale_with_var',
           'complete_scales_with_var','partial_scale_without_completion']
    for row in range(4):
        for col,name in enumerate(names):
            Image.fromarray(images[row,col]).save(native/f'example_{chr(97+row)}_{name}.png')
    shared=('Development-only qualitative examples at N = 1024 and SNR = 13 dB. '
        f'Rows follow entries {(group-1)*4+1} through {group*4} of the pre-existing fixed16 source order '
        f'(indices {", ".join(map(str,selected))}). '
        'All columns match the same original uint8 source pixels. Neither source selection nor noise selection '
        'uses reconstruction quality. The fixed first noise realization is used: seed 2001 for the continuous, '
        'SwinJSCC and HiFi-DiffCom systems, and 6201 for digital transmission. SwinJSCC and HiFi-DiffCom share '
        'the exact received waveform; cross-family noise draws are different. These images are development '
        'examples and do not provide an independent evaluation on the unified 500-image holdout. '
        'All images come from verified existing reconstruction caches; no inference, channel simulation or '
        'metric evaluation was repeated. ')
    five=(shared+'Columns show the original, adapted SwinJSCC image transmission, continuous latent-space '
        'joint source-channel coding, adapted HiFi-DiffCom diffusion-prior recovery, and partial-scale digital '
        'transmission with VAR completion. SwinJSCC uses the frozen 80k-step model and a 256-use protected '
        'header plus 768 body uses. HiFi-DiffCom uses this received frame and the frozen unconditional '
        'diffusion prior with its full posterior schedule. It has no label or text side information. '
        'The digital configuration at this point is m = 8, K = 9; its selected policy and frozen visual-model '
        'identity match the final release. No official-optimal or full-convergence claim is made for the adaptations.')
    mechanism=(shared+'Columns show the original, complete-scale transmission with VAR completion, partial-scale '
        'transmission without generative completion, and partial-scale transmission with VAR completion. '
        'Complete-scale means the selected prefix of complete scales (m = 8, K = 0, 255 tokens), not all source '
        'scales. Partial-scale transmission sends m = 8, K = 9 (264 tokens). Both digital configurations have '
        'the same modulation and coding permissions. The last two columns use the same actual received '
        'tokens and frozen decoder Dc; absent residual contributions are zero in direct decoding, not '
        'codebook index zero, and no extra source-image ground truth is supplied.')
    (out/'captions.tex').write_text('\\newcommand{\\HiFiDevelopmentComparisonCaption}{'+five+'}\n\n'
                                  '\\newcommand{\\HiFiDevelopmentMechanismCaption}{'+mechanism+'}\n',encoding='utf-8')
    (out/'README.md').write_text('# HiFi-matched development examples\n\n'
        '**Population: development, not the unified 500-image holdout.** All 16 historical fixed sources '
        'have matching N1024 / 13 dB cache entries for the five requested columns. '
        f'Group {group} of four consecutive entries in the original order was exported; no quality-based reselection.\n\n'
        +shared+'\n\n'+five[len(shared):]+'\n\n'+mechanism[len(shared):]+'\n\n'
        '## Files\n\n`availability_16.csv`: complete matched-source inventory; pixel verification is explicitly '
        'marked for the four exported sources. `provenance.csv`: exact archive/key/slot and original float-image '
        'hash per column. `selection_and_integrity.json`: source bindings, fixed noise and same-reception proofs. '
        '`individual/`: 28 native 256×256 PNG images. PDF/SVG preserve vector text over raster photographs; '
        'PNG previews are 600 dpi. No scientific result file or prior holdout figure was changed.\n\n'
        '## Reproduction\n\nWith NumPy, Pillow and matplotlib installed, from the repository root:\n\n'
        f'```sh\npython scripts/plot_paper_hifi_development_examples.py --group {group} --render\n```\n\n'
        'The small display bundle is portable. On the original cache host add `--collect` to revalidate '
        'the source caches before rendering. Internal method names remain in the scripts and provenance, '
        'not on the figure. The companion `plot_paper_visual_comparisons.py` supplies display helpers and labels.\n',encoding='utf-8')
    dump(out/'completion.json',dict(status='COMPLETE_DEVELOPMENT_CACHED_VISUALIZATION',N=1024,snr_db=13,
        matched_sources=16,displayed_sources=selected,group=group,population='development',hifi_missing_cells=0,
        new_model_inference=0,new_channel_simulations=0,new_metric_calls=0,
        outputs={p.name:sha(p) for p in outputs}))
    print(json.dumps({'render':'complete','exports':6,'hifi_cells':4}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--collect',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--group',type=int,choices=range(1,5),default=1);a=p.parse_args()
    if not (a.collect or a.render):p.error('Select --collect and/or --render')
    out=a.root/(OUT if a.group==1 else OUT+f'_group{a.group:02d}');out.mkdir(parents=True,exist_ok=True)
    if a.collect:collect(a.root,out,a.group)
    if a.render:render(out)
