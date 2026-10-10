#!/usr/bin/env python3
"""Render actual fixed16 missing-frame completions; never calls a model or PHY."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
REMOTE='/home/liulu/projects/VAR_COMM'
INPUTS={}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(p,expected=None):
    p=Path(p);actual=sha(p);require(expected is None or expected==actual,'Changed actual input: '+str(p))
    INPUTS[str(p.resolve())]=actual;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def one(rows):
    require(len(rows)==1,'Expected one exact source/method/workpoint record');return rows[0]
def display(x):
    require(x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1,'Actual float32 CHW RGB256 required')
    return np.rint(x.transpose(1,2,0)*np.float32(255)).astype(np.uint8)
def pixels(p):
    with Image.open(bind(p)) as im:a=np.asarray(im.convert('RGB')).copy()
    require(a.shape==(256,256,3),'Existing display must be256x256');return a
def csvout(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def mapped(remote,remote_base,local_base):
    path=Path(remote);base=Path(remote_base)
    require(path.is_relative_to(base),'Input is outside its registered artifact root: '+str(path))
    return Path(local_base)/path.relative_to(base)
def old_display(root):
    options=[root/'outputs/MAIN-RAW64-20261007/development_fixed16_cached_display_v2',
        root/'.research/main_raw64_20261007/take_over_v1/current/development_fixed16_cached_display_v2']
    folder=next((p for p in options if p.exists()),None);require(folder is not None,'Original fixed16 display asset directory required')
    done=read(folder/'completion.json');manifest=read(folder/'examples_manifest.json')
    require(manifest['source_indices']==list(FIXED) and not done['holdout_used'] and done['all_reported_methods_frozen_scope'],'Original fixed development identity differs')
    original_root=REMOTE+'/outputs/MAIN-RAW64-20261007/development_fixed16_cached_display_v2'
    for path,h in done['outputs'].items():bind(mapped(path,original_root,folder),h)
    images={};samples=[]
    public={'WHOLE VAR':('whole','Complete-scale transmission + VAR'),'PARTIAL direct Dc':('direct','Partial-scale transmission without completion'),
        'PARTIAL VAR':('partial','Partial-scale transmission + VAR (NeST-Com)'),
        'P baseline':('latent','Latent-space continuous JSCC'),'Swin80k':('swin','SwinJSCC-80k (adapted)')}
    for ordinal,page in enumerate(manifest['pages']):
        i=page['source_index'];require(i==FIXED[ordinal],'Original fixed order changed');folder_i=folder/f'{ordinal:02d}_source_{i:04d}'
        image=pixels(folder_i/'reference.png');images[i,'original',None]=image
        samples.append(dict(source_index=i,source_id=page['source_id'],fixed_order=ordinal+1,public_method='Original image',internal_method='source',
            snr_db='',noise_seed='',archive=str(folder_i/'reference.png'),reuse='EXISTING_DISPLAY',pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        for internal,(alias,label) in public.items():
            snrs=(1,4,10,19) if alias in ('whole','direct','partial') else (1,10,19) if alias=='latent' else (1,)
            for snr in snrs:
                frame=one([r for r in page['frames'] if r['column']==internal and r['snr_db']==snr])
                require(frame['noise_seed']==(6201 if alias in ('whole','direct','partial') else 2001),'Original fixed noise changed')
                p=folder_i/f'snr{snr}_{internal.replace(" ","_")}.png';image=pixels(p);images[i,alias,snr]=image
                samples.append(dict(source_index=i,source_id=page['source_id'],fixed_order=ordinal+1,public_method=label,internal_method=internal,
                    snr_db=snr,noise_seed=frame['noise_seed'],archive=str(p),original_archive=frame.get('archive'),array_key=frame.get('image_key'),
                    array_slot=frame.get('image_slot'),actual_state_sha256=frame.get('actual_state_sha256'),image_sha256=frame.get('image_sha256'),
                    reuse='EXISTING_DISPLAY',pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        for snr in (1,4,10,19):
            a=one([r for r in page['frames'] if r['column']=='PARTIAL VAR' and r['snr_db']==snr])
            b=one([r for r in page['frames'] if r['column']=='PARTIAL direct Dc' and r['snr_db']==snr])
            require(a['actual_state_sha256']==b['actual_state_sha256'],'Raw partial arms must share actual received tokens')
    return images,samples,manifest
def branch_completion(new_assets,request,method,wait_record):
    folder=new_assets/method;done=read(folder/'completion.json');attempt=read(folder/'attempt.json')
    require(done['status']=='T5_MISSING_FIXED16_BRANCH_COMPLETE' and done['method']==method and done['frame_count']==request['frame_caps'][method]
        and done['request_sha256']==sha(new_assets/'request.json')==attempt['request_sha256'] and done['packet_budget']['unresolved']==0,'Actual completed missing-frame branch required')
    launch=read(Path(wait_record)/'launch.json');exited=read(Path(wait_record)/'exit.json')
    require(launch['child_pid']==attempt['pid'] and launch['argv'][-2:]==['--method',method]
        and request['out']+'/request.json' in launch['argv'] and exited['actual_child_waited'] and exited['exit_code']==0,
        'Actual exact-child wait/exit0 proof required')
    require(launch['owner_script_sha256']=='dfeba4b30aa7cc3febe67253d584c18c6e1ede11f41dbbe84f6a7fdfdd829d89','Pinned actual-wait observer required')
    return done
def new_images(args,images,samples,manifest):
    base=Path(args.new_assets);r=read(base/'request.json')
    require(r['schema']=='WCL_T5_MISSING_FIXED16_FRAMES_V1' and r['source_indices']==list(FIXED) and r['snrs']==[1,10,19]
        and r['noise_seeds']==[2001] and not r['holdout_used'] and r['N']==1024,'Exact registered new fixed16 scope')
    source_ids={p['source_index']:p['source_id'] for p in manifest['pages']}
    for method,alias,label,wait in [('ADAPTIVE_BPG','bpg','Adaptive BPG + LDPC',args.bpg_wait_record),
                                    ('SWIN80K','swin','SwinJSCC-80k (adapted)',args.swin_wait_record)]:
        done=branch_completion(base,r,method,wait)
        frames=[f for f in r['frames'] if f['method']==method and f['action']=='GENERATE']
        require(done['frame_keys']==[f['key'] for f in frames],'Actual new frame order/coverage differs')
        for frame in frames:
            i,snr=frame['source_index'],frame['snr_db'];remote=r['out']+'/'+method+'/frames/'+frame['key']+'/completion.json'
            receipt=read(mapped(remote,r['out'],base),done['outputs'][remote]);require(receipt['frame']==frame and receipt['source_id']==source_ids[i],'New image source/workpoint differs')
            archive=r['out']+'/'+method+'/frames/'+frame['key']+'/reconstruction.npz';expected=receipt['outputs'][archive]
            require(done['outputs'][archive]==expected,'Actual frame archive not sealed in branch completion')
            p=bind(mapped(archive,r['out'],base),expected)
            with np.load(p,allow_pickle=False) as z:rgb=z['rgb'].copy();reference=z['source_rgb'].copy()
            require(np.array_equal(display(reference),images[i,'original',None]),'Actual new-method source differs from displayed source')
            image=display(rgb);images[i,alias,snr]=image
            samples.append(dict(source_index=i,source_id=source_ids[i],fixed_order=FIXED.index(i)+1,public_method=label,internal_method=method,
                snr_db=snr,noise_seed=2001,archive=str(p),original_archive=archive,array_key='rgb',array_slot='',
                request_sha256=done['request_sha256'],actual_receipt_sha256=sha(mapped(remote,r['out'],base)),status=receipt['status'],
                reuse='ACTUAL_NEW_MISSING_FRAME_RESULT',pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                outside_training_range=method=='SWIN80K' and snr==19))
    for i in FIXED:
        for snr in (1,10,19):
            for alias in ('bpg','swin','latent','partial'):require((i,alias,snr) in images,'Exact external comparison missing: '+str((i,alias,snr)))
    return r
def entropy_images(path,root,images,samples,manifest,freeze_path=None):
    m=read(path);require(m['status']=='T5_FROZEN_ENTROPY_FIXED16_DISPLAY_COMPLETE' and m['N']==1024 and m['source_indices']==list(FIXED)
        and m['snrs']==[4,10,19] and m['family'] in ('EC_STATIC_WHOLE','EC_VAR_WHOLE') and m['population']=='development_fixed16'
        and m['holdout_used'] is False and len(m['rows'])==48,'Complete frozen same-family4/10/19 development entropy manifest required')
    base=Path(root) if root else Path(path).parent
    def local(p):return Path(p) if Path(p).is_file() else mapped(p,m['artifact_root'],base)
    done=read(local(m['completion']['path']),m['completion']['sha256'])
    frozen=read(Path(freeze_path) if freeze_path else local(m['freeze']['path']),m['freeze']['sha256'])
    require(done['actual_children_waited'] and all(x==0 for x in done['worker_exit_codes']) and frozen['holdout_used_for_selection'] is False,
        'Actual-wait entropy completion and calibration-only frozen policy required')
    ids={p['source_index']:p['source_id'] for p in manifest['pages']};keys=set()
    for row in m['rows']:
        i,snr=row['source_index'],row['snr_db'];k=i,snr
        require(k not in keys and i in FIXED and snr in (4,10,19) and row['source_id']==ids[i] and row['family']==m['family'],'Exact single-family fixed entropy mapping')
        require(row['noise_seed']==m['noise_seed'],'One preregistered entropy noise seed per source/workpoint');keys.add(k)
        d=row['archive'];require(done['outputs'][d['path']]==d['sha256'],'Entropy reconstruction must be sealed by actual completion')
        p=bind(local(d['path']),d['sha256'])
        with np.load(p,allow_pickle=False) as z:
            array=z[row['array_key']];image=array.copy() if array.ndim==3 else array[int(row['array_slot'])].copy()
            reference=z[row['reference_key']].copy() if row.get('reference_key') else None
        if reference is not None:require(np.array_equal(display(reference),images[i,'original',None]),'Entropy source reference differs')
        require(m.get('image_hash_domain','T1_shared_image_sha')=='T1_shared_image_sha','Use the existing T1 actual image hash domain')
        a=np.ascontiguousarray(image)
        require(hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()==row['image_sha256'], 'Entropy actual float image identity differs')
        rgb=display(image);images[i,'entropy',snr]=rgb
        samples.append(dict(source_index=i,source_id=ids[i],fixed_order=FIXED.index(i)+1,public_method='Entropy-coded complete-scale transmission + VAR',
            internal_method=m['family'],snr_db=snr,noise_seed=row['noise_seed'],archive=str(p),array_key=row['array_key'],array_slot=row.get('array_slot'),
            actual_receipt_sha256=m['completion']['sha256'],frozen_policy_sha256=m['freeze']['sha256'],reuse='ACTUAL_FROZEN_ENTROPY_DISPLAY',
            pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),status=row.get('source_status'),
            actual_state_sha256=row.get('received_state_sha256'),visual_reuse=row.get('visual_reuse'),gray=row.get('gray')))
    require(keys=={(i,s) for i in FIXED for s in (4,10,19)},'No missing/favorable entropy examples')
    return m
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'svg.hashsalt':'wcl-t5-completed-fixed16-v1','figure.facecolor':'white','savefig.facecolor':'white'})
def page(images,snr,start,mechanism=False):
    aliases=['original','whole','direct','partial','entropy'] if mechanism else ['original','bpg','swin','latent','partial']
    labels=['Original image','Complete-scale\ntransmission\n+ VAR completion','Partial-scale\ntransmission\nwithout completion','Partial-scale\n+ VAR completion\n(NeST-Com)','Entropy-coded\ncomplete-scale\n+ VAR completion'] if mechanism else ['Original image','Adaptive BPG\n+ LDPC','SwinJSCC-80k\n(adapted)','Latent-space\ncontinuous JSCC','Partial-scale + VAR\n(NeST-Com)']
    width=7.;left=.29;right=.04;gap=.045;rowgap=.055;top=1.02 if mechanism else .86;bottom=.29 if not mechanism and snr==19 else .13
    cell=(width-left-right-4*gap)/5;height=top+bottom+4*cell+3*rowgap;fig=plt.figure(figsize=(width,height))
    fig.text(.515,1-.15/height,'Fixed development examples',ha='center',fontsize=9)
    fig.text(.515,1-.34/height,f'N = 1024, SNR = {snr} dB; original positions {start+1}–{start+4}',ha='center',fontsize=8.5)
    for c,label in enumerate(labels):fig.text((left+c*(cell+gap)+cell/2)/width,1-(.76 if mechanism else .65)/height,label,ha='center',va='center',fontsize=8.0,linespacing=1.15)
    for r,index in enumerate(FIXED[start:start+4]):
        y=bottom+(3-r)*(cell+rowgap);fig.text(.13/width,(y+cell/2)/height,f'{start+r+1:02d}',ha='center',va='center',fontsize=8)
        for c,alias in enumerate(aliases):
            ax=fig.add_axes([(left+c*(cell+gap))/width,y/height,cell/width,cell/height]);ax.set_axis_off()
            ax.imshow(images[index,alias,None if alias=='original' else snr],interpolation='nearest',rasterized=True)
    if not mechanism and snr==19:fig.text(.515,.12/height,"19 dB is outside Swin's training and calibration range.",ha='center',fontsize=7.7)
    return fig
def render(images,out,mechanism):
    snrs=(4,10,19) if mechanism else (1,10,19)
    for snr in snrs:
        stem=('fig_t5_mechanism_entropy' if mechanism else 'fig_t5_external_completed')+f'_N1024_{snr}dB'
        with PdfPages(out/(stem+'_all16.pdf')) as pdf:
            for start in range(0,16,4):
                fig=page(images,snr,start,mechanism);name=stem+f'_page{start//4+1:02d}'
                for ext in ('pdf','svg','png'):fig.savefig(out/(name+'.'+ext),dpi=600 if ext=='png' else 150)
                pdf.savefig(fig);plt.close(fig)
CAPTIONS=r'''% All examples retain fixed development ordering; no best-noise selection.
\newcommand{\WCLCompletedExternalCaption}{The sixteen previously fixed development sources at $N=1024$ and SNR 1, 10 or 19~dB, in the original order, with the first four entries designated for the main-text page. Columns show the source, adaptive BPG + LDPC, SwinJSCC-80k (adapted), latent-space continuous joint source--channel coding, and partial-scale digital transmission with VAR completion (NeST-Com). Previously completed images are reused; only registered missing Swin and adaptive BPG frames were generated under their original frozen policies. External methods use seed 2001 and the raw-token system uses its original seed 6201, each in its own frozen noise namespace. Matched sources and workpoints do not imply identical received observations. Failed and gray outputs remain, and no method-specific favorable noise is selected. The adapted Swin checkpoint is the user-selected 80k milestone, not an official optimum or a convergence claim. 19~dB is outside Swin's training and calibration range. HiFi examples are not substituted from another SNR.}
\newcommand{\WCLCompletedMechanismCaption}{Fixed development examples at $N=1024$ and 4, 10 or 19~dB. Columns show the source, complete-scale raw transmission with VAR completion, partial-scale raw transmission without completion, partial-scale raw transmission with VAR completion (NeST-Com), and the separately frozen complete-scale entropy-coded baseline with VAR completion. The two partial-scale arms share the same actually received tokens and frozen $D_c$; the direct arm sets untransmitted residual contributions to zero, without filling unknown tokens by codebook index zero or receiving extra source truth. The entropy column uses only its independently registered, completed same-source and same-workpoint frames. Its exact family, frozen policy and noise seed are recorded in the sample manifest. No 1~dB entropy result is inferred from 4~dB. Source pairing across methods does not claim identical received waveforms. All sixteen sources are shown, including failures, and none is selected by quality.}
'''
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--new-assets',type=Path)
    p.add_argument('--bpg-wait-record');p.add_argument('--swin-wait-record')
    p.add_argument('--entropy-manifest');p.add_argument('--entropy-assets');p.add_argument('--entropy-freeze')
    p.add_argument('--mode',choices=('both','external','mechanism'),default='both')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'Fresh figure directory required; old90files are immutable');style()
    require(a.mode=='mechanism' or all((a.new_assets,a.bpg_wait_record,a.swin_wait_record)),'External figures require completed missing-frame assets and both actual-wait records')
    require(a.mode!='mechanism' or a.entropy_manifest,'Mechanism-only output requires completed entropy manifest')
    images,samples,manifest=old_display(a.root.resolve())
    if a.mode!='mechanism':new_images(a,images,samples,manifest)
    entropy=entropy_images(a.entropy_manifest,a.entropy_assets,images,samples,manifest,a.entropy_freeze) if a.entropy_manifest else None
    a.out.mkdir(parents=True)
    if a.mode!='mechanism':render(images,a.out,False)
    if entropy and a.mode!='external':render(images,a.out,True)
    csvout(a.out/'sample_manifest.csv',samples);(a.out/'captions.tex').write_text(CAPTIONS,encoding='utf-8')
    (a.out/'README.md').write_text('''# Completed fixed development examples

These figures read existing fixed16 display pixels and actual completed missing-frame NPZ/receipts. New inference, channel draws, metric calls and bootstrap in this plotting script: zero. Original source order, each method's frozen workpoint and noise are retained. Images remain native256-pixel observations; PDF/SVG labels stay vector/editable and PNG previews are600dpi. No old figure or scientific result is overwritten.

All16 examples appear on four pages perSNR; page01 is the stable first-four main-text subset. External1/10/19 includes adaptive BPG, Swin, latent continuous JSCC and NeST-Com. Swin19 explicitly retains its out-of-training-range note. HiFi is not substituted from another SNR.

The optional entropy mechanism figures require a complete separately frozen same-family development16x4/10/19 manifest and actual wait/exit evidence. Without that manifest no fifth column or mechanism figure is fabricated. The original raw4dB images exist and are reused;1dB is never filled by4dB. Internal mappings and seeds remain in sample_manifest.csv. The two partial arms share actual received tokens; distinct methods retain distinct historical noise namespaces.

Reproduce external: python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples.py --mode external --root <repository> --new-assets <missing_v1 mirror> --bpg-wait-record <actual BPG observer directory> --swin-wait-record <actual Swin observer directory> --out <new empty output directory>

Reproduce mechanism only: python experiments/wcl-evidence-closure-20261009/scripts/plot_t5_completed_examples.py --mode mechanism --root <repository> --entropy-manifest <actual completed display_manifest.json> --entropy-assets <artifact mirror> --entropy-freeze <exact SHA-matched local selected_entropy_family.json> --out <new empty output directory>
'''+f'\nThis output mode: {a.mode}. Entropy family: {entropy["family"] if entropy else "not supplied"}.\n',encoding='utf-8')
    completion=dict(status='T5_ACTUAL_COMPLETED_EXAMPLES_RENDERED',script_sha256=sha(__file__),source_bindings=INPUTS,
        outputs={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},source_indices=list(FIXED),population='development',N=1024,
        external_snrs=[1,10,19] if a.mode!='mechanism' else [],entropy_mechanism_snrs=[4,10,19] if entropy and a.mode!='external' else [],entropy_missing=entropy is None,
        render_mode=a.mode,
        entropy_family=entropy['family'] if entropy else None,
        entropy_frozen_policy=entropy['freeze'] if entropy else None,
        new_inference=0,new_channel_draws=0,new_metrics=0,new_bootstrap=0,png_dpi=600,svg_text_editable=True,photos_are_raster_observations=True,
        no_old_figure_changed=True)
    write(a.out/'completion.json',completion);print(json.dumps(dict(status=completion['status'],files=len(completion['outputs']),entropy_missing=entropy is None)))
if __name__=='__main__':main()
