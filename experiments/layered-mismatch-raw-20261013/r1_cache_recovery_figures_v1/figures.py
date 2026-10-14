"""Fixed R1 calibration200 figures, only after authenticated CPU delivery.

No science, bootstrap, candidate ranking, or GO decision is computed here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
MASTER_SHA='ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c'
DELIVERY_SHA='f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01'
PUBLICATION_HELPER_SHA='94b6986e1db8a6082f0a02a6a91855fd79b926a390bb1551a1bc785d06bfd1b3'
ORIGINAL_MASTER_SHA='70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47'
FAILURE_AUDIT_SHA='daad4d721815607b9b9186ed33ded6b778c9a91b91d169afbfb5000f15add7d4'
DISPATCH_SHA='38a7a17dc2d6473972472b05a1fccd9c6954bba0269d013625ee25c28ce7207d'
HATS=(7,10)
METHODS=('U_two','S_nom','S_back3','S_robust','E_two')
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine')
LABELS={'U_two':'Two-group unequal protection','S_nom':'Single-group nominal',
        'S_back3':'Single-group 3 dB backoff','S_robust':'Single-group robust',
        'E_two':'Two-group equal protection'}
STYLE={'U_two':('#0F4D92','o','-'),'S_nom':('#B64342','s','--'),
       'S_back3':('#9A4D8E','^',':'),'S_robust':('#42949E','D','-.'),'E_two':('#8BCF8B','v',(0,(5,2,1,2)))}
AXIS_LABELS={'psnr_db':'PSNR (dB) ↑','lpips_alex':'LPIPS–Alex ↓','dinov2_vitl14_cosine':'DINOv2-L cosine ↑'}
LIMIT=10_000_000


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path=Path(path).absolute();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def checked(value):
    path=Path(value['path']);require(path.is_absolute() and '..' not in path.parts
        and all(not p.is_symlink() for p in (path,*path.parents)) and path.is_file(),'Invalid pinned file')
    require(sha(path)==value['sha256'],'Pinned SHA changed: '+str(path))
    if 'bytes' in value:require(type(value['bytes']) is int and value['bytes']==path.stat().st_size,'Pinned byte size changed')
    return path


def same_pin(left,right):return left['path']==right['path'] and left['sha256']==right['sha256']


def process_identity(pid):
    try:
        value=(Path('/proc')/str(pid)/'stat').read_text();tail=value[value.rfind(')')+2:].split()
        return dict(pid=int(pid),start_ticks=int(tail[19]),state=tail[0])
    except FileNotFoundError:return None


def identity(value):
    require(all(type(value.get(k)) is int and value[k]>0 for k in ('pid','start_ticks')),'PID and start ticks require positive integer evidence')
    return value['pid'],value['start_ticks']


def gone(value,probe):
    pid,ticks=identity(value);current=probe(pid)
    require(current is None or identity(current)!=(pid,ticks),'Original delivery process remains present, including zombie')


def unheld_lock(path):
    require(sys.platform.startswith('linux') and path.is_file() and not path.is_symlink(),'Missing actual Linux dispatcher lock')
    import fcntl
    with path.open('rb') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def zero_counters(value):
    return all(type(value.get(k)) is int and value[k]==0
        for k in ('scientific_calls','bootstrap_calls','reselection_calls'))


def recovery_provenance(audit,root,master):
    """Validate the already completed CPU audit, never reopen scientific runtime."""
    closure=audit['process_closure']
    original=dict(path=str(root/'results/layered_mismatch_raw_20261013/r1_execution_v1/registered/registration.json'),
                  sha256=ORIGINAL_MASTER_SHA)
    require(same_pin(closure['recovery_registration'],master)
        and same_pin(closure['original_registration'],original)
        and type(closure['original_owner_actual_exit']) is int and closure['original_owner_actual_exit']==1
        and type(closure['new_owner_actual_exit']) is int and closure['new_owner_actual_exit']==0
        and closure['all_original_and_new_processes_gone'] is True and zero_counters(closure),
        'Different mixed-stage CPU closure')
    for key,expected in (('inherited_successful_stages',('phy','projection')),
                         ('new_successful_stages',('visual','score','selection'))):
        stages=closure[key]
        require([row['stage'] for row in stages]==list(expected)
            and all(type(row['actual_exit']) is int and row['actual_exit']==0 for row in stages),
            'Incomplete recovered stage closure')
    failed=closure['original_failed_visual'];failure=closure['original_failure']
    require(failed['stage']=='visual' and type(failed['actual_exit']) is int and failed['actual_exit']==1
        and failure['original_failure_preserved'] is True
        and type(failure['failed_model_load_reservations']) is int and failure['failed_model_load_reservations']==1
        and all(type(failure[k]) is int and failure[k]==0 for k in
                ('completed_original_model_loads','completed_original_forward_calls','partial_images'))
        and failure['audit']['sha256']==FAILURE_AUDIT_SHA,'Original failure disclosure differs')


def guard_delivery(root,temporary,control,probe=process_identity):
    """Small exit/identity metadata first; then all manifest hashes; no quality parse."""
    root,temporary,control=map(Path,(root,temporary,control))
    require(control==temporary/'r1_cache_recovery_delivery_dispatch_v1','Unexpected delivery dispatch directory')
    entry=root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/deliver.py'
    master=root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_v1/registered/registration.json'
    expected_entry=dict(path=str(entry),sha256=DELIVERY_SHA);expected_master=dict(path=str(master),sha256=MASTER_SHA)
    checked(expected_entry);checked(expected_master)
    output=root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1';heavy=temporary/'r1_cache_recovery_delivery_v1'
    require(not (temporary/'r1_cache_recovery_delivery_v1_publication_failure.json').exists(),'Delivery publication failed')
    require(not (control/'failure.json').exists(),'Delivery dispatcher retained a failure')
    prepared=read(control/'prepared.json');launch=read(control/'launch.json');wait=read(control/'actual_wait.json')
    expected_dispatch=dict(path=str(root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_dispatch_v1/dispatch_once.py'),sha256=DISPATCH_SHA)
    expected_argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(entry),'--expected-registration',MASTER_SHA]
    for receipt in (prepared,launch,wait):
        require(receipt['schema']=='R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1'
                and same_pin(receipt['registration'],expected_master)
                and same_pin(receipt['delivery_entry'],expected_entry)
                and receipt['argv']==expected_argv and same_pin(receipt['dispatcher'],expected_dispatch),'Delivery execution binding differs')
    require(identity(launch['owner'])==identity(wait['owner']),'Delivery parent identity differs')
    require(identity(prepared['owner'])==identity(launch['owner']),'Prepared parent identity differs')
    for receipt in (prepared,launch,wait):
        require(receipt['CUDA_VISIBLE_DEVICES']=='' and type(receipt['scientific_calls']) is int
                and receipt['scientific_calls']==0 and receipt['max_seconds']==7200,
                'Unexpected delivery execution scope')
        checked(receipt['dispatcher'])
    require(same_pin(launch['dispatcher'],wait['dispatcher']),'Actual delivery dispatcher binding differs')
    require(wait['actual_wait'] is True and type(wait['returncode']) is int and wait['returncode']==0,
            'Actual delivery has not exited successfully')
    require(not any(wait.get(k,False) for k in ('terminated','killed','timed_out','timeout','stopped')),'Interrupted delivery')
    require(wait['metadata_error'] is None,'Delivery retained a metadata error')
    require(identity(wait['child'])==identity(launch['child']),'Wait does not match the actual child')
    gone(launch['owner'],probe);gone(launch['child'],probe);unheld_lock(control/'dispatch.lock')
    require(Path(wait['stdout']['path'])==control/'stdout.txt' and Path(wait['stderr']['path'])==control/'stderr.txt','Different delivery log paths')
    result=read(checked(wait['stdout']));checked(wait['stderr'])
    require(result['status']=='DELIVERED_R1_RECOVERED_CALIBRATION_ONLY' and result['output']==str(output)
            and result['heavy']==str(heavy) and zero_counters(result),'Delivery stdout is not a successful receipt')
    draft=Path(result['original_staging_preserved'])
    require(draft.parent==temporary and draft.name.startswith('r1_cache_recovery_delivery_v1.preparing_') and draft.is_dir()
            and not draft.is_symlink(),'Unexpected original delivery staging')
    require(not (draft/'publication_failure.json').exists(),'Preserved delivery staging records failure')
    complete=read(output/'completion.json');pending=output/'completion.pending.json'
    require(pending.is_file() and not pending.is_symlink() and not (output/'completion.json').is_symlink()
            and pending.stat().st_ino==(output/'completion.json').stat().st_ino
            and pending.stat().st_dev==(output/'completion.json').stat().st_dev
            and sha(pending)==sha(output/'completion.json'),'Final completion marker differs from local pending receipt')
    require(complete['status']=='COMPLETE_R1_CPU_DELIVERY' and complete['requires_actual_delivery_exit_zero'] is True
            and same_pin(complete['registration'],expected_master) and complete['R2']=='NOT_REGISTERED_NOT_RUN'
            and zero_counters(complete),'Different CPU delivery')
    manifest_path=checked(result['manifest']);require(manifest_path==output/'delivery_manifest.json','Unexpected delivery manifest')
    require(complete['manifest_sha256']==sha(manifest_path),'Delivery completion manifest SHA changed')
    manifest=read(manifest_path);require(same_pin(manifest['registration'],expected_master),'Manifest master differs')
    require(manifest['heavy_not_for_git']==str(heavy) and manifest['scientific_arrays_copied'] is False,'Manifest storage differs')
    seen=set();files={}
    for item in manifest['files']:
        path=Path(item['path'])
        require(path.parent in (output,heavy) and str(path) not in seen,'Foreign or duplicate manifest path')
        checked(item);seen.add(str(path));files[path.name]=item
    require(len(files)==len(seen),'Ambiguous delivery basenames')
    inputs={}
    for key in ('frozen_winner_summary_csv','selection'):
        name=manifest['copy_keys'][key];require(name in files,'Required selected output is absent')
        inputs[key]=files[name]
    audit=read(checked(files['closure_audit.json']))
    require(audit['status']=='PASS_R1_COMPLETED_CPU_DELIVERY' and same_pin(audit['registration'],expected_master)
            and zero_counters(audit)
            and audit['old_scientific_results_modified'] is False,'CPU delivery audit differs')
    recovery_provenance(audit,root,expected_master)
    return inputs,dict(registration=expected_master,delivery_entry=expected_entry,launch=pin(control/'launch.json'),
        actual_wait=pin(control/'actual_wait.json'),completion=pin(output/'completion.json'),manifest=pin(manifest_path),
        all_manifest_files_sha_verified=len(seen),actual_delivery_exit_zero=True,source_science_reaudited=False)


def load_tables(inputs):
    # Parse saved quality only after guard_delivery has returned successfully.
    with checked(inputs['frozen_winner_summary_csv']).open(newline='',encoding='utf-8-sig') as stream:raw=list(csv.DictReader(stream))
    selected=read(checked(inputs['selection']))
    require(selected['status']=='COMPLETE_R1_CALIBRATION_SELECTION' and selected['authenticated_actual_closure'] is True
            and selected['source_count']==200 and selected['noise_count']==3 and selected['no_bootstrap'] is True
            and selected['R2']=='NOT_REGISTERED_NOT_RUN' and selected['registration']['sha256']==MASTER_SHA,'Different calibration selection')
    winners={(r['method'],r['snr_hat']):r for r in selected['winners']}
    require(len(selected['winners'])==len(winners)==10 and set(winners)=={(m,h) for m in METHODS for h in HATS},'Frozen winner grid differs')
    expected={(m,h,t,k) for m in METHODS for h in HATS for t in (h-3,h,h+3) for k in METRICS};values={}
    fields={'method','snr_hat','snr_true','metric','mean','source_count','noise_count','candidate_id','receiver_rule'}
    require(len(raw)==90,'Expected exactly 90 saved metric cells')
    for row in raw:
        require(set(row)==fields and row['source_count']=='200' and row['noise_count']=='3','Unexpected quality CSV schema/denominators')
        key=(row['method'],int(row['snr_hat']),int(row['snr_true']),row['metric']);require(key in expected and key not in values,'Duplicate/foreign quality cell')
        value=float(row['mean']);require(math.isfinite(value),'Nonfinite metric mean')
        winner=winners[(key[0],key[1])]
        require(row['candidate_id']==winner['candidate_id'] and row['receiver_rule']==winner['receiver_rule'],'Curve does not use the saved frozen winner')
        values[key]=value
    require(set(values)==expected,'Incomplete quality grid')
    gate=selected['gate'];require(gate['status'] in ('GO_R2','NO_GO_STOP','ONE_POINT_PROMISING'),'Unknown saved gate')
    require(set(gate['comparisons'])=={'S_robust','S_back3'},'Saved GO comparison scope differs')
    go_rows=[]
    for reference in ('S_robust','S_back3'):
        comparison=gate['comparisons'][reference];require(set(comparison['per_hat'])=={'7','10'},'Saved per-hat comparison differs')
        for label,value in [('7',comparison['per_hat']['7']),('10',comparison['per_hat']['10']),('mean',comparison['mean'])]:
            require(type(value) in (int,float) and math.isfinite(value),'Nonfinite saved J difference')
            go_rows.append(dict(comparison='U_two_minus_'+reference,estimate=label,stored_DINO_J_difference=value,stored_gate=gate['status']))
    return dict(raw=raw,values=values,selection=selected,go_rows=go_rows,inputs=inputs)


def axis_limits(values):
    low,high=min(values),max(values);span=high-low
    pad=.08*span if span>0 else max(abs(low)*.08,.05)
    return low-pad,high+pad


def save_figure(fig,folder,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(folder/(name+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.035,
                    metadata={'Creator':'R1 calibration200 frozen-result plotting'} if ext=='pdf' else None)


def render(data,folder):
    # The caller has already fixed MPLCONFIGDIR to personal tmp and audited all data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['DejaVu Sans','Arial','Helvetica','sans-serif'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'legend.fontsize':8,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.usetex':False,
        'savefig.dpi':600,'axes.unicode_minus':True})
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.4),sharex='col',sharey='row')
    handles={};values=data['values']
    for i,metric in enumerate(METRICS):
        limits=axis_limits([v for (m,h,t,k),v in values.items() if k==metric])
        for j,hat in enumerate(HATS):
            ax=axes[i,j];xs=(hat-3,hat,hat+3)
            for method in METHODS:
                color,marker,line=STYLE[method]
                handle,=ax.plot(xs,[values[(method,hat,t,metric)] for t in xs],color=color,marker=marker,
                    linestyle=line,linewidth=1.35,markersize=4.5,markeredgewidth=.8,
                    markerfacecolor='white' if method=='E_two' else color,
                    markeredgecolor='#356235' if method=='E_two' else color,
                    zorder=5 if method=='U_two' else 3,label=LABELS[method])
                handles.setdefault(method,handle)
            ax.set_xticks(xs);ax.set_ylim(*limits);ax.grid(axis='y',color='#e1e1e1',linewidth=.55)
            ax.axvline(hat,color='#b0b0b0',linestyle=':',linewidth=.6,zorder=1)
            ax.set_axisbelow(True);ax.tick_params(labelsize=8)
            if j==0:ax.set_ylabel(AXIS_LABELS[metric])
            if i==0:ax.set_title(r'Estimated SNR $\hat{\gamma}$ = '+str(hat)+' dB')
            if i==2:ax.set_xlabel('True SNR (dB)')
    fig.legend([handles[m] for m in METHODS],[LABELS[m] for m in METHODS],loc='upper center',
               bbox_to_anchor=(.5,.997),ncol=2,columnspacing=1.2,handlelength=2.5)
    fig.text(.5,.005,'Calibration200 means; 3 noises per source; no confidence intervals',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.88),h_pad=1.3,w_pad=1.4)
    save_figure(fig,folder,'r1_calibration_quality');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,3.15))
    for reference,color,marker in (('S_robust','#42949E','D'),('S_back3','#9A4D8E','^')):
        rows=[r for r in data['go_rows'] if r['comparison']=='U_two_minus_'+reference]
        ax.plot((0,1,2),[r['stored_DINO_J_difference'] for r in rows],linestyle='none',
                marker=marker,markersize=6,color=color,label='Unequal protection minus '+LABELS[reference].lower())
    ax.set_xticks((0,1,2),('Estimate 7 dB','Estimate 10 dB','Two-estimate mean'))
    ax.set_ylim(*axis_limits([0.]+[r['stored_DINO_J_difference'] for r in data['go_rows']]))
    ax.axhline(0,color='#777777',linewidth=.65);ax.set_ylabel('Stored DINO-J difference')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=1)
    fig.text(.5,.01,'Stored calibration gate: '+data['selection']['gate']['status']+' | no confidence intervals',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.91));save_figure(fig,folder,'r1_saved_gate_comparisons');plt.close(fig)


def caption(data):
    return ('R1 calibration200 quality. The three rows report PSNR, LPIPS–Alex and DINOv2-L cosine; '
        'the columns fix the transmitter SNR estimate at 7 and 10 dB. The horizontal coordinate is the actual SNR. '
        'The dotted vertical guide marks the matched condition; the left point is a 3 dB overestimate at the transmitter, and the right point is a 3 dB underestimate. '
        'Each value is the mean over the fixed 200 calibration sources after averaging three registered noise realizations within each source. '
        'All five policies and receiver rules are the original saved calibration choices; no method was reselected for plotting. '
        'CRC-rejected, fixed-gray and legal wrong-ID receptions remain in the original means. Curves can coincide when saved policies or outcomes coincide; no jitter is applied. '
        'LPIPS retains its original sign and lower-is-better direction. These are non-blind calibration results, with no confidence intervals, bootstrap or ConvNeXt metric.\n\n'
        'Saved calibration-gate comparisons. The points reproduce the stored two-group minus robust-single and two-group minus fixed-backoff DINO-J differences for each estimate and their saved mean. '
        'J weights matched and transmitter-overestimate-by-3-dB conditions by 0.75 and 0.25. The true-SNR-above-estimate condition is pressure reporting only. '
        'The original stored gate is '+data['selection']['gate']['status']+'. This figure does not recompute the gate or establish a development/holdout claim, and does not run R2.\n')


def run(root,temporary,control):
    inputs,proof=guard_delivery(root,temporary,control);data=load_tables(inputs)
    entry=root/'experiments/layered-mismatch-raw-20261013/r1_delivery_v1/deliver.py'
    require(sha(entry)==PUBLICATION_HELPER_SHA,'Frozen publication helper changed')
    spec=importlib.util.spec_from_file_location('_r1_frozen_publish_helpers',entry);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    destination=root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_figures_v1';helper.require_absent(destination)
    claim=temporary/'r1_cache_recovery_figures_v1.claim.json';helper.write_json(claim,dict(registration=proof['registration'],scientific_calls=0,automatic_retry=False))
    helper.fsync_file(claim);claim_pin=dict(file=helper.pin(claim),identity=helper.filesystem_identity(claim,'file'))
    staging=temporary/('r1_cache_recovery_figures_v1.preparing_'+str(os.getpid())+'_'+str(time.time_ns()));staging.mkdir();light=staging/'light';light.mkdir()
    cache=staging/'matplotlib_cache';cache.mkdir();os.environ['MPLCONFIGDIR']=str(cache)
    phase='render'
    try:
        require(not any(x in sys.modules for x in ('torch','tensorflow','sionna','lpips')),'Plotting must use a clean CPU interpreter')
        render(data,light)
        shutil.copyfile(checked(inputs['frozen_winner_summary_csv']),light/'plot_data_quality.csv')
        with (light/'plot_data_gate.csv').open('x',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data['go_rows'][0]));writer.writeheader();writer.writerows(data['go_rows'])
        text=caption(data);(light/'CAPTIONS.txt').write_text(text,encoding='utf8')
        (light/'caption.tex').write_text(text.replace('_',r'\_').replace('%',r'\%').replace('&',r'\&'),encoding='utf8')
        helper.write_json(light/'input_audit.json',dict(proof=proof,inputs=inputs,quality_rows=90,stored_gate=data['selection']['gate']['status'],
            scientific_calls=0,bootstrap_calls=0,reselection_calls=0,gate_recomputed=False,visual_review='PENDING_REQUIRED'))
        artifacts=[]
        for path in sorted(light.iterdir()):
            require(path.is_file() and not path.is_symlink() and path.stat().st_size<LIMIT,'Artifact exceeds unchanged publication size limit')
            artifacts.append(dict(path=str(destination/path.name),sha256=sha(path),bytes=path.stat().st_size))
        helper.write_json(light/'figure_manifest.json',dict(registration=proof['registration'],files=artifacts,input_manifest=proof['manifest'],
            exports=['vector_PDF','editable_text_SVG','600dpi_PNG'],visual_review='PENDING_REQUIRED',scientific_calls=0))
        helper.write_json(light/'completion.json',dict(status='COMPLETE_R1_FIGURE_FILES_VISUAL_REVIEW_PENDING',registration=proof['registration'],
            manifest_sha256=sha(light/'figure_manifest.json'),requires_actual_plot_process_exit_zero=True,scientific_calls=0))
        phase='publish';directory_identity=helper.publish_directory(light,destination,claim_pin,exclude=('completion.json',))
        for item in artifacts:checked(item)
        require(sha(destination/'figure_manifest.json')==sha(light/'figure_manifest.json'),'Published figure manifest changed')
        helper.publish_completion(light/'completion.json',destination/'completion.json',directory_identity,claim_pin)
    except BaseException as error:
        helper.write_json(staging/'failure.json',dict(phase=phase,error=repr(error),errno=getattr(error,'errno',None),automatic_retry=False,
            scientific_calls=0,source_results_untouched=True));raise
    return dict(status='R1_FIGURE_FILES_READY_VISUAL_REVIEW_REQUIRED',output=str(destination),staging_preserved=str(staging),scientific_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-registration',required=True);args=parser.parse_args()
    require(args.expected_registration==MASTER_SHA and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixed master and hidden CUDA required')
    require(sys.platform.startswith('linux'),'Actual run requires the original Linux CPU environment')
    print(json.dumps(run(ROOT,TMP,TMP/'r1_cache_recovery_delivery_dispatch_v1'),sort_keys=True))


if __name__=='__main__':main()
