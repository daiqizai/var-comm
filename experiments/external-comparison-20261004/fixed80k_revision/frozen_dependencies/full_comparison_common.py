"""Frozen contracts for the remaining five-method comparison and publication."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

VERSION='FULL-COMPARISON-DELIVERY-20261004-R1'
HERE=Path(__file__).resolve().parent
STAGES=('own_controls','score','cost_own','cost_external','cost_report','full_report','publish')
OWN_STAGES=('qualify','screen','calibrate','development','export-n1024')
FAMILIES=('P','D_U','M1','SwinJSCC','HiFiDiffCom')
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim',
    'ms_ssim','dino_specificity','resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024*1024),b''):digest.update(value)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def seal(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Immutable record changed: '+str(path))
    else:write(path,value)


def verify(bindings):
    require(isinstance(bindings,dict),'Missing file bindings')
    for path,digest in bindings.items():
        require(Path(path).is_absolute() and Path(path).is_file() and sha(path)==digest,'Bound input changed: '+path)


def process(pid):
    require(type(pid) is int and pid>1,'Invalid exact process owner')
    folder=Path('/proc')/str(pid)
    try:
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],state=fields[0],
            command=[v for v in (folder/'cmdline').read_bytes().decode(errors='replace').split('\0') if v])
    except FileNotFoundError:return None


def live(record,reader=process):
    require(type(record.get('pid')) is int and record['pid']>1 and str(record.get('start_ticks','')).isdigit(),
        'Missing exact PID/start_ticks')
    current=reader(record['pid'])
    return bool(current and current['state']!='Z' and str(current['start_ticks'])==str(record['start_ticks']))


def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    values=[v.strip() for v in value.splitlines() if v.strip()]
    require(all(v.isdecimal() for v in values),'Unreadable GPU ownership query')
    return [int(v) for v in values]


def locations(root):
    root=Path(root).resolve();out=root/'outputs/EXTERNAL-COMPARISON-20261004'
    return dict(root=root,out=out,delivery=out/'full_delivery',result=root/'results/external_comparison_20261004',
        final=out/'final_comparison/final_stage',cost=out/'receiver_cost',own=out/'own_controls')


def default_config(root):
    p=locations(root);root=p['root'];out=p['out']
    native=root/'outputs/UNIFIED-METRICS-20261002/environment/bin/python'
    swin=root/'experiments/external-baseline-positioning-20260916/.venv/bin/python'
    original=read(root/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002/m1_calibration_registration.json')
    outer_path=out/'controller_config.json';outer=read(outer_path)
    return dict(version=VERSION,root=str(root),output=str(p['delivery']),runtime=str(HERE),
        external_controller_config=str(outer_path),external_eval_config=outer['evaluation']['config'],
        fixed_examples=str(out/'fixed_examples.json'),native_python=str(native),metric_python=str(native),
        report_python=str(native),swin_python=str(swin),poll_seconds=15,
        native_environment={'CUDA_VISIBLE_DEVICES':'0','VAR_COMM_DECODER_GATE':original['identity']['decoder_gate'],
            'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'6',
            'PYTHONPATH':str(root/'experiments/var-latent-enhancement-20260917/mechanisms/src')},
        metric_environment=dict(outer['evaluation']['score_environment']),
        swin_environment=dict(outer['evaluation']['reconstruction_environment']))


def required_sources(config):
    runtime=Path(config['runtime'])
    names=('own_controls.py','own_controls_score.py','own_controls_cost.py','own_controls_report.py',
           'full_comparison_common.py','full_comparison_delivery.py','publish_full_comparison.py','test_full_comparison.py')
    require(all((runtime/n).is_file() for n in names),'All final stage sources must exist before registration')
    paths={runtime/n for n in names}
    for pattern in ('own_controls*','full_comparison*','publish_full_comparison*','external_*.py','hifi_*.py','swin_*'):
        paths.update(p for p in runtime.glob(pattern) if p.is_file() and p.suffix in ('.py','.md','.json'))
    paths.update(Path(config[k]) for k in ('external_controller_config','external_eval_config','fixed_examples',
        'native_python','metric_python','report_python','swin_python'))
    root=Path(config['root'])
    paths.update(root/p for p in ('tools/update_repository_manifest.py','tools/verify_repository.py',
        'tools/run_cpu_checks.py','experiments/unified-metrics-20261002/publish.py'))
    outer=read(config['external_controller_config']);verify(outer['bindings'])
    return {**outer['bindings'],**{str(p.resolve()):sha(p) for p in paths}}


def validate_config(config,check_files=True):
    p=locations(config['root'])
    require(config.get('version')==VERSION and Path(config['output'])==p['delivery']
        and Path(config['runtime']).is_absolute(),'Unexpected full comparison configuration')
    require(1<=config.get('poll_seconds',0)<=60,'Poll interval must be 1..60 seconds')
    for kind in ('native','metric','swin'):
        env=config[kind+'_environment']
        require(env.get('CUDA_VISIBLE_DEVICES')=='0' and all(isinstance(k,str) and isinstance(v,str) for k,v in env.items()),
            'GPU stages require a frozen CUDA0 environment')
        require('LD_PRELOAD' not in env and 'HISTORICAL_TORCH_METADATA_MANIFEST' not in env,'Unregistered runtime preload')
    for key in ('native_python','metric_python','report_python','swin_python','external_controller_config','external_eval_config','fixed_examples'):
        require(Path(config[key]).is_absolute(),'Explicit absolute dependency required')
    if check_files:
        verify(config['bindings'])
        require(required_sources(config)==config['bindings'],'Full comparison source/dependency inventory changed')
    return config


def stage_spec(config,stage,token):
    p=locations(config['root']);runtime=Path(config['runtime']);root=str(p['root'])
    if stage=='own_controls':
        args=['--root',root,'--stage','all','--protocol',str(runtime/'own_controls_protocol.json')]
        script='own_controls.py';kind='native';gpu=True
    elif stage=='score':args=['--root',root,'--launch-id',token];script='own_controls_score.py';kind='metric';gpu=True
    elif stage.startswith('cost_'):
        part=stage[5:];args=['--root',root,'--config',config['external_eval_config'],'--stage',part,'--launch-id',token]
        script='own_controls_cost.py';kind={'own':'native','external':'swin','report':'report'}[part];gpu=part!='report'
    elif stage=='full_report':
        args=['--config',config['external_eval_config'],'--fixed-examples',config['fixed_examples'],
              '--receiver-cost',str(p['cost']/'completion.json'),'--launch-id',token]
        script='own_controls_report.py';kind='report';gpu=False
    elif stage=='publish':
        args=['--config',str(p['delivery']/'config.json')];script='publish_full_comparison.py';kind='report';gpu=False
    else:raise RuntimeError('Unregistered comparison stage')
    env=dict(config.get(kind+'_environment',{})) if gpu else dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    env['PYTHONDONTWRITEBYTECODE']='1'
    return [config[kind+'_python'],'-B','-u',str(runtime/script),*args],env,gpu


def receipt_bindings(value):
    bindings={}
    for key in ('inputs','input_bindings','bindings','source_bindings','outputs'):
        if key in value:
            verify(value[key]);bindings.update(value[key])
    return bindings


def proof_path(config,stage):
    p=locations(config['root'])
    return {'score':p['own']/'score_completion.json','cost_own':p['cost']/'own_completion.json',
        'cost_external':p['cost']/'external_completion.json','cost_report':p['cost']/'completion.json',
        'full_report':p['final']/'completion.json','publish':p['delivery']/'publication.json'}.get(stage)


def stage_proof(config,stage,missing_ok=False):
    p=locations(config['root'])
    paths=([p['own']/(s+'_completion.json') for s in OWN_STAGES] if stage=='own_controls' else [proof_path(config,stage)])
    if not all(path.is_file() for path in paths):
        if missing_ok:return None
        raise RuntimeError('Missing complete stage: '+stage)
    proofs={}
    for path in paths:
        value=read(path)
        if stage=='own_controls':
            name=path.name.removesuffix('_completion.json')
            require(value.get('status')=='OWN_CONTROL_STAGE_COMPLETE' and value.get('stage')==name
                and value.get('synthetic') is False and value.get('training_updates')==0 and value.get('holdout_access') is False,
                'Own reconstruction stage is incomplete')
            if name in ('screen','calibrate'):require(value.get('development_read') is False,'Calibration accessed development')
            if name=='development':require(value.get('sources')==100 and value.get('rows')==1800,'N2048 coverage differs')
            if name=='export-n1024':require(value.get('sources')==100 and value.get('rows')==2700
                and value.get('exact_previous_rgb') is True,'Frozen N1024 coverage/parity differs')
        elif stage=='score':
            require(value.get('status')=='OWN_CONTROLS_METRICS_COMPLETE' and value.get('sources')==100
                and value.get('rows')==5400 and value.get('metric_groups')==18 and value.get('synthetic') is False
                and value.get('selection_uses_development') is False and value.get('holdout_access') is False
                and value.get('new_metric_offset_applied') is False,'Own unified scoring incomplete')
        elif stage in ('cost_own','cost_external'):
            expected=('OWN_RECEIVER_BENCHMARK_COMPLETE',72) if stage=='cost_own' else ('EXTERNAL_RECEIVER_BENCHMARK_COMPLETE',48)
            require(value.get('status')==expected[0] and value.get('rows')==expected[1]
                and value.get('synthetic') is False,'Real receiver benchmark incomplete')
        elif stage=='cost_report':
            require(value.get('status')=='OWN_CONTROLS_RECEIVER_COST_COMPLETE' and value.get('rows')==120
                and value.get('same_population_all5') is True and value.get('synthetic') is False
                and value.get('source_indices')==[0,1,2,3] and value.get('receiver_boundary')=='received_full_waveform_to_RGB'
                and value.get('cuda_synchronized') is True and value.get('includes_header_decode') is True
                and value.get('includes_TX') is False and value.get('includes_model_loading') is False
                and value.get('includes_metric_scoring') is False and len(value.get('summary',[]))==30,
                'All five receivers require the same 120-frame measured cost population')
        elif stage=='full_report':
            require(value.get('status')=='MATCHED_FIVE_METHOD_REPORT_COMPLETE' and value.get('synthetic') is False
                and value.get('full_plan_complete') is True and value.get('remaining_requirements')==[]
                and value.get('sources')==100 and value.get('rows')==9000 and value.get('method_groups')==30
                and value.get('metrics_per_group')==13 and value.get('paired_comparisons')==780
                and value.get('source_bootstrap_replicates')==10000 and value.get('fixed_sources')==16
                and value.get('png_figures')==24 and value.get('pdf_figures')==6 and value.get('figure_cells')==480
                and value.get('resource_figures')==2 and value.get('receiver_cost_complete') is True
                and value.get('reference_conditions_complete') is True and value.get('sample_selection_uses_quality') is False
                and value.get('holdout_access') is False,'Complete 9000-row matched report/figures/cost required')
        else:
            require(value.get('status')=='PUSHED' and value.get('checks')=='PASS' and value.get('full_plan_complete') is True
                and value.get('commit')==value.get('remote_commit'),'Final normal publication is not verified')
        # Publication's repository blobs are checked by its publisher, so a
        # later independent release_manifest edit is not a scientific change.
        receipt_bindings(value)
        proofs[str(path)]=dict(sha256=sha(path),status=value['status'])
    return proofs


def final_table_gate(config):
    path=locations(config['root'])['result']/'final_comparison/metrics_per_frame.csv'
    wanted={(i,n,s,seed,m) for i in range(100) for n in (1024,2048) for s in (1,7,13)
            for seed in (2001,2002,2003) for m in FAMILIES}
    seen=set()
    with path.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key=(int(row['source_index']),int(row['N']),int(row['snr_db']),int(row['noise_seed']),row['comparison_family'])
            require(key in wanted and key not in seen,'Final table contains duplicate or unregistered rows');seen.add(key)
            require(row['synthetic']=='False' and row['label_conditioned']=='False' and float(row['E'])==2*key[1],
                'Final table is synthetic, class conditioned or outside energy budget')
            require(all(math.isfinite(float(row[m])) for m in METRICS),'Missing/nonfinite final quality metric')
    require(seen==wanted,'Final table does not contain all 9000 actual paired rows')
    return {str(path):sha(path)}
