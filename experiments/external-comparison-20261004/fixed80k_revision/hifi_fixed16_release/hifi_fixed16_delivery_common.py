"""Control contracts for the registered 64-frame paired release, then stop."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
VERSION='HIFI-FIXED16-DELIVERY-20261004-R1'
STAGES=('reconstruct','score','report','publish')
OWN_FILES=('hifi_fixed16_delivery_common.py','hifi_fixed16_delivery.py','test_hifi_fixed16_delivery.py')
FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
METHODS=('SwinJSCC_new_shared','HiFiDiffCom_SwinJSCC')

def require(value,message):
    if not value:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def seal(path,value):
    if Path(path).exists():require(read(path)==value,'Immutable record changed: '+str(path))
    else:write(path,value)

def verify(bindings):
    require(isinstance(bindings,dict),'Missing immutable file bindings')
    for path,digest in bindings.items():require(Path(path).is_file() and sha(path)==digest,'Bound file changed: '+path)

def receipt_bindings(value):
    bindings={}
    for key in ('bindings','inputs','input_bindings','source_bindings','outputs'):
        if key in value:verify(value[key]);bindings.update(value[key])
    return bindings

def locations(root):
    root=Path(root).resolve();out=root/'outputs/EXTERNAL-COMPARISON-20261004';parent=out/'fixed80k_revision'
    child=parent/'hifi_fixed16_release'
    return dict(root=root,out=out,parent=parent,child=child,delivery=child/'delivery',evaluation=child/'evaluation',
        parent_config=parent/'delivery/config.json',early=parent/'swin_early_release')

def process(pid):
    require(type(pid)is int and pid>1,'Exact process PID required')
    try:
        folder=Path('/proc')/str(pid);fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],state=fields[0],command=[v for v in (folder/'cmdline').read_bytes().decode(errors='replace').split('\0') if v])
    except (FileNotFoundError,ProcessLookupError):return None

def live(owner,reader=process):
    require(type(owner.get('pid'))is int and str(owner.get('start_ticks','')).isdigit(),'Missing exact PID/start ticks')
    current=reader(owner['pid'])
    return bool(current and current['state']!='Z' and str(current['start_ticks'])==str(owner['start_ticks']))

def frozen_parent(config):
    p=locations(config['root']);sys.path.insert(0,str(p['parent']))
    import fixed80k_delivery_common as parent
    require(Path(parent.__file__).resolve()==p['parent']/'fixed80k_delivery_common.py','Unexpected parent import')
    require(sha(config['parent_config'])==config['parent_config_sha256'],'Frozen parent configuration changed')
    value=parent.validate_config(read(config['parent_config']));parent.assert_original_paused(value);return value

def old_queue_paths(config):
    p=locations(config['root'])
    return ((p['early']/'delivery/controller_launch.json',p['early']/'delivery/status.json'),
        (p['parent']/'delivery/controller_launch.json',p['parent']/'delivery/status.json'),
        (p['parent']/'delivery/launches/reconstruct_current.json',p['parent']/'evaluation/reconstruct_status.json'))

def paused_queues(config,reader=process):
    """Only read old queues: this controller has no resume path for them."""
    admission=read(config['pause_admission']);snapshot=read(config['pause_evidence'])
    require(admission.get('status')=='FULL_QUEUES_SAFELY_PAUSED_FOR_FIXED16'
        and admission.get('training_resume_allowed') is False,'Wrong queue pause admission')
    require(admission['pause_evidence_sha256']==sha(config['pause_evidence']),'Pause evidence changed')
    owners=admission['owners'];require(len(owners)==3,'All three paused owner identities are required')
    snapshot_keys=('swin_early_release/delivery/status.json','delivery/status.json','evaluation/reconstruct_status.json')
    for saved,(launch_path,status_path),key in zip(owners,old_queue_paths(config),snapshot_keys):
        current=read(launch_path);status=read(status_path)
        require(current['pid']==saved['pid'] and str(current['start_ticks'])==str(saved['start_ticks']),
            'A full queue owner was replaced; do not resume it during the fixed16 release')
        require(not live(saved,reader) and status.get('status')=='PAUSED' and status.get('pid')==saved['pid'],
            'Every old full queue must remain safely paused and exited')
        proof=snapshot['processes'][str(saved['pid'])]
        require(proof.get('exists') is False and str(proof.get('start_ticks'))==str(saved['start_ticks']),
            'Pause proof did not observe exact owner exit')
        require(snapshot[key].get('status')=='PAUSED' and snapshot[key].get('pid')==saved['pid']
            and snapshot[key].get('safe_pause_handler_installed') is True,'Missing recorded safe-pause boundary')
    return admission

def validate(config):
    p=locations(config['root'])
    require(config.get('version')==VERSION and Path(config['output'])==p['delivery']
        and Path(config['science_config'])==p['child']/'config.json'
        and Path(config['parent_config'])==p['parent_config'],'Wrong fixed16 delivery paths')
    require(config.get('selected_step')==80000 and config.get('training_resume_allowed') is False
        and config.get('resume_full_queue_allowed') is False and config.get('stop_after_publication') is True,
        'This release must stop after publication without resuming training or the full queue')
    require(config.get('sources')==16 and config.get('physical_frames')==64 and config.get('rows')==128
        and config.get('source_indices')==list(FIXED) and config.get('snrs')==[7,13] and config.get('noise_seeds')==[2001]
        and config.get('budgets')==[1024,2048],'Fixed16 scientific scope differs')
    require(1<=config.get('poll_seconds',0)<=60,'Polling must be 1..60 seconds')
    verify(config['bindings']);frozen_parent(config);paused_queues(config);return config

def stage_spec(config,stage,token):
    p=locations(config['root']);parent=read(config['parent_config'])
    if stage in ('reconstruct','score'):
        script='hifi16_eval.py';args=['--root',config['root'],'--stage',stage,'--launch-id',token]
        kind='swin' if stage=='reconstruct' else 'metric';gpu=True
    elif stage=='report':script='hifi_fixed16_report.py';args=['--root',config['root'],'--launch-id',token];kind='report';gpu=False
    elif stage=='publish':
        script='hifi_fixed16_publish.py';args=['--root',config['root'],'--delivery-config',str(p['delivery']/'config.json')];kind='report';gpu=False
    else:raise RuntimeError('There is no registered stage or resume action named '+stage)
    env=dict(parent[kind+'_environment']) if gpu else dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    env['PYTHONDONTWRITEBYTECODE']='1'
    return [parent[kind+'_python'],'-B','-u',str(p['child']/script),*args],env,gpu

def receipt_path(config,stage):
    p=locations(config['root'])
    return {'reconstruct':p['evaluation']/'reconstruction_completion.json','score':p['evaluation']/'completion.json',
        'report':p['evaluation']/'report_completion.json','publish':p['child']/'release_publication.json'}[stage]

def stage_proof(config,stage,missing_ok=False):
    path=receipt_path(config,stage)
    if not path.exists():
        if missing_ok:return None
        raise RuntimeError('Missing completed stage: '+stage)
    value=read(path)
    expected={'reconstruct':'HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE','score':'HIFI_FIXED16_EVALUATION_COMPLETE',
        'report':'HIFI_FIXED16_REPORT_COMPLETE','publish':'PUSHED'}[stage]
    require(value.get('status')==expected and value.get('rows')==128,'Wrong fixed16 completion status or rows')
    if stage!='publish':
        require(value.get('synthetic') is False and value.get('sources')==16,'Real fixed16 completion required')
        if stage=='reconstruct':require(value.get('physical_frames')==64,'Reconstruction budget coverage differs')
        if stage=='score':require(value.get('metric_groups')==8,'Both methods and four conditions must be scored')
        if stage=='report':require(value.get('metric_groups')==8 and value.get('png_figures')==16
            and value.get('pdf_figures')==4 and value.get('figure_cells')==128
            and value.get('physical_frames')==64 and value.get('fixed_sources')==16 and value.get('paired_metric_groups')==52
            and value.get('selected_step')==80000 and value.get('exploratory_fixed_subset') is True
            and value.get('full_evaluation_resume_authorized') is False
            and value.get('full_comparison_complete') is False,'Fixed16 paired examples/report scope differs')
    else:require(value.get('checks')=='PASS' and value.get('phase')=='hifi_fixed16_release'
        and value.get('full_comparison_complete') is False and value.get('commit')==value.get('remote_commit'),
        'Verified partial-scope normal publication required')
    receipt_bindings(value);return dict(path=str(path),sha256=sha(path),status=value['status'])

def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    rows=[line.strip() for line in value.splitlines() if line.strip()];require(all(line.isdigit() for line in rows),'Unreadable GPU ownership')
    return list(map(int,rows))
