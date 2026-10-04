"""Control-plane contracts for an early real Swin release and cold HiFi resume."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
VERSION='SWIN-EARLY-DELIVERY-20261004-R1'
STAGES=('reconstruct','score','report','publish','resume_hifi')
OWN_FILES=('swin_early_delivery_common.py','swin_early_delivery.py','hifi_resume.py','test_swin_early_delivery.py')

def require(value,message):
    if not value:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def seal(path,value):
    if Path(path).exists():require(read(path)==value,'Immutable record changed: '+str(path))
    else:write(path,value)

def verify(bindings):
    require(isinstance(bindings,dict),'Missing immutable bindings')
    for path,digest in bindings.items():require(Path(path).is_file() and sha(path)==digest,'Bound file changed: '+path)

def receipt_bindings(value):
    bindings={}
    for name in ('bindings','inputs','input_bindings','source_bindings','outputs'):
        if name in value:verify(value[name]);bindings.update(value[name])
    return bindings

def locations(root):
    root=Path(root).resolve();out=root/'outputs/EXTERNAL-COMPARISON-20261004';parent=out/'fixed80k_revision';child=parent/'swin_early_release'
    return dict(root=root,out=out,parent=parent,child=child,delivery=child/'delivery',evaluation=child/'evaluation',
        parent_config=parent/'delivery/config.json',parent_delivery=parent/'delivery')

def process(pid):
    require(type(pid)is int and pid>1,'Exact positive PID required')
    try:
        folder=Path('/proc')/str(pid);fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],state=fields[0],command=[v for v in (folder/'cmdline').read_bytes().decode(errors='replace').split('\0') if v])
    except (FileNotFoundError,ProcessLookupError):return None

def live(owner,reader=process):
    require(type(owner.get('pid'))is int and str(owner.get('start_ticks','')).isdigit(),'Missing PID/start tick identity')
    current=reader(owner['pid'])
    return bool(current and current['state']!='Z' and str(current['start_ticks'])==str(owner['start_ticks']))

def parent_modules(config):
    folder=locations(config['root'])['parent'];sys.path.insert(0,str(folder))
    import fixed80k_delivery_common as common
    import fixed80k_delivery as delivery
    require(Path(common.__file__).resolve()==folder/'fixed80k_delivery_common.py'
        and Path(delivery.__file__).resolve()==folder/'fixed80k_delivery.py','Unexpected frozen parent import')
    return common,delivery

def original_pause_gate(config):
    common,_=parent_modules(config);parent=common.validate_config(read(config['parent_config']))
    require(sha(config['parent_config'])==config['parent_config_sha256'],'Original delivery config changed')
    common.assert_original_paused(parent);return parent

def snapshot_gate(config,reader=process):
    """Historical pause evidence stays immutable after new live status changes."""
    value=read(config['pause_evidence']);admission=read(config['pause_admission'])
    parent=value['delivery/status.json'];paired=value['evaluation/reconstruct_status.json']
    require(parent.get('status')==paired.get('status')=='PAUSED'
        and parent.get('config_sha256')==config['parent_config_sha256']
        and parent.get('training_resumed') is False and paired.get('safe_pause_handler_installed') is True,
        'Both original fixed-80k owners must have safely paused')
    require(value.get('completed_physical_frames')==admission['completed_physical_frames']
        and value['completed_physical_frames']>0,'Saved paired frame count differs')
    owners=admission['owners']
    require({parent['pid'],paired['pid']} <= {owner['pid'] for owner in owners},'Pause owner coverage differs')
    for owner in owners:
        require(not live(owner,reader),'An original paired owner remains alive')
        if str(owner['pid']) in value['processes']:
            require(value['processes'][str(owner['pid'])].get('exists') is False,'Pause proof did not observe exact process exit')
    return admission

def validate(config):
    p=locations(config['root'])
    require(config.get('version')==VERSION and Path(config['output'])==p['delivery']
        and Path(config['science_config'])==p['child']/'config.json'
        and Path(config['parent_config'])==p['parent_config'],'Wrong early-release paths or protocol')
    require(config.get('selected_step')==80000 and config.get('training_resume_allowed') is False
        and config.get('resume_unchanged_hifi_after_publication') is True,'Changed authorized continuation scope')
    require(1<=config.get('poll_seconds',0)<=60,'Polling must be 1..60 seconds')
    verify(config['bindings']);original_pause_gate(config);snapshot_gate(config);return config

def stage_spec(config,stage,token):
    p=locations(config['root']);parent=read(config['parent_config'])
    if stage in ('reconstruct','score'):
        script='swin_early.py';args=['--root',config['root'],'--stage',stage,'--launch-id',token]
        kind='swin' if stage=='reconstruct' else 'metric';gpu=True
    elif stage=='report':script='swin_early_report.py';args=['--root',config['root'],'--launch-id',token];kind='report';gpu=False
    elif stage=='publish':
        script='swin_early_publish.py';args=['--root',config['root'],'--delivery-config',str(p['delivery']/'config.json')];kind='report';gpu=False
    elif stage=='resume_hifi':
        script='hifi_resume.py';args=['--config',str(p['delivery']/'config.json'),'--launch-id',token];kind='report';gpu=False
    else:raise RuntimeError('Unregistered stage')
    env=dict(parent[kind+'_environment']) if gpu else dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    env['PYTHONDONTWRITEBYTECODE']='1'
    return [parent[kind+'_python'],'-B','-u',str(p['child']/script),*args],env,gpu

def receipt_path(config,stage):
    p=locations(config['root'])
    return {'reconstruct':p['evaluation']/'reconstruction_completion.json','score':p['evaluation']/'completion.json',
        'report':p['evaluation']/'report_completion.json','publish':p['child']/'release_publication.json',
        'resume_hifi':p['delivery']/'hifi_resume_completion.json'}[stage]

def stage_proof(config,stage,missing_ok=False):
    path=receipt_path(config,stage)
    if not path.exists():
        if missing_ok:return None
        raise RuntimeError('Missing completed stage: '+stage)
    value=read(path)
    status={'reconstruct':'SWIN_EARLY_RECONSTRUCTIONS_COMPLETE','score':'SWIN_EARLY_EVALUATION_COMPLETE',
        'report':'SWIN_EARLY_REPORT_COMPLETE','publish':'PUSHED','resume_hifi':'UNCHANGED_FIXED80K_PARENT_REPORTED_AND_PUSHED'}[stage]
    require(value.get('status')==status,'Unexpected complete stage: '+stage)
    if stage in ('reconstruct','score','report'):
        require(value.get('synthetic') is False and value.get('rows')==1800 and value.get('sources')==100,
            'All 1800 real Swin rows are required')
        if stage=='score':require(value.get('metric_groups')==6 and value.get('full_comparison_complete') is False,'Swin score scope differs')
        if stage=='report':require(value.get('metric_groups')==6 and value.get('png_figures')==24
            and value.get('pdf_figures')==6 and value.get('figure_cells')==96
            and value.get('full_comparison_complete') is False,'Swin fixed examples/report scope differs')
    elif stage=='publish':require(value.get('checks')=='PASS' and value.get('phase')=='swin_early_release'
        and value.get('full_comparison_complete') is False and value.get('rows')==1800
        and value.get('commit')==value.get('remote_commit'),'Verified standalone Swin push required')
    else:require(value.get('training_resumed') is False and value.get('selected_step')==80000 and value.get('full_plan_complete') is True
        and value.get('parent_config_sha256')==config['parent_config_sha256'],'Parent completion identity differs')
    receipt_bindings(value);return dict(path=str(path),sha256=sha(path),status=value['status'])

def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    rows=[row.strip() for row in value.splitlines() if row.strip()]
    require(all(row.isdigit() for row in rows),'Unexpected GPU ownership result');return list(map(int,rows))

def verify_publication_blobs(root,record,check_remote=True):
    require(record.get('status')=='PUSHED' and record.get('checks')=='PASS'
        and record.get('commit')==record.get('remote_commit'),'Normal push is not verified')
    root=Path(root)
    for path,digest in record['published_files'].items():
        relative=Path(path).relative_to(root).as_posix()
        data=subprocess.check_output(['git','show',record['commit']+':'+relative],cwd=root)
        require(hashlib.sha256(data).hexdigest()==digest,'Published Git artifact differs: '+relative)
    if check_remote:
        subprocess.run(['git','fetch','origin','main'],cwd=root,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','merge-base','--is-ancestor',record['commit'],'origin/main'],cwd=root,check=True)
