"""Independent delivery of the user-requested exact 80k evaluation.

The original training controller remains paused. No completion of that training
protocol is synthesized. Existing own-method science and paid PHY are unchanged.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
VERSION='USER-FIXED80K-DELIVERY-20261004-R1'
STAGES=('hifi_qualification','reconstruct','external_score','external_report','own_controls','score',
        'cost_own','cost_external','cost_report','full_report','publish')

def require(value,message):
    if not value:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(bindings):
    require(isinstance(bindings,dict),'Missing identity bindings')
    for path,digest in bindings.items():require(sha(path)==digest,'Bound file changed: '+path)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def seal(path,value):
    if Path(path).exists():require(read(path)==value,'Immutable record changed: '+str(path))
    else:write(path,value)

def locations(root):
    root=Path(root).resolve();out=root/'outputs/EXTERNAL-COMPARISON-20261004';revision=out/'fixed80k_revision'
    return dict(root=root,out=out,revision=revision,delivery=revision/'delivery',
        result=root/'results/external_comparison_20261004',own=out/'own_controls',cost=out/'receiver_cost',
        final=out/'final_comparison/final_stage')

def original_modules(config):
    runtime=Path(config['runtime']);sys.path.insert(0,str(runtime))
    import full_comparison_common as original
    return original

def validate_config(config):
    p=locations(config['root'])
    require(config.get('version')==VERSION and Path(config['output'])==p['delivery'],'Wrong independent delivery configuration')
    require(config.get('requested_step')==80000 and config.get('training_resume_allowed') is False,
        'Exact user milestone and paused training required')
    require(config.get('full_plan_scope')=='original_five_method_comparison_with_user_fixed_80k_external_checkpoint',
        'Unregistered delivery scope')
    require(1<=config['poll_seconds']<=60,'Invalid polling interval');verify(config['bindings'])
    return config

def receipt_bindings(value):
    bindings={}
    for field in ('bindings','input_bindings','inputs','source_bindings','outputs'):
        if field in value:verify(value[field]);bindings.update(value[field])
    return bindings

def stage_proof(config,stage,missing_ok=False):
    p=locations(config['root']);ev=read(config['external_eval_config'])
    external={'hifi_qualification':Path(ev['hifi_qualification_path']),
        'reconstruct':Path(ev['output'])/'reconstruction_completion.json',
        'external_score':Path(ev['output'])/'completion.json',
        'external_report':Path(ev['output'])/'report_completion.json'}
    if stage in external:
        path=external[stage]
        if not path.exists():
            if missing_ok:return None
            raise RuntimeError('Missing completed stage: '+stage)
        value=read(path)
        expected={'hifi_qualification':'HIFI_SWIN_QUALIFICATION_PASS','reconstruct':'EXTERNAL_RECONSTRUCTIONS_COMPLETE',
            'external_score':'EXTERNAL_EVALUATION_COMPLETE','external_report':'EXTERNAL_TWO_METHOD_REPORT_COMPLETE'}[stage]
        require(value.get('status')==expected,'Wrong external receipt: '+stage)
        if stage=='hifi_qualification':
            selection=read(Path(ev['training_output'])/'selected_swin.json')
            require(value.get('selected_checkpoint_sha256')==selection['checkpoint_sha256'],'Qualification checkpoint differs')
            require(all(value.get(k) is True for k in ('full_input_gradient_pass','native_forward_parity_pass','model_parameters_unchanged')),
                'Real gradient qualification incomplete')
        else:
            require(value.get('synthetic') is False,'Synthetic evaluation forbidden')
            require(value.get('sources')==100 and value.get('rows')==3600,'Incomplete external frame coverage')
            if stage!='external_report':require(value.get('sampler_step_limit') is None,'Shortened sampler forbidden')
        receipt_bindings(value)
        return {str(path):dict(sha256=sha(path),status=value['status'])}
    original=original_modules(config)
    if stage=='publish':
        path=p['delivery']/'publication.json'
        if not path.exists():
            if missing_ok:return None
            raise RuntimeError('Missing checked publication')
        value=read(path);require(value.get('status')=='PUSHED' and value.get('checks')=='PASS'
            and value.get('commit')==value.get('remote_commit') and value.get('full_plan_complete') is True,
            'Normal push not verified')
        verify(value['inputs']);return {str(path):dict(sha256=sha(path),status=value['status'])}
    return original.stage_proof(config,stage,missing_ok)

def final_table_gate(config):return original_modules(config).final_table_gate(config)

def process(pid):
    require(type(pid)is int and pid>1,'Invalid exact process owner')
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],state=fields[0],command=[v for v in (p/'cmdline').read_bytes().decode(errors='replace').split('\0') if v])
    except FileNotFoundError:return None

def live(owner):
    current=process(owner['pid'])
    return bool(current and current['state']!='Z' and str(current['start_ticks'])==str(owner['start_ticks']))

def assert_original_paused(config):
    p=locations(config['root']);snapshot=read(config['pause_evidence'])
    require(snapshot['swin_training/status.json']['status']=='PAUSED'
        and snapshot['swin_training/status.json']['step']==81551
        and snapshot['checkpoint_80000']['sha256']=='8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21',
        'Original pause evidence differs')
    require(all(value.get('exists') is False for value in snapshot['processes'].values()),'Original owner did not exit')
    require(read(p['out']/'swin_training/status.json').get('status')=='PAUSED','Original trainer has resumed')
    for filename in ('controller/controller_launch.json','controller/training_current_launch.json','delivery/launch.json','full_delivery/controller_launch.json'):
        require(not live(read(p['out']/filename)),'Original delivery/controller has resumed')
    require(not (p['out']/'swin_training/completion.json').exists(),'Original training completed unexpectedly')
    return snapshot
