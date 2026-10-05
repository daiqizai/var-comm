"""Register (never launch) the separately supervised original800 source codec.

The CPU asset batch must be successful and reaped. Old200 bitstreams are reused
only through exact-ID proofs. This registrar never constructs models, opens
source arrays, renders images, scores metrics or invokes a physical decoder.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEADLINE=1791564605.9549868
REQUEST_NAMES=('assets_registrar_module','owner_module','wait_module','asset_module','source_module','static_closure_module',
    'assets_config','assets_registration','assets_owner_config','assets_owner_launch','assets_owner_completion',
    'assets_completion','assets_manifest','visual_reference_config','static_reference','numerical_reference','prepared_qualification')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed immutable binding: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p);result[p]=s
    return result
def pin(record):
    p=Path(record['path']);require(p.is_absolute() and sha(p)==record['sha256'],'Pinned input changed: '+str(p));return p
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def verify_assets_closed(a,w,t,paths,state_reader):
    oc,ar,ac=read(paths['assets_owner_config']),read(paths['assets_registration']),read(paths['assets_config'])
    a.validate_config(oc,ar,sha(paths['assets_owner_config']))
    require(oc['registration']==str(paths['assets_registration']) and ar['allowed_stage_ids']==['source']
            and ar['source_stage_scope']=='FULL1000_CPU_ASSETS_ONLY' and oc['phase_limits']==t.PHASES
            and oc['overall_deadline_unix']==DEADLINE,'Wrong prerequisite asset owner scope')
    require(len(oc['stages'])==1 and oc['stages'][0]['id']=='source'
            and oc['stages'][0]['resource']=='cpu' and len(oc['stages'][0]['jobs'])==1,'One CPU assets predecessor required')
    job=oc['stages'][0]['jobs'][0];base=Path(oc['owner_out'])
    expected=[job['argv'][0],'-B',str(paths['asset_module']),'--config',str(paths['assets_config'])]
    require(job['id']=='full1000_assets' and job['argv']==expected and job['out']==ac['out']
            and job['completion']==str(paths['assets_completion'])
            and ac['registration']==str(paths['assets_registration']) and ac['root']==oc['root']
            and paths['assets_owner_completion']==base/'completion.json'
            and paths['assets_manifest']==Path(ac['out'])/'manifest.json','Actual CPU asset command/config differs')
    for p in (Path(oc['out'])/'STOP',base/'STOP',Path(ac['out'])/'STOP'):
        require(not p.exists(),'STOP blocks source registration: '+str(p))
    require(not (base/'failure.json').exists() and not (Path(ac['out'])/'failure.json').exists(),
            'CPU assets failure cannot be reclassified as success')
    launch=read(paths['assets_owner_launch']);ident=launch['identity']
    expected=[launch['argv'][0],'-B',str(paths['owner_module']),'--config',str(paths['assets_owner_config'])]
    require(ident['argv']==launch['argv']==expected and launch['registration_sha256']==sha(paths['assets_registration'])
            and launch['owner_config_sha256']==sha(paths['assets_owner_config']),'Assets owner launch differs')
    ip=base/'owner_identity.json';actual=read(ip)
    require(all(actual[k]==ident[k] for k in ('pid','uid','start_ticks','argv'))
            and actual['config_sha256']==sha(paths['assets_owner_config'])
            and actual['registration_sha256']==sha(paths['assets_registration']),'Assets owner identity differs')
    require(w.exited(ident,state_reader),'CPU assets owner still live or unreaped')
    closed=w.verify_batch(a,paths['assets_owner_config'],paths['assets_registration'],ident['uid'],state_reader)
    done=read(paths['assets_completion']);manifest=read(paths['assets_manifest'])
    for name,value in dict(status='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',source_count=1000,
        reused_codec_sources=200,pending_source_encoding_count=800,source1000_codec_complete=False,
        new_source_encoding=False,GPU_used=False,new_visual_inference=0,new_packet_decodes=0,
        development_used=False,holdout_used=False,full1000_calibration_complete=False).items():
        require(done.get(name)==value,'Wrong CPU asset completion scope: '+name)
    require(done['registration_sha256']==sha(paths['assets_registration'])
            and done['outputs'].get(str(paths['assets_manifest']))==sha(paths['assets_manifest'])
            and closed['bindings'].get(str(paths['assets_completion']))==sha(paths['assets_completion']),
            'Original CPU completion/manifest not sealed')
    for values in (ar['source_bindings'],ar['input_bindings'],done['source_bindings'],done['input_bindings'],done['outputs']):verify(values)
    for name in ('assets_config','assets_registrar_module','owner_module','wait_module','asset_module'):
        require(merge(ar['source_bindings'],ar['input_bindings']).get(str(paths[name]))==sha(paths[name]),
                'CPU asset execution does not bind dependency: '+name)
    t.check_budget(read(paths['assets_owner_completion'])['budget'])
    closed['bindings']=merge(closed['bindings'],bind((ip,*[paths[k] for k in ('assets_owner_config','assets_registration',
        'assets_config','assets_owner_launch','assets_owner_completion','assets_completion','assets_manifest')])))
    closed.update(owner_identity=ident,owner_exited=True,original_render_owner_success=False)
    return dict(old=oc,assets_config=ac,assets_reg=ar,assets_done=done,manifest=manifest,closure=closed)


def static_source_closure(helper,root,visual,reference,oldbound):
    """Recompute after final source files are tracked; never weaken a guard."""
    require(reference['status']=='EXACT_SOURCE_CLOSURE_MATCH','Original visual prelaunch source audit was not PASS')
    frozen=reference['inputs']
    for key,value in (('root',root),('runtime',visual['native_runtime']),('quality_runtime',visual['uep_runtime'])):
        require(frozen[key]==value,'Original native source location differs: '+key)
    for key,subdir,pattern in (('var_source','models','*.py'),('dino_source','dinov2','**/*.py')):
        directory=Path(frozen[key]);files=list((directory/subdir).glob(pattern))
        require(directory.is_absolute() and files,'Original external source subtree missing: '+key)
        for p in files:require(oldbound.get(str(p))==sha(p),'External model source is not from original closure: '+str(p))
    actual=helper.collect_bindings(root,visual['native_runtime'],frozen['var_source'],frozen['dino_source'],visual['uep_runtime'])
    changes=helper.compare_bindings(actual,oldbound)
    require(not changes['changed'],'An existing source changed; diagnose before registering')
    # New tracked helpers may be omitted from the old revision. Register exact
    # newly observed bytes, while retaining every historical source unchanged.
    combined=merge(oldbound,actual)
    require(helper.compare_bindings(actual,combined)['status']=='EXACT_SOURCE_CLOSURE_MATCH','Visual closure incomplete')
    return actual,changes


def make_configs(r,ctx):
    paths=ctx['paths'];old=ctx['old'];ac=ctx['assets_config'];visual=ctx['visual'];frozen=ctx['static_reference']['inputs']
    execution=Path(r['execution_dir']);rp=execution/'execution_registration.json';cp=execution/'source_config.json'
    cfg=dict(schema='H_FULL1000_SOURCE_CODEC_CONFIG_V1',root=old['root'],H_out=old['out'],out=r['source_out'],
        registration=str(rp),source_owner_config=str(execution/'owner_config.json'),
        protocol=ac['protocol'],calibration_registration=ac['calibration_registration'],
        runtime_dir=visual['runtime_dir'],native_runtime=visual['native_runtime'],uep_runtime=visual['uep_runtime'],
        var_source=frozen['var_source'],dino_source=frozen['dino_source'],source_driver_module=visual['source_driver_module'],
        ledger=old['budget_path'],budget_registration=old['budget_registration'],phase_limits=copy.deepcopy(old['phase_limits']),
        stop_file=str(Path(old['out'])/'STOP'),max_seconds=r['max_seconds'],overall_deadline_unix=DEADLINE,
        numerical_reference_field=['numerical_runtime'])
    rename={'assets_module':'asset_module','wait_module':'wait_module'}
    for key in ('assets_module','assets_config','assets_registration','assets_completion','assets_manifest',
                'assets_owner_config','assets_owner_launch','owner_module','wait_module','static_closure_module','numerical_reference'):
        cfg[key]=str(paths[rename.get(key,key)])
    oc=copy.deepcopy(old);oc.update(owner_out=str(execution),registration=str(rp))
    oc['stages']=[dict(id='source',resource='gpu',requires=[],max_seconds=r['max_seconds'],jobs=[dict(
        id='full1000_source',argv=[r['python'],'-B',str(paths['source_module']),'--config',str(cp)],cwd=old['root'],
        out=r['source_out'],completion=str(Path(r['source_out'])/'completion.json'),accepted_statuses=['H_FULL1000_SOURCE_CODEC_COMPLETE'],
        receipt_expect=dict(source_count=1000,reused_codec_sources=200,newly_encoded_sources=800,
            pending_source_encoding_count=0,source1000_codec_complete=True,new_packet_decodes=0,
            new_image_renders=0,new_metric_calls=0,development_used=False,holdout_used=False,
            training_updates=0,full1000_calibration_complete=False,GPU_used=True))])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).resolve();r=read(request_path)
    require(r['schema']=='H_FULL1000_SOURCE_REGISTRATION_REQUEST_V1','Wrong source registration request')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    require(time.time()<DEADLINE,'Original H qualification deadline expired')
    paths={k:pin(r[k]) for k in REQUEST_NAMES}
    t=module(paths['assets_registrar_module'],'source_registration_cpu_assets_toolkit')
    a=module(paths['owner_module'],'source_registration_original_owner')
    w=module(paths['wait_module'],'source_registration_original_wait')
    ctx=verify_assets_closed(a,w,t,paths,a.raw_process_state);old,ar,ac=ctx['old'],ctx['assets_reg'],ctx['assets_config']
    bound=merge(ar['source_bindings'],ar['input_bindings'])
    for name in ('visual_reference_config','static_reference'):
        require(bound.get(str(paths[name]))==sha(paths[name]),'Original visual reference is not in CPU provenance: '+name)
    visual=read(paths['visual_reference_config']);static=read(paths['static_reference'])
    hc=read(ac['h_source_config'])
    require(visual['root']==hc['root']==old['root'] and visual['H_out']==hc['out']==old['out']
            and visual['native_runtime']==hc['native_runtime'] and visual['uep_runtime']==hc['uep_runtime']
            and visual['calibration_registration']==hc['calibration_registration']==ac['calibration_registration']
            and visual['protocol']==ac['protocol'],'Original visual runtime/data scope differs')
    require(Path(visual['source_driver_module']).parent==Path(visual['runtime_dir'])
            and bound.get(visual['source_driver_module'])==sha(visual['source_driver_module']),
            'Frozen original source codec entry missing')
    numerical=read(paths['numerical_reference']);cal=read(ac['calibration_registration'])
    require(numerical['status']=='REAL_NATIVE_QUALIFICATION_PASS' and numerical['synthetic'] is False
            and numerical['frozen_identity']==cal['identity'],'Original native qualification identity differs')
    flags=numerical['numerical_runtime']
    require(flags['threads']==old['gpu_threads']==6 and flags['deterministic'] is True
            and flags['matmul_tf32'] is False and flags['cudnn_tf32'] is False
            and flags['cudnn_benchmark'] is False and flags['precision']=='highest','Original FP32 numerical policy differs')
    q=read(paths['prepared_qualification']);verify(q['source_bindings'])
    require(q['status']=='H_FULL1000_SOURCE_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False
            and q['new_packet_decodes']==0 and q['results'] and all(row['exit_code']==0 for row in q['results']),
            'New source implementation has not passed CPU qualification')
    for p in (Path(__file__).resolve(),paths['source_module'],paths['static_closure_module']):
        require(q['source_bindings'].get(str(p))==sha(p),'Qualification misses current source: '+str(p))
    helper=module(paths['static_closure_module'],'source_registration_complete_static_graph')
    actual,delta=static_source_closure(helper,old['root'],visual,static,bound)
    sources=merge(ar['source_bindings'],q['source_bindings'],actual,bind((Path(__file__).resolve(),)))
    inputs=merge(ar['input_bindings'],ctx['closure']['bindings'],ctx['assets_done']['outputs'],
        ctx['assets_done']['input_bindings'],bind((request_path,*paths.values())))
    before=a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True);t.check_budget(before)
    ctx.update(paths=paths,visual=visual,static_reference=static)
    execution,out,h=Path(r['execution_dir']),Path(r['source_out']),Path(old['out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H path required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'New outputs overlap')
    require(type(r['max_seconds']) is int and 0<r['max_seconds']<=86400 and Path(r['python']).is_absolute()
            and Path(r['python']).is_file(),'Missing interpreter or finite source time cap')
    require(set(old['gpu_affinity'])<=set(os.sched_getaffinity(0)),'Original visual CPU affinity unavailable')
    cfg,oc=make_configs(r,ctx);verify(sources);verify(inputs)
    driver=module(paths['source_module'],'source_registration_pure_driver')
    driver.validate_manifest(ctx['manifest'],ctx['assets_done'],cal)
    execution.mkdir()
    try:
        cp,op,rp=execution/'source_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        gate=execution/'predecessor_closure.json';save(gate,dict(ctx['closure'],budget=before))
        graph=execution/'visual_source_closure.json';save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',
            source_bindings=actual,source_enumeration_only=True,GPU_used=False,new_packet_decodes=0,
            original_registered_files_changed=False,newly_bound_sources=delta['missing'],
            inputs=static['inputs'],helper_sha256=sha(paths['static_closure_module'])))
        save(cp,cfg);save(op,oc);inputs=merge(inputs,bind((cp,op,gate,graph)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            source_stage_scope='FULL1000_SOURCE_CODEC_ONLY',allowed_stage_ids=['source'],owner_config_sha256=sha(op),
            phase_limits=t.PHASES,source_bindings=sources,input_bindings=inputs,budget_before=before,
            budget_registration_sha256=old['budget_registration_sha256'],scientific_protocol_sha256=sha(cfg['protocol']),
            source_count=1000,reused_codec_sources=200,newly_encoded_sources_expected=800,
            original_render_owner_success=False,CPU_assets_complete=True,source1000_codec_complete=False,
            GPU_jobs=1,GPU_threads=6,GPU_affinity=old['gpu_affinity'],new_packet_decodes=0,
            new_image_renders=0,new_metric_calls=0,training_updates=0,development_used=False,holdout_started=False,
            full1000_calibration_complete=False,H_full_delivery_claimed=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time())
        a.validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(helper.collect_bindings(old['root'],visual['native_runtime'],static['inputs']['var_source'],
            static['inputs']['dino_source'],visual['uep_runtime'])==actual,'Tracked source set changed during registration')
        require(not (h/'STOP').exists() and not (execution/'STOP').exists() and time.time()<DEADLINE,'STOP/deadline changed')
        require(a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
                'Original shared packet budget changed during registration')
        save(rp,reg);driver.load_registered(cp)  # SHA/metadata/closure only, no model construction.
        done=dict(status='H_FULL1000_SOURCE_CODEC_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),
            owner_config_sha256=sha(op),outputs=bind((cp,op,rp,gate,graph)),workers_started=False,GPU_used=False,
            new_packet_decodes=0,new_source_encodings=0,new_image_renders=0,source1000_codec_complete=False,future_stage_automatic=False)
        save(execution/'registration_completion.json',done);return done
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);args=p.parse_args()
    value=register(args.request);print(json.dumps({k:value[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
