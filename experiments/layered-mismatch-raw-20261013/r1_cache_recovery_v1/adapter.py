"""Private routing adapters. Only receiver cache setup changes executable model setup.

All mathematical definitions, model bundles and frozen numerical flags are reused.
Metadata explicitly distinguishes old successful PHY/projection from new stages.
"""
from __future__ import annotations
import ast, copy, importlib.util, os, sys, tempfile, types
from pathlib import Path
import common as c

CACHE_OLD="out=Path(reg['runtime']['visual_directory']);require(not out.exists(),'Visual attempt cannot replay');out.mkdir()"
CACHE_NEW=CACHE_OLD+';_prepare_receiver_cache(out)'

def receiver_source(text):
    c.require(text.count(CACHE_OLD)==1 and '_prepare_receiver_cache' not in text,'Original receiver cache insertion anchor changed')
    result=text.replace(CACHE_OLD,CACHE_NEW)
    c.require(ast.dump(ast.parse(result.replace(CACHE_NEW,CACHE_OLD)))==ast.dump(ast.parse(text)),'Receiver inverse AST differs')
    return result

def prepare_receiver_cache(out):
    out=Path(out)
    c.require(out==c.RUN/'visual' and out.is_dir() and not out.is_symlink(),'Exact new visual output directory required')
    cache=out/'runtime_cache';c.require(not cache.exists() and not cache.is_symlink(),'No cache reuse/overwrite')
    cache.mkdir(mode=0o700)
    for name in ('tmp','inductor','triton'):(cache/name).mkdir(mode=0o700)
    for name in ('TMPDIR','TEMP','TMP'):os.environ[name]=str(cache/'tmp')
    # tempfile can have cached a pre-import path; make the admitted path explicit.
    tempfile.tempdir=str(cache/'tmp')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor')
    os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
    return dict(path=str(cache),source_truth_access_added=False,mathematical_changes=0)

def load_source(name,path,text=None):
    module=types.ModuleType(name);module.__file__=str(path);module.__package__=''
    sys.modules[name]=module
    exec(compile(path.read_text(encoding='utf8') if text is None else text,str(path),'exec'),module.__dict__)
    return module

def validate_runtime(runtime,original):
    c.require(runtime==c.frozen_runtime(original),'Only reviewed output routing may differ from original runtime')
    c.require(runtime['deadline_unix']==c.DEADLINE,'Recovery cannot reset deadline')

def validate_visual_runtime(runtime):
    # Fixed source-free equivalent of the original master-derived runtime;
    # no original master/source descriptor is opened in the visual child.
    expected=dict(root=str(c.RUN),python_cpu='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python',
        python_visual='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python',
        target_gpu=3,cpu_affinity=[0,1],phy_cpu_affinity=[2,3],max_seconds=43200,deadline_unix=c.DEADLINE,
        stop_file=str(c.OLD_OUT/'STOP'))
    expected.update({stage+'_directory':str((c.ANCESTOR if stage in ('phy','projection') else c.RUN)/stage)
                     for stage in ('phy','projection','visual','score','selection')})
    c.require(runtime==expected,'Source-free runtime differs from exact original resource and mixed storage routing')

def original_wait(directory,s,original_function):
    directory=Path(directory)
    if directory.parent!=c.ANCESTOR:return original_function(directory)
    c.require(directory.name in ('phy','projection'),'Failed ancestor cannot be used as completed')
    c.require(not (directory/'failure.json').exists(),'Inherited successful stage has a failure marker')
    launch=c.read(c.ANCESTOR/(directory.name+'_launch.json'));wait=c.read(directory/'actual_wait.json');owner=c.read(c.ANCESTOR/'owner_launch.json')
    c.require(c.same(launch['registration'],c.ORIGINAL) and c.same(owner['registration'],c.ORIGINAL)
              and wait==c.read(c.ANCESTOR/(directory.name+'_actual_wait.json')),'Inherited real wait provenance differs')
    c.actual(wait,launch['process'],0);c.gone(launch['process']);c.gone(owner['process'])
    return c.pin(directory/'actual_wait.json')

def receiver_inherited_wait(directory,projection,original_function):
    """Source-free predecessor identity, projected by the CPU registration builder."""
    directory=Path(directory)
    if directory!=c.ANCESTOR/'projection':return original_function(directory)
    c.require(not (directory/'failure.json').exists(),'Inherited projection has a failure marker')
    c.require(projection['schema']=='R1_RX_ONLY_INHERITED_PROJECTION_V1'
        and projection['original_master_sha256']==c.ORIGINAL['sha256'] and projection['source_truth_access_added'] is False,
        'Unreviewed inherited RX control receipt')
    c.require(projection['actual_wait']['path']==str(directory/'actual_wait.json'),'Wrong inherited wait location')
    wait=c.readpin(projection['actual_wait']);c.actual(wait,projection['projection_process'],0)
    c.gone(projection['projection_process']);c.gone(projection['owner_process'])
    return c.pin(directory/'actual_wait.json')

def install(registration,role):
    c.require(role in ('owner','visual','score','selection','dispatch'),'Only missing recovery stages allowed')
    reg=c.readpin(registration);visual=role=='visual'
    expected=c.OUT/'registered'/('visual_registration.json' if visual else 'registration.json')
    c.require(Path(registration['path'])==expected,'Wrong recovery registration path')
    for name,digest in c.OLD_SHA.items():c.require(c.sha(c.OLD/name)==digest,'Original binding changed: '+name)
    for path,digest in reg['code_bindings'].items():c.require(c.sha(path)==digest,'Registered source changed: '+path)
    # Metadata wrapper remains the already reviewed optional-byte-count adapter.
    for p in (c.E,c.OLD,c.E/'r1_source_identity_revision_v1'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    if not visual:
        spec=importlib.util.spec_from_file_location('_cache_recovery_original_pin_adapter',c.E/'r1_source_identity_revision_v1/pin_adapter.py')
        old_adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_adapter);old_adapter.install_phy()
    shared=load_source('shared',c.OLD/'shared.py');shared.TMP=c.RUN;shared.OUT=c.OUT
    if visual:shared.validate_runtime=validate_visual_runtime
    else:
        ancestor=c.readpin(c.ORIGINAL)
        shared.validate_runtime=lambda runtime:validate_runtime(runtime,ancestor['runtime'])
    original_validate=shared.validate_registration
    def validate(value):
        v=original_validate(value)
        c.require(Path(value['path'])==expected,'Foreign registration for this process role')
        if v['schema']==shared.MASTER_SCHEMA:
            c.require(v['recovery']['original_registration']==c.ORIGINAL and v['recovery']['schema']==c.RECOVERY_SCHEMA,'Missing ancestor binding')
            auth=c.validate_authorization(c.readpin(v['recovery']['authorization']))
            c.require(all(v[k]==ancestor[k] for k in ancestor if k not in ('runtime','code_bindings','visual_registration')),
                      'Original master scientific or policy field changed')
            c.require(v['code_bindings']==dict(ancestor['code_bindings'],**auth['code_bindings']),'Recovery code scope expanded')
        else:
            old_visual=c.readpin(c.ORIGINAL_VISUAL)
            c.require(all(v[k]==old_visual[k] for k in old_visual if k not in ('runtime','code_bindings')),
                      'Original source-free visual/model field changed')
            c.require(v['cache_revision']==dict(schema=c.RECOVERY_SCHEMA,private_cache=str(c.RUN/'visual/runtime_cache'),
                source_truth_access_added=False,mathematical_changes=0),'Unreviewed source-free cache scope')
            c.require('recovery' not in v,'Visual registration must not expose failure/source master descriptor')
            c.require(set(v['inherited_projection'])=={'schema','original_master_sha256','actual_wait','projection_process','owner_process','source_truth_access_added'},
                      'Source-free predecessor fields expanded')
        return v
    shared.validate_registration=validate
    receiver=load_source('receiver_visual',c.OLD/'receiver_visual.py',receiver_source((c.OLD/'receiver_visual.py').read_text(encoding='utf8')))
    receiver._prepare_receiver_cache=prepare_receiver_cache
    old_wait=receiver.require_successful_wait
    receiver.require_successful_wait=(lambda directory:receiver_inherited_wait(directory,reg['inherited_projection'],old_wait)) if visual else (
        lambda directory:original_wait(directory,shared,old_wait))
    if visual:
        # Pre-firewall original_wait may authenticate old owner/dispatch. The
        # model itself still sees only the original RX inputs and model pins.
        return shared,receiver,None,None
    import score_provenance
    score=load_source('score_selection',c.OLD/'score_selection.py')
    score_provenance.install(score)
    owner=load_source('_cache_recovery_original_owner',c.OLD/'owner.py')
    owner.STAGES=c.STAGES
    original_window=owner.acquire_window
    owner.acquire_window=lambda r,stage,flock:original_window(r,'visual' if stage=='phy' else stage,flock)
    old_command=owner.stage_command
    def stage_command(r,p,stage,parent,ticks):
        c.require(stage in c.STAGES,'Recovery cannot launch PHY/projection')
        command=old_command(r,p,stage,parent,ticks)
        # Keep original script/action/flags visible for original provenance checks.
        return command[:3]+[str(c.HERE/'runner.py'),'--original',command[3]]+command[4:]
    owner.stage_command=stage_command
    return shared,receiver,score,owner
