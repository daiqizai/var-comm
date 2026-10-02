"""Frozen M2 runtime adapter; never edits the original experiment source."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_REL=Path('experiments/scale-causal-partial-residual-20261002')
OUT_REL=Path('outputs/METRIC-SPEED-20261002')
PARENT_REL=Path('outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002')
QUALIFICATION='m2_speed_qualification.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def source_bindings(here=HERE):
    return {str(p.resolve()):sha(p) for p in sorted(Path(here).iterdir()) if p.suffix in ('.py','.md')}


def verify(bindings):
    if not isinstance(bindings,dict) or not bindings:raise RuntimeError('Nonempty immutable bindings required')
    for path,digest in bindings.items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed: '+path)


def is_resource_busy(error,common):
    """Only the original dedicated exception type permits automatic retry."""
    assets=getattr(common,'assets',None)
    old=getattr(assets,'old',None)
    base=getattr(old,'b',None)
    dedicated=getattr(base,'ResourceBusy',None)
    return isinstance(dedicated,type) and isinstance(error,dedicated)


def run_stage(runner,common,stage,out):
    try:return runner.main(stage)
    except Exception as error:
        if not is_resource_busy(error,common):raise
        write(Path(out)/'resource_wait.json',dict(stage=stage,status='WAITING_FOR_SAFE_RESOURCE',
            error=str(error),time=time.time(),exit_code=75))
        raise SystemExit(75) from error


def require_m1(root=ROOT):
    parent=Path(root)/PARENT_REL
    completion=parent/'m1_complete.json';publication=parent/'m1_publication.json'
    done=read(completion);pub=read(publication)
    commit=pub.get('commit','')
    if (done.get('status')!='M1_COMPLETE' or done.get('synthetic') is not False or done.get('training_updates')!=0
            or done.get('publication')!=pub or pub.get('status')!='PUSHED' or pub.get('checks')!='PASS'
            or not re.fullmatch('[0-9a-f]{40}',commit) or pub.get('remote_commit')!=commit):
        raise RuntimeError('Verified real M1 publication must precede M2 qualification/runtime')
    verify(pub['source_bindings'])
    return {str(p.resolve()):sha(p) for p in (completion,publication)}


def original_gpu_workers(root=ROOT,proc=Path('/proc')):
    """Known original GPU entrypoints; parent/controller processes are separate."""
    base=str((Path(root)/BASE_REL).resolve())+'/'
    names=('m1_runner.py','m2_runner.py','qualification.py','engineering_probe.py')
    active=[]
    if not Path(proc).is_dir():raise RuntimeError('Original worker exit check requires Linux /proc')
    for p in Path(proc).iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        if any(base+name in arg.decode(errors='replace') for arg in args for name in names):active.append(int(p.name))
    return active


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load_original(root=ROOT):
    base=Path(root)/BASE_REL
    sys.path.insert(0,str(base))
    common=_module('_metric_speed_original_common',base/'common.py');sys.modules['common']=common
    receiver=_module('_metric_speed_original_receiver',base/'residual_receiver.py');sys.modules['residual_receiver']=receiver
    runner=_module('_metric_speed_original_m2',base/'m2_runner.py')
    return common,receiver,runner


def numerical_backend(torch,device):
    return dict(torch_version=str(torch.__version__),device=str(device),
        cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision())


def validate_selection(qualification,publication,own):
    """Pure receipt contract, also exercised by CPU fixtures."""
    if qualification.get('status') not in ('QUALIFIED','USE_ORIGINAL'):
        raise RuntimeError('No completed implementation qualification')
    selected='accelerated' if qualification['status']=='QUALIFIED' else 'original'
    if qualification.get('selected_implementation')!=selected:
        raise RuntimeError('Implementation choice contradicts qualification status')
    if qualification.get('development_read') is not False or qualification.get('training_updates')!=0:
        raise RuntimeError('Qualification scope must be calibration-only and zero-training')
    if qualification.get('source_bindings')!=own or publication.get('source_bindings')!=own:
        raise RuntimeError('All final speed source files must match qualification/publication')
    if not qualification.get('base_source_bindings'):raise RuntimeError('Original source bindings missing')
    if (publication.get('status')!='PUSHED' or publication.get('checks')!='PASS'
            or not re.fullmatch('[0-9a-f]{40}',publication.get('commit',''))
            or publication.get('commit')!=publication.get('remote_commit')):
        raise RuntimeError('Checked speed source publication missing')
    if selected=='accelerated':
        try:speedup=float(qualification.get('speedup',0))
        except (TypeError,ValueError):raise RuntimeError('Finite measured speedup required') from None
        if qualification.get('strict_equality_passed') is not True or not math.isfinite(speedup) or speedup<1.10:
            raise RuntimeError('Strict equality and10% aggregate speedup required')
    return selected


def install_runtime(common,receiver,runner,qualification,profile,bindings,accelerator):
    """Patch only this process; retain original inference on USE_ORIGINAL."""
    selected=profile['selected_implementation']
    if selected=='accelerated':
        def accelerated(*args,**kwargs):return accelerator.infer(receiver,*args,**kwargs)
        receiver.infer=accelerated
    elif selected!='original':raise RuntimeError('Unknown implementation')
    original_sources=common.source_bindings
    def extended_sources():
        actual=original_sources()
        for path,digest in bindings.items():
            if path in actual and actual[path]!=digest:raise RuntimeError('Runtime/source binding collision')
            actual[path]=digest
        verify(bindings)
        return actual
    common.source_bindings=extended_sources
    original_registration=common.registration
    def registration(loaded,data,stage,inputs=None):
        # Intercept only this synchronous original registration's seal call.
        # Keeping its implementation avoids duplicating its scientific fields.
        old_seal=common.seal
        target=common.OUT/(stage+'_registration.json')
        def seal(path,value):
            if Path(path)==target:value=dict(value,runtime_profile=profile)
            return old_seal(path,value)
        common.seal=seal
        try:
            paths=list(inputs or [])
            paths.extend(Path(path) for path in profile['receipt_inputs'])
            return original_registration(loaded,data,stage,paths)
        finally:common.seal=old_seal
    common.registration=registration
    original_setup=common.setup
    def setup():
        loaded=original_setup()
        if loaded['identity']!=qualification['model_identity']:raise RuntimeError('Qualified visual model identity differs')
        if numerical_backend(receiver.torch,loaded['device'])!=qualification['numerical_backend']:
            raise RuntimeError('Qualified numerical backend/precision flags differ')
        return loaded
    common.setup=setup
    return profile


def main(stage):
    own=source_bindings();parent=require_m1();out=ROOT/OUT_REL
    if original_gpu_workers():raise RuntimeError('An original GPU worker remains active')
    qpath=out/QUALIFICATION;ppath=out/'source_publication.json'
    q=read(qpath);pub=read(ppath);selected=validate_selection(q,pub,own)
    qsha=sha(qpath)
    if pub.get('qualification_sha256')!=qsha:raise RuntimeError('Published qualification SHA differs')
    verify(own);verify(parent);verify(q['base_source_bindings']);verify(q['m1_publication_bindings'])
    receipt_inputs={str(qpath):qsha,str(ppath):sha(ppath)}
    profile=dict(name='M2_SEQUENTIAL_SYNCHRONIZATION_20261002',selected_implementation=selected,
        qualification_sha256=qsha,source_publication_commit=pub['commit'],receipt_inputs=receipt_inputs,
        training_updates=0,scientific_policy_unchanged=True,source_bindings=own)
    common,receiver,runner=load_original()
    import acceleration
    install_runtime(common,receiver,runner,q,profile,{**own,**receipt_inputs},acceleration)
    stages=('calibration','evaluation','actual','timing') if stage=='all' else (stage,)
    for name in stages:
        completion=common.OUT/f'm2_{name}_complete.json'
        if completion.exists():
            registration=common.OUT/f'm2_{name}_registration.json'
            # Gate-skipped actual has no development registration; its frozen
            # calibration registration still proves the implementation choice.
            if not registration.exists():registration=common.OUT/'m2_calibration_registration.json'
            if read(registration).get('runtime_profile')!=profile:
                raise RuntimeError('Completed M2 stage belongs to another runtime profile')
    run_stage(runner,common,stage,out)
    verify(own);verify(parent);verify(receipt_inputs);verify(q['base_source_bindings'])
    write(out/f'{stage}_runtime_completion.json',dict(status='COMPLETE',stage=stage,runtime_profile=profile,
        source_bindings=own,training_updates=0,qualification_sha256=qsha))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,
        choices=('calibration','evaluation','actual','timing','all'))
    main(parser.parse_args().stage)
