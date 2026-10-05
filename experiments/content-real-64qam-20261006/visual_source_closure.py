"""Read-only reconstruction of the frozen native visual source bindings.

No model modules are imported. The only subprocess is the same read-only
``git ls-files`` query used by the historical common.source_bindings().
This helper reports omissions/changes; it does not edit a registration or run
an experiment. The original renderer's complete binding check stays intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

GIT_PATTERNS = ('*.py', '*.cpp', 'configs/*.yaml', 'configs/*.json')
COMMON_DIRECTORY = 'experiments/scale-causal-partial-residual-20261002'
OLD_DIRECTORY = 'experiments/extreme-bandwidth-20261001-N1024'
QUALITY_DIRECTORY = 'experiments/prior-aware-grouped-mcs-v1'
EXPLICIT_RELATIVE = (
    'src/var_comm/next_scale_prior.py', 'src/var_comm/scale_channel.py',
    'src/var_comm/token_trellis.cpp', 'src/var_comm/study.py', 'src/var_comm/quality.py',
    'experiments/rx-posterior-step1-20260929/probe.py',
    'experiments/rx-posterior-step1-20260929/rx_v3_common.py',
    'experiments/rx-posterior-step1-20260929/run_preflight.py',
    'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',
)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def _directory(path, label, *, resolve=False):
    path = Path(path)
    require(path.is_absolute(), label+' must be absolute')
    if resolve: path = path.resolve()
    require(path.is_dir(), label+' does not exist: '+str(path))
    return path


def tracked_paths(root):
    """Use exactly the historical tracked-path patterns, not an import graph."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    result = subprocess.check_output(['git','ls-files',*GIT_PATTERNS],
                                     cwd=root, text=True, env=env)
    return result.splitlines()


def _collect_paths(root, runtime, var_source, dino_source, quality_runtime, tracked):
    """Pure path expansion. ``tracked`` is injected only by tests/internal call."""
    root = _directory(root,'root',resolve=True)
    runtime = _directory(runtime,'runtime',resolve=True)
    var_source = _directory(var_source,'var_source')
    dino_source = _directory(dino_source,'dino_source')
    quality_runtime = _directory(quality_runtime or root/QUALITY_DIRECTORY,'quality_runtime',resolve=True)
    here = root/COMMON_DIRECTORY; old = root/OLD_DIRECTORY
    require(here.is_dir() and old.is_dir(), 'Original common/OLD source directories missing')
    # Reproduce quality_driver.build_native, then common.source_bindings.
    paths = list(runtime.glob('m1_*.py'))
    paths += [(quality_runtime/'quality_driver.py').resolve(), (quality_runtime/'source_quality.py').resolve()]
    paths += list(here.glob('*.py')) + list(here.glob('*.json')) + list(here.glob('*.md'))
    for relative in tracked:
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and relative,
                'Invalid relative tracked path: '+str(relative))
        paths.append(root/p)
    paths += list(old.glob('*.py')) + [root/p for p in EXPLICIT_RELATIVE]
    paths += list((var_source/'models').glob('*.py'))
    paths += list((dino_source/'dinov2').rglob('*.py'))
    require((var_source/'models').is_dir() and (dino_source/'dinov2').is_dir(),
            'Original VAR/DINO source subtree missing')
    for p in paths: require(p.is_file(), 'Expected bound source file is missing: '+str(p))
    return sorted(set(paths),key=str)


def collect_bindings(root, runtime, var_source, dino_source, quality_runtime=None):
    root = _directory(root,'root',resolve=True)
    paths = _collect_paths(root,runtime,var_source,dino_source,quality_runtime,tracked_paths(root))
    return {str(p):sha(p) for p in paths}


def merge_registration(registration):
    require(isinstance(registration.get('source_bindings'),dict)
            and isinstance(registration.get('input_bindings'),dict), 'Registration source/input maps required')
    result = {}
    for group in ('input_bindings','source_bindings'):
        for path, value in registration[group].items():
            require(path not in result or result[path] == value, 'Conflicting registered SHA: '+path)
            result[path] = value
    return result


def compare_bindings(actual, registered):
    """Missing is an omission; changed is a distinct hard diagnostic category."""
    missing = {p:actual[p] for p in sorted(actual) if p not in registered}
    changed = {p:dict(registered_sha256=registered[p],actual_sha256=actual[p])
               for p in sorted(actual) if p in registered and registered[p] != actual[p]}
    extra = {p:registered[p] for p in sorted(registered) if p not in actual}
    return dict(status='EXACT_SOURCE_CLOSURE_MATCH' if not missing and not changed else 'SOURCE_CLOSURE_MISMATCH',
        required_files=len(actual), matched_files=len(actual)-len(missing)-len(changed),
        missing=missing,changed=changed,extra_registered_bindings=extra,
        extra_scope='Expected: registration also binds data, receipts and other execution sources; extras are not removed',
        registration_modified=False,renderer_guard_unchanged=True)


def audit(root,runtime,var_source,dino_source,registration_path,quality_runtime=None):
    registration_path = Path(registration_path)
    require(registration_path.is_absolute(), 'Absolute registration path required')
    registration = json.loads(registration_path.read_text(encoding='utf-8-sig'))
    actual = collect_bindings(root,runtime,var_source,dino_source,quality_runtime)
    result = compare_bindings(actual,merge_registration(registration))
    result.update(schema='STATIC_NATIVE_VISUAL_SOURCE_CLOSURE_V1',
        registration_path=str(registration_path),registration_sha256=sha(registration_path),
        source_bindings=actual,helper_sha256=sha(__file__),
        inputs=dict(root=str(root),runtime=str(runtime),var_source=str(var_source),dino_source=str(dino_source),
                    quality_runtime=str(quality_runtime or Path(root)/QUALITY_DIRECTORY)),
        git_patterns=list(GIT_PATTERNS),source_enumeration_only=True,
        model_imports=False,GPU_used=False,actual_packet_decodes=0,experiment_started=False,
        guard='A changed existing SHA requires diagnosis; this helper never adds, replaces or removes any binding')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','var-source','dino-source','registration','out'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--quality-runtime')
    args=parser.parse_args()
    result=audit(args.root,args.runtime,args.var_source,args.dino_source,args.registration,args.quality_runtime)
    path=Path(args.out)
    require(path.is_absolute() and path.parent.is_dir(), 'Existing absolute output parent required')
    # Failure/older audits are immutable. A fresh name is required on every run.
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps({k:result[k] for k in ('status','required_files','matched_files')},sort_keys=True))


if __name__=='__main__':main()
