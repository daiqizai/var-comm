"""One explicit metadata-only compatibility probe; no tensor/model/channel work.

This is not a scientific qualification and creates no R1 registration or stage.
It reads existing software metadata to exercise the renderer's Python-open gate.
"""
import ast
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import receiver_visual as v
s=v.s


def main():
    # Actual CUDA initialization and all model constructors remain forbidden.
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    s.require(1<=args.attempt<=6,'Finite metadata probe attempt label required')
    out=s.TMP/('preparation_zero_model_import_probe_v'+str(args.attempt))
    s.require(not out.exists(),'Metadata probe already attempted; inspect saved evidence')
    out.mkdir(parents=True);(out/'cache').mkdir()
    os.environ.update(TMPDIR=str(out/'cache'),XDG_CACHE_HOME=str(out/'cache'))
    prior=s.read(s.ROOT/'results/entropy_main_common500_20261013/registration_v4.json')
    request=s.readpin(prior['visual_request']);spec=request['spec'];env=s.readpin(spec['environment'])
    software=env['files']+env['projection_generated_files']+env['host_native_library_facts']
    model_files=[p for p in spec['files'] if p['sha256'] in set(v.WEIGHTS.values())|{x[1] for x in v.MATH.values()}
        or Path(p['path']).parent==Path(spec['native_dependency_candidate']['root'])/'models']
    dist=Path(spec['native_dependency_candidate']['root'])/'dist.py'
    receipt=s.RT/'receipts/h800_all_devices_readonly_20261011_v1/completion.json'
    code=dict(prior['code_bindings'])
    for p in (Path(v.__file__),Path(s.__file__),s.EXPERIMENT/'contract.py',Path(__file__)):
        code[str(p)]=s.sha(p)
    firewall=v.SourceReadFirewall([p['path'] for p in software+model_files]+[dist],out,[receipt],code)
    sys.addaudithook(firewall.hook)
    result=dict(schema='R1_RX_ZERO_MODEL_IMPORT_PROBE_V1',scientific_calls=0,tensor_operations_requested=0,
        model_constructors=0,source_archive_reads=0,source_token_reads=0,channel_calls=0,
        actual_R1_registration_created=False,qualification_frames=0,CUDA_VISIBLE_DEVICES='',
        entry_sha256=code[str(Path(v.__file__))],shared_sha256=code[str(Path(s.__file__))],
        probe_sha256=code[str(Path(__file__))])
    try:
        import numpy as np
        import torch
        s.require(not torch.cuda.is_initialized(),'CUDA initialized during metadata imports')
        mapped={str(Path(p['path']).resolve()):p for p in software}
        result['actual_runtime_files']=[]
        for path in (sys.executable,np.__file__,torch.__file__,torch._C.__file__):
            actual=str(Path(path).resolve());s.require(actual in mapped,'Runtime import outside original projection: '+actual)
            # Resolve an existing environment symlink to its exact physical file;
            # no new symlink or alternate scientific bytes are introduced.
            pin=dict(mapped[actual],path=actual);s.checked(pin);result['actual_runtime_files'].append(pin)
        ns=dict(torch=torch,Path=Path,sys=sys,np=np,hashlib=__import__('hashlib'),PATCH_NUMS=v.SIZES,VOCAB_SIZE=4096)
        find=lambda h:next(p for p in model_files if p['sha256']==h)
        result['AST_proofs']={'loader':v.definitions(find(v.MATH['loader'][1]),{'load_models','state_sha256'},ns)}
        dn=dict(torch=torch,nn=torch.nn,copy=__import__('copy'))
        result['AST_proofs']['decoder']=v.definitions(find(v.MATH['decoder'][1]),{'ContinuousDecoder'},dn)
        pn=dict(torch=torch,np=np,SIZES=v.SIZES)
        result['AST_proofs']['prior']=v.definitions(find(v.MATH['prior'][1]),{'_Prior'},pn)
        rn=dict(np=np,time=time,SIZES=v.SIZES,require=s.require,token_count=lambda m,K=0:v.OFFSETS[m]+K)
        result['AST_proofs']['render']=v.definitions(find(v.MATH['render'][1]),{'render_received'},rn)
        # Imports only: build_vae_var/load_models/ContinuousDecoder/_Prior and
        # render_received are deliberately never called by this probe.
        s.require('dist' not in sys.modules and 'models' not in sys.modules,'Fresh author namespaces required')
        s.require(s.sha(dist)==v.DIST_SHA,'Reviewed dist differs');s.load(dist,'dist')
        sys.path.insert(0,str(dist.parent));__import__('models')
        s.require(not torch.cuda.is_initialized(),'Author import initialized CUDA')
        fake_reg=dict(device_receipt=dict(path=str(receipt),sha256=s.sha(receipt)),runtime=dict(target_gpu=3),probe_only=True)
        device,g=s.resource_device(fake_reg,out)
        result['device_metadata_binding']=device.device_identity_binding()
        # NVML/process/host metadata only; resource busy is an admissible probe
        # outcome and never causes a model or alternate resource launch.
        try:
            snapshot,_=device.resource_snapshot(g.helper(),prelaunch=True)
            result['resource_probe']=dict(status='READY_METADATA_ONLY',snapshot=snapshot)
        except Exception as error:
            if type(error).__name__!='ResourceBusy':raise
            result['resource_probe']=dict(status='BUSY_METADATA_ONLY',reason=str(error))
        s.require(not torch.cuda.is_initialized(),'Metadata resource probe initialized CUDA')
        s.require(not firewall.blocked,'Metadata imports swallowed a forbidden Python-open attempt')
        result.update(status='PASS_ZERO_MODEL_IMPORTS',CUDA_initialized=False,blocked=firewall.blocked,
            opened_paths=dict(firewall.reads),limitation='Python-open audit, not OS sandbox or numerical model qualification')
        s.write(out/'completion.json',result)
    except BaseException as error:
        result.update(status='FAILED_METADATA_ONLY',error=repr(error),blocked=firewall.blocked,opened_paths=dict(firewall.reads))
        s.write(out/'failure.json',result);raise
    print(json.dumps(dict(status=result['status'],completion=s.pin(out/'completion.json'),scientific_calls=0)))


if __name__=='__main__':main()
