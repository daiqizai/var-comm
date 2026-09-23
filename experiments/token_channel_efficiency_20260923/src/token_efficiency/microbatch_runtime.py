"""Explicit execution lineage for a user-authorized microbatch change."""
import copy,io
from pathlib import Path
import torch
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from var_comm.next_scale_prior import state_sha256
from short_prefix.train import update,forward,same_tree
from .common import EXP,read,register,bindings

def runtime_registration(out,spec,historical_sha,entry):
    if spec['effective_batch']!=16 or spec['budgets']!={'2048':8} or spec['precision_changed'] is not False:raise RuntimeError('unqualified microbatch configuration')
    bench=Path(spec['benchmark_acceptance'])
    if digest(bench)!=spec['benchmark_sha256']:raise RuntimeError('benchmark identity')
    r=read(bench)
    if r['synthetic'] is not False or not r['summary']['8']['equivalence_pass']:raise RuntimeError('real microbatch acceptance required')
    path=out/'execution'/spec['version']/'registration.json'
    record={'historical_registration_sha256':historical_sha,'historical_microbatch':4,'actual_microbatch':8,'effective_batch':16,'precision_changed':False,'exact_old_trajectory_claimed':False,'spec':spec,'bindings':bindings([entry,__file__,EXP/'microbatch_runtime.json',bench])}
    register(path,record)
    return {'path':str(path),'sha256':digest(path),'version':spec['version'],'microbatch':8}

def validate_execution_checkpoint(payload,execution,spec,checkpoint):
    prior=payload.get('execution_identity')
    if prior is None:
        origin=spec['initial_resume']
        if payload['state']['step']!=origin['step'] or digest(checkpoint)!=origin['sha256']:raise RuntimeError('initial execution migration checkpoint identity')
    else:
        if prior!=execution:raise RuntimeError('execution version mismatch on resume')
        verify_execution(prior)

def verify_execution(identity):
    path=Path(identity['path'])
    if digest(path)!=identity['sha256']:raise RuntimeError('execution receipt hash')
    r=read(path)
    if identity['version']!=r['spec']['version'] or identity['microbatch']!=r['actual_microbatch']:raise RuntimeError('execution receipt context')
    verify_snapshot(r['bindings'])
    return r

def verify_selected_execution(selected,payload):
    a=selected.get('execution_identity');b=payload.get('execution_identity')
    if a!=b:raise RuntimeError('selected/checkpoint execution lineage mismatch')
    if b is not None:
        r=verify_execution(b)
        if r['historical_registration_sha256']!=payload['registration_sha256']:raise RuntimeError('execution historical registration mismatch')

# The same real update/forward functions as the original qualification, with
# the actual selected microbatch passed explicitly to every update.
def qualify_microbatch(models,b,decoder,lp,scale,out,micro):
    if len(b['F'])!=16 or micro!=8:raise RuntimeError('actual effective batch16/micro8 qualification required')
    frozen=state_sha256(decoder);probe=copy.deepcopy(models);opts={n:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for n,m in probe.items()}
    update(probe,opts,b,decoder,lp,scale,micro)
    names=list(probe);last=names[-1];before=copy.deepcopy(probe[last].state_dict());opt_before=copy.deepcopy(opts[last].state_dict())
    first=names[0];update({first:probe[first]},{first:opts[first]},b,decoder,lp,scale,micro)
    assert same_tree(before,probe[last].state_dict()) and same_tree(opt_before,opts[last].state_dict())
    for model in probe.values():
        _,wave=forward(model,b);torch.testing.assert_close(wave.square().sum((1,2)),torch.full((len(wave),),2.*model.uses,device=wave.device),atol=.02,rtol=1e-5)
        assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
    buffer=io.BytesIO();torch.save({'models':probe.state_dict(),'opts':{n:o.state_dict() for n,o in opts.items()}},buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=True);restored=copy.deepcopy(probe);restored.load_state_dict(saved['models'])
    resumed_opts={n:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for n,m in restored.items()}
    for n in names:resumed_opts[n].load_state_dict(copy.deepcopy(saved['opts'][n]))
    update(probe,opts,b,decoder,lp,scale,micro);update(restored,resumed_opts,b,decoder,lp,scale,micro)
    assert same_tree(probe.state_dict(),restored.state_dict()) and all(same_tree(opts[n].state_dict(),resumed_opts[n].state_dict()) for n in names)
    assert state_sha256(decoder)==frozen and all(p.grad is None for p in decoder.parameters())
    write_json(out/'qualification.json',{'status':'REAL_MICROBATCH8_OPTIMIZER_ISOLATION_AND_BITWISE_RESUME_PASS','microbatch':micro,'effective_batch':len(b['F']),'decoder_sha256':frozen,'arms':names,'synthetic':False,'probe_updates_discarded':True})
