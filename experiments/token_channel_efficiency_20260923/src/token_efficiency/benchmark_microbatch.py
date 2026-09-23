"""Disposable real-weight microbatch benchmark. Never saves trained weights."""
import copy,json,time
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import require_available,model_paths,perceptual_model,write_json,digest,verify_snapshot
from latent_enhancement_b.common import scale_statistics,load_decoder
from latent_followup.run_identity import checked_checkpoint
from var_comm.next_scale_prior import load_models,state_sha256
from short_prefix.data import Population
from short_prefix.train import update,losses
from .common import OUT,EXP,ROOT,configure_runtime,read,bindings
from .models import BudgetContinuous
from .thermal_guard import active_owned_queue,sample,hot

TOL={'loss_abs':1e-5,'gradient_relative_l2':1e-3,'parameter_update_relative_l2':1e-2}

def vector(items):return torch.cat([v.detach().float().reshape(-1).cpu() for v in items])

def relative(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-20))

def main():
    configure_runtime();require_available();assert not active_owned_queue()
    out=OUT/'microbatch_benchmark_v1';out.mkdir(exist_ok=True)
    device=torch.device('cuda:0');N=2048;name='P2048'
    latest=read(OUT/'training/P2048_seed2026092304/latest.json');checkpoint=checked_checkpoint(latest,ROOT)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    reg=read(OUT/'training/P2048_seed2026092304/registration.json');verify_snapshot(reg['bindings'])
    assert latest['reason']=='safe_pause'
    vae,var=load_models(model_paths(),device);del var;decoder=load_decoder(vae,device);del vae
    scale=scale_statistics(device);lp=perceptual_model(device);cal=Population('calibration',N=N)
    ids=torch.arange(16);b=cal.batch(ids,ids%5,ids%3,[4101+i%3 for i in range(16)],device)
    weights={k[len(name)+1:]:v for k,v in payload['models'].items() if k.startswith(name+'.')}
    original=vector(weights.values());frozen=state_sha256(decoder)
    def build():
        m=BudgetContinuous(scale.cpu(),N).to(device);m.load_state_dict(weights)
        o=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4);o.load_state_dict(copy.deepcopy(payload['optimizers'][name]));return m,o
    def one(m,o,micro):update({name:m},{name:o},b,decoder,lp,scale,micro)
    identity={'source_checkpoint':latest,'registration_sha256':payload['registration_sha256'],'decoder_sha256':frozen,'calibration_sources':cal.ids[:16],'effective_batch':16,'precision':'existing deterministic FP32; TF32 disabled','thresholds':TOL,'timing_rounds':[4,8,16,16,8,4],'warmup_updates':3,'timed_updates':10,'disposable_updates_only':True,'synthetic':False,'bindings':bindings([__file__,Path(__file__).with_name('models.py'),ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py'])}
    write_json(out/'registration.json',identity)
    equivalence=[];reference=None
    for micro in (4,8,16):
        m,o=build();torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():u=torch.cat([losses(m,{k:v[i:i+micro] for k,v in b.items()},decoder,lp,scale)[0].detach().cpu() for i in range(0,16,micro)])
        one(m,o,micro);grad=vector(p.grad for p in m.parameters());delta=vector(m.state_dict().values())-original
        if micro==4:reference=(u,grad,delta)
        row={'microbatch':micro,'loss_max_abs':float((u-reference[0]).abs().max()),'gradient_relative_l2':relative(grad,reference[1]),'parameter_update_relative_l2':relative(delta,reference[2]),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}
        row['pass']=row['loss_max_abs']<=TOL['loss_abs'] and row['gradient_relative_l2']<=TOL['gradient_relative_l2'] and row['parameter_update_relative_l2']<=TOL['parameter_update_relative_l2'];equivalence.append(row)
        del m,o,grad,delta;torch.cuda.empty_cache()
    trials=[]
    for micro in identity['timing_rounds']:
        before=sample()
        if hot(before):raise RuntimeError('thermal benchmark invalid; cool and rerun under a new identity')
        m,o=build()
        for _ in range(identity['warmup_updates']):one(m,o,micro)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
        for _ in range(identity['timed_updates']):one(m,o,micro)
        torch.cuda.synchronize();seconds=(time.perf_counter()-start)/identity['timed_updates'];after=sample()
        r={'microbatch':micro,'seconds_per_update':seconds,'images_per_second':16/seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'thermal_before':before,'thermal_after':after}
        trials.append(r);print('BENCHMARK',json.dumps(r),flush=True)
        del m,o;torch.cuda.empty_cache()
        if hot(after):raise RuntimeError('thermal benchmark invalid')
    assert state_sha256(decoder)==frozen and all(p.grad is None for p in decoder.parameters());verify_snapshot(identity['bindings']);assert digest(checkpoint)==latest['sha256']
    summary={str(m):{'seconds_per_update':float(np.mean([r['seconds_per_update'] for r in trials if r['microbatch']==m])),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in trials if r['microbatch']==m),'equivalence_pass':next(r['pass'] for r in equivalence if r['microbatch']==m)} for m in (4,8,16)}
    eligible=[m for m in (8,16) if summary[str(m)]['equivalence_pass'] and summary[str(m)]['seconds_per_update']<summary['4']['seconds_per_update']*.95]
    chosen=min(eligible,key=lambda m:summary[str(m)]['seconds_per_update']) if eligible else 4
    write_json(out/'acceptance.json',{'status':'REAL_MICROBATCH_BENCHMARK_COMPLETE','identity_sha256':digest(out/'registration.json'),'equivalence':equivalence,'trials':trials,'summary':summary,'recommended_microbatch':chosen,'decision':'at least5% lower update time and numerical tolerance pass; effective batch16 unchanged','synthetic':False,'formal_quality_metrics':False,'training_checkpoints_modified':False})
    print('MICROBATCH_RESULT',chosen,summary,flush=True)
if __name__=='__main__':main()
