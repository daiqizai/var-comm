"""Calibration-only continuation decisions and immutable extension lineage."""
from pathlib import Path
import math
from latent_enhancement.runtime import digest,verify_snapshot
from latent_followup.run_identity import checked_checkpoint
from .common import ROOT,OUT,read,register,bindings
from .microbatch_runtime import verify_execution

RULE={'interval':2500,'extend_updates':10000,'minimum_relative_improvement':.002,'both_intervals_required':True,'source':'registered short-prefix calibration extension criterion; no development input'}

def should_extend(values):
    if len(values)!=3 or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('three positive finite complete-calibration utilities required')
    improvements=[(a-b)/a for a,b in zip(values,values[1:])]
    return all(v>=RULE['minimum_relative_improvement'] for v in improvements),improvements

def next_budget_action(N):
    folder=OUT/f'training/P{N}_seed2026092304';latest=read(folder/'latest.json');cp=checked_checkpoint(latest,ROOT);step=latest['step']
    # Reuse an unfinished registered extension; never change its parent at resume.
    for planfile in sorted((folder/'extensions').glob('until_*/plan.json')):
        plan=read(planfile)
        if not (planfile.parent/'completion.json').exists():return {'kind':'extend','plan':str(planfile),'until':plan['until']}
    if step<20000 or step%10000:raise RuntimeError('full20k/10k boundary required before lifecycle decision')
    evidence={};values=[]
    for s in (step-5000,step-2500,step):
        p=folder/'calibration'/f'full_{s:05d}.json';r=read(p);csv=p.with_suffix('.csv')
        if r['step']!=s or r['rows']!=15000 or r['sha256']!=digest(csv):raise RuntimeError('full calibration receipt identity')
        values.append(r['summary'][f'P{N}']);evidence[str(p)]=digest(p);evidence[str(csv)]=digest(csv)
    extend,improvements=should_extend(values);selected=read(folder/f'selected_P{N}.json');checked_checkpoint(selected,ROOT)
    decision={'N':N,'step':step,'rule':RULE,'utilities':values,'relative_improvements':improvements,'extend':extend,'evidence_bindings':evidence,'selected':selected,'latest':latest,'development_used':False}
    decisionfile=folder/'decisions'/f'at_{step:05d}.json';register(decisionfile,decision)
    if extend:
        until=step+10000;plan={'N':N,'until':until,'parent':latest,'microbatch':8 if N==2048 else 4,'decision':str(decisionfile),'decision_sha256':digest(decisionfile),'evidence_bindings':evidence,'version':f'budget-extension-N{N}-until{until}'}
        planfile=folder/'extensions'/f'until_{until}'/'plan.json';register(planfile,plan)
        return {'kind':'extend','plan':str(planfile),'until':until}
    final={'status':'CALIBRATION_EXTENSION_RULE_STOP_SUPPORTED','N':N,'completed_step':step,'selected':selected,'decision':str(decisionfile),'decision_sha256':digest(decisionfile),'synthetic':False,'convergence_claimed':False}
    register(folder/'finalization.json',final);return {'kind':'final','path':str(folder/'finalization.json')}

def register_extension(out,plan,planfile,historical_sha,entry):
    if plan['N'] not in (2048,3060) or plan['microbatch']!=(8 if plan['N']==2048 else 4) or plan['until']!=plan['parent']['step']+10000:raise RuntimeError('registered extension scope')
    decision=read(plan['decision'])
    if digest(plan['decision'])!=plan['decision_sha256'] or not decision['extend'] or decision['latest']!=plan['parent']:raise RuntimeError('calibration decision binding')
    verify_snapshot(plan['evidence_bindings'])
    path=Path(planfile).parent/'registration.json'
    record={'historical_registration_sha256':historical_sha,'actual_microbatch':plan['microbatch'],'effective_batch':16,'spec':plan,'bindings':bindings([entry,__file__,Path(__file__).with_name('microbatch_runtime.py'),planfile,plan['decision']]),'parent_checkpoint':plan['parent'],'precision_changed':False}
    register(path,record);return {'path':str(path),'sha256':digest(path),'version':plan['version'],'microbatch':plan['microbatch']}

def validate_extension_resume(payload,execution,plan,checkpoint):
    previous=payload.get('execution_identity')
    if previous==execution:
        verify_execution(previous)
        if not plan['parent']['step']<=payload['state']['step']<=plan['until']:raise RuntimeError('extension step range')
    else:
        if previous:verify_execution(previous)
        if digest(checkpoint)!=plan['parent']['sha256'] or payload['state']['step']!=plan['parent']['step']:raise RuntimeError('extension requires exact parent checkpoint')
