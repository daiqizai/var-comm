"""Reports retain the original scientific calculations and honest pause provenance."""
from pathlib import Path
import sys
import json
from fixed80k_delivery_common import require

DISCLOSURE=('The user explicitly requested evaluation of the exact 80,000-step Swin checkpoint. '
    'Training reached 81,551 updates before the safe pause completed; those later weights are preserved but not evaluated. '
    'This is a fixed-step checkpoint rather than calibration-best selection. Original complete 80k calibration is verified, '
    'but the checkpoint choice is the explicit user request, without development-based selection. '
    'The original 240k maximum and registered plateau/LR protocol remain recorded and unfinished. '
    'budget_truncated=True; user_requested_pause=True; no convergence is claimed. Completion of this evaluation '
    'does not mean completion of the original training protocol. HiFi uses the same exact 80k checkpoint '
    'and full registered frozen-ADM posterior schedule.')

def install_reports(root):
    from fixed80k_adapter import install
    install(root)
    import own_controls_report as own
    import external_eval_report as external
    globals().update({name:getattr(own,name) for name in ('admitted_json','policy_table','c','METHODS','FAMILIES')})
    own.protocol_metadata=protocol_metadata
    original_text=external.report_text
    def text(*args,**kwargs):
        selected=args[4] if len(args)>4 else kwargs['selected']
        require(selected['step']==80000 and selected['selected_by']=='explicit_user_fixed_step','Report selected a different Swin')
        return original_text(*args,**kwargs)+'\n## Explicit user-selected 80k checkpoint\n\n'+DISCLOSURE+'\n'
    external.report_text=text
    original_full=own.text_report
    own.text_report=lambda *a,**k:original_full(*a,**k)+'\n## Explicit user-selected 80k checkpoint\n\n'+DISCLOSURE+'\n'
    original_seal=external.seal
    def seal(path,value):
        if Path(path).name=='report_registration.json':
            value['input_bindings'].update({str(Path(__file__).resolve()):external.sha(__file__)})
        return original_seal(path,value)
    external.seal=seal
    return external,own

# The following function keeps the original P, policy and resource admission
# code. Only the original Swin completion block is replaced by the explicit
# fixed milestone receipt. All input checks still bind the completed images.
def protocol_metadata(root,indexed,own_registration,external_completion):
    root=Path(root);own_inputs=dict(own_registration['input_bindings']);bindings={}
    # P512 is context only and is not constructed by the selected N1024 replay.
    # Its exact completion remains authenticated by the original completed
    # unified evaluation registration, which the own scorer already admits.
    old_registration=root/'results/unified_metrics_20261002/metrics_registration.json'
    published_original=admitted_json(old_registration,own_inputs)
    for path,digest in published_original['frozen_policy_bindings'].items():
        require(path not in own_inputs or own_inputs[path]==digest,'Conflicting original model/policy identity')
        own_inputs[path]=digest
    bindings[str(old_registration)]=c.sha(old_registration)
    old_dir=root/'results/extreme_bandwidth_20261001_R1_N1024'
    paths=[old_dir/'digital_selected_policy.json',root/'results/scale_causal_partial_residual_20261002/m1_policy.json',
           root/'results/external_comparison_20261004/own_controls/N2048_policy.json']
    policies=[admitted_json(p,own_inputs) for p in paths];bindings.update({str(p):c.sha(p) for p in paths})
    require(policies[2]==own_registration['policy'],'Score and report N2048 policy differ')
    policy_rows=policy_table(indexed,*policies)
    training=[]
    for n,name in ((512,'EXTREME-BW-20260930-R1'),(1024,'EXTREME-BW-20261001-R1-N1024')):
        folder=root/'outputs'/name/'training'/f'p{n}_2026093001'
        dp,rp=folder/'completion.json',folder/'registration.json'
        done,reg=admitted_json(dp,own_inputs),admitted_json(rp,own_inputs)
        require(done.get('status')=='CALIBRATION_ONLY_SELECTION_COMPLETE' and done.get('synthetic') is False
                and done.get('development_read') is False and done['registration_sha256']==c.sha(rp)
                and done['state']['finished'] is True,'Original continuous training is not complete')
        chosen=done['selected'][f'P{n}'];decision=done['state']['decisions'][-1]
        require(type(done.get('budget_truncated')) is bool and done['budget_truncated']==decision['budget_truncated']
                and chosen['training_seed']==2026093001,'Original budget/seed disclosure differs')
        training.append(dict(method=f'P{n}',N=n,main_comparison=n==1024,training_seed=chosen['training_seed'],
            selected_step=int(chosen['step']),completed_step=int(done['state']['step']),
            maximum_steps=int(reg['protocol']['maximum_updates']),budget_truncated=done['budget_truncated'],
            stop_reason=decision['reason'],convergence_claimed=False,checkpoint_sha256=chosen['checkpoint_sha256'],
            loss='RGB MSE + 0.1 LPIPS-Alex + 0.01 normalized latent error'))
        bindings.update({str(p):c.sha(p) for p in (dp,rp)})
    # The already authenticated P2048 row carries its original selected model,
    # actual initialization seed and completed training length. No new fit.
    p_metadata=[]
    for i in range(100):
        for s in c.SNRS:
            for seed in c.SEEDS:
                source=json.loads(indexed[i,2048,s,seed,'P']['original_row_json'])
                meta=json.loads(source['history_metadata_json'])
                p_metadata.append({k:meta[k] for k in ('training_seed','initialization_seed','selected_step','training_completed_step','checkpoint_sha256')})
    require(all(x==p_metadata[0] for x in p_metadata) and p_metadata[0]['training_seed']==2026092304,
            'P2048 rows use different preselected training runs')
    p=p_metadata[0]
    training.append(dict(method='P2048',N=2048,main_comparison=True,training_seed=p['training_seed'],
        initialization_seed=p['initialization_seed'],selected_step=p['selected_step'],completed_step=p['training_completed_step'],
        maximum_steps='',budget_truncated='not_reclassified_original_completed_run',stop_reason='original_selected_checkpoint',
        convergence_claimed=False,checkpoint_sha256=p['checkpoint_sha256'],
        loss='RGB MSE + 0.1 LPIPS-Alex + 0.01 normalized latent error'))
    from fixed80k_adapter import paths,validate_selection,READY
    view=paths(root)['view'];selected,checked=validate_selection(view,root)
    rp,sp,dp=view/'registration.json',view/'selected_swin.json',view/'completion.json'
    for path in (rp,sp,dp):admitted_json(path,external_completion['bindings'])
    done=c.read(dp);config=c.read(rp)['config']
    require(done['status']==READY and selected['step']==80000 and done['actual_training_pause_step']==81551
        and done['training_finished'] is False and done['scientific_convergence_proven'] is False,
        'Exact user-selected 80k disclosure differs')
    require(config['total_N']==[1024,2048] and config['data_N']==[768,1664] and config['header_N']==[256,384]
        and config['maximum_steps']==240000,'Original Swin training and paid resource protocol changed')
    training.append(dict(method='SwinJSCC shared; also frozen HiFi likelihood',N='1024+2048',main_comparison=True,
        training_seed=config['seed'],selected_step=80000,completed_step=81551,
        maximum_steps=config['maximum_steps'],budget_truncated=True,
        stop_reason='user_requested_pause; evaluate_exact_80000; actual_updates_81551; original_training_unfinished',
        convergence_claimed=False,checkpoint_sha256=selected['checkpoint_sha256'],loss='RGB MSE only'))
    bindings.update({**checked,str(Path(__file__).resolve()):c.sha(__file__)})
    resources=[dict(N=n,comparison_family=m,method=METHODS[n][FAMILIES.index(m)],
        body_N=n if m=='P' else n-68 if m in ('D_U','M1') else 768 if n==1024 else 1664,
        header_N=0 if m=='P' else 68 if m in ('D_U','M1') else 256 if n==1024 else 384,
        E=2*n,metadata=('none' if m=='P' else 'paid 10-bit class ignored by unconditional receiver; mode; CRC/tail/rate-matching'
            if m=='D_U' and n==1024 else 'm/K/order; CRC/tail/rate-matching; no class' if m in ('D_U','M1')
            else 'paid mask rank/power; CRC/tail/rate-matching')) for n in c.BUDGETS for m in FAMILIES]
    return dict(policies=policy_rows,training=training,resources=resources),bindings

