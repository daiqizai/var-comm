"""Independent CPU publication of N4084 additional-seed 10k extensions.

No training, GPU evaluation, active dependency mutation or scheduler receipt repair.
Preparation may precede initial20k; it explicitly reports absent boundary prerequisites.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from tools import publish_c_repeat_initial as initial
from tools.publish_c_initial_milestone import (
    ROOT, OUT, check, digest, read, write, csv_write,
)
from tools.publish_c_extension import completed_snapshot, verify_index, validate_decision
from tools.audit_c_terminal_boundary import validate_thermal_sequence

ARMS=initial.ARMS
SEEDS=initial.SEEDS


def scope(group,seed,until):
    src,command,base=initial.scope(group,seed)
    check(type(until) is int and until>=30000 and until%10000==0,
          'registered 10k boundary after initial20k')
    command[-1]=str(until)
    target=initial.RESULTS/'C_extensions'/f'N4084_{group}_seed{seed}_until{until}'
    return src,command,base,target


def checkpoint_path(src,group,step):
    suffix='_full_calibration' if group=='pure' else ''
    return src/'checkpoints'/f'step_{step:05d}{suffix}.pt'


def validate_boundary(done,regsha,arms,until):
    check(done['status']=='REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE' and
          initial.completion_is_real_schema(done,arms) and done['registration_sha256']==regsha and
          done['state']['step']==done['state']['last_full']==until and
          done['state']['updates']=={a:until for a in arms} and
          set(done['selected'])==set(arms),'complete real repeat extension boundary')


def validate_previous(audit,group,seed,until):
    check(audit['N']==4084 and audit['group']==group and audit['training_seed']==seed and
          audit['until']==until-10000 and audit['synthetic'] is False and
          audit['terminal_artifacts_verified'] is True and
          audit['parent'] is None and audit['parent_updates']==0 and
          audit['arms']==ARMS[group], 'same fresh repeat prior published boundary')
    x=audit['terminal_checkpoint']
    check(x['step']==until-10000 and digest(x['path'])==x['sha256'],
          'prior full checkpoint identity')
    d=audit['official_decision']
    check(d['step']==until-10000 and d['extend'] is True and d['until']==until,
          'original prior decision authorizes this extension')
    return x


def history(group,seed,until,regsha):
    src,command,base,target=scope(group,seed,until)
    paths=[base]+[scope(group,seed,s)[3] for s in range(30000,until,10000)]
    refs=[]
    for i,p in enumerate(paths):
        refs.append(verify_index(p))
        check(digest(p/'training/registration.json')==regsha,'published registration identity')
        validate_previous(read(p/'audit.json'),group,seed,30000+i*10000)
    parent=validate_previous(read(paths[-1]/'audit.json'),group,seed,until)
    return refs,paths,parent


def retained_sources(files,group,seed,until):
    _,_,base,_=scope(group,seed,until)
    retained={}
    for n,p in files.items():
        if n.startswith('calibration/'):
            step=int(Path(n).stem.split('_')[1])
            if step<=until-10000:
                owner=base if step<=20000 else scope(group,seed,((step-1)//10000+1)*10000)[3]
                check(digest(p)==digest(owner/n),'historical calibration copy changed')
                continue
        if n in ('training/qualification.json','shared/candidate.json'):
            check(digest(p)==digest(base/n),'initial qualification/candidate identity')
            continue
        retained[n]=p
    return retained


def boundary(src, command, group, seed, regsha, allow_gap, until):
    chain = OUT/'delivery_chain_v1'
    key = f'C_N4084_{group}_seed{seed}_until{until}'
    sp = chain/'stages'/f'{key}.json'
    if sp.exists():
        stage = read(sp)
        check(stage['command'] == command, 'exact engine and seed stage command')
        done = completed_snapshot(stage,until,ARMS[group],regsha)
        validate_boundary(done,regsha,ARMS[group],until)
        return done,dict(complete_stage=True,process_returncode=0), {
            'training/stage.json':sp,'training/completion_snapshot.json':Path(stage['snapshot'])}
    check(allow_gap, 'original stage missing; explicit thermal artifact capture required')
    cp = src/'completion.json';done = read(cp)
    validate_boundary(done,regsha,ARMS[group],until)
    attempts = sorted((chain/'attempts').glob(key+'_*'))
    check(bool(attempts), 'original terminal attempt')
    inner = attempts[-1];launch = read(inner/'launch.json')
    check(launch['command'][3:] == command and
          launch['session_id'] == launch['process']['pid'], 'exact terminal launch')
    def exited(x):
        p = Path('/proc')/str(x['process']['pid'])/'stat'
        if p.exists():
            check(p.read_text().split()[21] != str(x['process']['start_ticks']),
                  'original terminal process must actually exit')
    exited(launch)
    outers = [(p,read(p/'launch.json')) for p in sorted((chain/'attempts').glob('C_followups_*'))]
    before = [(p,x) for p,x in outers if x['time'] < launch['time']]
    check(bool(before), 'enclosing original runner')
    outerpath,outer = before[-1];exited(outer)
    after = [(p,x) for p,x in outers if x['time'] > cp.stat().st_mtime]
    check(bool(after), 'original runner resumed after terminal completion')
    resumedpath,resumed = after[0]
    stop=read(inner/'stop_request.json');outerstop=read(outerpath/'stop_request.json')
    dp=src/'delivery_decisions'/f'at_{until}.json'
    validate_thermal_sequence(launch,stop,outer,outerstop,resumed,
                              cp.stat().st_mtime,dp.stat().st_mtime)
    log=(inner/'console.log').read_text()
    check(f'full calibration {until} ' in log and
          not any(x in log for x in ('Traceback','RuntimeError','FAILED_')),
          'complete terminal calibration without unknown failure')
    files={'training/completion_snapshot.json':cp}
    for name,p in [('inner_launch.json',inner/'launch.json'),
                   ('inner_stop.json',inner/'stop_request.json'),
                   ('inner_console.log',inner/'console.log'),
                   ('outer_launch.json',outerpath/'launch.json'),
                   ('outer_stop.json',outerpath/'stop_request.json'),
                   ('resumed_launch.json',resumedpath/'launch.json')]:
        files['thermal/'+name]=p
    return done,dict(complete_stage=False,process_returncode=None,
        completion_mtime=cp.stat().st_mtime,original_command=command,
        explanation='Verified terminal assets after original nested thermal pause; no scheduler receipt, exit code or finalization reconstructed.'),files



def prepare(group,seed,until):
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','mask CUDA before CPU audit')
    src,command,base,target=scope(group,seed,until)
    reg,regsha,inputs=initial.audit_inputs(group,seed)
    curves,snrs,means,files,total=initial.calibration(src,reg,group,until,True)
    selected={a:read(src/f'selected_{a}.json') for a in ARMS[group]}
    initial.validate_selected(selected,curves,regsha,ARMS[group])
    checkpoints={x['checkpoint']:initial.cpu_checkpoint(x['checkpoint'],x['step'],regsha,ARMS[group])
                 for x in selected.values()}
    prerequisites=[base]+[scope(group,seed,s)[3] for s in range(30000,until,10000)]
    missing=[str(p) for p in prerequisites if not (p/'index.json').exists()]
    refs=[];parent=None
    if not missing:
        refs,_,prior=history(group,seed,until,regsha)
        parent=initial.cpu_checkpoint(prior['path'],until-10000,regsha,ARMS[group])
        check(parent==prior,'prior CPU checkpoint unchanged')
        retained_sources(files,group,seed,until)
        check(max(means)>=until-10000,'complete previous calibration prefix')
    audit=dict(status='C_REPEAT_EXTENSION_CPU_PREPARATION_ONLY',N=4084,group=group,
        training_seed=seed,until=until,registration_sha256=regsha,
        sealed_steps=sorted(means),calibration_rows=total,selected=selected,
        checkpoints=list(checkpoints.values()),parent_checkpoint=parent,
        missing_previous_publications=missing,previous_boundary_verified=not missing,
        history_references=refs,**inputs,boundary_published=False,
        tool_sha256=digest(__file__),GPU='NOT_RUN_BY_THIS_CPU_TOOL')
    path=OUT/'monitoring'/f'C_repeat_extension_preparation_{group}_{seed}_{time.time_ns()}.json'
    write(path,audit);print(audit['status'],path,'rows',total,'missing',len(missing))
    return audit


def collect(group,seed,until,allow_gap=False,audit_only=False):
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','mask CUDA before CPU audit')
    src,command,base,target=scope(group,seed,until)
    check(audit_only or not target.exists(),'immutable publication already exists')
    reg,regsha,inputs=initial.audit_inputs(group,seed)
    refs,paths,prior=history(group,seed,until,regsha)
    done,evidence,files=boundary(src,command,group,seed,regsha,allow_gap,until)
    curves,snrs,means,cals,total=initial.calibration(src,reg,group,until,False)
    check(sorted(means)==list(range(0,until+1,2500)) and
          total==(until//2500+1)*15000*len(ARMS[group]),'complete all-history calibration')
    files.update(cals)
    initial.validate_selected(done['selected'],curves,regsha,ARMS[group])
    parent=initial.cpu_checkpoint(prior['path'],until-10000,regsha,ARMS[group])
    check(parent==prior,'previous complete CPU optimizer/RNG state unchanged')
    terminal=initial.cpu_checkpoint(checkpoint_path(src,group,until),until,regsha,ARMS[group])
    check(terminal['state']==done['state'],'terminal CPU state matches exact boundary')
    checkpoints={terminal['path']:terminal}
    for x in done['selected'].values():
        if x['checkpoint'] not in checkpoints:
            checkpoints[x['checkpoint']]=initial.cpu_checkpoint(
                x['checkpoint'],x['step'],regsha,ARMS[group])
    for step in range(20000,until+1,10000):
        dp=src/'delivery_decisions'/f'at_{step}.json';d=read(dp)
        validate_decision(d,step,curves,ARMS[group])
        for p,h in d['evidence_bindings'].items():
            check(digest(p)==h,'official decision evidence')
        if step<until:
            owner=base if step==20000 else scope(group,seed,step)[3]
            check(digest(dp)==digest(owner/'decisions'/dp.name),'prior decision unchanged')
            check(d['extend'] is True and d['until']==step+10000,'authorized continuation')
        else:
            decision=d;files['decisions/'+dp.name]=dp
    files.update({'training/registration.json':src/'registration.json',
                  'training/qualification.json':src/'qualification.json',
                  'shared/candidate.json':OUT/'C_followups/candidate.json'})
    retained=retained_sources(files,group,seed,until)
    originals={n:dict(source=str(p),sha256=digest(p),bytes=p.stat().st_size)
               for n,p in files.items()}
    audit=dict(status='REAL_C_REPEAT_EXTENSION_CALIBRATION_VERIFIED'+
        ('' if evidence['complete_stage'] else '_STAGE_RECEIPT_MISSING'),
        N=4084,group=group,training_seed=seed,until=until,arms=ARMS[group],
        parent=None,parent_updates=0,engine=command[0],synthetic=False,
        calibration_rows=total,new_calibration_rows=4*15000*len(ARMS[group]),
        history_references=refs,parent_checkpoint=parent,terminal_checkpoint=terminal,
        selected=done['selected'],checkpoints=list(checkpoints.values()),
        official_decision=decision,boundary_evidence=evidence,
        complete_stage=evidence['complete_stage'],process_returncode=evidence['process_returncode'],
        terminal_artifacts_verified=True,originals=originals,**inputs,
        tool_sha256=digest(__file__),source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        new_holdout=False,content_selector=False,development_used=False,
        scope='Cached calibration rescoring, not independent PHY; registered E8168 is not new measured per-frame energy. Source bootstrap and training-seed variation are separate.',
        GPU='NOT_RUN_BY_THIS_CPU_TOOL')
    if audit_only:
        p=OUT/'monitoring'/f'C_repeat_extension_boundary_{group}_{seed}_{until}_{time.time_ns()}.json'
        write(p,audit);print('AUDIT_ONLY',p,audit['status']);return audit
    ledger=read(base/'resource_ledger.json')
    ledger.update(training_seed=seed,parent_updates=0,energy_kind='registered E8168, not new measured per-frame energy',
                  online_timing='NOT_RUN for these selected checkpoints')
    check(all(digest(p)==originals[n]['sha256'] for n,p in files.items()),
          'captured sources changed during audit')
    check(all(p.stat().st_size<10000000 for p in retained.values()),'lightweight captured files only')
    target.mkdir(parents=True)
    for n,p in retained.items():
        dst=target/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        check(digest(dst)==originals[n]['sha256'],'copied identity')
    write(target/'selected.json',done['selected']);write(target/'boundary_evidence.json',evidence)
    write(target/'history_references.json',refs);write(target/'resource_ledger.json',ledger)
    csv_write(target/'calibration_curve.csv',curves);csv_write(target/'calibration_by_snr.csv',snrs)
    for step,rr in means.items():
        if step>until-10000:csv_write(target/f'source_means_{step:05d}.csv',rr)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fields,name in [(('utility','U_image'),'selection'),(('mse','lpips_alex','normalized_latent'),'components')]:
        fig,axs=plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False)
        for ax,k in zip(axs[0],fields):
            for arm in ARMS[group]:
                rr=[r for r in curves if r['method']==arm]
                ax.plot([r['step'] for r in rr],[r[k] for r in rr],marker='o',label=arm)
            ax.set(xlabel='Updates',ylabel=k,title=f'N4084 / seed {seed}')
            ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(target/f'calibration_{name}.svg');plt.close(fig)
    for p in target.glob('*.svg'):
        p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    write(target/'audit.json',audit)
    index={str(p.relative_to(target)):dict(sha256=digest(p),bytes=p.stat().st_size)
           for p in sorted(target.rglob('*')) if p.is_file()}
    check(all(x['bytes']<10000000 for x in index.values()),'lightweight publication only')
    write(target/'index.json',dict(status=audit['status'],files=index))
    print(audit['status'],target,len(index));return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=tuple(ARMS),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--until',type=int,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--audit-only',action='store_true')
    p.add_argument('--allow-thermal-receipt-gap',action='store_true')
    a=p.parse_args()
    if a.prepare_only:prepare(a.group,a.seed,a.until)
    else:collect(a.group,a.seed,a.until,a.allow_thermal_receipt_gap,a.audit_only)


if __name__=='__main__':
    main()
