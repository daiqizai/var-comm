"""Independent CPU publication for the two registered additional C seeds.

Only initial 20k boundaries at N4084 are supported. Hybrid repeats use
short_prefix.train; pure repeats use fresh token_efficiency.C_train.
No experiment output, active dependency, scheduler receipt or exit code is edited.
"""
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from tools.publish_c_initial_milestone import (
    ROOT, SHORT, OUT, METRICS, check, digest, read, write, csv_write, validate_rows,
)
from tools.publish_c_extension import completed_snapshot, verify_index, validate_decision
from tools.publish_c_pure_milestone import validate_pure_rows
from tools.audit_c_terminal_boundary import validate_thermal_sequence

SEEDS = (2026092404, 2026092504)
ARMS = {'m6':['H6-V','H6-P'], 'm8':['H8-V'], 'pure':['P4084']}
RESULTS = ROOT/'results/token_channel_efficiency_20260923'


def scope(group, seed):
    check(group in ARMS and type(seed) is int and seed in SEEDS,
          'only registered N4084 additional seeds and frozen m6/m8/pure groups')
    src = (SHORT/'scoped_N4084' if group == 'pure' else SHORT)/'training'/f'{group}_seed{seed}'
    command = (['token_efficiency.C_train','--N','4084'] if group == 'pure'
               else ['short_prefix.train'])
    command += ['--group',group,'--seed',str(seed),'--until','20000']
    return src, command, RESULTS/'C_initial_milestones'/f'N4084_{group}_seed{seed}_20k'


def validate_scope(reg, group, seed):
    scope(group, seed)
    check(reg['group'] == group and reg['seed'] == seed and reg['parent'] is None,
          'fresh repeat scope; no inherited first-seed pure parent')
    cfg = reg['protocol']
    check(cfg['repeats']['additional_training_seeds'] == list(SEEDS),
          'original repeat seed protocol')
    check(set(reg['parameter_counts']) == set(ARMS[group]), 'registered repeat arms')
    if group == 'pure':
        check(reg['N'] == 4084 and reg['fresh_pure_repeat'] is True,
              'pure must use fresh scoped N4084 engine')
        check(reg['cache_calibration'] == reg['cache_train'] ==
              {'source':'original F, checked shard hashes'}, 'original F cache only')
    else:
        check('N' not in reg and 'fresh_pure_repeat' not in reg,
              'hybrid must use original short_prefix engine')
        ledger = reg['cache_calibration']['scope']['ledger']
        nd = 1200 if group == 'm6' else 2992
        check([ledger[k] for k in ('N','NH','ND','NA','E')] ==
              [4084,68,nd,4084-68-nd,8168], 'hybrid paid allocation')
    check(reg['order_seed'] == cfg['data_seed']+seed-cfg['initialization_seed'] and
          reg['channel_seed'] == cfg['channel_seed']+seed-cfg['initialization_seed'],
          'repeat data/channel RNG lineage')


def validate_boundary(done, regsha, arms):
    check(done['status'] == 'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE' and
          done['synthetic'] is False and done['registration_sha256'] == regsha and
          done['state']['step'] == done['state']['last_full'] == 20000 and
          done['state']['updates'] == {a:20000 for a in arms} and
          set(done['selected']) == set(arms), 'complete real initial paired/single boundary')


def checkpoint_payload(value, step, regsha, arms, full=True):
    st = value['state']
    check(value['registration_sha256'] == regsha and st['step'] == step and
          st['updates'] == {a:step for a in arms} and
          (not full or st['last_full'] == step), 'checkpoint update/registration identity')
    check(set(value['optimizers']) == set(arms) and
          all(k in value for k in ('models','order','rng','torch_rng','cuda_rng')) and
          bool(value['models']), 'complete model/optimizer/resume state')
    check(all(bool(value['optimizers'][a]['state']) == (step > 0) for a in arms),
          'optimizer empty only before first real update')


def cpu_checkpoint(path, step, regsha, arms, full=True):
    import torch
    x = torch.load(path, map_location='cpu', weights_only=True)
    checkpoint_payload(x,step,regsha,arms,full)
    result = dict(path=str(path),sha256=digest(path),step=step,
                  registration_sha256=regsha,state=x['state'],cpu_payload_verified=True)
    del x
    check(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return result


def validate_selected(selected, curves, regsha, arms):
    check(set(selected) == set(arms), 'all registered selected arms')
    for arm,x in selected.items():
        best = min((r for r in curves if r['method'] == arm),
                   key=lambda r:(r['utility'],r['step']))
        check(x['arm_key'] == arm and x['registration_sha256'] == regsha and
              x['step'] == x['total_updates'] == best['step'] and
              x['parent_updates'] == 0 and
              abs(x['utility']-best['utility']) < 1e-12, 'fresh full-history selection')
        check(digest(x['checkpoint']) == x['checkpoint_sha256'], 'selected checkpoint SHA')


def boundary(src, command, group, seed, regsha, allow_gap):
    chain = OUT/'delivery_chain_v1'
    key = f'C_N4084_{group}_seed{seed}_until20000'
    sp = chain/'stages'/f'{key}.json'
    if sp.exists():
        stage = read(sp)
        check(stage['command'] == command, 'exact engine and seed stage command')
        done = completed_snapshot(stage,20000,ARMS[group],regsha)
        validate_boundary(done,regsha,ARMS[group])
        return done,dict(complete_stage=True,process_returncode=0), {
            'training/stage.json':sp,'training/completion_snapshot.json':Path(stage['snapshot'])}
    check(allow_gap, 'original stage missing; explicit thermal artifact capture required')
    cp = src/'completion.json';done = read(cp)
    validate_boundary(done,regsha,ARMS[group])
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
    dp=src/'delivery_decisions/at_20000.json'
    validate_thermal_sequence(launch,stop,outer,outerstop,resumed,
                              cp.stat().st_mtime,dp.stat().st_mtime)
    log=(inner/'console.log').read_text()
    check('full calibration 20000 ' in log and
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


def audit_inputs(group, seed):
    src,command,target=scope(group,seed)
    regpath=src/'registration.json';reg=read(regpath);regsha=digest(regpath)
    validate_scope(reg,group,seed)
    cfg=reg['protocol'];sources=reg['source_image_bindings']['calibration']
    training=reg['source_image_bindings']['train']
    check(len(sources)==len({x['image_id'] for x in sources})==1000 and
          len(training)==len({x['image_id'] for x in training})==20000 and
          not ({x['image_id'] for x in sources}&{x['image_id'] for x in training}),
          'original disjoint populations')
    base=RESULTS/'C_initial_milestones'/f'{group}_seed2026092304_20k'
    ref=verify_index(base);oldreg=read(base/'training/registration.json')
    check(reg['protocol']==oldreg['protocol'] and
          reg['source_image_bindings']==oldreg['source_image_bindings'] and
          reg['decoder_state_sha256']==oldreg['decoder_state_sha256'],
          'same fixed protocol/source/preprocessing/Decoder as original seed')
    choicepath=OUT/'C_followups/candidate.json';choice=read(choicepath)
    reviewed=RESULTS/'C_initial_milestones/N3060_m6_seed2026092304_20k/shared/candidate.json'
    check(digest(choicepath)==digest(reviewed) and choice['m']==6 and
          choice['additional_seeds']==list(SEEDS), 'frozen calibration-only repeat choice')
    qual=read(src/'qualification.json')
    check(qual['status']=='REAL_GPU_GRADIENT_ENERGY_POPULATED_OPTIMIZER_ISOLATION_AND_BITWISE_RESUME_PASS'
          and qual['arms']==ARMS[group] and qual['probe_updates_discarded'] and
          qual['synthetic'] is False and
          qual['decoder_sha256']==reg['decoder_state_sha256'], 'actual repeat qualification')
    bindings=dict(reg['bindings'])
    runtime=read(OUT/'C_followups/cache_runtime.json');bindings.update(runtime['bindings'])
    oldaudit=read(base/'audit.json');calhashes={};shards={'train':0,'calibration':0}
    for role in shards:
        check(reg['cache_'+role]==oldreg['cache_'+role], 'cache lineage reused unchanged')
    for n,m in oldaudit['originals'].items():
        if not n.startswith('cache/'):
            continue
        p=Path(m['source'])
        check(digest(p)==m['sha256']==digest(base/n), 'original cache receipt changed')
        if p.name=='registration.json':
            bindings.update(read(p)['bindings'])
        if p.name.startswith('shard_'):
            role=p.parent.name;shards[role]+=1;meta=read(p)
            if group!='pure':
                check(meta['identity']['scope']==reg['cache_'+role]['scope'],
                      'cache sample scope')
            if role=='calibration':
                tensor=p.with_suffix('.pt');check(digest(tensor)==meta['sha256'], 'calibration tensor SHA')
                calhashes[str(tensor)]=meta['sha256']
    check(shards=={'train':200,'calibration':10}, 'complete cache shard manifest coverage')
    for p,h in bindings.items():
        check(digest(p)==h, 'bound asset/source changed: '+p)
    return reg,regsha,dict(history_reference=ref,checked_bindings=len(bindings),
        calibration_tensor_hashes_rechecked=calhashes,cache_shards=shards,
        training_tensors='Original loader verification; publisher did not rehash all train tensors',
        candidate_sha256=digest(choicepath))


def calibration(src, reg, group, maximum, partial):
    cfg=reg['protocol'];sources=reg['source_image_bindings']['calibration']
    curves=[];snrs=[];means={};files={};total=0
    for step in range(0,maximum+1,2500):
        p=src/'calibration'/f'full_{step:05d}.json'
        if partial and not p.exists():
            break
        meta=read(p);c=p.with_suffix('.csv')
        check(meta['step']==step and digest(c)==meta['sha256'], 'sealed calibration identity')
        rows=list(csv.DictReader(c.open()))
        check(len(rows)==meta['rows']==15000*len(ARMS[group]), 'full calibration population')
        if group=='pure':
            validate_pure_rows(rows,sources,cfg['snrs_db'],cfg['calibration_seeds'])
        else:
            validate_rows(rows,ARMS[group],sources,cfg['snrs_db'],cfg['calibration_seeds'])
        total+=len(rows)
        def summary(rr):
            return dict(**{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS},
                header_failures=sum(r['header_ok']=='0' for r in rr),
                body_crc_failures=('not_applicable' if group=='pure' else
                                   sum(r['body_crc_ok']=='0' for r in rr)))
        for arm in ARMS[group]:
            item=dict(step=step,method=arm,**summary([r for r in rows if r['method']==arm]))
            check(abs(item['utility']-meta['summary'][arm])<1e-12, 'full mean')
            curves.append(item)
        ss=defaultdict(list);mm=defaultdict(list)
        for r in rows:
            ss[(r['method'],int(r['snr_db']))].append(r)
            mm[(r['method'],int(r['source_index']),int(r['snr_db']))].append(r)
        for (arm,snr),rr in sorted(ss.items()):
            snrs.append(dict(step=step,method=arm,snr_db=snr,**summary(rr)))
        means[step]=[dict(step=step,method=arm,source_index=i,image_id=sources[i]['image_id'],
             snr_db=snr,noise_count=len(rr),
             **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS})
             for (arm,i,snr),rr in sorted(mm.items())]
        files['calibration/'+p.name]=p;files['calibration/'+c.name]=c
    check(bool(curves), 'at least one sealed full calibration required')
    return curves,snrs,means,files,total


def run(group,seed,prepare=False,audit_only=False,allow_gap=False):
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'mask CUDA before CPU audit')
    src,command,target=scope(group,seed)
    check(prepare or audit_only or not target.exists(), 'immutable publication already exists')
    reg,regsha,inputs=audit_inputs(group,seed)
    done=None;boundary_info=None;files={}
    if not prepare:
        done,boundary_info,files=boundary(src,command,group,seed,regsha,allow_gap)
    curves,snrs,means,cals,total=calibration(src,reg,group,20000,prepare)
    files.update(cals)
    selected=({a:read(src/f'selected_{a}.json') for a in ARMS[group]} if prepare else done['selected'])
    validate_selected(selected,curves,regsha,ARMS[group])
    checkpoints={}
    for x in selected.values():
        checkpoints[x['checkpoint']]=cpu_checkpoint(x['checkpoint'],x['step'],regsha,ARMS[group])
    decision=None;terminal=None
    if not prepare:
        suffix='_full_calibration' if group=='pure' else ''
        path=src/'checkpoints'/f'step_20000{suffix}.pt'
        terminal=cpu_checkpoint(path,20000,regsha,ARMS[group])
        check(terminal['state']==done['state'], 'terminal CPU state equals boundary')
        checkpoints[str(path)]=terminal
        dp=src/'delivery_decisions/at_20000.json';decision=read(dp)
        validate_decision(decision,20000,curves,ARMS[group])
        for p,h in decision['evidence_bindings'].items():
            check(digest(p)==h, 'formal decision evidence SHA')
        files['decisions/at_20000.json']=dp
    files.update({'training/registration.json':src/'registration.json',
                  'training/qualification.json':src/'qualification.json',
                  'shared/candidate.json':OUT/'C_followups/candidate.json'})
    originals={n:dict(source=str(p),sha256=digest(p),bytes=p.stat().st_size) for n,p in files.items()}
    status=('C_REPEAT_INITIAL_CPU_PREPARATION_ONLY' if prepare else
            'REAL_C_REPEAT_INITIAL20K_CALIBRATION_VERIFIED'+
            ('' if boundary_info['complete_stage'] else '_STAGE_RECEIPT_MISSING'))
    audit=dict(status=status,N=4084,group=group,training_seed=seed,until=20000,
        engine=command[0],parent=None,parent_updates=0,arms=ARMS[group],synthetic=False,
        calibration_rows=total,sealed_steps=sorted(means),selected=selected,
        terminal_checkpoint=terminal,checkpoints=list(checkpoints.values()),
        official_decision=decision,boundary_evidence=boundary_info,originals=originals,
        complete_stage=boundary_info['complete_stage'] if boundary_info else False,
        process_returncode=boundary_info['process_returncode'] if boundary_info else None,
        terminal_artifacts_verified=not prepare,**inputs,
        tool_sha256=digest(__file__),source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        new_holdout=False,content_selector=False,development_used=False,
        scope='Cached calibration scoring; registered E8168 is not a new per-frame energy measurement. Formal selected development and full online timing remain pending. Image bootstrap and training-seed variability are distinct.',
        GPU='NOT_RUN_BY_THIS_CPU_TOOL')
    if prepare or audit_only:
        p=OUT/'monitoring'/f'C_repeat_{group}_seed{seed}_{time.time_ns()}.json'
        write(p,audit);print(status,p,'rows',total);return audit
    check(total==9*15000*len(ARMS[group]) and sorted(means)==list(range(0,20001,2500)),
          'nine complete fresh training calibration rounds')
    check(all(digest(p)==originals[n]['sha256'] for n,p in files.items()), 'sources changed during audit')
    target.mkdir(parents=True)
    for n,p in files.items():
        dst=target/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        check(digest(dst)==originals[n]['sha256'], 'copied asset identity')
    csv_write(target/'calibration_curve.csv',curves);csv_write(target/'calibration_by_snr.csv',snrs)
    for step,rows in means.items():csv_write(target/f'source_means_{step:05d}.csv',rows)
    write(target/'selected.json',selected);write(target/'boundary_evidence.json',boundary_info)
    write(target/'history_reference.json',inputs['history_reference'])
    base=Path(inputs['history_reference']['path'])
    ledger=read(base/'resource_ledger.json')
    ledger.update(energy_kind='registered E8168, not new measured per-frame energy',
                  online_timing='NOT_RUN for these selected checkpoints',
                  training_seed=seed,parent_updates=0)
    write(target/'resource_ledger.json',ledger)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fields,name in [(('utility','U_image'),'selection'),(('mse','lpips_alex','normalized_latent'),'components')]:
        fig,axs=plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False)
        for ax,k in zip(axs[0],fields):
            for arm in ARMS[group]:
                rr=[x for x in curves if x['method']==arm]
                ax.plot([x['step'] for x in rr],[x[k] for x in rr],marker='o',label=arm)
            ax.set(xlabel='Updates',ylabel=k,title=f'N4084 / seed {seed}')
            ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(target/f'calibration_{name}.svg');plt.close(fig)
    for p in target.glob('*.svg'):
        p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    write(target/'audit.json',audit)
    index={str(p.relative_to(target)):dict(sha256=digest(p),bytes=p.stat().st_size)
           for p in sorted(target.rglob('*')) if p.is_file()}
    check(all(x['bytes']<10000000 for x in index.values()), 'lightweight publication only')
    write(target/'index.json',dict(status=status,files=index))
    print(status,target,len(index));return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=tuple(ARMS),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--allow-thermal-receipt-gap',action='store_true')
    a=p.parse_args()
    run(a.group,a.seed,a.prepare_only,a.audit_only,a.allow_thermal_receipt_gap)


if __name__=='__main__':
    main()
