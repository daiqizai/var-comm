"""CPU-only publication of original N4084 C seed 2026092304 10k extensions."""
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
    ROOT, SHORT, OUT, METRICS, check, digest, read, write, csv_write,
    validate_rows, extension_evidence,
)
from tools.publish_c_pure_milestone import validate_pure_rows

SEED = 2026092304
RESULTS = ROOT/'results/token_channel_efficiency_20260923'


def arms_for(group):
    check(group in ('m6', 'm7', 'm8', 'pure'), 'original registered group')
    return ['P4084'] if group == 'pure' else (
        [group.upper().replace('M', 'H')+'-V'] +
        ([] if group == 'm8' else [group.upper().replace('M', 'H')+'-P']))


def verify_index(folder):
    index = read(folder/'index.json')
    for name, meta in index['files'].items():
        p = folder/name
        check(p.resolve().is_relative_to(folder.resolve()), 'index path outside publication')
        check(p.stat().st_size == meta['bytes'] < 10000000 and
              digest(p) == meta['sha256'], 'published file identity: '+name)
    return dict(path=str(folder), index_sha256=digest(folder/'index.json'),
                files=index['files'])


def validate_decision(decision, step, curves, arms):
    rule = decision['rule']
    check(rule['interval'] == 2500 and rule['extend_updates'] == 10000 and
          rule['minimum_relative_improvement'] == .002 and
          rule['both_intervals_required'] is True, 'original extension rule')
    check(decision['step'] == step and decision['development_used'] is False and
          decision['paired_arms_extend_together'] is True and
          set(decision['arms']) == set(decision['values']) == set(arms),
          'paired calibration-only decision identity')
    support = []
    for arm in arms:
        rr = sorted((r for r in curves if r['method'] == arm and
                     r['step'] in (step-5000, step-2500, step)), key=lambda r:r['step'])
        check([r['step'] for r in rr] == [step-5000, step-2500, step],
              'three decision checkpoints required')
        vals = [r['utility'] for r in rr]
        check(len(decision['values'][arm]) == 3 and
              all(abs(a-b) < 1e-12 for a,b in zip(vals, decision['values'][arm])),
              'decision means')
        gains = [(a-b)/a for a,b in zip(vals, vals[1:])]
        actual = decision['arms'][arm]
        check(len(actual['relative_improvements']) == 2 and
              all(abs(a-b) < 1e-12 for a,b in zip(gains, actual['relative_improvements'])),
              'decision improvements')
        extend = all(g >= .002 for g in gains)
        check(actual['extend'] is extend, 'both intervals required for one arm')
        support.append(extend)
    extend = any(support)
    check(decision['extend'] is extend and
          decision['until'] == (step+10000 if extend else step), 'paired extension outcome')
    return extend


def completed_snapshot(stage, until, arms, regsha):
    check(stage['returncode'] == 0 and digest(stage['snapshot']) ==
          stage['completion_sha256'], 'successful immutable stage snapshot')
    done = read(stage['snapshot'])
    check(done['status'] == 'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE' and
          done['registration_sha256'] == regsha and
          done['state']['step'] == done['state']['last_full'] == until and
          done['state']['updates'] == {a:until for a in arms} and
          set(done['selected']) == set(arms), 'completed paired update boundary')
    # stage['completion'] is a live path and can legitimately change on the next extension.
    return done


def validate_selection(selected, curves, arms, regsha, group):
    check(set(selected) == set(arms), 'selected arms')
    for arm in arms:
        item = selected[arm]
        best = min((r for r in curves if r['method'] == arm),
                   key=lambda r:(r['utility'], r['step']))
        check(item['arm_key'] == arm and item['registration_sha256'] == regsha and
              item['step'] == best['step'] and item['total_updates'] == best['step'] and
              item['parent_updates'] == (10000 if group == 'pure' else 0) and
              abs(item['utility']-best['utility']) < 1e-12, 'all-history calibration selection')
        check(digest(item['checkpoint']) == item['checkpoint_sha256'], 'selected checkpoint SHA')


def checkpoint_identity(path, step, regsha, arms):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    check(value['registration_sha256'] == regsha and value['state']['step'] == step and
          value['state']['last_full'] == step and
          value['state']['updates'] == {a:step for a in arms}, 'full checkpoint boundary')
    check(set(value['optimizers']) == set(arms) and
          all(value['optimizers'][a]['state'] for a in arms), 'populated arm optimizers')
    check(all(k in value for k in ('models','order','rng','torch_rng','cuda_rng')),
          'complete resumable checkpoint')
    result = dict(path=str(path), sha256=digest(path), step=step,
                  registration_sha256=regsha, state=value['state'],
                  cpu_payload_verified=True)
    del value
    check(not torch.cuda.is_initialized(), 'CPU verification initialized CUDA')
    return result


def collect(group, until, audit_only=False):
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be masked before CPU audit')
    check(until >= 30000 and until % 10000 == 0, '10k extension boundary after initial 20k')
    arms = arms_for(group)
    run = f'{group}_seed{SEED}'
    src = SHORT/'training'/run
    target = RESULTS/'C_extensions'/f'{run}_until{until}'
    check(audit_only or not target.exists(), 'immutable publication already exists')
    initial = RESULTS/'C_initial_milestones'/f'{run}_20k'
    references = [verify_index(initial)]
    regpath = src/'registration.json'
    regsha = digest(regpath)
    check(regsha == digest(initial/'training/registration.json'), 'initial registration unchanged')
    reg = read(regpath)
    check(reg['group'] == group and reg['seed'] == SEED, 'registered run')
    cfg = reg['protocol']
    sources = reg['source_image_bindings']['calibration']
    check(len(sources) == len({r['image_id'] for r in sources}) == 1000, '1000 calibration sources')
    bindings = dict(reg['bindings'])
    ready = ROOT/'experiments/token_channel_efficiency_20260923/C_followups_ready.json'
    bindings.update(read(ready)['bindings'])
    bindings.update(read(OUT/'C_followups/cache_runtime.json')['bindings'])
    for p,h in bindings.items():
        check(digest(p) == h, 'bound dependency changed: '+p)
    initial_audit = read(initial/'audit.json')
    tensors = {}
    for name,h in initial_audit['calibration_tensor_hashes_rechecked'].items():
        p = Path(name) if Path(name).is_absolute() else Path(
            initial_audit['originals']['cache/calibration/completion.json']['source']).parent/name
        check(digest(p) == h, 'calibration tensor identity')
        tensors[str(p)] = h
    check(len(tensors) == 10, 'ten calibration tensors')
    previous = initial if until == 30000 else RESULTS/'C_extensions'/f'{run}_until{until-10000}'
    if until > 30000:
        for prior_until in range(30000, until, 10000):
            references.append(verify_index(RESULTS/'C_extensions'/f'{run}_until{prior_until}'))
        prior = read(previous/'audit.json')
        check(prior['complete_stage'] and prior['until'] == until-10000 and
              prior['group'] == group, 'previous extension publication required')
    files = {'training/registration.json':regpath, 'scheduler/readiness.json':ready}
    stagepath = OUT/'delivery_chain_v1/stages'/f'C_N4084_{run}_until{until}.json'
    done = None
    if stagepath.exists():
        stage = read(stagepath)
        expected = ['short_prefix.train','--group',group,'--seed',str(SEED),'--until',str(until)]
        check(stage['command'] == expected, 'original training stage command')
        done = completed_snapshot(stage, until, arms, regsha)
        files.update({'training/stage.json':stagepath,
                      'training/completion_snapshot.json':Path(stage['snapshot']),
                      'training/final_attempt_launch.json':Path(stage['attempt'])/'launch.json'})
    check(audit_only or done is not None, 'real completed stage required before publication')
    start = 10000 if group == 'pure' else 0
    steps = []
    for step in range(start, until+1, 2500):
        if not (src/'calibration'/f'full_{step:05d}.json').exists():
            break
        steps.append(step)
    check(steps and steps[-1] >= 20000, 'initial calibration history required')
    check(audit_only or steps[-1] == until, 'complete extension calibration history')
    curves, by_snr, source_means, total = [], [], {}, 0
    for step in steps:
        meta_path = src/'calibration'/f'full_{step:05d}.json'
        p = meta_path.with_suffix('.csv')
        meta = read(meta_path)
        check(meta['step'] == step and meta['sha256'] == digest(p), 'calibration receipt/hash')
        rows = list(csv.DictReader(p.open()))
        check(meta['rows'] == len(rows) == 15000*len(arms), 'complete calibration row count')
        if group == 'pure':
            validate_pure_rows(rows, sources, cfg['snrs_db'], cfg['calibration_seeds'])
        else:
            validate_rows(rows, arms, sources, cfg['snrs_db'], cfg['calibration_seeds'])
        total += len(rows)
        for arm in arms:
            rr = [r for r in rows if r['method'] == arm]
            item = dict(step=step, method=arm,
                        **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS})
            check(abs(item['utility']-meta['summary'][arm]) < 1e-12, 'calibration mean')
            item.update(header_failures=0 if group == 'pure' else sum(r['header_ok']=='0' for r in rr),
                        body_crc_failures='not_applicable' if group == 'pure' else sum(r['body_crc_ok']=='0' for r in rr))
            curves.append(item)
        if step <= until-10000:
            anchor = initial if step <= 20000 else RESULTS/'C_extensions'/f'{run}_until{((step-1)//10000+1)*10000}'
            for x in (meta_path,p):
                check(digest(anchor/'calibration'/x.name) == digest(x), 'published historical calibration changed')
            continue
        files['calibration/'+p.name] = p
        files['calibration/'+meta_path.name] = meta_path
        groups = defaultdict(list)
        for row in rows:
            groups[(row['method'],int(row['snr_db']))].append(row)
        for (arm,snr),rr in sorted(groups.items()):
            by_snr.append(dict(step=step,method=arm,snr_db=snr,
                              **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS},
                              header_failures=0 if group=='pure' else sum(r['header_ok']=='0' for r in rr),
                              body_crc_failures='not_applicable' if group=='pure' else sum(r['body_crc_ok']=='0' for r in rr)))
        groups = defaultdict(list)
        for row in rows:
            groups[(row['method'],int(row['source_index']),int(row['snr_db']))].append(row)
        source_means[step] = [dict(step=step,method=a,source_index=i,image_id=sources[i]['image_id'],
                                  snr_db=snr,noise_count=len(rr),
                                  **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS})
                              for (a,i,snr),rr in sorted(groups.items())]
    decisions = {}
    for boundary in range(20000, until+1, 10000):
        p = src/'delivery_decisions'/f'at_{boundary:05d}.json'
        if boundary > steps[-1] or not p.exists():
            check(audit_only, 'formal scheduler decision required')
            break
        dec = read(p)
        extend = validate_decision(dec, boundary, curves, arms)
        if boundary < until:
            check(extend, 'extension must be authorized by prior calibration decision')
        for q,h in dec['evidence_bindings'].items():
            check(digest(q) == h, 'decision evidence binding')
        decisions[str(boundary)] = dec
        files['decisions/'+p.name] = p
    parent = checkpoint_identity(src/'checkpoints'/f'step_{until-10000:05d}.pt',
                                 until-10000, regsha, arms)
    if until == 30000:
        initial_selected = read(initial/'selected.json')
        boundary = [s for s in initial_selected.values() if s['step'] == 20000]
        check(boundary and all(s['checkpoint_sha256'] == parent['sha256'] for s in boundary),
              'published initial boundary checkpoint unchanged')
    if until > 30000:
        check(parent['sha256'] == read(previous/'audit.json')['terminal_checkpoint']['sha256'],
              'previous completed boundary checkpoint unchanged')
    terminal = None
    if done:
        validate_selection(done['selected'], curves, arms, regsha, group)
        terminal = checkpoint_identity(src/'checkpoints'/f'step_{until:05d}.pt',until,regsha,arms)
        check(terminal['state'] == done['state'], 'terminal payload/completion state')
    audit = dict(status='C_EXTENSION_CPU_PREPARATION' if audit_only else 'REAL_C_EXTENSION_CALIBRATION_VERIFIED',
                 synthetic=False, complete_stage=done is not None, group=group, until=until,
                 training_seed=SEED, N=4084, calibration_steps=steps, calibration_rows=total,
                 new_calibration_rows=len(source_means)*15000*len(arms), arms=arms,
                 selected=done['selected'] if done else None, parent_checkpoint=parent,
                 terminal_checkpoint=terminal, checked_bindings=len(bindings),
                 calibration_tensor_hashes_rechecked=tensors,
                 training_tensors='Original real loader verification and published manifests; not rehashed by this publisher',
                 extension_evidence=extension_evidence(curves,arms), decisions=decisions,
                 decoder_state_sha256=reg['decoder_state_sha256'], references=references,
                 new_holdout=False, content_selector=False, development_used=False,
                 energy='Registered constraint only; calibration has no new per-frame E measurement',
                 online_timing='NOT_RUN by this publisher; calibration/engine seconds are not online TX/RX',
                 pending=['calibration-driven remaining C extensions','N3060 and additional training seeds',
                          'selected development, paired statistics and online timing',
                          'four registered historical GPU workers and final merged delivery'],
                 tool_sha256=digest(__file__),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if audit_only:
        output = OUT/'monitoring'/f'C_extension_preparation_{group}_{time.time_ns()}.json'
        write(output,audit)
        print('PREPARATION_ONLY',output,total)
        return audit
    check(len(source_means) == 4, 'exactly four new full calibrations per 10k')
    target.mkdir(parents=True)
    for name,p in files.items():
        dst = target/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
    write(target/'references.json',references)
    write(target/'selected.json',done['selected'])
    ledger = read(initial/'resource_ledger.json')
    ledger['online_timing'] = 'NOT_RUN for these selected checkpoints by this publisher'
    write(target/'resource_ledger.json',ledger)
    csv_write(target/'calibration_curve.csv',curves)
    csv_write(target/'calibration_by_snr.csv',by_snr)
    for step,rows in source_means.items():
        csv_write(target/f'source_means_{step:05d}.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metrics,name in [(('utility','U_image'),'selection'),(('mse','lpips_alex','normalized_latent'),'components')]:
        fig,axs=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4),squeeze=False)
        for ax,key in zip(axs[0],metrics):
            for arm in arms:
                rr=[r for r in curves if r['method']==arm]
                ax.plot([r['step'] for r in rr],[r[key] for r in rr],marker='o',label=arm)
            ax.axvline(until-10000,color='grey',ls='--',alpha=.5)
            ax.set(xlabel='Total updates (pure inherits 10k)' if group=='pure' else 'Updates',
                   ylabel=key,title='1000 calibration sources / 15 noise cells')
            ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(target/f'calibration_{name}.svg');plt.close(fig)
    for p in target.glob('*.svg'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    audit['originals']={name:dict(source=str(p),sha256=digest(p)) for name,p in files.items()}
    write(target/'audit.json',audit)
    index={str(p.relative_to(target)):dict(sha256=digest(p),bytes=p.stat().st_size)
           for p in sorted(target.rglob('*')) if p.is_file()}
    check(all(v['bytes']<10000000 for v in index.values()),'publication file size limit')
    write(target/'index.json',dict(status=audit['status'],files=index))
    print(audit['status'],target,total)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True,choices=('m6','m7','m8','pure'))
    parser.add_argument('--until',required=True,type=int)
    parser.add_argument('--audit-only',action='store_true')
    a=parser.parse_args()
    collect(a.group,a.until,a.audit_only)


if __name__=='__main__':
    main()
