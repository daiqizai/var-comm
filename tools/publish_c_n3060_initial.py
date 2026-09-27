"""CPU publication of the first registered N3060/m6 C_train 20k boundary.

A thermal artifact capture is explicitly distinct from a successful stage receipt.
It never reconstructs a scheduler receipt, exit code, or finalization.
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
    extension_evidence,
)
from tools.publish_c_extension import (
    completed_snapshot, validate_selection, checkpoint_identity, validate_decision,
)
from tools.audit_c_terminal_boundary import validate_thermal_sequence

SEED = 2026092304
ARMS = ['H6-V', 'H6-P']
COMMAND = ['token_efficiency.C_train', '--N', '3060', '--group', 'm6',
           '--seed', str(SEED), '--until', '20000']
KEY = f'C_N3060_m6_seed{SEED}_until20000'
SRC = SHORT/'scoped_N3060/training'/f'm6_seed{SEED}'
TARGET = ROOT/'results/token_channel_efficiency_20260923/C_initial_milestones'/f'N3060_m6_seed{SEED}_20k'


def validate_scope(reg):
    check((reg['N'], reg['group'], reg['seed']) == (3060, 'm6', SEED) and
          reg['parent'] is None and reg['fresh_pure_repeat'] is False,
          'only original fresh N3060/m6 C_train run')
    ledger = reg['cache_calibration']['scope']['ledger']
    check([ledger[k] for k in ('N','NH','ND','NA','E')] == [3060,68,1200,1792,6120],
          'registered N3060 resource allocation')


def validate_boundary(done, regsha):
    check(done['status'] == 'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE' and
          done['synthetic'] is False and done['registration_sha256'] == regsha and
          done['state']['step'] == done['state']['last_full'] == 20000 and
          done['state']['updates'] == {a:20000 for a in ARMS} and
          set(done['selected']) == set(ARMS), 'real paired initial 20k boundary')


def capture_boundary(src, regsha, allow_gap):
    chain = OUT/'delivery_chain_v1'
    stagepath = chain/'stages'/f'{KEY}.json'
    if stagepath.exists():
        stage = read(stagepath)
        check(stage['command'] == COMMAND, 'registered scoped engine command')
        done = completed_snapshot(stage, 20000, ARMS, regsha)
        validate_boundary(done, regsha)
        return done, dict(complete_stage=True, process_returncode=0), {
            'training/stage.json':stagepath,
            'training/completion_snapshot.json':Path(stage['snapshot']),
        }
    check(allow_gap, 'original stage missing; explicit thermal artifact capture required')
    completion = src/'completion.json'
    done = read(completion)
    validate_boundary(done, regsha)
    terminal = src/'checkpoints/step_20000_full_calibration.pt'
    # Mutable latest/status may already belong to the authorized extension.
    # The original 20k completion and immutable full checkpoint remain mandatory.
    check(terminal.exists(), 'immutable 20k checkpoint required')
    attempts = sorted((chain/'attempts').glob(KEY+'_*'))
    check(bool(attempts), 'registered training attempts')
    innerdir = attempts[-1]
    launch = read(innerdir/'launch.json')
    check(launch['command'][3:] == COMMAND and
          launch['session_id'] == launch['process']['pid'], 'scoped terminal launch')
    def exited(x):
        p = Path('/proc')/str(x['process']['pid'])/'stat'
        if p.exists():
            check(p.read_text().split()[21] != str(x['process']['start_ticks']),
                  'terminal process must actually exit')
    exited(launch)
    outers = [(p,read(p/'launch.json')) for p in sorted((chain/'attempts').glob('C_followups_*'))]
    before = [(p,x) for p,x in outers if x['time'] < launch['time']]
    check(bool(before), 'enclosing registered runner')
    outerdir, outer = before[-1]
    exited(outer)
    completed_time = completion.stat().st_mtime
    after = [(p,x) for p,x in outers if x['time'] > completed_time]
    check(bool(after), 'original runner must have resumed')
    resumed_dir, resumed = after[0]
    decisionpath = src/'delivery_decisions/at_20000.json'
    stop = read(innerdir/'stop_request.json')
    outerstop = read(outerdir/'stop_request.json')
    validate_thermal_sequence(launch, stop, outer, outerstop, resumed,
                              completed_time, decisionpath.stat().st_mtime)
    console = (innerdir/'console.log').read_text()
    check('full calibration 20000 ' in console and
          not any(x in console for x in ('Traceback','RuntimeError','FAILED_')),
          'complete terminal calibration log without failure')
    files = {
        'training/completion_snapshot.json':completion,
        'thermal/inner_launch.json':innerdir/'launch.json',
        'thermal/inner_stop.json':innerdir/'stop_request.json',
        'thermal/inner_console.log':innerdir/'console.log',
        'thermal/outer_launch.json':outerdir/'launch.json',
        'thermal/outer_stop.json':outerdir/'stop_request.json',
        'thermal/resumed_launch.json':resumed_dir/'launch.json',
    }
    return done, dict(complete_stage=False, process_returncode=None,
        status='INDEPENDENT_INITIAL_BOUNDARY_ARTIFACT_CAPTURE_STAGE_RECEIPT_MISSING',
        completion_mtime=completed_time, original_command=COMMAND,
        explanation='Original C_train completed 20k while the enclosing runner stopped for thermal protection. The resumed lifecycle made its original calibration-only decision. This captures artifacts, not a scheduler receipt, process exit code, or finalization.'), files


def collect(allow_gap=False, audit_only=False):
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'mask CUDA before CPU audit')
    check(audit_only or not TARGET.exists(), 'immutable publication already exists')
    regpath = SRC/'registration.json'
    regsha = digest(regpath)
    reg = read(regpath)
    validate_scope(reg)
    done, boundary, files = capture_boundary(SRC, regsha, allow_gap)
    cfg = reg['protocol']
    sources = reg['source_image_bindings']['calibration']
    check(len(sources) == len({s['image_id'] for s in sources}) == 1000,
          'original calibration population')
    files.update({'training/registration.json':regpath,
                  'training/qualification.json':SRC/'qualification.json',
                  'shared/cache_runtime.json':OUT/'C_followups/cache_runtime.json',
                  'shared/pretrain_acceptance.json':OUT/'C_followups/N3060_m6_acceptance.json',
                  'shared/candidate.json':OUT/'C_followups/candidate.json',
                  'shared/P3060_reuse.json':OUT/'C_followups/P3060_reuse.json'})
    bindings = dict(reg['bindings'])
    bindings.update(read(files['shared/cache_runtime.json'])['bindings'])
    qual = read(files['training/qualification.json'])
    check(qual['status'] == 'REAL_GPU_GRADIENT_ENERGY_POPULATED_OPTIMIZER_ISOLATION_AND_BITWISE_RESUME_PASS' and
          qual['probe_updates_discarded'] and not qual['synthetic'] and
          qual['decoder_sha256'] == reg['decoder_state_sha256'], 'real training qualification')
    pre = read(files['shared/pretrain_acceptance.json'])
    check(pre['status'] == 'REAL_N3060_PRETRAIN_CACHE_ONLINE_CLEAN_RX_PASS' and
          not pre['synthetic'] and len(pre['checks']) == 8, 'real pretrain acceptance')
    bindings.update(pre['bindings'])
    for name in ('C_pretrain_N3060_m6','C_cache_N3060_m6_calibration','C_cache_N3060_m6_train'):
        p = OUT/'delivery_chain_v1/stages'/f'{name}.json'
        s = read(p)
        check(s['returncode'] == 0 and digest(s['snapshot']) == s['completion_sha256'],
              'pretrain/cache immutable stage')
        files[f'stages/{name}.json'] = p
        files[f'stages/{name}_completion.json'] = Path(s['snapshot'])
    tensors = {}
    for role in ('train','calibration'):
        cache = SHORT/'cache/N3060_m6_ND1200'/role
        comp = read(cache/'completion.json')
        creg = read(cache/'registration.json')
        check(comp == reg['cache_'+role] and not comp['synthetic'] and
              comp['registration_sha256'] == digest(cache/'registration.json') and
              comp['scope'] == creg['scope'], 'cache completion and registration')
        check(len(comp['hashes']) == (200 if role == 'train' else 10), 'cache shard coverage')
        bindings.update(creg['bindings'])
        files[f'cache/{role}/registration.json'] = cache/'registration.json'
        files[f'cache/{role}/completion.json'] = cache/'completion.json'
        for name, sha in comp['hashes'].items():
            p = cache/Path(name).with_suffix('.json')
            meta = read(p)
            check(meta['sha256'] == sha and meta['identity']['scope'] == comp['scope'],
                  'cache shard scope and identity')
            files[f'cache/{role}/{p.name}'] = p
            if role == 'calibration':
                check(digest(cache/name) == sha, 'calibration tensor SHA')
                tensors[str(cache/name)] = sha
    for p, sha in bindings.items():
        check(digest(p) == sha, 'bound source/asset identity: '+p)
    curves, snrs, means = [], [], {}
    for step in range(0,20001,2500):
        p = SRC/'calibration'/f'full_{step:05d}.json'
        meta = read(p)
        c = p.with_suffix('.csv')
        check(meta['step'] == step and digest(c) == meta['sha256'], 'calibration CSV SHA')
        rows = list(csv.DictReader(c.open()))
        check(meta['rows'] == len(rows) == 30000, 'complete paired calibration')
        validate_rows(rows, ARMS, sources, cfg['snrs_db'], cfg['calibration_seeds'])
        for arm in ARMS:
            rr = [r for r in rows if r['method'] == arm]
            item = dict(step=step, method=arm,
                        **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS},
                        header_failures=sum(r['header_ok']=='0' for r in rr),
                        body_crc_failures=sum(r['body_crc_ok']=='0' for r in rr))
            check(abs(item['utility']-meta['summary'][arm]) < 1e-12, 'full mean')
            curves.append(item)
        groups = defaultdict(list)
        sourcegroups = defaultdict(list)
        for r in rows:
            groups[(r['method'],int(r['snr_db']))].append(r)
            sourcegroups[(r['method'],int(r['source_index']),int(r['snr_db']))].append(r)
        for (arm,snr), rr in sorted(groups.items()):
            snrs.append(dict(step=step,method=arm,snr_db=snr,
                **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS},
                header_failures=sum(r['header_ok']=='0' for r in rr),
                body_crc_failures=sum(r['body_crc_ok']=='0' for r in rr)))
        means[step] = [dict(step=step,method=arm,source_index=i,image_id=sources[i]['image_id'],
            snr_db=snr,noise_count=len(rr),
            **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS})
            for (arm,i,snr),rr in sorted(sourcegroups.items())]
        files['calibration/'+p.name] = p
        files['calibration/'+c.name] = c
    validate_selection(done['selected'], curves, ARMS, regsha, 'm6')
    terminal = checkpoint_identity(SRC/'checkpoints/step_20000_full_calibration.pt',
                                   20000, regsha, ARMS)
    check(terminal['state'] == done['state'], 'terminal CPU state equals original completion')
    checkpoints = {terminal['path']:terminal}
    for s in done['selected'].values():
        if s['checkpoint'] not in checkpoints:
            checkpoints[s['checkpoint']] = checkpoint_identity(s['checkpoint'],s['step'],regsha,ARMS)
    dp = SRC/'delivery_decisions/at_20000.json'
    decision = read(dp)
    validate_decision(decision, 20000, curves, ARMS)
    for p, sha in decision['evidence_bindings'].items():
        check(digest(p) == sha, 'official decision evidence')
    files['decisions/at_20000.json'] = dp
    originals = {n:dict(source=str(p),sha256=digest(p),bytes=p.stat().st_size) for n,p in files.items()}
    audit = dict(status='REAL_N3060_INITIAL_20K_CALIBRATION_VERIFIED'+
                 ('' if boundary['complete_stage'] else '_STAGE_RECEIPT_MISSING'),
        N=3060,group='m6',training_seed=SEED,until=20000,arms=ARMS,calibration_rows=270000,
        synthetic=False,terminal_artifacts_verified=True,boundary_evidence=boundary,
        complete_stage=boundary['complete_stage'],process_returncode=boundary['process_returncode'],
        selected=done['selected'],terminal_checkpoint=terminal,checkpoints=list(checkpoints.values()),
        checked_bindings=len(bindings),calibration_tensor_hashes_rechecked=tensors,
        training_tensors='All shard manifests retained; actual training loader verification; publisher did not rehash all training tensors',
        extension_evidence=extension_evidence(curves,ARMS),official_decision=decision,
        decoder_state_sha256=reg['decoder_state_sha256'],originals=originals,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tool_sha256=digest(__file__),new_holdout=False,content_selector=False,development_used=False,
        scope='270000 cached calibration scoring rows, not independent digital PHY transmissions; registered E, not new per-frame energy measurements; no selected development or online timing claim.',
        pending=['N3060 calibration-driven extensions','two additional training seeds',
                 'selected development, diagnostics and online timing','four historical GPU workers',
                 'final all-method paired statistics and merged delivery'])
    if audit_only:
        p = OUT/'monitoring'/f'N3060_initial_publication_preparation_{time.time_ns()}.json'
        write(p,audit)
        print('AUDIT_ONLY',p,audit['status'])
        return audit
    # Verify all captured sources remain unchanged before creating an immutable publication.
    check(all(digest(p) == originals[n]['sha256'] for n,p in files.items()), 'source changed during audit')
    TARGET.mkdir(parents=True)
    for n,p in files.items():
        dst = TARGET/n
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
        check(digest(dst) == originals[n]['sha256'], 'copied asset identity')
    write(TARGET/'selected.json',done['selected'])
    write(TARGET/'boundary_evidence.json',boundary)
    ledger = dict(reg['cache_calibration']['scope']['ledger'])
    ledger.update(source_payload_bits=1092,header_class_bits=10,header_mode_bits=2,
        header_crc_bits=16,header_tail_bits=6,header_mother_bits=68,header_coded_slots=136,
        body_crc_bits=16,body_tail_bits=6,body_mother_bits=2228,body_coded_slots=2400,
        body_effective_information_rate=1114/2400,
        energy_kind='registered E6120 constraint; no new per-frame E in calibration',
        online_timing='NOT_RUN for these selected checkpoints',noise=cfg['noise'])
    write(TARGET/'resource_ledger.json',ledger)
    csv_write(TARGET/'calibration_curve.csv',curves)
    csv_write(TARGET/'calibration_by_snr.csv',snrs)
    for step, rr in means.items():
        csv_write(TARGET/f'source_means_{step:05d}.csv',rr)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fields,name in [(('utility','U_image'),'selection'),(('mse','lpips_alex','normalized_latent'),'components')]:
        fig,axs=plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False)
        for ax,k in zip(axs[0],fields):
            for arm in ARMS:
                rr=[r for r in curves if r['method']==arm]
                ax.plot([r['step'] for r in rr],[r[k] for r in rr],marker='o',label=arm)
            ax.set(xlabel='Updates',ylabel=k,title='N3060 / 1000 calibration sources')
            ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(TARGET/f'calibration_{name}.svg');plt.close(fig)
    for p in TARGET.glob('*.svg'):
        p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    write(TARGET/'audit.json',audit)
    index={str(p.relative_to(TARGET)):dict(sha256=digest(p),bytes=p.stat().st_size)
           for p in sorted(TARGET.rglob('*')) if p.is_file()}
    check(all(x['bytes'] < 10000000 for x in index.values()), 'publication file size')
    write(TARGET/'index.json',dict(status=audit['status'],files=index))
    print(audit['status'],TARGET,len(index))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-thermal-receipt-gap',action='store_true',
        help='Capture only the verified initial N3060 boundary artifacts; never infer stage receipt or exit code')
    parser.add_argument('--audit-only',action='store_true')
    a=parser.parse_args()
    collect(a.allow_thermal_receipt_gap,a.audit_only)


if __name__=='__main__':
    main()

