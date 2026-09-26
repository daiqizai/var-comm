"""Audit/publish the original P4084 10k -> 20k continuation, without GPU work."""
import argparse
import csv
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from tools.publish_c_initial_milestone import (
    ROOT, SHORT, OUT, METRICS, check, digest, read, write, csv_write,
    validate_rows, extension_evidence,
)

ARM = 'P4084'
RUN = 'pure_seed2026092304'
STEPS = tuple(range(10000, 20001, 2500))


def validate_pure_rows(rows, sources, snrs, seeds):
    # Reuse complete key/formula validation only after checking the pure schema.
    check(all(r['method'] == ARM and r['header_ok'] == '1' and
              r['body_crc_ok'] == 'not_applicable' for r in rows),
          'pure has no digital header or body CRC')
    validate_rows([dict(r, body_crc_ok='1') for r in rows],
                  [ARM], sources, snrs, seeds)


def validate_completion(done, regsha):
    check(done['status'] == 'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE',
          'successful initial milestone required')
    check(done['registration_sha256'] == regsha, 'completion registration')
    state = done['state']
    check(state['step'] == state['last_full'] == 20000 and
          state['updates'] == {ARM: 20000}, 'pure total 20k completion')
    check(set(done['selected']) == {ARM}, 'single pure arm')
    selected = done['selected'][ARM]
    check(selected['step'] in STEPS and selected['total_updates'] == selected['step']
          and selected['parent_updates'] == 10000 and selected['arm_key'] == ARM
          and selected['registration_sha256'] == regsha, 'pure selected lineage')


def same_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def validate_parent_payloads(parent, initial, parent_regsha, regsha):
    check(parent['registration_sha256'] == parent_regsha and
          initial['registration_sha256'] == regsha, 'checkpoint registrations')
    check(parent['state']['step'] == 10000 and
          parent['state']['updates']['pure_continuous'] == 10000 and
          initial['state']['step'] == 10000 and
          initial['state']['updates'] == {ARM: 10000}, '10k parent accounting')
    prefix = 'pure_continuous.'
    expected = {ARM+'.'+k[len(prefix):]: v for k, v in parent['models'].items()
                if k.startswith(prefix)}
    check(bool(expected) and same_tree(expected, initial['models']),
          'inherited model parameters/buffers')
    check(same_tree(parent['optimizers']['pure_continuous'], initial['optimizers'][ARM]),
          'inherited populated optimizer')
    for name in ('order', 'rng'):
        check(same_tree(parent[name], initial[name]), 'inherited '+name)


def collect(audit_only=False):
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit requires CUDA_VISIBLE_DEVICES empty')
    import torch
    check(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    src = SHORT/'training'/RUN
    target = ROOT/'results/token_channel_efficiency_20260923/C_initial_milestones'/f'{RUN}_20k'
    if not audit_only:
        check(not target.exists(), 'immutable publication already exists')
    regpath = src/'registration.json'
    reg, regsha = read(regpath), digest(regpath)
    check(reg['group'] == 'pure' and reg['seed'] == 2026092304, 'registered pure run')
    check(reg['parameter_counts'] == {ARM: 522880}, 'registered pure parameter count')
    cfg, sources = reg['protocol'], reg['source_image_bindings']['calibration']
    train_sources = reg['source_image_bindings']['train']
    check(len(sources) == len({x['image_id'] for x in sources}) == 1000, 'calibration population')
    check(len(train_sources) == len({x['image_id'] for x in train_sources}) == 20000,
          'training population')
    check(not ({x['image_id'] for x in sources} & {x['image_id'] for x in train_sources}),
          'population overlap')
    files = {'training/registration.json': regpath, 'training/qualification.json': src/'qualification.json'}
    qual = read(files['training/qualification.json'])
    check(qual['status'] == 'REAL_GPU_GRADIENT_ENERGY_POPULATED_OPTIMIZER_ISOLATION_AND_BITWISE_RESUME_PASS'
          and qual['arms'] == [ARM] and qual['synthetic'] is False
          and qual['probe_updates_discarded'] and qual['decoder_sha256'] == reg['decoder_state_sha256'],
          'actual pure training qualification')
    parentpath = Path(reg['parent']['checkpoint'])
    parentreg = parentpath.parent.parent/'registration.json'
    check(digest(parentpath) == reg['parent']['sha256'] and
          digest(parentreg) == reg['parent']['registration_sha256'], 'parent SHA')
    bindings = dict(reg['bindings'])
    bindings.update(read(parentreg)['bindings'])
    for path, sha in bindings.items():
        check(digest(path) == sha, 'bound source/asset changed: '+path)
    initialpath = src/'checkpoints/step_10000.pt'
    parent = torch.load(parentpath, map_location='cpu', weights_only=True)
    initial = torch.load(initialpath, map_location='cpu', weights_only=True)
    validate_parent_payloads(parent, initial, digest(parentreg), regsha)
    del parent, initial
    files['parent/registration.json'] = parentreg
    cache = ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/cache_v1'
    complete = read(cache/'completion.json')
    check(digest(cache/'registration.json') == complete['registration_sha256'] and
          digest(cache/'training_statistics.json') == complete['statistics_sha256'], 'original F cache identity')
    for name in ('completion.json', 'registration.json', 'training_statistics.json'):
        files['cache/'+name] = cache/name
    rehashed = {}
    for role, expected_sources in [('train', train_sources), ('calibration', sources)]:
        check(reg['cache_'+role] == {'source': 'original F, checked shard hashes'}, 'pure cache kind')
        count, ids = 0, []
        for p in sorted((cache/role).glob('shard_*.json')):
            meta = read(p)
            count += meta['count']
            files[f'cache/{role}/{p.name}'] = p
            if role == 'calibration':
                tensor = p.with_suffix('.pt')
                check(digest(tensor) == meta['sha256'], 'calibration F tensor SHA')
                payload = torch.load(tensor, map_location='cpu', weights_only=True)
                ids.extend(payload['image_ids'])
                check(len(payload['image_ids']) == meta['count'], 'calibration F shard count')
                rehashed[str(tensor)] = meta['sha256']
        check(count == len(expected_sources) == complete['source_counts'][role], 'F cache source count')
        if role == 'calibration':
            check(ids == [x['image_id'] for x in expected_sources], 'F cache source order')
    done = None
    if not audit_only:
        stage = OUT/'delivery_chain_v1/stages/C_initial_pure.json'
        sr = read(stage)
        check(sr['returncode'] == 0 and digest(sr['snapshot']) == sr['completion_sha256'],
              'successful immutable stage snapshot')
        done = read(sr['snapshot'])
        validate_completion(done, regsha)
        files['training/stage.json'] = stage
        files['training/completion_snapshot.json'] = Path(sr['snapshot'])
    steps = [s for s in STEPS if (src/'calibration'/f'full_{s:05d}.json').exists()]
    check(steps and steps == list(STEPS[:len(steps)]), 'contiguous original continuation calibration')
    if not audit_only:
        check(steps == list(STEPS), 'five complete continuation calibration rounds')
    curves, snr_rows, source_rows = [], [], []
    for step in steps:
        receipt = src/'calibration'/f'full_{step:05d}.json'
        path, meta = receipt.with_suffix('.csv'), read(receipt)
        check(meta['step'] == step and digest(path) == meta['sha256'], 'calibration receipt SHA')
        with path.open() as f:
            rows = list(csv.DictReader(f))
        check(meta['rows'] == len(rows) == 15000, 'calibration row count')
        validate_pure_rows(rows, sources, cfg['snrs_db'], cfg['calibration_seeds'])
        item = dict(step=step, method=ARM, **{k:statistics.fmean(float(r[k]) for r in rows) for k in METRICS})
        check(abs(item['utility'] - meta['summary'][ARM]) < 1e-12, 'calibration mean')
        curves.append(item)
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row['snr_db'])].append(row)
        for snr, rr in sorted(grouped.items()):
            snr_rows.append(dict(step=step, method=ARM, snr_db=snr,
                **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS}))
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row['source_index']), int(row['snr_db'])].append(row)
        for (i, snr), rr in sorted(grouped.items()):
            source_rows.append(dict(step=step, method=ARM, source_index=i, image_id=sources[i]['image_id'],
                snr_db=snr, noise_count=len(rr), **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS}))
        for p in (path, receipt):
            files['calibration/'+p.name] = p
    selected = done['selected'][ARM] if done else read(src/'selected_P4084.json')
    best = min(curves, key=lambda r:(r['utility'], r['step']))
    # Audit-only may observe a newly sealed calibration before selected is updated.
    if not audit_only:
        check(best['step'] == selected['step'] and abs(best['utility']-selected['utility']) < 1e-12,
              'minimum complete calibration selection')
    check(digest(selected['checkpoint']) == selected['checkpoint_sha256'] and
          selected['registration_sha256'] == regsha and selected['parent_updates'] == 10000,
          'selected checkpoint identity')
    check(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    audit = dict(status='PURE_CONTINUATION_CPU_PREPARATION_PASS_PENDING_20K' if audit_only else
        'REAL_C_PURE_INITIAL_20K_CONTINUATION_VERIFIED', synthetic=False, complete_stage=not audit_only,
        training_seed=2026092304, calibration_steps=steps, calibration_rows=15000*len(steps),
        source_count=1000, checked_bindings=len(bindings), decoder_state_sha256=reg['decoder_state_sha256'],
        parent=reg['parent'], parent_updates=10000, milestone_total_updates=20000,
        milestone_new_updates=10000, parent_model_optimizer_order_rng_exact=True,
        initial_checkpoint_sha256=digest(initialpath), selected=selected,
        calibration_tensor_hashes_rechecked=rehashed,
        training_tensor_hashes='metadata retained; checked by actual training loader, not rehashed here',
        digital_crc='not_applicable; no digital transmissions in this pure arm',
        tool_sha256=digest(__file__), helper_sha256=digest(ROOT/'tools/publish_c_initial_milestone.py'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        new_holdout=False, content_selector=False, development_used=False,
        pending=['scheduler calibration-only extension decision', 'selected development/actual E/online timing',
                 'N3060 and additional seeds/diagnostics', 'historical GPU workers and final merged delivery'])
    if not audit_only:
        audit['extension_evidence'] = extension_evidence(curves, [ARM])
        audit['official_extension_decision'] = 'Read original scheduler decisions; this publisher does not choose or launch.'
    return target, files, audit, curves, snr_rows, source_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true',
                        help='validate completed calibration prefix, save only local evidence; never publish a milestone')
    args = parser.parse_args()
    target, files, audit, curves, snrs, sources = collect(args.audit_only)
    if args.audit_only:
        from datetime import datetime, timezone
        name = datetime.now(timezone.utc).strftime('pure_publication_preparation_%Y%m%dT%H%M%SZ.json')
        write(OUT/'monitoring'/name, audit)
        print(audit['status'], audit['calibration_steps'], audit['calibration_rows'])
        return
    target.mkdir(parents=True)
    audit['originals'] = {}
    for name, path in files.items():
        dst = target/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        audit['originals'][name] = dict(source=str(path), sha256=digest(dst), bytes=dst.stat().st_size)
    write(target/'selected.json', {ARM:audit['selected']})
    csv_write(target/'calibration_curve.csv', curves)
    csv_write(target/'calibration_by_snr.csv', snrs)
    for step in STEPS:
        csv_write(target/f'source_means_{step:05d}.csv', [r for r in sources if r['step'] == step])
    write(target/'resource_ledger.json', dict(N=4084, registered_E=8168, NH=0, ND=0, NA=4084,
        source_payload_bits=None, control_bits=0, FEC_mother_bits=0, coded_slots=0,
        ledger_kind='continuous waveform; zero serialized bits does not mean zero channel uses',
        energy_kind='registered 2N constraint; calibration does not export per-frame measured E',
        parameters=522880, decoder_frozen=True, digital_crc='not_applicable',
        noise_namespace='VAR-CONTINUOUS-4084|image_id', calibration_seeds=[4101,4102,4103],
        variance_per_real_coordinate='10**(-snr_db/10)', online_timing='NOT_RUN for these checkpoints'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fields, name in [(('utility','U_image'),'selection'),
                         (('mse','lpips_alex','normalized_latent'),'components')]:
        fig, axs = plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False)
        for ax, key in zip(axs[0],fields):
            ax.plot([r['step'] for r in curves],[r[key] for r in curves],marker='o',label=ARM)
            ax.set(xlabel='Total updates (10k inherited)',ylabel=key,title='1000 calibration sources / 15 noise cells')
            ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(target/f'calibration_{name}.svg'); plt.close(fig)
    for p in target.glob('*.svg'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    write(target/'audit.json',audit)
    index={str(p.relative_to(target)):dict(sha256=digest(p),bytes=p.stat().st_size)
           for p in sorted(target.rglob('*')) if p.is_file()}
    check(all(v['bytes'] < 10_000_000 for v in index.values()), 'publication file size limit')
    write(target/'index.json',dict(status=audit['status'],files=index))
    print(audit['status'], audit['calibration_rows'], len(index))


if __name__ == '__main__':
    main()
