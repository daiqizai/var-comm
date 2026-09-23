"""Publish a verified initial budget milestone; never launch or change a run."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', type=int, choices=(2048, 3060), required=True)
    args = parser.parse_args()
    run = f'P{args.N}_seed2026092304'
    source = ROOT / 'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/training' / run
    target = ROOT / 'results/token_channel_efficiency_20260923/budget_milestones' / (run + '_20k')
    check(not target.exists(), 'immutable publication already exists')
    completion = read(source / 'completion.json')
    selected = read(source / f'selected_P{args.N}.json')
    check(completion['state']['step'] == 20000 and not completion['synthetic'], 'real initial20k required')
    check(completion['selected'][f'P{args.N}'] == selected, 'selected receipt mismatch')
    check(digest(selected['checkpoint']) == selected['checkpoint_sha256'], 'selected checkpoint hash')
    check(digest(source / 'registration.json') == selected['registration_sha256'], 'training registration hash')
    qualification = read(source / 'qualification.json')
    check(not qualification['synthetic'] and qualification['status'].endswith('_PASS'), 'actual qualification required')
    files = [source / n for n in ['completion.json', 'registration.json', 'qualification.json', f'selected_P{args.N}.json']]
    registrations = [read(source / 'registration.json')]
    execution = selected.get('execution_identity')
    if execution:
        ep = Path(execution['path'])
        check(digest(ep) == execution['sha256'], 'execution hash')
        registrations.append(read(ep))
        files.extend([ep, ep.parent / 'qualification.json'])
    count = 0
    for registration in registrations:
        for path, sha in registration['bindings'].items():
            check(digest(path) == sha, f'bound file changed: {path}')
            count += 1
    curves = []
    source_ids = None
    full_keys = None
    for step in range(0, 20001, 2500):
        receipt = source / 'calibration' / f'full_{step:05d}.json'
        path = receipt.with_suffix('.csv')
        r = read(receipt)
        check(r['step'] == step and r['sha256'] == digest(path), 'calibration receipt/hash')
        with path.open() as f:
            rows = list(csv.DictReader(f))
        keys = [(x['image_id'], int(x['snr_db']), int(x['seed'])) for x in rows]
        ids = {x['image_id']: int(x['source_index']) for x in rows}
        check(len(rows) == r['rows'] == 15000 and len(ids) == 1000, 'full calibration size')
        check(set(ids.values()) == set(range(1000)), 'source index coverage')
        expected = set(itertools.product(ids, (1, 4, 7, 13, 19), (4101, 4102, 4103)))
        check(len(set(keys)) == len(keys) and set(keys) == expected, 'source/SNR/noise coverage')
        if source_ids is None:
            source_ids, full_keys = ids, set(keys)
        check(ids == source_ids and set(keys) == full_keys, 'calibration identity changes')
        for row in rows:
            check(row['method'] == f'P{args.N}' and ids[row['image_id']] == int(row['source_index']), 'arm/source identity')
            check(all(math.isfinite(float(row[k])) for k in ('mse', 'lpips_alex', 'normalized_latent', 'utility', 'U_image')), 'nonfinite calibration')
            check(abs(float(row['U_image']) - float(row['mse']) - .1 * float(row['lpips_alex'])) < 1e-7, 'U_image mismatch')
        mean = statistics.fmean(float(x['utility']) for x in rows)
        check(abs(mean - r['summary'][f'P{args.N}']) < 1e-12, 'selection utility summary mismatch')
        curves.append(dict(step=step, selection_utility=mean, U_image=statistics.fmean(float(x['U_image']) for x in rows), mse=statistics.fmean(float(x['mse']) for x in rows), lpips_alex=statistics.fmean(float(x['lpips_alex']) for x in rows)))
        files.extend([receipt, path])
    best = min(curves, key=lambda x: (x['selection_utility'], x['step']))
    check(selected['step'] == best['step'] and abs(selected['utility'] - best['selection_utility']) < 1e-12, 'calibration-selected checkpoint mismatch')
    values = [c['selection_utility'] for c in curves[-3:]]
    improvements = [(a-b)/a for a,b in zip(values, values[1:])]
    target.mkdir(parents=True)
    published = {}
    for path in files:
        relative = path.relative_to(source)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        published[str(relative)] = dict(source=str(path.relative_to(ROOT)), sha256=digest(dest), bytes=dest.stat().st_size)
    with (target / 'calibration_curve.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot([c['step'] for c in curves], [c['selection_utility'] for c in curves], 'o-', label='Registered selection utility')
    ax.plot([c['step'] for c in curves], [c['U_image'] for c in curves], 's--', label='MSE + 0.1 LPIPS (report only)')
    ax.set(xlabel='Training updates', ylabel='Calibration mean (lower is better)', title=f'P{args.N}: initial 20k, 1000 sources / 15 noise cells')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(target / 'calibration_curve.svg'); plt.close(fig)
    svg = target / 'calibration_curve.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    audit = dict(status='REAL_INITIAL_20K_MILESTONE_VERIFIED_NOT_CONVERGENCE', run_id=run, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), tool_sha256=digest(__file__), N=args.N, E=2*args.N, source_count=1000, calibration_rows=135000, selected=selected, checked_bound_files=count, source_snr_noise_keys_checked=True, selection_metric='registered utility including latent auxiliary; U_image is reported separately', relative_improvements=improvements, initial_rule_supports_10k_extension=all(x >= .002 for x in improvements), extension_launched_by_this_tool=False, synthetic=False, development_used=False, new_holdout_used=False, files=published, pending=['scheduler extension decision after both initial budgets', 'actual selected development and shared online timing', 'B1/B2/C and historical reference delivery'])
    (target / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k:audit[k] for k in ['status','run_id','checked_bound_files','relative_improvements','initial_rule_supports_10k_extension']},indent=2))

if __name__ == '__main__':
    main()
