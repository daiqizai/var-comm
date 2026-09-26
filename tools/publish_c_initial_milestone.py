"""Publish an immutable C initial-20k calibration milestone without launching GPU work."""
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
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SHORT = ROOT / 'outputs/SHORT-PREFIX-20260923'
OUT = ROOT / 'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
METRICS = ('mse', 'lpips_alex', 'normalized_latent', 'utility', 'U_image')

def check(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def validate_rows(rows, arms, sources, snrs, seeds):
    expected = set(itertools.product(arms, range(len(sources)), snrs, seeds))
    seen = set()
    flags = {}
    for row in rows:
        key = (row['method'], int(row['source_index']), int(row['snr_db']), int(row['seed']))
        check(key in expected and key not in seen, 'duplicate or unexpected calibration key')
        seen.add(key)
        check(row['image_id'] == sources[key[1]]['image_id'], 'source order/identity mismatch')
        check(all(math.isfinite(float(row[k])) for k in METRICS), 'nonfinite calibration')
        check(row['header_ok'] in ('0', '1') and row['body_crc_ok'] in ('0', '1'), 'failure flags')
        mse, lp, latent, utility, image = (float(row[k]) for k in METRICS)
        check(abs(image - mse - .1*lp) < 1e-7, 'U_image formula')
        check(abs(utility - image - .01*latent*int(row['header_ok'])) < 1e-7, 'masked utility formula')
        pair = key[1:]
        current = (row['header_ok'], row['body_crc_ok'])
        check(pair not in flags or flags[pair] == current, 'paired digital failure mismatch')
        flags[pair] = current
    check(seen == expected, 'incomplete calibration grid')

def extension_evidence(curves, arms):
    result = {}
    for arm in arms:
        values = [r['utility'] for r in curves if r['method'] == arm][-3:]
        gains = [(a-b)/a for a,b in zip(values, values[1:])]
        result[arm] = dict(values=values, relative_improvements=gains,
                           supports_extension=all(g >= .002 for g in gains))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int, choices=(6,7,8), required=True)
    a = parser.parse_args()
    run = f'm{a.m}_seed2026092304'
    src = SHORT/'training'/run
    target = ROOT/'results/token_channel_efficiency_20260923/C_initial_milestones'/f'{run}_20k'
    check(not target.exists(), 'immutable publication already exists')
    stage = OUT/'delivery_chain_v1/stages'/f'C_initial_train_m{a.m}.json'
    sr = read(stage)
    snapshot = Path(sr['snapshot'])
    check(sr['returncode'] == 0 and digest(snapshot) == sr['completion_sha256'], 'completed stage snapshot')
    done = read(snapshot)
    check(done['state']['step'] == done['state']['last_full'] == 20000, 'initial20k completion')
    arms = sorted(done['selected'])
    check(arms == sorted([f'H{a.m}-V'] + ([f'H{a.m}-P'] if a.m != 8 else [])), 'registered arms')
    check(all(done['state']['updates'][arm] == 20000 for arm in arms), 'paired update opportunity')
    reg = read(src/'registration.json')
    check(digest(src/'registration.json') == done['registration_sha256'], 'registration SHA')
    check(reg['group'] == f'm{a.m}' and reg['seed'] == 2026092304, 'run identity')
    cfg = reg['protocol']
    sources = reg['source_image_bindings']['calibration']
    check(len(sources) == len({r['image_id'] for r in sources}) == 1000, 'calibration population')
    files = {'training/registration.json':src/'registration.json',
             'training/completion_snapshot.json':snapshot,
             'training/stage.json':stage,
             'training/qualification.json':src/'qualification.json',
             'shared/qualification_execution.json':SHORT/'qualification_execution.json',
             'shared/cache_runtime.json':OUT/'C_followups/cache_runtime.json'}
    bindings = dict(reg['bindings'])
    for name in ('training/qualification.json', 'shared/qualification_execution.json'):
        q = read(files[name])
        check(q['status'].endswith('_PASS') and q.get('synthetic', False) is False
              and q['probe_updates_discarded'], 'real qualification')
        if name.startswith('training'):
            check(q['decoder_sha256'] == reg['decoder_state_sha256'], 'qualified Decoder identity')
        if name.startswith('shared'):
            check(q['decoder_unchanged'], 'shared frozen Decoder')
            bindings.update(q['bindings'])
    for stage_name in (f'C_initial_qualification_m{a.m}', 'C_initial_execution_qualification',
                       f'C_initial_cache_m{a.m}_calibration', f'C_initial_cache_m{a.m}_train'):
        stage_path = OUT/'delivery_chain_v1/stages'/f'{stage_name}.json'
        record = read(stage_path)
        check(record['returncode']==0 and digest(record['snapshot'])==record['completion_sha256'],
              'qualification/cache stage snapshot')
        check(digest(record['completion'])==record['completion_sha256'], 'qualification/cache completion unchanged')
        files[f'stages/{stage_name}.json'] = stage_path
        files[f'stages/{stage_name}_completion.json'] = Path(record['snapshot'])
    bindings.update(read(files['shared/cache_runtime.json'])['bindings'])
    for path, sha in bindings.items():
        check(digest(path) == sha, 'bound source/asset changed: '+path)
    cache_shards = {}
    for role in ('train', 'calibration'):
        cache = SHORT/f'cache/N4084_m{a.m}_ND{cfg["digital_data_uses"][str(a.m)]}'/role
        completion = read(cache/'completion.json')
        creg = read(cache/'registration.json')
        check(completion == reg['cache_'+role] and not completion['synthetic'], 'cache completion identity')
        check(digest(cache/'registration.json') == completion['registration_sha256'], 'cache registration')
        check(creg['scope'] == completion['scope'], 'cache scope')
        for path, sha in creg['bindings'].items():
            check(digest(path) == sha, 'cache source binding')
        files[f'cache/{role}/completion.json'] = cache/'completion.json'
        files[f'cache/{role}/registration.json'] = cache/'registration.json'
        for name, sha in completion['hashes'].items():
            meta = cache/Path(name).with_suffix('.json')
            record = read(meta)
            check(record['sha256'] == sha and record['identity']['scope'] == completion['scope'], 'cache shard receipt')
            # Training tensors were verified by the real training loader. Retain its immutable
            # manifest; independently rehash calibration tensors used by all reported rows.
            if role == 'calibration':
                check(digest(cache/name) == sha, 'calibration cache tensor')
                cache_shards[name] = sha
            files[f'cache/{role}/{meta.name}'] = meta
    curves, snr_rows, source_rows = [], [], []
    raw_rows = 0
    for step in range(0,20001,2500):
        receipt = src/'calibration'/f'full_{step:05d}.json'
        path = receipt.with_suffix('.csv')
        meta = read(receipt)
        check(meta['step'] == step and digest(path) == meta['sha256'], 'full calibration hash')
        with path.open() as f:
            rows = list(csv.DictReader(f))
        check(meta['rows'] == len(rows) == 15000*len(arms), 'full calibration size')
        validate_rows(rows, arms, sources, cfg['snrs_db'], cfg['calibration_seeds'])
        raw_rows += len(rows)
        for arm in arms:
            ar = [r for r in rows if r['method'] == arm]
            item = dict(step=step, method=arm, **{k:statistics.fmean(float(r[k]) for r in ar) for k in METRICS})
            check(abs(item['utility'] - meta['summary'][arm]) < 1e-12, 'calibration summary')
            item.update(header_failures=sum(r['header_ok']=='0' for r in ar),
                        body_crc_failures=sum(r['body_crc_ok']=='0' for r in ar))
            curves.append(item)
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row['method'],int(row['snr_db']))].append(row)
        for (arm,snr), rr in sorted(grouped.items()):
            snr_rows.append(dict(step=step, method=arm, snr_db=snr,
                **{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS},
                header_failures=sum(r['header_ok']=='0' for r in rr),
                body_crc_failures=sum(r['body_crc_ok']=='0' for r in rr)))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row['method'], int(row['source_index']),int(row['snr_db']))].append(row)
        for (arm,i,snr),rr in sorted(grouped.items()):
            source_rows.append(dict(step=step,method=arm,source_index=i,image_id=sources[i]['image_id'],
                snr_db=snr,noise_count=len(rr),**{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS}))
        files['calibration/'+receipt.name] = receipt
        files['calibration/'+path.name] = path
    for arm, selected in done['selected'].items():
        best = min((r for r in curves if r['method']==arm), key=lambda r:(r['utility'],r['step']))
        check(best['step']==selected['step'] and abs(best['utility']-selected['utility'])<1e-12, 'minimum calibration selection')
        check(selected['registration_sha256']==done['registration_sha256'] and
              digest(selected['checkpoint'])==selected['checkpoint_sha256'], 'selected checkpoint identity')
    evidence = extension_evidence(curves, arms)
    target.mkdir(parents=True)
    originals = {}
    for name,path in files.items():
        dst = target/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dst)
        originals[name] = dict(source=str(path),sha256=digest(dst),bytes=dst.stat().st_size)
    write(target/'selected.json',done['selected'])
    csv_write(target/'calibration_curve.csv',curves)
    csv_write(target/'calibration_by_snr.csv',snr_rows)
    # One file per checkpoint stays below the repository's10MB artifact limit.
    for step in range(0,20001,2500):
        csv_write(target/f'source_means_{step:05d}.csv',[r for r in source_rows if r['step']==step])
    ledger = dict(reg['cache_calibration']['scope']['ledger'])
    payload = 12*sum(s*s for s in (1,2,3,4,5,6,8,10)[:a.m])
    ledger.update(source_payload_bits=payload,header_class_bits=10,header_mode_bits=2,
        header_crc_bits=16,header_tail_bits=6,header_mother_bits=68,header_coded_slots=136,
        body_crc_bits=16,body_tail_bits=6,body_mother_bits=2*(payload+22),
        body_coded_slots=2*ledger['ND'],body_effective_information_rate=(payload+22)/(2*ledger['ND']),
        energy_kind='registered per-frame2N constraint; per-frame measured E not exported by calibration',
        online_timing='NOT_RUN for these selected checkpoints',noise=cfg['noise'])
    write(target/'resource_ledger.json',ledger)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fields,name in [(('utility','U_image'),'selection'),(('mse','lpips_alex','normalized_latent'),'components')]:
        fig,axs=plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False)
        for ax,k in zip(axs[0],fields):
            for arm in arms:
                rr=[r for r in curves if r['method']==arm]
                ax.plot([r['step'] for r in rr],[r[k] for r in rr],marker='o',label=arm)
            ax.set(xlabel='Updates',ylabel=k,title='1000 calibration sources / 15 noise cells')
            ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(target/f'calibration_{name}.svg');plt.close(fig)
    for path in target.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    audit = dict(status='REAL_C_INITIAL_20K_CALIBRATION_MILESTONE_VERIFIED',m=a.m,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tool_sha256=digest(__file__),synthetic=False,training_seed=2026092304,
        calibration_rows=raw_rows,source_count=1000,arms=arms,selected=done['selected'],
        decoder_state_sha256=reg['decoder_state_sha256'],checked_bindings=len(bindings),
        calibration_tensor_hashes_rechecked=cache_shards,
        training_cache_tensor_hashes='retained registration/completion/shard manifests; real loader verified at training, not rehashed by this publisher',
        originals=originals,extension_evidence=evidence,
        calibration_rule_supports_paired_10k=any(v['supports_extension'] for v in evidence.values()),
        official_extension_decision='PENDING existing scheduler after initial matrix; publisher does not launch or register a decision',
        new_holdout=False,content_selector=False,development_used=False,
        pending=['C remaining initial groups and paired extensions','N3060 calibration choice and two additional training seeds',
                 'selected development quality/actual per-frame E/online timing/paired bootstrap','four historical GPU workers and final merged publication'],
        scope_note='Original completion mentions new test; supplemental protocol explicitly defers it. Calibration utility is not development quality; no N savings or training-seed inference.')
    write(target/'audit.json',audit)
    index={str(p.relative_to(target)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(target.rglob('*')) if p.is_file()}
    write(target/'index.json',dict(status=audit['status'],files=index))
    print(json.dumps({k:audit[k] for k in ('status','m','calibration_rows','checked_bindings','extension_evidence','calibration_rule_supports_paired_10k')},indent=2))

if __name__=='__main__':
    main()
