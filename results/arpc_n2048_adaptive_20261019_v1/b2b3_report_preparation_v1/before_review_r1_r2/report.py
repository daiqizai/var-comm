"""Read-only scientific delivery assembly. No model, channel or resampling code."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import shutil
import time

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
O = R/'results/arpc_n2048_adaptive_20261019_v1'
L = BASE/'arpc_n2048_adaptive_20261019_v1'
PRIOR = R/'results/arpc_n2048_validation_20261019_v1'
DEST = O/'b2b3_report_v1'
METRICS = (
    ('psnr_db', 'PSNR (dB)', 1, 'higher'),
    ('lpips_alex', 'LPIPS', 1, 'lower'),
    ('dreamsim', 'DreamSim', 1, 'lower'),
    ('convnext_top1_source_prediction', 'Agreement (%)', 100, 'higher'),
    ('dinov2_vitl14_cosine', 'DINO similarity', 1, 'higher'),
    ('dists', 'DISTS', 1, 'lower'),
    ('ms_ssim', 'MS-SSIM', 1, 'higher'),
)
METHODS = ('NEST_ADAPTIVE_N2048', 'ARPC_ADAPTIVE_N2048')
FIXED = ('NEST_FULL_M10_N2048', 'ARPC_FULL_E7_N2048')
LABELS = ('NeST-Com (entropy-coded)', 'ARPC + LDPC–QAM (adapted)')
B2_CELLS = (
    (0, 4, 'nest_t24_q2_r1-2'), (0, 4, 'nest_t24_q2_r2-3'),
    (0, 10, 'nest_t24_q2_r2-3'), (0, 10, 'nest_t24_q2_r3-4'),
    (0, 10, 'nest_t24_q2_r5-6'), (0, 10, 'nest_t24_q4_r1-2'),
    (0, 10, 'nest_t24_q4_r2-3'), (1, 4, 'arpc_t7_q2_r2-3'),
    (1, 10, 'arpc_t7_q4_r2-3'),
)
STYLE = dict(colors=['#3775BA', '#B64342'], markers=['o', 's'],
             lines=['-', '--'], marker_pt=4.5, linewidth_pt=[1.7, 1.5],
             label_pt=8.5, tick_pt=7.5, legend_pt=8, panel_pt=9,
             quality_mm=[180, 102], calibration_mm=[180, 64], dpi=600,
             grayscale_dpi=300, svg_fonttype='none', pdf_fonttype=42)
EXPECTED = {
    'B2_transport': O/'b2b3_revision_v1/B2/transport_completion.json',
    'B3_transport': O/'b2b3_revision_v1/B3/transport_completion.json',
    'B2_native': O/'b2b3_analysis_v1/B2/native_completion.json',
    'B3_native': O/'b2b3_analysis_v1/B3/native_completion.json',
    'B2_score': O/'b2b3_analysis_v1/B2/score_completion.json',
    'selection': O/'b2b3_analysis_v1/B2/selection.json',
    'B3_metrics': O/'b3_metrics_v1/completion.json',
    'B3_registration': O/'b3_metrics_v1/registration.json',
    'fixed_delivery': PRIOR/'final_delivery_manifest.json',
}
INPUTS = {}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'unsafe absolute path')
    for q in (p, *p.parents):
        require(not q.is_symlink(), 'symlink forbidden: '+str(q))
    a = p.stat(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    b = p.stat()
    require((a.st_ino, a.st_size, a.st_mtime_ns) ==
            (b.st_ino, b.st_size, b.st_mtime_ns), 'changed while reading')
    return dict(path=str(p), bytes=a.st_size, sha256=h.hexdigest())


def checked(d):
    require(set(d) >= {'path', 'bytes', 'sha256'}, 'incomplete descriptor')
    p = Path(d['path'])
    require(p.is_relative_to(R) or p.is_relative_to(BASE), 'input outside R/PFS runtime')
    actual = pin(p)
    require(actual['bytes'] == d['bytes'] and actual['sha256'] == d['sha256'], 'SHA/bytes mismatch: '+str(p))
    INPUTS[str(p)] = actual
    return p


def load(d):
    return json.loads(checked(d).read_bytes())


def known(path):
    return load(pin(path))


def save(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def csvread(path):
    checked(pin(path))
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows):
    require(bool(rows), 'empty table')
    with Path(path).open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())


def outputs(done):
    for d in done['outputs']:
        checked(d)


def successful(w):
    require(type(w['returncode']) is int and w['returncode'] == 0, 'not actual exit 0')
    require(all(w.get(k, False) is False for k in ('timed_out', 'terminated', 'killed')),
            'interrupted child')
    require(w.get('metadata_error') is None, 'wait metadata error')


def gone(i):
    require(i['host'] == os.uname().nodename, 'cross-host /proc forbidden')
    p = Path('/proc')/str(i['pid'])/'stat'
    if not p.exists():
        return True
    a = p.read_text().rsplit(')', 1)[1].split()
    return int(a[19]) != i['start_ticks'] or a[0] == 'Z'


def wait_closed(path):
    p = Path(path); w = known(p); successful(w)
    launch = known(p.with_name('launch.json'))
    require(launch['process'] == w['process'], 'wait/launch child mismatch')
    require(gone(w['process']) and gone(launch['parent_process']), 'live child/control parent')
    require(not p.with_name('failure.json').exists(), 'stage failure present')
    return w


def observation(d):
    x = load(d); require(x['status'] == 'CLOSED_ON_EXECUTION_HOST', 'missing host closure')
    w, launch = load(x['actual_wait']), load(x['launch']); successful(w)
    require(w['process'] == launch['process'] == x['child'], 'host child mismatch')
    require(launch['parent_process'] == x['parent'], 'host parent mismatch')
    require(x['host'] == x['child']['host'] == x['parent']['host'], 'host mismatch')
    require(x['child_gone'] is True and x['parent_gone'] is True, 'host live process')
    require(x['observed_unix'] >= x['not_before_unix'], 'premature closure')
    if x['host'] == os.uname().nodename:
        require(gone(x['child']) and gone(x['parent']), 'process revived/reused incorrectly')


def ledger_ok(v):
    require(v['unresolved'] == 0, 'unresolved ledger')
    require(all(r['status'] == 'COMPLETE' for r in v['counts']), 'non-complete ledger')


def validate_selection(s):
    require(s['schema'] == 'N2048_B2_DINO_FOUR_POLICIES_V1' and
            s['status'] == 'FOUR_POLICIES_FROZEN', 'selection not frozen')
    require(s['metric'] == 'dinov2_vitl14_cosine' and s['calibration_bootstrap_calls'] == 0
            and s['no_additional_candidates'] is True, 'selection scope')
    candidates = s['candidate_means']; chosen = s['rows']
    require(len(candidates) == 9 and
            {(int(r['codec']), int(r['snr_db']), r['policy_id']) for r in candidates} == set(B2_CELLS),
            'not the nine authorized candidates')
    require(len(chosen) == 4 and {(int(r['codec']), int(r['snr_db'])) for r in chosen} ==
            {(c, n) for c in (0, 1) for n in (4, 10)}, 'not four policies')
    index = {(r['codec'], r['snr_db'], r['policy_id']): r for r in candidates}
    for r in candidates:
        require(r['source_mean_count'] == 1000 and r['noise_per_source'] == 3 and
                math.isfinite(r['calibration_mean_DINO']), 'candidate denominator/value')
    for r in chosen:
        k = (r['codec'], r['snr_db'], r['policy_id'])
        require(k in index and r['calibration_mean_DINO'] == index[k]['calibration_mean_DINO']
                and bool(r['tie_break']), 'saved selection inconsistent')
    # Consume the saved selection, never rerun select_four or pick a policy here.
    return candidates, chosen


def verify_closure(auth):
    require(auth['schema'] == 'N2048_B2B3_REPORT_AUTHORIZATION_V1' and
            auth['report_only'] is True and auth['science_calls'] == 0, 'report authorization')
    require(set(auth['inputs']) == set(EXPECTED), 'required root inputs')
    data = {}
    for name, path in EXPECTED.items():
        require(Path(auth['inputs'][name]['path']) == path, 'wrong input: '+name)
        data[name] = load(auth['inputs'][name])
    for d in auth['source_files']:
        checked(d)
    required_sources = {str(Path(__file__).resolve().parent/n)
                        for n in ('report.py', 'test_report.py', 'README.md')}
    require({d['path'] for d in auth['source_files']} == required_sources, 'report source binding')
    require(set(auth['outer_waits']) >= {f'{phase}_{role}' for phase in ('B2','B3')
            for role in ('tx','rx','audit')} | {'B2_DINO','B3_score','B3_statistics'},
            'root outer closure coverage incomplete')
    for d in auth['outer_waits'].values():
        checked(d); wait_closed(d['path'])
    for d in auth.get('additional_host_observations', []):
        observation(d)
    exceptions = []
    for item in auth['engineering_exceptions']:
        require(item['science_calls'] == 0 and item['scope'] == 'engineering_only'
                and bool(item['description']) and bool(item['evidence']), 'unbounded exception')
        for d in item['evidence']:
            checked(d)
        exceptions.append(item)
    native_regs = {}; transport_regs = {}
    for phase, count in (('B2', 27000), ('B3', 1200)):
        t = data[phase+'_transport']; n = data[phase+'_native']
        require(t['status'] == 'TRANSPORT_COMPLETE_NATIVE_AND_SCORE_SEPARATE' and
                t['logical'] == count and t['subframes'] == t['transmitted_subframes'] == count*2,
                'transport scope/closure')
        require(t['decoder_attempts'] == t['header_decodes']+t['body_decodes'] <= count*4
                and t['header_decodes'] == count*2, 'actual decode budget')
        for role in ('tx', 'rx', 'audit'):
            wait_closed(L/'b2b3_revision_v1/controls'/(phase+'_'+role)/'actual_wait.json')
            require(not (L/'b2b3_revision_v1'/phase/role/'failure.json').exists(), 'transport failure')
        for role in ('tx', 'rx'):
            v = known(L/'b2b3_revision_v1'/phase/role/'completion.json'); ledger_ok(v['ledger'])
        r = load(t['registration']); transport_regs[phase] = r
        for d in r['code']:
            checked(d)
        checked(r['source_fallback']); checked(r['private_sources'])
        require(n['status'] == 'COMPLETE' and n['logical_objects'] == count and
                n['process_closure_status'] == 'ALL_NATIVE_PROCESSES_GONE', 'native not closed')
        require(n['transport_completion'] == auth['inputs'][phase+'_transport'], 'native transport binding')
        m = load(n['manifest'])
        require(m['inputs'] == n['inputs'] and m['outputs'] == n['outputs'], 'native manifest mismatch')
        for d in m['inputs']+m['outputs']:
            checked(d)
        for d in n['process_closure_evidence']:
            observation(d)
        nr = load(n['registration']); native_regs[phase] = nr
        for d in nr['code']:
            checked(d)
        for name in ('nest', 'arpc'):
            jobs = load(nr['jobs'][name])
            require(len(jobs) == nr['new_native_calls'][name], 'native actual job count')
            control = L/'b2b3_analysis_v1/controls'/(phase+'_'+name)
            require(not(control/'failure.json').exists() and
                    not(L/'b2b3_analysis_v1'/phase/name/'rx/failure.json').exists(), 'native failure')
            if not jobs:
                require(known(control/'zero_call_completion.json')['new_model_calls'] == 0, 'missing zero receipt')
            else:
                cp = known(L/'b2b3_analysis_v1'/phase/name/'rx/completion.json')
                if 'ledger' in cp:
                    ledger_ok(cp['ledger'])
    b2 = data['B2_score']; require(b2['status'] == 'COMPLETE' and b2['logical_frames'] == 27000
                                and b2['bootstrap_calls'] == 0, 'B2 score not closed')
    outputs(b2); ledger_ok(b2['ledger']); checked(b2['actual_wait']); wait_closed(b2['actual_wait']['path'])
    sel = data['selection']; validate_selection(sel)
    for key in ('B2_registration', 'B2_transport_completion', 'source_evidence', 'metric_evidence',
                'selection_evidence', 'score_completion', 'score_actual_wait'):
        checked(sel[key])
    require(sel['score_completion'] == auth['inputs']['B2_score'], 'selection score binding')
    require(transport_regs['B3']['selection'] == auth['inputs']['selection'], 'B3 selected policy binding')
    metric = data['B3_metrics']; mr = data['B3_registration']
    require(metric['status'] == 'N2048_ADAPTIVE_TWO_METHOD_COMPARISON_COMPLETE' and
            metric['sources'] == 100 and metric['logical_frames'] == 1200 and metric['snr_db'] == [4, 10],
            'B3 metric scope')
    require(set(metric['metrics']) == {m[0] for m in METRICS}, 'seven metric coverage')
    for d in mr['inputs']+mr['code']+mr['original_metric_sources']:
        checked(d)
    require(auth['inputs']['B3_native'] in mr['inputs'], 'metric/native binding')
    outputs(metric)
    for role in ('score', 'statistics'):
        wait_closed(L/'b3_metrics_v1/controls'/role/'actual_wait.json')
    scored = known(O/'b3_metrics_v1/score_completion.json'); outputs(scored); ledger_ok(scored['ledger'])
    require(scored['logical_frames'] == 1200 and scored['new_pairs'] == mr['new_seven_metric_pairs']
            and scored['reused_pairs'] == mr['exact_prior_seven_metric_pairs'], 'metric reuse counts')
    boot = metric['bootstrap']
    require(boot['seed'] == 2026101902 and boot['replicates'] == 10000 and boot['draw_calls'] == 1,
            'wrong saved bootstrap')
    checked(boot['indices'])  # Hash only; never load draws or calculate intervals.
    f = data['fixed_delivery']; require(f['status'] == 'SCIENTIFIC_AND_VISUAL_DELIVERY_COMPLETE', 'fixed reference not delivered')
    science = load(f['science']); outputs(science); fixed = load(science['metrics']); outputs(fixed)
    fixed_scores = known(PRIOR/'metrics_v1/score_completion.json'); outputs(fixed_scores); ledger_ok(fixed_scores['ledger'])
    require(fixed['status'] == 'N2048_TWO_METHOD_COMPARISON_COMPLETE' and fixed['logical_frames'] == 1800,
            'fixed reference scope')
    receipts = load(f['closed_receipts'])
    for pair in receipts['copies']:
        p = checked(pair['copy'])
        require(pair['copy']['sha256'] == pair['source']['sha256'], 'receipt copy mismatch')
        # Keep original nonzero engineering receipts without relabelling them.
    for key in ('actual_visual_review', 'final_figure', 'registration'):
        checked(f[key])
    for name in ('process_closure_main.json','process_closure_leo.json'):
        old_observation=known(PRIOR/'final_readiness'/name)
        require(old_observation['status']=='PASS_ALL_RECORDED_LOCAL_IDENTITIES_GONE' and
                old_observation['execution_lock_unheld'] is True, 'old reference process closure')
        for item in old_observation['identities']:
            require(item['gone'] is True and item['identity']['host']==old_observation['host'],
                    'old reference process not closed')
            checked(item['receipt'])
            if old_observation['host']==os.uname().nodename:
                require(gone(item['identity']),'old reference identity live')
    return data, transport_regs, native_regs, scored, exceptions


def number(x):
    if x in ('True', 'False'):
        return float(x == 'True')
    y = float(x); require(math.isfinite(y), 'non-finite table value'); return y


def validate_metric_tables(frames, sources, means, paired, ids, methods, snrs):
    metrics = {m[0] for m in METRICS}; ids = set(ids)
    require(len(ids) == 100, 'source count')
    keys = [(r['method'], r['source_id'], int(r['snr_db']), int(r['noise_seed'])) for r in frames]
    expect = {(m, s, n, z) for m in methods for s in ids for n in snrs for z in (6201, 6202, 6203)}
    require(len(keys) == len(set(keys)) and set(keys) == expect, 'logical frame coverage')
    for r in frames:
        for metric in metrics:
            number(r[metric])
    sourcekeys = {(r['method'], r['source_id'], int(r['snr_db']), r['metric']) for r in sources}
    require(len(sources) == len(sourcekeys) == len(ids)*len(methods)*len(snrs)*7, 'source metric count')
    require(sourcekeys == {(m, s, n, k) for m in methods for s in ids for n in snrs for k in metrics}, 'source grid')
    for r in sources:
        require(int(r['noise_count']) == 3, 'not three-noise source mean'); number(r['source_mean'])
    mean_keys = {(r['method'], int(r['snr_db']), r['metric']) for r in means}
    require(len(means) == len(mean_keys) == len(methods)*len(snrs)*7, 'summary count')
    require(mean_keys == {(m, n, k) for m in methods for n in snrs for k in metrics}, 'summary grid')
    require(len(paired) == len(snrs)*7 and {(int(r['snr_db']), r['metric']) for r in paired} ==
            {(n, k) for n in snrs for k in metrics}, 'paired grid')
    directions = {m[0]: m[3] for m in METRICS}
    for r in means+paired:
        require(int(r['source_count']) == 100 and r['direction'] == directions[r['metric']], 'summary identity')
        v, lo, hi = (number(r[k]) for k in ('mean', 'ci_low', 'ci_high'))
        require(lo <= hi, 'reversed interval')
        if r in paired:
            require((r['interval_crosses_or_touches_zero'] == 'True') == (lo <= 0 <= hi), 'paired zero flag')
    return wide_source_rows(sources)


def wide_source_rows(rows):
    out = {}
    for r in rows:
        k = (r['method'], r['source_id'], int(r['snr_db']))
        item = out.setdefault(k, dict(method=k[0], source_id=k[1], snr_db=k[2], noise_count=3))
        require(r['metric'] not in item, 'duplicate source metric')
        item[r['metric']] = r['source_mean']
    return [out[k] for k in sorted(out)]


def enrich_frames(frames, pixels, fallback, capacities, choices):
    px = {x['key']: x for x in pixels}; fb = {(x['source_id'], x['codec'], x['snr_db']): x for x in fallback}
    require(len(frames) == len(px) == 1200 and len(fb) == 400, 'B3 transport/pixel/fallback count')
    policies = {(x['codec'], x['snr_db']): x['policy_id'] for x in choices}; out = []
    for f in frames:
        p, b = px[f['key']], fb[f['source_id'], f['codec'], f['snr_db']]
        require(f['policy_id'] == b['policy_id'] == policies[f['codec'], f['snr_db']], 'unfrozen B3 policy')
        require(f['sent_endpoint'] == b['actual_endpoint'] and f['object_bits'] == b['object_bits'], 'actual fallback mismatch')
        g = capacities[f['layout_id']]
        require(g['total_complex_symbols'] == 2048 and g['header_symbols_per_fragment'] == 68 and
                g['length_bits_per_fragment'] == 13 and g['crc_bits_per_fragment'] == 16 and
                g['reassembly_bits_per_fragment'] == 96, 'charged geometry changed')
        require(g['object_bits'] == b['capacity_bits'] == 16*g['fragment_bytes'], 'byte-rounded source capacity')
        bits, cap = f['object_bits'], g['object_bits']
        require(0 <= bits <= cap and f['N'] == 2048 and f['nominal_Es'] == 2, 'source budget')
        require(p['source_id'] == f['source_id'] and p['source_input_key'] == f['source_input_key']
                and p['accepted_wrong_post_exit'] == f['accepted_wrong'], 'pixel provenance mismatch')
        correct = bool(f['exact_post_exit'] and p['receiver_status'] != 'SOURCE_PARSE_FAIL')
        reason = (f['failure_reason'] if not f['accepted'] else
                  ('ACCEPTED_WRONG_SOURCE' if f['accepted_wrong'] else
                   ('SOURCE_PARSE_FAIL' if p['receiver_status'] == 'SOURCE_PARSE_FAIL' else 'CORRECT')))
        out.append(dict(key=f['key'], method=METHODS[f['codec']], source_id=f['source_id'],
            source_index=f['source_index'], snr_db=f['snr_db'], noise_seed=f['noise_seed'],
            policy_id=f['policy_id'], layout_id=f['layout_id'], target=f['target'],
            sent_endpoint=f['sent_endpoint'], actual_received_codec=f['actual_received_codec'],
            actual_received_endpoint=f['actual_received_endpoint'], source_no_fit=f['source_no_fit'],
            q=g['q'], nominal_rate=g['nominal_rate'], n_per_fragment=g['n'], k_per_fragment=g['k'],
            object_bits=bits, source_capacity_bits=cap, source_occupancy=bits/cap,
            source_byte_padding_bits=8*((bits+7)//8)-bits,
            total_length_bits=26, total_crc_bits=32, total_reassembly_bits=192,
            total_header_symbols=136, total_body_symbols=1912, N=2048, nominal_Es=2,
            accepted=f['accepted'], exact_post_exit=f['exact_post_exit'], accepted_wrong=f['accepted_wrong'],
            receiver_status=p['receiver_status'], correct_delivery=correct, outcome=reason,
            header_statuses=json.dumps(f['header_statuses']), fragment_statuses=json.dumps(f['fragment_statuses']),
            source_input_key=f['source_input_key'], image_float_sha256=p['image_float_sha256']))
    return out


def aggregate_transport(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['method'], r['snr_db']].append(r)
    summary, counts = [], []
    for (method, snr), group in sorted(grouped.items()):
        require(len(group) == 300 and len({r['source_id'] for r in group}) == 100, 'transport denominator')
        summary.append(dict(method=method, snr_db=snr, logical_frames=300, sources=100,
            accepted=sum(r['accepted'] for r in group), correct_delivery=sum(r['correct_delivery'] for r in group),
            correct_delivery_rate=sum(r['correct_delivery'] for r in group)/300,
            accepted_wrong=sum(r['accepted_wrong'] for r in group), source_no_fit=sum(r['source_no_fit'] for r in group),
            mean_source_occupancy=sum(r['source_occupancy'] for r in group)/300))
        for field in ('outcome', 'sent_endpoint', 'actual_received_endpoint', 'layout_id', 'receiver_status'):
            for value, count in sorted(collections.Counter(str(r[field]) for r in group).items()):
                counts.append(dict(method=method, snr_db=snr, field=field, value=value, count=count, denominator=300))
    return summary, counts


def format_ci(r, factor=1):
    return f"{number(r['mean'])*factor:.4f} [{number(r['ci_low'])*factor:.4f}, {number(r['ci_high'])*factor:.4f}]"


def latex_table(path, rows, methods, caption):
    lookup = {(r['method'], int(r['snr_db']), r['metric']): r for r in rows}
    snrs = sorted({int(r['snr_db']) for r in rows})
    text = ['% Saved point estimates and original intervals only; no new statistics.',
            r'\begin{table*}[t]', r'\centering\small', r'\caption{'+caption+'}',
            r'\begin{tabular}{lllr}', r'\toprule', r'SNR & Method & Metric & Mean [95\% CI] \\', r'\midrule']
    for snr in snrs:
        for i, method in enumerate(methods):
            for metric, label, factor, _ in METRICS:
                text.append(f"{snr} dB & {LABELS[i].replace('–','--')} & {label.replace('%', chr(92)+'%')} & {format_ci(lookup[method,snr,metric],factor)} " + r'\\')
    text += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    path.write_text('\n'.join(text), encoding='utf8')


def plots(out, means, paired, candidates, chosen):
    mpl_cache=Path(os.environ['MPLCONFIGDIR'])
    require(mpl_cache.is_relative_to(L) and mpl_cache.is_dir(),'private MPLCONFIGDIR under runtime required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':8.5, 'xtick.labelsize':7.5, 'ytick.labelsize':7.5,
        'legend.fontsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.65, 'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
        'savefig.facecolor':'white'})
    manifest = []
    def export(fig, name, size):
        for suffix in ('pdf', 'svg', 'png'):
            p = out/(name+'.'+suffix)
            fig.savefig(p, dpi=600, facecolor='white')  # fixed page size; no tight bounding-box resizing
            manifest.append(dict(**pin(p), size_mm=size, raster_dpi=600 if suffix=='png' else None))
        return
    def data_plot(ispaired, gray=False):
        fig, axes = plt.subplots(2,4,figsize=(180/25.4,102/25.4))
        fig.subplots_adjust(left=.065,right=.99,bottom=.12,top=.87,wspace=.46,hspace=.65)
        for i,(metric,label,factor,direction) in enumerate(METRICS):
            ax=axes.flat[i]; ax.set_title(f"({chr(97+i)}) {label}",loc='left',fontsize=9,pad=5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xticks([4,10]);ax.set_xlim(3,11);ax.set_xlabel('SNR (dB)')
            ax.grid(axis='y',color='.9',lw=.45,zorder=0)
            if ispaired:
                rr=sorted([r for r in paired if r['metric']==metric],key=lambda r:int(r['snr_db']))
                yy=[number(r['mean'])*factor for r in rr]
                lo=[number(r['ci_low'])*factor for r in rr];hi=[number(r['ci_high'])*factor for r in rr]
                color='0.15' if gray else '#3775BA'
                ax.axhline(0,color='.45',lw=.7,ls=':',zorder=1)
                ax.vlines([4,10],lo,hi,color=color,lw=1,zorder=2)
                for x,a,b in zip((4,10),lo,hi):ax.hlines([a,b],x-.12,x+.12,color=color,lw=.9,zorder=2)
                ax.plot([4,10],yy,'o-',color=color,ms=4.5,lw=1.5,zorder=3)
                ax.set_title(f"({chr(97+i)}) Δ{label}",loc='left',fontsize=9,pad=5)
            else:
                for m,method in enumerate(METHODS):
                    rr=sorted([r for r in means if r['metric']==metric and r['method']==method],key=lambda r:int(r['snr_db']))
                    yy=[number(r['mean'])*factor for r in rr];lo=[number(r['ci_low'])*factor for r in rr];hi=[number(r['ci_high'])*factor for r in rr]
                    color=['.12','.45'][m] if gray else STYLE['colors'][m]
                    ax.vlines([4,10],lo,hi,color=color,lw=1,zorder=2)
                    for x,a,b in zip((4,10),lo,hi):ax.hlines([a,b],x-.12,x+.12,color=color,lw=.9,zorder=2)
                    ax.plot([4,10],yy,color=color,marker=STYLE['markers'][m],ls=STYLE['lines'][m],
                            ms=4.5,mfc=color if m==0 else 'white',mec=color,lw=STYLE['linewidth_pt'][m],zorder=3,label=LABELS[m])
            ax.margins(y=.15)
        axes.flat[7].set_visible(False)
        if ispaired:
            fig.text(.5,.96,'Direct paired difference: NeST-Com − ARPC',ha='center',fontsize=8.5)
            fig.text(.5,.925,'Positive favors NeST for PSNR, agreement, DINO, MS-SSIM; negative for LPIPS, DreamSim, DISTS.',ha='center',fontsize=7)
        else:
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,handlelength=2.4)
        return fig
    for paired_flag,name in ((False,'adaptive_quality'),(True,'direct_paired')):
        fig=data_plot(paired_flag);export(fig,name,[180,102]);plt.close(fig)
        fig=data_plot(paired_flag,True);fig.savefig(out/(name+'_gray.png'),dpi=300);plt.close(fig)
    selected={(r['codec'],r['snr_db']):r['policy_id'] for r in chosen}
    for gray in (False,True):
        fig,axes=plt.subplots(1,4,figsize=(180/25.4,64/25.4))
        fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.77,wspace=.52)
        for i,(codec,snr) in enumerate(((0,4),(0,10),(1,4),(1,10))):
            ax=axes[i];rr=[r for r in candidates if (r['codec'],r['snr_db'])==(codec,snr)]
            color=['.15','.45'][codec] if gray else STYLE['colors'][codec]
            names=[r['policy_id'].split('_q')[1].replace('_r',' / ').replace('-','/') for r in rr]
            for j,r in enumerate(rr):
                win=r['policy_id']==selected[codec,snr]
                ax.plot(r['calibration_mean_DINO'],j,marker='o' if win else 's',mfc=color if win else 'white',
                        mec=color,ms=5 if win else 4.5,color=color,zorder=3)
            ax.set_yticks(range(len(rr)),names);ax.set_ylim(-.6,len(rr)-.4);ax.invert_yaxis()
            ax.set_xlabel('Mean DINO');ax.set_title(f"({chr(97+i)}) {'NeST-Com' if codec==0 else 'ARPC'}\n{snr} dB",loc='left',fontsize=9)
            ax.grid(axis='x',color='.9',lw=.45);ax.margins(x=.12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        fig.text(.5,.97,'Nine registered calibration cells; filled circle = saved selected policy',ha='center',fontsize=8)
        fig.text(.5,.04,'Row labels: QAM bits/symbol q / nominal code rate. Each point averages 1,000 sources after three noises.',ha='center',fontsize=7)
        if gray:fig.savefig(out/'calibration_gray.png',dpi=300)
        else:export(fig,'calibration',[180,64])
        plt.close(fig)
    save(out/'figure_manifest.json',dict(status='EXPORTED_AWAITING_ACTUAL_VISUAL_REVIEW',files=manifest,
        vector_pdf=True,editable_svg_text=True,draws_or_statistics_recomputed=0,style=STYLE))


def run(auth):
    data, tr, nr, scored, exceptions = verify_closure(auth)
    candidates, chosen=validate_selection(data['selection'])
    candidate_evidence=load(data['selection']['selection_evidence'])
    require(candidate_evidence==candidates,'saved candidate evidence mismatch')
    b2_source_path=next(d for d in data['B2_score']['outputs'] if Path(d['path']).name=='source_noise_means.csv')
    b2sources=csvread(checked(b2_source_path))
    b2ids={s['source_id'] for s in load(tr['B2']['private_sources'])}
    require(len(b2ids)==1000 and len(b2sources)==9000 and
            {(x['source_id'],int(x['codec']),int(x['snr_db']),x['policy_id']) for x in b2sources} ==
            {(sid,c,s,p) for sid in b2ids for c,s,p in B2_CELLS},
            'B2 saved source denominator')
    for row in b2sources:number(row['DINO_source_mean'])
    root=O/'b3_metrics_v1'
    frames=csvread(root/'logical_scores.csv');sources=csvread(root/'source_noise_means.csv')
    means=csvread(root/'summary.csv');paired=csvread(root/'paired_differences.csv')
    current_sources=load(tr['B3']['private_sources'])
    ids=[r['source_id'] for r in current_sources]
    wide=validate_metric_tables(frames,sources,means,paired,ids,METHODS,(4,10))
    require(len(wide)==400,'wide source count')
    fixedroot=PRIOR/'metrics_v1'
    ff=csvread(fixedroot/'logical_scores.csv');fs=csvread(fixedroot/'source_noise_means.csv')
    fm=csvread(fixedroot/'summary.csv');fp=csvread(fixedroot/'paired_differences.csv')
    prior_plan=known(fixedroot/'pair_plan.json')
    require({(s['source_id'],s['source_index'],s['archive']['sha256']) for s in prior_plan['sources']} ==
            {(s['source_id'],s['source_index'],s['source']['sha256']) for s in current_sources},
            'fixed/adaptive source pixel identities differ')
    validate_metric_tables(ff,fs,fm,fp,ids,FIXED,(4,10,19))
    require({r['comparison'] for r in paired}=={'NeST_adaptive_minus_ARPC_adaptive'},'paired contrast')
    trows=load(data['B3_transport']['logical_frames']);pixels=load(data['B3_native']['logical_pixels'])
    enriched=enrich_frames(trows,pixels,load(tr['B3']['source_fallback']),tr['B3']['capacities'],chosen)
    score_by_key={r['key']:r for r in frames}
    for r in enriched:
        s=score_by_key[r['key']]
        require((s['method'],s['source_id'],int(s['snr_db']),int(s['noise_seed']))==
                (r['method'],r['source_id'],r['snr_db'],r['noise_seed']), 'score frame identity')
        r.update({k:s[k] for k,_,_,_ in METRICS})
    transport_summary,distributions=aggregate_transport(enriched)
    counts=[]
    for phase in ('B2','B3'):
        t=data[phase+'_transport']
        for kind,cap in (('logical',27000 if phase=='B2' else 1200),('subframes',54000 if phase=='B2' else 2400),('decoder_attempts',108000 if phase=='B2' else 4800)):
            counts.append(dict(phase=phase,kind=kind,actual=t[kind],cap=cap))
        for name,count in nr[phase]['new_native_calls'].items():
            counts.append(dict(phase=phase,kind='new_native_'+name,actual=count,cap='actual missing received objects'))
        counts.append(dict(phase=phase,kind='reused_native_objects',actual=nr[phase]['reused_native_objects'],cap='exact received object identity'))
    b2=data['B2_score']
    for kind,value in [('new_DINO_pairs',b2['new_DINO_pairs']),('exact_DINO_pair_reuse',b2['exact_pair_reuse']),('source_forward_calls',b2['source_forward_calls']),('bootstrap_calls',b2['bootstrap_calls'])]:
        counts.append(dict(phase='B2',kind=kind,actual=value,cap='registered'))
    for kind in ('unique_pairs','new_pairs','reused_pairs'):
        counts.append(dict(phase='B3',kind=kind,actual=scored[kind],cap='registered actual unique pixel pairs'))
    require(sum(r['actual'] for r in counts if r['kind']=='decoder_attempts')<=112800,'global decode cap')
    require(not DEST.exists(),'report output already exists; preserve and diagnose')
    DEST.mkdir();(DEST/'figures').mkdir();(DEST/'source').mkdir()
    for d in auth['source_files']:
        shutil.copyfile(checked(d),DEST/'source'/Path(d['path']).name)
    csvwrite(DEST/'b2_all_candidates.csv',candidates);csvwrite(DEST/'b2_four_selections.csv',chosen)
    csvwrite(DEST/'b2_source_DINO_means_9000.csv',b2sources)
    csvwrite(DEST/'b3_frames_1200.csv',enriched);csvwrite(DEST/'b3_source_means_400.csv',wide)
    csvwrite(DEST/'b3_source_metric_means_2800.csv',sources);csvwrite(DEST/'b3_summary_28.csv',means)
    csvwrite(DEST/'b3_direct_paired_14.csv',paired);csvwrite(DEST/'b3_transport_summary.csv',transport_summary)
    csvwrite(DEST/'b3_transport_distributions.csv',distributions);csvwrite(DEST/'budget_and_reuse.csv',counts)
    for snrs,name in (((4,10),'fixed_full_4_10_reference'),((19,),'fixed_full_19_reference')):
        sm=[r for r in fm if int(r['snr_db']) in snrs];pa=[r for r in fp if int(r['snr_db']) in snrs]
        csvwrite(DEST/(name+'_summary.csv'),sm);csvwrite(DEST/(name+'_paired.csv'),pa)
        latex_table(DEST/(name+'.tex'),sm,FIXED,'Previously completed fixed complete-representation reference; retained original estimates and intervals.')
    latex_table(DEST/'adaptive_quality.tex',means,METHODS,'Finite low-SNR adaptive evaluation on the fixed external 100 sources. Pointwise source-bootstrap intervals.')
    csvwrite(DEST/'fixed_vs_adaptive_saved_means.csv',
        [dict(regime='adaptive',**r) for r in means]+[dict(regime='fixed_full_reference',**r) for r in fm if int(r['snr_db']) in (4,10)])
    save(DEST/'engineering_exceptions.json',dict(exceptions=exceptions,scientific_failure_suppression=False))
    save(DEST/'actual_scoring_ledger.json',dict(B2=b2['ledger'],B3=scored['ledger']))
    save(DEST/'style.json',STYLE)
    plots(DEST/'figures',means,paired,candidates,chosen)
    text=['# N2048 有限低 SNR 自适应交付', '',
          '本报告只整理已经闭合的 B2/B3 结果；没有调用模型、信道、指标或 bootstrap。B2 使用 cal1000；B3 使用固定 external_common100_v1。两者不与 common500 或 new100 混合。', '',
          '## 固定范围与选择', '',
          'B2 共 9 个候选单元，每单元 1000 源、每源三次噪声，共 27000 个逻辑对象。四个策略直接读取已冻结选择；没有重新选择，也没有用区间重叠触发新候选。校准仅使用实际像素 DINO 均值，不宣称全局最优。', '']
    for r in chosen:text.append(f"- codec={r['codec']}、{r['snr_db']} dB：`{r['policy_id']}`，校准 DINO={r['calibration_mean_DINO']:.9f}。")
    text += ['', '## 实际测试与传输', '',
             'B3 为 2 方法 × 2 SNR × 100 源 × 3 噪声 = 1200 帧。每帧两次 68 符号头部、每组 13 bit 长度＋16 bit CRC＋96 bit 重组字段，正文两次 956 符号，总 N=2048、标称 Es=2。源码占用以实际终止源码 bit 数除以按字节装入后的源码容量；不能把这比值称为无线码率。', '',
             '正确交付率要求后验源码一致且独立解析没有失败；合法误收、NULL、头部/正文失败和解析失败均保留原分母。逐帧文件同时保存发送端实际回退终点与实际收到的终点，二者不混同。', '']
    for r in transport_summary:text.append(f"- {r['method']} / {r['snr_db']} dB：正确交付 {r['correct_delivery']}/300；误收 {r['accepted_wrong']}；装不下 {r['source_no_fit']}；平均源码占用 {r['mean_source_occupancy']:.6f}。")
    text += ['', '## 七指标与直接配对区间', '',
             '先在每源内部平均三次噪声，再读取已保存的来源级统计。400 个源×方法×SNR 组对应 2800 个指标源均值；每个 SNR 14 个方法均值、7 个直接配对区间，合计 28 均值与 14 区间。配对方向统一 NeST−ARPC；低值更好的指标出现负差时有利于 NeST。', '',
             'B3 仅使用已完成的 10000 次 source bootstrap（seed 2026101902），区间是逐点 percentile 95%，没有多重比较调整。本报告仅校验保存的索引 SHA，不加载索引或重算任何区间。跨零区间不等于等价证明。', '']
    for r in paired:
        metric=next(m for m in METRICS if m[0]==r['metric'])
        text.append(f"- {r['snr_db']} dB / {metric[1]}：NeST−ARPC {format_ci(r,metric[2])}；{'跨/触零' if r['interval_crosses_or_touches_zero']=='True' else '不跨零'}。")
    text += ['', '## 固定完整表示参照', '',
             '原固定完整表示的 4/10 dB 结果与当前有限自适应结果在独立表格中并列；只使用各自已保存均值和区间，不生成 adaptive−fixed 新区间，也不相减两个边际区间。原 19 dB 完整表示参照另列，不连成“全 SNR 均经本轮优化”的曲线。ARPC 全系统差异不能全部归因于尺度内端点；现有结果不支撑新的模型训练或运行速度结论。', '',
             '## 预算、异常与交付边界', '',
             '预算与实际调用见 budget_and_reuse.csv，实际评分 ledger 原样保留；计数区分逻辑帧、物理子帧、译码、独立接收恢复和去重评分。工程失败证据逐一登记，未被改成科学成功或删除。所有旧 B0/B1、N1024、19 dB 和计时保持原版本；大备份延后。', '',
             '图采用真实已保存点与区间；导出本身不构成视觉检查。实际审图及父子进程退出收据须由 root 在本入口真实完成后另行写入，原 completion 不覆盖。', '']
    (DEST/'REPORT_zh.md').write_text('\n'.join(text),encoding='utf8')
    (DEST/'captions_en.txt').write_text(
        'Calibration. Actual received-pixel DINO averaged over three noises within each source, then over 1,000 calibration sources. Nine registered candidates; filled circles denote the saved four selections. No calibration confidence intervals were calculated.\n\n'
        'Adaptive quality. Seven metrics for the frozen finite policies at 4 and 10 dB on external_common100_v1. Each source first averages three noises. Error bars are the saved pointwise 95% source-bootstrap intervals. Failures retain their original denominators and fixed receiver outputs. Agreement is the fraction of reconstructions whose ConvNeXt top-1 prediction matches that of the source.\n\n'
        'Direct paired differences. Saved NeST-Com minus ARPC paired estimates and 95% intervals. Positive values favor NeST-Com for PSNR, prediction agreement, DINO and MS-SSIM; negative values favor NeST-Com for LPIPS, DreamSim and DISTS. Intervals touching zero do not establish equivalence. The separate fixed complete-representation 19 dB table is not a low-SNR-adapted result.\n',encoding='utf8')
    save(DEST/'input_manifest.json',dict(schema='READ_ONLY_REPORT_INPUTS_V1',files=list(INPUTS.values()),
        source_bootstrap=data['B3_metrics']['bootstrap'],new_bootstrap_draws=0,scientific_calls=0))
    files=[pin(p) for p in sorted(DEST.rglob('*')) if p.is_file()]
    save(DEST/'manifest.json',dict(status='REPORT_EXPORTED_VISUAL_REVIEW_PENDING',files=files,
        counts=dict(B2_candidates=9,selected_policies=4,B3_logical_frames=1200,source_groups=400,
                    source_metric_means=2800,summary_means=28,direct_paired_intervals=14),
        scientific_calls=0,bootstrap_calls=0))
    save(DEST/'completion.json',dict(status='REPORT_EXPORTED_VISUAL_REVIEW_PENDING',manifest=pin(DEST/'manifest.json'),
        scientific_calls=0,new_bootstrap_calls=0,actual_visual_review_required=True))
    print(json.dumps(pin(DEST/'completion.json')),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',required=True)
    ap.add_argument('--authorization-sha256',required=True);ap.add_argument('--authorization-bytes',type=int,required=True)
    a=ap.parse_args();auth=load(dict(path=a.authorization,sha256=a.authorization_sha256,bytes=a.authorization_bytes))
    run(auth)
