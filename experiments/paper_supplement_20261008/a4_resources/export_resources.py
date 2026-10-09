"""A4: read-only original500 resource/failure/state-quality export. No model/PHY/RNG."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np

COMMIT = '252176e041758ecb2d3e81b6fde5b587e7e17bb7'
DONE_SHA = 'a76ed9aad61b09019382a335da4d51780412fd6fb0bc2a33b9d9bb2d73afe38d'
CONFIG_SHA = 'beccc520c572b53a74ee8e05a48022cf8028c509f261b9c8e02d20eb870d326c'
CONFIG_SOURCE_SHA = '7d33da6d74cbc415dc1bfe145605e7fc885c25e8f73f992efe8d0d3910cee06c'
SUMMARY_SHA = '5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859'
SNRS = [1, 4, 7, 10, 13, 19]
METRICS = ['psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction']
ARMS = {'VAR_completion': 'VAR', 'direct_Dc_missing_residual_zero': 'DIRECT'}
RAW_STATES = ['HEADER_REJECTED', 'HEADER_WRONG_ACCEPT', 'BODY_CRC_REJECT_KEEP',
              'BODY_CRC_WRONG_ACCEPT', 'BODY_CRC_ACCEPT_CORRECT']
NAMES = {'RAW_WHOLE': 'Full-scale digital transmission', 'RAW_PARTIAL': 'Partial-scale digital transmission',
         'P1024': 'Continuous latent JSCC', 'SwinJSCC80k': 'SwinJSCC-80k (adapted)'}
COLORS = {'RAW_WHOLE': '#0072B2', 'RAW_PARTIAL': '#009E73', 'P1024': '#E69F00', 'SwinJSCC80k': '#CC79A7'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    require(bool(rows), 'Empty CSV export: ' + str(path))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


class Reader:
    def __init__(self):
        self.bindings = {}

    def bytes(self, path, expected=None):
        path = Path(path)
        b = path.read_bytes()
        h = hashlib.sha256(b).hexdigest()
        require(expected is None or h == expected, 'Hash mismatch: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == h, 'File changed while exporting')
        self.bindings[str(path)] = h
        return b

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))


def config_rows(reader, path):
    rows = list(csv.DictReader(io.StringIO(reader.bytes(path, CONFIG_SHA).decode('utf-8-sig'))))
    require(len(rows) == 24, 'Expected four transmitter configurations at six SNRs')
    result = {}
    for r in rows:
        method, snr = r['method'], int(r['snr_db'])
        require((method, snr) not in result, 'Duplicate selected configuration')
        for key in ['header_complex_uses', 'body_transmitted_complex_uses', 'idle_complex_uses', 'total_complex_uses_N']:
            r[key] = int(r[key])
        require(sum(r[k] for k in ['header_complex_uses', 'body_transmitted_complex_uses', 'idle_complex_uses']) == r['total_complex_uses_N'] == 1024,
                'Selected frame budget differs')
        r['frame_effective_use_fraction'] = (r['header_complex_uses'] + r['body_transmitted_complex_uses']) / 1024
        r['source_information_bits_per_complex_use'] = int(r['source_bits']) / 1024 if method.startswith('RAW_') else None
        r['source_information_scope'] = '12-bit transmitted token payload; not correctly delivered bits' if method.startswith('RAW_') else 'NOT_APPLICABLE_CONTINUOUS_SYMBOLS'
        r['padding_rule'] = 'known QPSK Es2; carries no image information' if method.startswith('RAW_') else 'none'
        r['symbol_energy_convention'] = 'constellation average Es2; no per-frame RAW normalization' if method.startswith('RAW_') else 'frozen continuous normalization; realized energy reported separately'
        result[method, snr] = r
    require(set(result) == {(m, s) for m in NAMES for s in SNRS}, 'Selected configuration grid differs')
    return result


def receiver_diagnostic(e, tokens, cfg):
    """Offline truth comparison only; original receiver state is never changed."""
    tx_count = int(cfg['source_token_count'])
    truth = tokens[:tx_count]
    payload = ((truth[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).reshape(-1)
    require(array_sha(payload) == e['transmitted_payload_sha256'], 'TX token/payload hash differs')
    st = e['receiver_state']
    received = np.asarray([v for scale in st.get('prefix', []) for v in scale] + st.get('partial_values', []), dtype=np.int64)
    overlap = min(len(received), tx_count)
    wrong = int(np.count_nonzero(received[:overlap] != truth[:overlap]))
    missing, extra = max(tx_count - len(received), 0), max(len(received) - tx_count, 0)
    correct_prefix = 0
    for a, b in zip(received[:overlap], truth[:overlap]):
        if a != b:
            break
        correct_prefix += 1
    header_ok, body_ok = bool(e['header_ok']), bool(e['body_crc_accept'])
    header_wrong = header_ok and e['rx_profile_id'] != e['profile_id']
    body = e['body']
    payload_wrong = (body is not None and not np.array_equal(np.asarray(body['hard_payload'], dtype=np.uint8), payload))
    false_accept = header_ok and body_ok and payload_wrong
    if not header_ok:
        state = 'HEADER_REJECTED'
    elif header_wrong:
        state = 'HEADER_WRONG_ACCEPT'
    elif not body_ok:
        state = 'BODY_CRC_REJECT_KEEP'
    elif false_accept:
        state = 'BODY_CRC_WRONG_ACCEPT'
    else:
        state = 'BODY_CRC_ACCEPT_CORRECT'
    require(not e['gray'] or len(received) == 0, 'Gray output unexpectedly has received tokens')
    require(e['same_RX_visual_proof']['received_token_count'] == len(received), 'Received token count differs')
    require(not e['same_RX_visual_proof']['target_used_for_generation'] and not e['same_RX_visual_proof']['TX_truth_used_for_generation'], 'Source truth entered original generation')
    tx = e['transmission']
    require(tx['N'] == 1024 and tx['header_uses'] == cfg['header_complex_uses']
        and tx['body_uses'] == cfg['body_transmitted_complex_uses'] and tx['idle_uses'] == cfg['idle_complex_uses'], 'Actual/frozen frame budget differs')
    g = tx['groups'][0]
    require(g['source_bits'] == int(cfg['source_bits']) and g['k'] == int(cfg['ldpc_information_k'])
        and g['n'] == int(cfg['ldpc_transmitted_n']) and g['q'] * tx['body_uses'] == g['n'], 'Actual transmitted code dimensions differ')
    body_energy = sum(float(v) for v in g['actual_E'])
    idle_energy = 2 * tx['idle_uses']
    return dict(failure_state=state, header_accepted=header_ok, header_wrong_accept=header_wrong,
        body_crc_accepted=body_ok if header_ok else None, body_crc_rejected=header_ok and not body_ok,
        body_crc_wrong_accept=bool(false_accept), gray=bool(e['gray']),
        tx_profile_id=e['profile_id'], rx_profile_id=e['rx_profile_id'],
        tx_planned_token_count=tx_count, rx_token_count=len(received), compared_token_count=overlap,
        wrong_compared_tokens=wrong, token_error_fraction_compared=wrong / overlap if overlap else None,
        missing_planned_tokens=missing, extra_received_tokens=extra,
        wrong_or_missing_planned_fraction=(wrong + missing) / tx_count,
        rx_correct_contiguous_planned_prefix=correct_prefix,
        actual_frame_energy=float(tx['E']), actual_body_energy=body_energy,
        known_padding_energy=idle_energy, actual_header_energy=float(tx['E']) - body_energy - idle_energy,
        actual_mean_complex_energy=float(tx['E']) / 1024,
        received_sha256=e['received_sha256'], waveform_sha256=tx['waveform_sha256'],
        offline_truth_used_only_for_diagnostics=True)


def public_point(method, snr):
    if method.startswith('RAW_'):
        _, family, arm = method.split('_')
        return f'RAW64_{family}_{"VAR_COMPLETION" if arm == "VAR" else "DIRECT_DC"}_SNR_{snr}'
    return f'P1024_SNR_{snr}' if method == 'P1024' else f'SWIN80K_N1024_SNR_{snr}'


def summarize(frame_rows, quality_rows, published, configs):
    frame_groups = defaultdict(list)
    for r in frame_rows:
        frame_groups[r['method'], r['snr_db']].append(r)
    resource, failures, tokens = [], [], []
    for (method, snr), rows in sorted(frame_groups.items()):
        require(len(rows) == 1500 and len({r['source_id'] for r in rows}) == 500, 'Incomplete frame group')
        r = dict(configs[method, snr], source_count=500, noise_count=3, frame_count=1500)
        energies = np.asarray([x['actual_frame_energy'] for x in rows], dtype=np.float64)
        r.update(actual_frame_energy_mean=float(energies.mean()), actual_frame_energy_min=float(energies.min()),
            actual_frame_energy_p05=float(np.quantile(energies, .05)), actual_frame_energy_p50=float(np.quantile(energies, .5)),
            actual_frame_energy_p95=float(np.quantile(energies, .95)), actual_frame_energy_max=float(energies.max()),
            actual_frame_energy_std=float(energies.std(ddof=0)), actual_mean_complex_energy=float(energies.mean()/1024))
        resource.append(r)
        states = RAW_STATES if method.startswith('RAW_') else ['HEADER_REJECTED', 'HEADER_WRONG_ACCEPT', 'HEADER_ACCEPTED'] if method == 'SwinJSCC80k' else ['CONTINUOUS_OUTPUT']
        for state in states:
            selected = [x for x in rows if x['failure_state'] == state]
            failures.append(dict(method=method, snr_db=snr, failure_state=state, frame_count=len(selected),
                fraction_of_all_frames=len(selected)/1500, represented_source_count=len({x['source_id'] for x in selected}),
                total_source_count=500, total_frame_count=1500, gray_frame_count=sum(x['gray'] for x in selected)))
        if method.startswith('RAW_'):
            for field in ['token_error_fraction_compared', 'wrong_or_missing_planned_fraction', 'rx_correct_contiguous_planned_prefix', 'rx_token_count']:
                values = np.asarray([x[field] for x in rows if x[field] is not None], dtype=np.float64)
                tokens.append(dict(method=method, snr_db=snr, diagnostic=field, valid_frame_count=len(values),
                    missing_frame_count=1500-len(values), mean=float(values.mean()) if len(values) else None,
                    p05=float(np.quantile(values,.05)) if len(values) else None,
                    p50=float(np.quantile(values,.5)) if len(values) else None,
                    p95=float(np.quantile(values,.95)) if len(values) else None,
                    max=float(values.max()) if len(values) else None, offline_truth_only=True))
    groups = defaultdict(list)
    for r in quality_rows:
        groups[r['method'], r['snr_db']].append(r)
    state_quality, all_quality, checks = [], [], []
    for (method, snr), rows in sorted(groups.items()):
        require(len(rows) == 1500, 'Incomplete quality group')
        expected_states = RAW_STATES if method.startswith('RAW_') else ['HEADER_REJECTED', 'HEADER_WRONG_ACCEPT', 'HEADER_ACCEPTED'] if method == 'SwinJSCC80k' else ['CONTINUOUS_OUTPUT']
        for state in ['ALL_FRAMES', *expected_states]:
            selected = rows if state == 'ALL_FRAMES' else [r for r in rows if r['failure_state'] == state]
            by_source = defaultdict(list)
            for r in selected:
                by_source[r['source_id']].append(r)
            for metric in METRICS:
                frame_mean = float(np.mean([r[metric] for r in selected])) if selected else None
                source_mean = float(np.mean([np.mean([r[metric] for r in batch]) for batch in by_source.values()])) if selected else None
                row = dict(method=method, snr_db=snr, failure_state=state, metric=metric,
                    source_count=len(by_source), frame_count=len(selected), frame_weighted_mean=frame_mean,
                    source_balanced_conditional_mean=source_mean,
                    status='MEASURED_DESCRIPTIVE_SUBSET' if selected else 'NO_OBSERVATIONS', ci_low=None, ci_high=None,
                    interval_status='NOT_RECOMPUTED_NO_BOOTSTRAP')
                if state == 'ALL_FRAMES':
                    require(len(by_source) == 500 and all(len(v)==3 for v in by_source.values()), 'Not500 sources with3 noises')
                    pub = published[public_point(method, snr), metric]
                    difference = abs(source_mean - float(pub['mean']))
                    require(difference < 1e-10, 'Exported metric mean differs from published CSV: ' + str((method,snr,metric,difference)))
                    row.update(status='PUBLISHED_FULL_FRAME_RESULT_REUSED', ci_low=float(pub['ci_low']), ci_high=float(pub['ci_high']),
                        interval_status='ORIGINAL_PUBLISHED_SOURCE_BOOTSTRAP_REUSED')
                    all_quality.append(dict(method=method, snr_db=snr, metric=metric, mean=float(pub['mean']),
                        ci_low=float(pub['ci_low']), ci_high=float(pub['ci_high']), source_count=500, noise_count=3,
                        point_id=pub['point_id'], source_information_bits_per_complex_use=configs[method.rsplit('_',1)[0] if method.startswith('RAW_') else method,snr]['source_information_bits_per_complex_use']))
                    checks.append(dict(method=method,snr_db=snr,metric=metric,absolute_mean_difference=difference))
                state_quality.append(row)
    return resource, failures, tokens, state_quality, all_quality, checks


def figures(out, resource, failures, all_quality, state_quality):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8.5, 'axes.titlesize':9, 'legend.fontsize':7,
        'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':True, 'grid.color':'#e5e5e5',
        'grid.linewidth':.5, 'axes.axisbelow':True, 'savefig.facecolor':'white'})
    def finish(fig, name):
        for ext in ['pdf','svg','png']:
            fig.savefig(out / (name+'.'+ext), dpi=600 if ext=='png' else None, bbox_inches='tight', pad_inches=.07)
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(7,4.7),layout='constrained',sharex=True,sharey=True)
    for ax, method in zip(axes.flat, NAMES):
        rows = sorted([r for r in resource if r['method']==method], key=lambda r:int(r['snr_db']))
        bottom = np.zeros(6)
        for field,label,color,hatch in [('header_complex_uses','Protected header','#56B4E9','///'),
            ('body_transmitted_complex_uses','Image-bearing body',COLORS[method],None),
            ('idle_complex_uses','Known padding','#c5c5c5','xx')]:
            values = np.asarray([r[field] for r in rows])
            ax.bar(SNRS,values,bottom=bottom,width=1.55,label=label,color=color,edgecolor='#333333',linewidth=.35,hatch=hatch)
            bottom+=values
        ax.set_title(NAMES[method]);ax.set_xticks(SNRS);ax.set_ylim(0,1100);ax.set_yticks([0,256,512,768,1024])
        ax.set_xlabel('SNR (dB)');ax.set_ylabel('Complex channel uses')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    finish(fig,'figA4_budget_allocation')
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained',sharey=True)
    style=[('HEADER_REJECTED','Header rejected','#777777','///'),('HEADER_WRONG_ACCEPT','Wrong header accepted','#CC79A7','++'),
        ('BODY_CRC_REJECT_KEEP','Body CRC rejected; KEEP','#E69F00','xx'),('BODY_CRC_WRONG_ACCEPT','Wrong body accepted','#D55E00','..'),
        ('BODY_CRC_ACCEPT_CORRECT','Body accepted, correct','#009E73',None)]
    for ax,method in zip(axes,['RAW_WHOLE','RAW_PARTIAL']):
        bottom=np.zeros(6)
        for state,label,color,hatch in style:
            vals=np.asarray([next(r['fraction_of_all_frames'] for r in failures if r['method']==method and r['snr_db']==s and r['failure_state']==state)*100 for s in SNRS])
            ax.bar(SNRS,vals,bottom=bottom,width=1.6,label=label,color=color,edgecolor='#333333',linewidth=.3,hatch=hatch);bottom+=vals
        ax.set_title(NAMES[method]);ax.set_xlabel('SNR (dB)');ax.set_xticks(SNRS);ax.set_ylim(0,103);ax.set_ylabel('All frames (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    finish(fig,'figA4_failure_states')
    labels=['PSNR (dB) ↑','LPIPS ↓','DINOv2-L cosine similarity ↑','ConvNeXt prediction agreement (%) ↑']
    fig,axes=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
    for ax,metric,label in zip(axes.flat,METRICS,labels):
        for family,marker,line in [('RAW_WHOLE','s','--'),('RAW_PARTIAL','o','-')]:
            rows=sorted([r for r in all_quality if r['method']==family+'_VAR' and r['metric']==metric],key=lambda r:r['snr_db'])
            factor=100 if metric==METRICS[-1] else 1
            x=np.asarray([r['source_information_bits_per_complex_use'] for r in rows]);y=np.asarray([r['mean'] for r in rows])*factor
            lo=np.asarray([r['ci_low'] for r in rows])*factor;hi=np.asarray([r['ci_high'] for r in rows])*factor
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=marker+line,color=COLORS[family],markersize=4,capsize=2,linewidth=1,
                label=NAMES[family].replace(' digital transmission','')+' + VAR')
        ax.set_xlabel('Source bits / complex channel use');ax.set_ylabel(label)
        ax.margins(x=.08,y=.16)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=2)
    finish(fig,'figA4_source_information_quality')
    fig,axes=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
    selected_styles=[('ALL_FRAMES','All frames','#0072B2','o','-'),('BODY_CRC_ACCEPT_CORRECT','Body accepted, correct','#009E73','s','--'),
        ('BODY_CRC_REJECT_KEEP','Body CRC rejected; KEEP','#E69F00','^',':'),('HEADER_REJECTED','Header rejected','#777777','x','-.')]
    for ax,metric,label in zip(axes.flat,METRICS,labels):
        for state,name,color,marker,line in selected_styles:
            rows=[next(r for r in state_quality if r['method']=='RAW_PARTIAL_VAR' and r['metric']==metric and r['snr_db']==s and r['failure_state']==state) for s in SNRS]
            factor=100 if metric==METRICS[-1] else 1
            values=[r['frame_weighted_mean']*factor if r['frame_count'] else np.nan for r in rows]
            ax.plot(SNRS,values,label=name,color=color,marker=marker,linestyle=line,markersize=4,linewidth=1)
        ax.set_xlabel('SNR (dB)');ax.set_xticks(SNRS);ax.set_ylabel(label);ax.margins(y=.12)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=2)
    finish(fig,'figA4_partial_VAR_state_quality')


def figure_captions(failures):
    reject=next(r for r in failures if r['method']=='RAW_PARTIAL' and r['snr_db']==7 and r['failure_state']=='BODY_CRC_REJECT_KEEP')
    return '''% A4 figure caption fragments; no newly estimated confidence intervals.
\\caption{Frozen N1024 allocation to protected header, image-bearing body and known padding. RAW padding consists of energized known QPSK symbols and carries no image information. Selected full-scale and partial-scale schemes had the same modulation/coding permissions; a selected idle allocation does not imply that stronger full-budget protection was forbidden.}
\\caption{Actual receiver states for all 500 sources and three noise realizations per SNR. Body CRC rejection under KEEP retains actual hard tokens and is not automatically a gray-image failure. Wrong acceptance is diagnosed offline against transmitted truth and never changes receiver decisions.}
\\caption{Published full-frame quality versus transmitted raw source bits per complex channel use. Each line connects the six frozen operating points in SNR order: 1, 4, 7, 10, 13 and 19 dB. Exact point identities and coordinates are retained in published_quality_reused.csv. Points vary in both SNR and frozen policy, so the lines are descriptive rather than a controlled causal allocation curve. Error bars reuse the original source-level 95\\% intervals. ConvNeXt reports source-prediction agreement, not classification accuracy.}
\\caption{Descriptive state-conditioned quality of partial-scale digital transmission with VAR completion. Curves show frame-weighted subset means; zero-count states have undefined means and remain explicit in the accompanying tables. Subsets have different source/noise membership and different sample counts at each SNR; counts are given in subset_counts.csv. At 7 dB, the body-CRC-rejected KEEP subset contains only COUNT7 of 1500 frames, so its agreement value must not be interpreted as a reliable improvement or a population success rate. No conditional confidence intervals, gain intervals or additional bootstrap replicates were computed.}
'''.replace('COUNT7',str(reject['frame_count']))


def closed_outputs(out):
    # Logs and launcher metadata can still change after the result is closed.
    return {str(p):sha(p) for p in out.iterdir() if p.is_file()
            and p.name!='completion.json' and p.suffix!='.log'
            and not p.name.startswith(('launch','console'))}


def render_existing(args):
    source=Path(args.render_existing).resolve()
    out=Path(args.output).resolve() if args.output else source/'figures_r2'
    require(out!=source,'Render output must not overwrite the scientific result directory')
    require(not out.exists() or not any(out.iterdir()),'Use an empty render output directory')
    completion_path=source/'completion.json'
    completion=json.loads(completion_path.read_text(encoding='utf-8'))
    require(completion['status']=='COMPLETE' and completion['published_commit']==COMMIT
            and completion['source_count']==500 and completion['noise_count']==3,
            'Only the completed published500 export can be rendered')
    bindings={str(completion_path):sha(completion_path)}
    tables={}
    for name in ['resource_summary.csv','failure_state_counts.csv','published_quality_reused.csv','state_quality.csv']:
        path=source/name
        expected=[h for p,h in completion['outputs'].items() if Path(p).name==name]
        require(len(expected)==1 and sha(path)==expected[0],'Original CSV differs: '+name)
        bindings[str(path)]=expected[0]
        rows=list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        for row in rows:
            row['snr_db']=int(row['snr_db'])
            for key in ['header_complex_uses','body_transmitted_complex_uses','idle_complex_uses',
                        'frame_count','source_count','noise_count']:
                if key in row:
                    row[key]=int(row[key])
            for key in ['fraction_of_all_frames','source_information_bits_per_complex_use',
                        'mean','ci_low','ci_high','frame_weighted_mean']:
                if key in row:
                    row[key]=float(row[key]) if row[key] else None
        tables[name]=rows
    resource=tables['resource_summary.csv'];failures=tables['failure_state_counts.csv']
    quality=tables['published_quality_reused.csv'];states=tables['state_quality.csv']
    require(len(resource)==24 and len(quality)==144,'Existing figure data grid differs')
    out.mkdir(parents=True,exist_ok=True)
    figures(out,resource,failures,quality,states)
    # These are literal existing count fields, not recomputed subsets or statistics.
    counts=[{k:r[k] for k in ['snr_db','failure_state','source_count','frame_count']}
            for r in states if r['method']=='RAW_PARTIAL_VAR' and r['metric']==METRICS[0]]
    csv_write(out/'subset_counts.csv',counts)
    (out/'captions.tex').write_text(figure_captions(failures),encoding='utf-8')
    (out/'README.md').write_text(
        '# A4 rendering revision 2\n\n'
        'This directory only rerenders four existing, completion-hash-verified CSVs. '
        'Original data, statistics and first-render files remain unchanged. '
        'No source checkpoints, model, channel, RNG or bootstrap are accessed.\n\n'
        'The source-information figure omits point labels that overlapped at 10/19 dB; '
        'the caption supplies SNR ordering and the original table retains exact coordinates. '
        'The state-quality caption now calls out varying subset counts and the single '
        'CRC-rejected KEEP frame at 7 dB. subset_counts.csv copies existing count fields; '
        'conditional curves have no gain or subset confidence intervals.\n\n'
        'Reproduce: `python experiments/paper_supplement_20261008/a4_resources/export_resources.py '
        '--render-existing '+str(source)+' --output <new-empty-render-directory>`\n',encoding='utf-8')
    save(out/'input_bindings.json',dict(published_commit=COMMIT,read_files=bindings,
        script_sha256=sha(__file__),operation='RENDER_EXISTING_TABLES_ONLY',
        new_statistics=0,new_inference=0,new_channel_simulations=0,new_bootstrap_replicates=0))
    # Verify even the input bytes remain unchanged after rendering.
    require(all(sha(p)==h for p,h in bindings.items()),'An existing input changed during rendering')
    save(out/'completion.json',dict(status='COMPLETE',operation='RENDER_EXISTING_TABLES_ONLY',
        published_commit=COMMIT,source_count=500,noise_count=3,outputs=closed_outputs(out),
        new_statistics=0,new_inference=0,new_channel_simulations=0,new_bootstrap_replicates=0))
    print(json.dumps(dict(status='COMPLETE',operation='RENDER_EXISTING_TABLES_ONLY',output=str(out))),flush=True)


def run(args):
    root,out=Path(args.root).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/'completion.json').exists(),'Completed export exists; reuse it or choose a fresh output directory')
    r=Reader();base=root/'outputs/MAIN-RAW64-20261007';metrics=base/'unified500_metrics_r6'
    done=r.json(metrics/'completion.json',DONE_SHA)
    require(done['source_count']==500 and done['raw_metric_row_count']==33000 and done['baseline_metric_row_count']==18000,'Wrong original500 metric completion')
    require(done['logical_family_metric_row_count']==36000 and done['noise_seeds']==[6201,6202,6203],'RAW family/noise scope differs')
    configs=config_rows(r,root/'paper/tables/mainraw64_supplement_20261008/selected_configs.csv')
    provenance=r.json(root/'paper/tables/mainraw64_supplement_20261008/sources.json',CONFIG_SOURCE_SHA)
    require(provenance['published_commit']==COMMIT,'Configuration provenance commit differs')
    save(out/'configuration_provenance.json',provenance)
    summary_path=root/'results/main_raw64_20261007/final_common500_r6/summary.csv'
    published_rows=list(csv.DictReader(io.StringIO(r.bytes(summary_path,SUMMARY_SHA).decode('utf-8-sig'))))
    published={(x['point_id'],x['metric']):x for x in published_rows}
    require(len(published)==len(published_rows),'Duplicated published summary identity')
    frame_rows,quality_rows=[],[]
    began=time.time()
    for i,sid in enumerate(done['source_ids']):
        cp_path=metrics/'source_checkpoints'/f'{i:04d}.json'
        cp=r.json(cp_path,done['outputs'][str(cp_path)])
        rows=r.json(cp['rows'],done['outputs'][cp['rows']])
        require(cp['source_index']==i and cp['source_id']==sid and len(rows)==66,'RAW source metric scope differs')
        ref=cp['reference_evidence'];asset=r.bytes(ref['archive'],ref['archive_sha256'])
        with np.load(io.BytesIO(asset),allow_pickle=False) as z:
            token=z['tokens'].copy()
        require(token.dtype==np.int64 and token.shape==(680,), 'Original token archive layout differs')
        scp=r.json(ref['checkpoint'],ref['checkpoint_sha256'])
        # Source assets use their own domain; received payloads use array_sha.
        source_token_sha=hashlib.sha256(b'int64:680\0'+token.astype('<i8').tobytes()).hexdigest()
        require(source_token_sha==scp['tokens_sha256'] and scp['source_id']==sid,'Original source token hash differs')
        raw_path=base/'unified500_same_rx_parallel_images_r2/sources'/f'{i:04d}.json'
        original=r.json(raw_path,cp['input_bindings'][str(raw_path)])
        old={(e['snr_db'],e['candidate_id'],e['noise_seed']):e for e in original}
        require(len(old)==33,'Wrong physical RAW frame count')
        diagnostic={};seen=set()
        for m in rows:
            require(m['population_role']=='holdout' and m['source_id']==sid and m['N']==1024 and m['metrics_scored'],'Mixed population or unscored image')
            key=(m['snr_db'],m['candidate_id'],m['noise_seed'])
            e=old[key];arm=m['decoder_arm'];unique=(*key,arm)
            require(unique not in seen,'Duplicate RAW metric row');seen.add(unique)
            require(m['render_evidence']==e and m['image_sha256']==e['same_RX_images'][arm]['image_sha256'],'Metric/image/actual receiver binding differs')
            require(e['source_tokens_sha256']==scp['tokens_sha256'],'Received row source token identity differs')
            for family in m['family_memberships']:
                transmitter='RAW_'+family;cfg=configs[transmitter,m['snr_db']]
                require(int(cfg['profile_id'])==e['profile_id'] and cfg['candidate_id']==e['candidate_id'],'Wrong frozen transmitter policy')
                dkey=(transmitter,*key)
                if dkey not in diagnostic:
                    d=receiver_diagnostic(e,token,cfg)
                    f=dict(source_index=i,source_id=sid,method=transmitter,snr_db=m['snr_db'],noise_seed=m['noise_seed'],
                        candidate_id=e['candidate_id'],header_complex_uses=cfg['header_complex_uses'],
                        body_complex_uses=cfg['body_transmitted_complex_uses'],padding_complex_uses=cfg['idle_complex_uses'],
                        source_bits=int(cfg['source_bits']),frame_effective_use_fraction=cfg['frame_effective_use_fraction'],
                        source_information_bits_per_complex_use=cfg['source_information_bits_per_complex_use'],**d)
                    frame_rows.append(f);diagnostic[dkey]=f
                values={metric:float(m[metric]) for metric in METRICS}
                require(all(math.isfinite(v) for v in values.values()),'Missing/nonfinite original metric')
                quality_rows.append(dict(source_index=i,source_id=sid,method=transmitter+'_'+ARMS[arm],snr_db=m['snr_db'],
                    noise_seed=m['noise_seed'],failure_state=diagnostic[dkey]['failure_state'],image_sha256=m['image_sha256'],**values))
        require(len(diagnostic)==36 and len(seen)==66,'RAW family/arm coverage differs')
        for method in ['P1024','SwinJSCC80k']:
            p=metrics/'baselines'/method/'source_checkpoints'/f'{i:04d}.json'
            bc=r.json(p,done['outputs'][str(p)])
            rp=metrics/'baselines'/method/'sources'/f'{i:04d}.json'
            baseline=r.json(rp,done['outputs'][str(rp)])
            require(bc['source_id']==sid and len(baseline)==18,'Baseline source metric scope differs')
            seen_b=set()
            for m in baseline:
                e=m['render_evidence'];snr=m['snr_db'];seed=m['noise_seed']
                require(m['source_id']==sid and m['population_role']=='holdout' and m['N']==1024,'Mixed baseline population')
                require((snr,seed) not in seen_b,'Duplicate baseline frame');seen_b.add((snr,seed))
                cfg=configs[method,snr]
                header=e.get('header_accepted');wrong=bool(e.get('header_metadata_mismatch',False))
                state='CONTINUOUS_OUTPUT' if method=='P1024' else 'HEADER_REJECTED' if not header else 'HEADER_WRONG_ACCEPT' if wrong else 'HEADER_ACCEPTED'
                frame_rows.append(dict(source_index=i,source_id=sid,method=method,snr_db=snr,noise_seed=seed,
                    failure_state=state,header_accepted=header,header_wrong_accept=wrong,
                    gray=method=='SwinJSCC80k' and not header,actual_frame_energy=float(e['E']),
                    actual_mean_complex_energy=float(e['E'])/1024,header_complex_uses=cfg['header_complex_uses'],
                    body_complex_uses=cfg['body_transmitted_complex_uses'],padding_complex_uses=0,
                    frame_effective_use_fraction=cfg['frame_effective_use_fraction'],waveform_sha256=e['waveform_sha256'],
                    received_sha256=e.get('observation_sha256'),offline_truth_used_only_for_diagnostics=True))
                values={metric:float(m[metric]) for metric in METRICS}
                require(all(math.isfinite(v) for v in values.values()),'Missing/nonfinite baseline metric')
                quality_rows.append(dict(source_index=i,source_id=sid,method=method,snr_db=snr,noise_seed=seed,
                    failure_state=state,image_sha256=m['image_sha256'],**values))
            require(seen_b=={(s,n) for s in SNRS for n in [2001,2002,2003]},'Baseline SNR/noise grid differs')
        if i%50==0 or i==499:
            print(json.dumps(dict(sources_read=i+1,total=500,seconds=time.time()-began)),flush=True)
    require(len(frame_rows)==36000 and len(quality_rows)==54000,'Complete physical-family/image population required')
    resource,failures,tokens,state_quality,all_quality,checks=summarize(frame_rows,quality_rows,published,configs)
    for name,rows in [('resource_summary.csv',resource),('failure_state_counts.csv',failures),('token_error_summary.csv',tokens),
        ('state_quality.csv',state_quality),('published_quality_reused.csv',all_quality),('published_mean_checks.csv',checks),
        ('frame_diagnostics.csv',frame_rows),('quality_by_frame.csv',quality_rows)]:
        csv_write(out/name,rows)
    figures(out,resource,failures,all_quality,state_quality)
    save(out/'input_bindings.json',dict(published_commit=COMMIT,read_files=r.bindings,script_sha256=sha(__file__),
        new_inference=0,new_channel_simulations=0,new_packet_decodes=0,new_bootstrap_replicates=0,
        note='Read original floating metrics, receiver evidence, frozen resources, and original token arrays; no image rendering or source encoding.'))
    report=['# A4: resources, failures and state-conditioned quality','',
        f'Original published result commit: `{COMMIT}`. All 500 sources and three noises per SNR are retained.',
        'No inference, channel simulation, packet decoding, policy selection or bootstrap was performed.',
        '', 'The 16,500 original RAW physical frames expand to 18,000 WHOLE/PARTIAL memberships because the 7 dB policy is shared. '+
        'They have two reconstruction arms. The continuous and adapted Swin baselines contribute 9,000 original frames each.',
        '', '`resource_summary.csv` contains source bits, CRC/control fields, actual LDPC k/n, nominal/full-budget allocation, header/body/padding, frame-use efficiency and measured energy quantiles. '+
        'WHOLE retains the same legal MCS/allocation permission as PARTIAL. Its selected configuration and padding do not establish that extra protection was prohibited.',
        '', 'Known RAW padding is energized QPSK carrying no image information; frame energy is the saved actual waveform energy, not an imposed 2048 value.',
        '', '`token_error_fraction_compared` excludes absent token positions; no-overlap frames are missing for that diagnostic. '+
        '`wrong_or_missing_planned_fraction` separately counts wrong or absent transmitted-prefix tokens. Neither is the transmitter planned prefix itself. '+
        'All truth comparisons are offline diagnostics and do not change receiver decisions.',
        '', 'Body CRC rejection under KEEP is not image failure. Header rejection, wrong accepted header, CRC rejected KEEP, wrong accepted body and correct accepted body are retained separately. '+
        'Zero-count states remain in the CSVs.',
        '', 'State-quality tables contain both frame-weighted subset means and means balanced over sources represented in that state. '+
        'They are descriptive, have varying subset membership, and have no newly computed intervals. ALL_FRAMES retains the original published source-level intervals. '+
        'ConvNeXt agreement is source-prediction agreement, not label accuracy; plotted values multiply the stored fractions by100. LPIPS is not sign-flipped.',
        '', 'The source-information/quality figure changes SNR along each line as well as the selected policy. It is descriptive and does not isolate a causal bit-allocation effect. '+
        'Its error bars are the original published intervals. The state-quality figure uses frame-weighted subset means and omits only undefined zero-count means, preserving the zero-count CSV rows.',
        '', 'SwinJSCC is an adapted, budget-truncated 80k checkpoint. 19 dB remains outside its training and calibration range.',
        '', '## Selected 10/19 dB facts','']
    for snr in [10,19]:
        for method in ['RAW_WHOLE','RAW_PARTIAL']:
            x=next(z for z in resource if z['method']==method and int(z['snr_db'])==snr)
            gray=sum(z['gray_frame_count'] for z in failures if z['method']==method and z['snr_db']==snr)
            reject=next(z['frame_count'] for z in failures if z['method']==method and z['snr_db']==snr and z['failure_state']=='BODY_CRC_REJECT_KEEP')
            report.append(f'- {NAMES[method]}, {snr} dB: m={x["m"]}, K={x["K"]}, {x["body_modulation"]}, '+
                f'actual LDPC {x["ldpc_effective_rate_fraction"]}, header/body/padding={x["header_complex_uses"]}/{x["body_transmitted_complex_uses"]}/{x["idle_complex_uses"]}; '+
                f'CRC-rejected KEEP={reject}/1500, gray={gray}/1500, mean actual energy={x["actual_frame_energy_mean"]:.6f}.')
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (out/'captions.tex').write_text(figure_captions(failures),encoding='utf-8')
    subset_counts=[{k:r[k] for k in ['snr_db','failure_state','source_count','frame_count']}
        for r in state_quality if r['method']=='RAW_PARTIAL_VAR' and r['metric']==METRICS[0]]
    csv_write(out/'subset_counts.csv',subset_counts)
    outputs=closed_outputs(out)
    save(out/'completion.json',dict(status='COMPLETE',published_commit=COMMIT,source_count=500,noise_count=3,
        frame_memberships=36000,quality_rows=54000,configuration_rows=24,
        maximum_published_mean_difference=max(x['absolute_mean_difference'] for x in checks),outputs=outputs,
        new_inference=0,new_channel_simulations=0,new_packet_decodes=0,new_bootstrap_replicates=0,seconds=time.time()-began))
    print(json.dumps(dict(status='COMPLETE',output=str(out),sources=500,new_packet_decodes=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--render-existing',help='Only rerender an existing completed export; default output is its figures_r2 subdirectory')
    args=parser.parse_args()
    if args.render_existing:
        render_existing(args)
    else:
        require(args.root and args.output,'Export requires --root and --output')
        run(args)
