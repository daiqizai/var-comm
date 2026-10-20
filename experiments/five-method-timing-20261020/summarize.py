"""Summarize actual completed timing calls. No model, PHY or bootstrap imports."""
import argparse
import csv
import math
import statistics
from collections import Counter
from contract import *

NAMES=dict(NEST='NeST-Com (entropy-coded, m10 corrected)',P100000='Latent JSCC (100k)',
           Swin200000='SwinJSCC (adapted, 200k)',BPG='Adaptive BPG + LDPC',
           ARPC='ARPC + LDPC-QAM (adapted)')


def percentile(values,q):
    a=sorted(values)
    pos=(len(a)-1)*q
    lo=int(math.floor(pos));hi=int(math.ceil(pos))
    return a[lo]+(a[hi]-a[lo])*(pos-lo)


def csvsave(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main(expected):
    r=binding(expected)
    outer=read(T/'dispatch/actual_wait.json')
    assert type(outer['returncode']) is int and outer['returncode']==0 and not outer['timed_out']
    for d in (T/'dispatch/launch.json',T/'control/launch.json'):
        ident=read(d)['process']
        now=process(ident['pid'])
        assert now is None or now['start_ticks']!=ident['start_ticks']
    assert not (T/'control/failure.json').exists()
    assert read(T/'control/completion.json')['formal_calls']==720
    out=O/'delivery';out.mkdir()
    rows=[];bindings=[];resources=[];compute={}
    for method in METHODS:
        phases={}
        roles=('tx','rx') if method in ('P100000','Swin200000') else ('tx','phy','rx')
        for role in roles:
            d=T/method/role
            assert not (d/'failure.json').exists()
            wait=read(T/'control'/(method+'_'+role)/'actual_wait.json')
            assert type(wait['returncode']) is int and wait['returncode']==0 and not wait['timed_out']
            old=wait['child'];now=process(old['pid'])
            assert now is None or now['start_ticks']!=old['start_ticks']
            c=read(d/'completion.json')
            assert c['calls']==288 and c['formal']==144 and not c['blocked_reads']
            phases[role]={x['key']:x for x in c['results']}
            bindings.append(pin(d/'completion.json'))
            resources.append(dict(method=method,role=role,setup=read(d/'setup.json'),
                before=read(d/'resource_before.json'),after=c['resource_after']))
        for case in r['grid']:
            key=case['key'];tx=phases['tx'][key];rx=phases['rx'][key]
            phy=phases.get('phy',{}).get(key,dict(seconds={}))
            txs=tx['seconds']['source_tx']+phy['seconds'].get('phy_tx',0.)
            rxs=rx['seconds']['native_rx']+phy['seconds'].get('phy_rx',0.)
            assert math.isfinite(txs) and math.isfinite(rxs) and txs>=0 and rxs>=0
            rows.append(dict(method=method,display_name=NAMES[method],**case,
                source_id=tx['source_id'],tx_ms=txs*1000,rx_ms=rxs*1000,
                total_ms=(txs+rxs)*1000,status=rx['evidence']['status'],
                output_sha256=rx['evidence']['image_sha256']))
        compute[method]=dict(full_calls=288,formal_calls=144,warmups=144,
            source_calls=sum(v['evidence'].get('candidate_encodes',1) for v in phases['tx'].values()),
            source_candidate_decodes=sum(v['evidence'].get('candidate_decodes',0) for v in phases['tx'].values()),
            receiver_statuses=dict(Counter(v['evidence']['status'] for v in phases['rx'].values())),
            unique_output_hashes=len({v['evidence']['image_sha256'] for v in phases['rx'].values()}),
            cached_scientific_reuse=0)
    summary=[]
    for method in METHODS:
        for snr in SNRS:
            selected=[x for x in rows if x['method']==method and x['snr_db']==snr and x['phase']=='measure']
            assert len(selected)==48 and len({x['source_index'] for x in selected})==16
            rec=dict(method=method,display_name=NAMES[method],snr_db=snr,sources=16,measurements=48)
            for field in ['tx_ms','rx_ms','total_ms']:
                values=[x[field] for x in selected]
                rec[field+'_mean']=statistics.fmean(values)
                rec[field+'_median']=statistics.median(values)
                rec[field+'_p95']=percentile(values,.95)
            summary.append(rec)
    csvsave(out/'per_call.csv',rows)
    csvsave(out/'summary_ms_per_image.csv',summary)
    save(out/'compute_and_resources.json',dict(compute=compute,resources=resources,
        protocol=r['timing'],registration=pin(O/'registration.json'),completed_inputs=bindings,
        quality_evaluations=0,bootstrap=0,training=0))
    text=['# Current five-method fixed16 N1024 timing','',
          'Each method/SNR uses the same 16 development sources, batch size 1, '
          'three fresh warmups and three measurements per source. All 48 measurements '
          'per method/SNR are retained, including rejected/gray outcomes.','',
          'TX and RX are independently measured in separate processes on the same '
          'registered CPU/GPU pair. Total software processing is their sum; model '
          'loading, AWGN synthesis, cross-process handoff and audit/result persistence '
          'are outside this total. BPG subprocess and necessary codec file conversion '
          'costs are included. This is not radio airtime or an end-to-end RPC timer.','',
          'Original per-method Python/framework stacks are retained and disclosed '
          'in compute_and_resources.json. These values are not merged with the '
          'historical H800 six-method experiment, which predates m10 and excludes '
          'BPG and ARPC. Shared load is sampled, not claimed absent.','',
          '| Method | SNR (dB) | TX (ms/image) | RX (ms/image) | Total (ms/image) |',
          '|---|---:|---:|---:|---:|']
    for v in summary:
        text.append(f"| {v['display_name']} | {v['snr_db']} | {v['tx_ms_mean']:.2f} | {v['rx_ms_mean']:.2f} | {v['total_ms_mean']:.2f} |")
    text+=['','The table shows arithmetic means. Unrounded per-call values, medians '
           'and p95 values are supplied as CSV. No quality metrics or bootstrap '
           'were recomputed. Output hash matches do not reduce the actual measured-call count.','']
    (out/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    save(out/'manifest.json',dict(files=[pin(p) for p in out.iterdir() if p.is_file()]))
    save(out/'completion.json',dict(status='COMPLETE',formal_calls=720,full_calls=1440,
        registration_sha256=expected,manifest=pin(out/'manifest.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True)
    main(ap.parse_args().registration)
