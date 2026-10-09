"""Read completed adaptive-BPG metadata only; no codec, PHY, model or bootstrap."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SNRS = [1, 4, 7, 10, 13, 19]
SEEDS = [2001, 2002, 2003]
STATES = ['SOURCE_UNFIT', 'HEADER_REJECT', 'BODY_CRC_REJECT', 'BODY_PARSER_REJECT',
          'BPG_DECODER_REJECT', 'BPG_SOURCE_FORMAT_REJECT', 'BPG_DECODED']
HEADER_SHA = '1f1fad07942e30cbc844e0cc3c2d7da7c747b568639d27115dc33eaec49cc65b'
SCALE_SHA = '35cae80a94d3997584342b6e4566a6cff53cf4914ca4745a59b0a9bba48c8145'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    data = Path(path).read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Changed bound input: '+str(path))
    return json.loads(data)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows):
    require(bool(rows), 'Empty export: '+str(path))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def summary(values, prefix):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    require(not len(a) or np.isfinite(a).all(), 'Nonfinite resource measurement')
    result = {prefix+'_count':len(a)}
    result.update({prefix+'_'+key:float(fn(a)) if len(a) else None for key,fn in [
        ('mean', np.mean), ('min', np.min), ('p05', lambda x:np.quantile(x,.05)),
        ('p50', lambda x:np.quantile(x,.5)), ('p95', lambda x:np.quantile(x,.95)), ('max', np.max)]})
    return result


def frame_resources(row, profile, selected):
    """Metadata normalization also used for checking real calibration fixtures."""
    q, k, n = [profile[x] for x in ['q','k','n']]
    require(row['profile_id']==profile['profile_id'] and all(row[x]==profile[x] for x in ['q','k','n','capacity_bytes']),
            'Receiver row/frozen transmitter profile differs')
    require(q in [2,4,6] and n==956*q and profile['N']==1024 and profile['header_symbols']==68
            and profile['body_symbols']==956 and profile['capacity_bytes']==(k-29)//8,
            'Unexpected complete-file budget')
    require([row[x] for x in ['allocated_symbols','header_symbols','body_symbols','padding_symbols']]==[1024,68,956,0],
            'Unexpected recorded frame allocation')
    fit = selected is not None
    require(row['source_encoding_fit']==fit and row['status'] in STATES
            and row['gray_substitution']==(row['status']!='BPG_DECODED'), 'Source/state semantics differ')
    h, b = row['header'], row['body']
    result = {key:row[key] for key in ['frame_id','source_index','source_id','snr_db','noise_seed','profile_id',
        'source_encoding_fit','selected_resolution','selected_qp','source_stream_bytes','status','gray_substitution',
        'packet_decoder_calls','psnr_db','undetected_payload_difference','image_sha256','reference_sha256']}
    result.update(q=q, modulation={2:'QPSK',4:'16QAM',6:'64QAM'}[q], nominal_rate_label=profile['rate'],
        ldpc_information_bits_k=k, ldpc_transmitted_bits_n=n, actual_ldpc_rate=k/n,
        actual_ldpc_rate_fraction=f'{k}/{n}', capacity_bytes=profile['capacity_bytes'],
        N_allocated=1024, header_allocated_uses=68, body_allocated_uses=956, padding_allocated_uses=0,
        actual_transmitted_uses=1024 if fit else 0, actual_header_uses=68 if fit else 0,
        actual_body_uses=956 if fit else 0, actual_padding_uses=0,
        header_information_bits=12, header_crc_bits=16,
        header_tail_bits=6, header_transmitted_bits=136 if fit else None,
        actual_body_transmitted_bits=n if fit else None, body_length_field_bits=13, body_crc_bits=16,
        body_source_bits=8*row['source_stream_bytes'] if fit else None,
        body_known_zero_information_padding_bits=k-29-8*row['source_stream_bytes'] if fit else None,
        source_information_bits_per_allocated_use=8*row['source_stream_bytes']/1024 if fit else None,
        frame_effective_use_fraction_when_transmitted=1.0 if fit else None,
        actual_frame_energy=row['actual_frame_energy'],
        header_energy_derived=136.0 if fit else None,
        body_energy_derived=row['actual_frame_energy']-136.0 if fit else None,
        padding_energy_derived=0.0 if fit else None,
        energy_scope='SAVED_FRAME_TOTAL_MINUS_EXACT_PINNED_CONSTANT_MODULUS_HEADER' if fit else 'NO_TRANSMISSION',
        header_rejected=False, header_wrong_accept=False, body_crc_rejected=False,
        body_crc_accepted_parser_rejected=False, parser_accepted_payload_different=False,
        body_crc_accepted_detected_wrong_content=False, image_decoder_or_format_rejected=False,
        received_profile_id=None, received_BPG_sha256=None)
    if not fit:
        require(row['status']=='SOURCE_UNFIT' and h is None and b is None and row['packet_decoder_calls']==0
                and row['actual_frame_energy'] is None and row['undetected_payload_difference'] is None
                and all(row[x] is None for x in ['selected_resolution','selected_qp','source_stream_bytes']),
                'SOURCE_UNFIT must have no transmitted/received frame')
        return result
    require(selected['resolution'] in [256,128,64,32] and 0<=selected['qp']<=51
            and 0<selected['complete_BPG_bytes']<=profile['capacity_bytes'] and selected['independent_decode'],
            'Invalid frozen source selection')
    require((row['selected_resolution'],row['selected_qp'],row['source_stream_bytes'])==
            (selected['resolution'],selected['qp'],selected['complete_BPG_bytes']), 'Source selection changed across stages')
    require(math.isfinite(row['actual_frame_energy']) and row['actual_frame_energy']>136 and h is not None,
            'Missing actual transmitted energy/header record')
    result['header_rejected']=not h['header_ok']
    result['header_wrong_accept']=bool(h['header_ok'] and h['profile_id']!=profile['profile_id'])
    result['received_profile_id']=h['profile_id']
    if not h['header_ok']:
        require(row['status']=='HEADER_REJECT' and b is None and row['packet_decoder_calls']==1,
                'Header rejection state differs')
        return result
    require(b is not None and row['packet_decoder_calls']==2 and b['received_profile_id']==h['profile_id'],
            'Actual received-profile/body record differs')
    result['body_crc_rejected']=not b['crc_accepted']
    result['body_crc_accepted_parser_rejected']=bool(b['crc_accepted'] and not b['parser_accepted'])
    if not b['parser_accepted']:
        require(row['status']==('BODY_CRC_REJECT' if not b['crc_accepted'] else 'BODY_PARSER_REJECT')
                and row['undetected_payload_difference'] is None, 'Rejected body status differs')
    else:
        require(b['crc_accepted'] and row['status'] in ['BPG_DECODED','BPG_DECODER_REJECT','BPG_SOURCE_FORMAT_REJECT']
                and type(row['undetected_payload_difference']) is bool, 'Parsed stream outcome differs')
        require((b['received_BPG_sha256']!=selected['stream_sha256'])==row['undetected_payload_difference'],
                'Offline actual-byte mismatch diagnostic differs')
        result['received_BPG_sha256']=b['received_BPG_sha256']
        result['parser_accepted_payload_different']=row['undetected_payload_difference']
    result['body_crc_accepted_detected_wrong_content']=bool(result['body_crc_accepted_parser_rejected']
        or result['parser_accepted_payload_different'])
    result['image_decoder_or_format_rejected']=row['status'] in ['BPG_DECODER_REJECT','BPG_SOURCE_FORMAT_REJECT']
    return result


def run(args):
    request=Path(args.request).resolve(); cfg=read(request); request_sha=sha(request)
    require(cfg['schema']=='A1B_ADAPTIVE_BPG_REQUEST_V1' and cfg['SNRs']==SNRS and cfg['noise_seeds']==SEEDS
            and len(cfg['records']['calibration'])==100 and len(cfg['records']['holdout'])==500
            and not cfg['holdout_used_for_selection'] and not cfg['old_main_results_modified'], 'Wrong frozen adaptive scope')
    base=Path(cfg['out']); out=Path(args.output) if args.output else base/'resources_v1'
    require(out.resolve()!=base.resolve() and (not out.exists() or not any(out.iterdir())), 'Use a fresh output directory')
    controls={str(request):request_sha}
    def stage(name):
        path=base/name/'completion.json'; d=read(path); controls[str(path)]=sha(path)
        require(d['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1' and d['stage']==name and d['request_sha256']==request_sha
                and d['actual_children_waited'] and all(x==0 for x in d['worker_exit_codes']), 'Stage not actually complete: '+name)
        return d
    done=stage('holdout'); freeze_done=stage('freeze'); qualification_done=stage('qualification')
    require(done['frame_count']==9000 and done['root_budget_before']==done['root_budget_after'], 'Incomplete final holdout')
    freeze_path=base/'freeze/worker_0.json'; freeze=read(freeze_path,freeze_done['outputs'][str(freeze_path)])
    qual_path=base/'qualification/worker_0.json'; qual=read(qual_path,qualification_done['outputs'][str(qual_path)])
    controls.update({str(freeze_path):sha(freeze_path),str(qual_path):sha(qual_path)})
    require(freeze['request_sha256']==request_sha and freeze['calibration_source_count']==100
            and freeze['noise_count']==3 and not freeze['holdout_used_for_selection'], 'Wrong calibration/freeze scope')
    # Read source files as bytes only. Fixed +/-1 header coordinates are exactly
    # preserved by the original float64-to-float32 conversion, so E_header=136.
    for suffix,expected in [('uep_phy.py',HEADER_SHA),('scale_channel.py',SCALE_SHA)]:
        paths=[p for p,h in cfg['phy_dependency_bindings'].items() if p.endswith('/'+suffix)]
        require(len(paths)==1 and cfg['phy_dependency_bindings'][paths[0]]==expected and sha(paths[0])==expected,
                'The proven constant-modulus header implementation changed')
        controls[paths[0]]=expected
    profiles={s:next(p for p in cfg['catalogue'] if p['profile_id']==freeze['selected_profile_ids'][str(s)]) for s in SNRS}
    normalized, sources, samples=[],[],[]
    for i,record in enumerate(cfg['records']['holdout']):
        cp_path=base/'holdout/sources'/f'{i:04d}.json'; cp=read(cp_path,done['outputs'][str(cp_path)])
        require(cp['status']=='ADAPTIVE_BPG_SOURCE_RX_COMPLETE_V1' and cp['request_sha256']==request_sha
                and cp['source_index']==i and cp['source_id']==record['source_id'] and cp['preprocessing_id']==record['preprocessing_id']
                and cp['method']=='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024' and cp['runtime_identity']==qual['backend_identity']
                and len(cp['rows'])==18, 'Wrong complete holdout source')
        coded_path=base/'codec/holdout'/f'{i:04d}'/'completion.json'; coded=read(coded_path,cp['outputs'][str(coded_path)])
        require(coded['status']=='ADAPTIVE_BPG_SOURCE_CODEC_COMPLETE_V1' and coded['request_sha256']==request_sha
                and coded['source_id']==record['source_id'] and coded['source_index']==i, 'Wrong source codec identity')
        seen=set()
        for row in cp['rows']:
            s,seed=row['snr_db'],row['noise_seed']; require((s,seed) not in seen, 'Duplicate received row');seen.add((s,seed))
            require(s in SNRS and seed in SEEDS and row['source_id']==record['source_id'] and row['source_index']==i
                    and not row['quality_selection'] and row['preprocessing_id']==record['preprocessing_id'], 'Mixed source/population row')
            p=profiles[s]; selected=coded['fits'][str(p['profile_id'])]['selected']
            normalized.append(frame_resources(row,p,selected))
        require(seen=={(s,n) for s in SNRS for n in SEEDS}, 'Incomplete frozen grid')
        for s in SNRS:
            p=profiles[s]; chosen=coded['fits'][str(p['profile_id'])]['selected']; fit=chosen is not None
            sources.append(dict(source_index=i,source_id=record['source_id'],snr_db=s,profile_id=p['profile_id'],
                status='FIT' if fit else 'SOURCE_UNFIT',resolution=chosen['resolution'] if fit else None,
                qp=chosen['qp'] if fit else None,complete_BPG_bytes=chosen['complete_BPG_bytes'] if fit else None,
                complete_BPG_sha256=chosen['stream_sha256'] if fit else None,capacity_bytes=p['capacity_bytes'],
                source_bits=chosen['complete_BPG_bytes']*8 if fit else None,
                original_stream=chosen['stream'] if fit else None))
            if i==0:
                sample=dict(sources[-1],sample_role='source_index_0_at_all_SNRs_complete_TX_BPG_before_FEC',copied_file=None)
                if fit:
                    data=Path(chosen['stream']).read_bytes()
                    require(len(data)==chosen['complete_BPG_bytes'] and hashlib.sha256(data).hexdigest()==chosen['stream_sha256'],
                            'Fixed actual stream sample differs')
                    sample['_bytes']=data
                samples.append(sample)
    require(len(normalized)==9000 and len(sources)==3000, 'Missing final population')
    resource, failures, distributions=[],[],[]
    diagnostic_events=['header_rejected','header_wrong_accept','body_crc_rejected','body_crc_accepted_parser_rejected',
        'parser_accepted_payload_different','body_crc_accepted_detected_wrong_content','image_decoder_or_format_rejected','gray_substitution']
    for s in SNRS:
        frames=[r for r in normalized if r['snr_db']==s]; sr=[r for r in sources if r['snr_db']==s]
        fit=[r for r in sr if r['status']=='FIT']; tx=[r for r in frames if r['source_encoding_fit']]
        p=profiles[s]; layout=qual['layouts'][str(p['profile_id'])]
        require(layout['k']==p['k'] and layout['n']==p['n'] and abs(layout['actual_effective_rate']-p['k']/p['n'])<1e-12,
                'Actual qualified LDPC layout differs')
        r=dict(snr_db=s,source_count=500,noise_count=3,frame_count=1500,source_fit_count=len(fit),source_unfit_count=500-len(fit),
            actual_transmitted_frame_count=len(tx),profile_id=p['profile_id'],q=p['q'],modulation={2:'QPSK',4:'16QAM',6:'64QAM'}[p['q']],
            nominal_rate_label=p['rate'],k=p['k'],n=p['n'],actual_rate_fraction=f"{p['k']}/{p['n']}",actual_rate=p['k']/p['n'],
            internal_k_with_filler=layout['k_ldpc'],filler_bits=layout['k_filler'],puncturing_bits=layout['puncturing_bits'],
            repetition_bits=layout['repetition_bits'],layout_id=layout['layout_id'],capacity_bytes=p['capacity_bytes'],
            header_payload_bits=12,header_crc_bits=16,header_tail_bits=6,header_transmitted_bits=136,
            body_length_field_bits=13,body_crc_bits=16,N_allocated=1024,header_allocated_uses=68,body_allocated_uses=956,
            padding_allocated_uses=0,allocation_mode='full_budget_fixed_container',frame_effective_use_fraction_when_transmitted=1.0,
            source_information_scope='Complete BPG bytes including mandatory format header; not correctly delivered information',
            actual_energy_scope='Actual transmitted frames only; SOURCE_UNFIT is no transmission, not a synthetic zero-energy waveform')
        for key in ['complete_BPG_bytes','source_bits']:
            r.update(summary([x[key] for x in fit],key+'_fit_sources'))
        r.update(summary([x['source_bits']/1024 for x in fit],'source_bits_per_allocated_use_fit_sources'))
        for key in ['actual_frame_energy','header_energy_derived','body_energy_derived','padding_energy_derived']:
            r.update(summary([x[key] for x in tx],key))
        r['gray_frame_count']=sum(x['gray_substitution'] for x in frames); resource.append(r)
        for state in STATES:
            selected=[x for x in frames if x['status']==state]
            failures.append(dict(snr_db=s,event=state,mutually_exclusive_final_status=True,frame_count=len(selected),
                fraction_of_all_frames=len(selected)/1500,represented_source_count=len({x['source_id'] for x in selected}),
                denominator_frames=1500,denominator_sources=500,gray_frame_count=sum(x['gray_substitution'] for x in selected)))
        for event in diagnostic_events:
            selected=[x for x in frames if x[event]]
            failures.append(dict(snr_db=s,event=event,mutually_exclusive_final_status=False,frame_count=len(selected),
                fraction_of_all_frames=len(selected)/1500,represented_source_count=len({x['source_id'] for x in selected}),
                denominator_frames=1500,denominator_sources=500,gray_frame_count=sum(x['gray_substitution'] for x in selected)))
        for dimension,values in [('resolution',[32,64,128,256]),('qp',list(range(52)))]:
            count=Counter(x[dimension] for x in fit)
            for value in values:
                distributions.append(dict(snr_db=s,dimension=dimension,value=value,source_count=count[value],
                    fraction_of_all_sources=count[value]/500,fit_source_count=len(fit),denominator_sources=500))
        count=Counter((x['resolution'],x['qp']) for x in fit)
        for (resolution,qp),num in sorted(count.items()):
            distributions.append(dict(snr_db=s,dimension='resolution_and_qp',value=f'{resolution}:{qp}',source_count=num,
                fraction_of_all_sources=num/500,fit_source_count=len(fit),denominator_sources=500))
        for label,count in [('FIT',len(fit)),('SOURCE_UNFIT',500-len(fit))]:
            distributions.append(dict(snr_db=s,dimension='source_fit',value=label,source_count=count,
                fraction_of_all_sources=count/500,fit_source_count=len(fit),denominator_sources=500))
    out.mkdir(parents=True,exist_ok=True); (out/'streams').mkdir()
    for sample in samples:
        if '_bytes' in sample:
            path=out/'streams'/f"source0000_snr{sample['snr_db']:02d}.bpg"
            path.write_bytes(sample.pop('_bytes')); sample['copied_file']=str(path)
    for filename,data in [('frame_resources.csv',normalized),('source_configurations.csv',sources),('resource_summary.csv',resource),
        ('failure_counts.csv',failures),('source_setting_distribution.csv',distributions),('stream_samples.csv',samples)]:
        csv_write(out/filename,data)
    (out/'README.md').write_text('''# Adaptive BPG holdout resources and failures

This is the later-added adaptive-downsampling BPG + LDPC baseline, distinct from
the original native256 BPG experiment. The frozen source rule chooses among
256/128/64/32-pixel BPG inputs and QP candidates using source-side MSE and actual
complete-file size; restored images use fixed Pillow bicubic to 256. MCS was
selected on exactly 100 original calibration sources with three noises, not
1000 sources and not the holdout. All 500 holdout sources x six SNRs x three
noises (9000 rows) are retained, including every failed source or link outcome.

This export reads JSON and fixed existing BPG bytes only. It imports no source
codec, model or PHY implementation, reads no reconstruction NPZ, draws no noise,
reruns no decoder, and performs no bootstrap. Quality and paired confidence
intervals are delivered by the separate metrics stage. Original results and
selection files are never changed.

resource_summary.csv records each frozen profile, qualified actual LDPC k/n,
nominal label, filler/rate-matching facts, complete-file byte distribution and
frame energy. B_src is complete BPG file bytes x8, including its format header.
The container also pays 13 length bits, known zero information padding, and a
16-bit body CRC. k includes all of these, rather than being pure source bits.
The header pays 12 profile bits, 16 CRC bits and 6 tail bits, encoded into136 bits.
All transmitted frames allocate68 header +956 body +0 physical-padding complex
uses. Information-container zero bits are encoded inside the body; they are not
extra unused physical symbols. Frame-use fraction is1 for actually transmitted
frames. SOURCE_UNFIT has no waveform/PHY call; energy and codeword quantities
specific to its actual transmission are NA and actual_transmitted_uses is0.

The saved actual_frame_energy is the measured transmitter waveform sum. Header
energy136 is an exact derivation from the hash-pinned Header/scale_channel
mapping:136 real coordinates are +/-1, exactly preserved in float32. Derived
body energy is the saved total minus136; physical padding energy is0. No signal
is regenerated to obtain these components. Energy distributions exclude absent
SOURCE_UNFIT frames and explicitly report their counts. These are waveform
energy units, not hardware joules or a claim of strict per-frame energy equality.
The selected profile k/n and allocated budget remain listed even for SOURCE_UNFIT;
the separate actual transmission fields show zero uses and no transmitted bits.

source_configurations.csv has3000 source/SNR entries, avoiding triple-counting
the same source-side choice across noise seeds. source_setting_distribution.csv
reports QP and resolution distributions over500 sources at each SNR; its separate
dimensions are alternative descriptions and must not be summed together.

failure_counts.csv distinguishes mutually exclusive final statuses from
overlapping diagnostics. Header rejection, body CRC rejection, parser rejection,
BPG decoder rejection and unsupported decoded format retain their original
statuses. header_wrong_accept compares actual accepted profile with the frozen
transmitter profile offline. parser_accepted_payload_different compares actual
received bytes with the selected TX stream offline. A CRC-accepted parser failure
or byte mismatch is reported as detected wrong content; a parser-rejected body
has no saved received BPG file to pretend was decoded. These diagnostics never
rescue, replace or change a receiver output. A wrongly accepted stream which
decodes remains a non-gray decoded image, with its wrong-content flag retained.
Zero-count states remain explicit; they do not establish zero population risk.

streams/ contains the original complete TX BPG files for source index0 at each
of the six SNRs, fixed by index without looking at quality or noise outcomes.
stream_samples.csv gives source identity, resolution, QP, original path and hash.
If source0 is unfit, the manifest retains that fact without selecting another
source. No best noise sample is chosen: these are pre-FEC source code streams.

Reproduce after actual holdout completion:

    python experiments/paper_supplement_20261008/a1_bpg_adaptive/export_resources.py --request <frozen-a1b-request.json>

The default output is resources_v1 under the frozen experiment directory; an
explicit --output must be a new empty directory. Existing publication filters
are unchanged; binary BPG examples remain local when the filter excludes them.
''',encoding='utf-8')
    require(all(sha(p)==h for p,h in controls.items()), 'Control input changed during export')
    outputs={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.log' and not p.name.startswith(('launch','console'))}
    write(out/'completion.json',dict(status='ADAPTIVE_BPG_READ_ONLY_RESOURCE_EXPORT_COMPLETE_V1',source_count=500,noise_count=3,
        frame_count=9000,source_configuration_count=3000,calibration_source_count=100,controls=controls,outputs=outputs,
        script_sha256=sha(__file__),new_PHY_decodes=0,new_codec_calls=0,new_channel_draws=0,new_bootstrap_replicates=0,
        original_science_modified=False,stream_sample_source_index=0,stream_sample_count=sum(x['copied_file'] is not None for x in samples)))
    print(json.dumps(dict(status='COMPLETE',output=str(out),sources=500,frames=9000)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--output')
    run(parser.parse_args())
