"""Read-only T4 resource/timing assembly; no model, PHY, noise or bootstrap."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np

METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
MISSING=(None,'','NA','N/A','None','null')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def num(v):
    if v in MISSING:return None
    if v in ('True','true'):return 1.
    if v in ('False','false'):return 0.
    value=float(v);require(math.isfinite(value),'Nonfinite value must not silently disappear');return value
def boolean(v):
    if v in MISSING:return None
    require(v in (True,False,'True','False','true','false',0,1),'Expected observed boolean')
    return v in (True,'True','true',1)
def first(r,*keys):
    for k in keys:
        if r.get(k) not in MISSING:return r[k]
    return None
def stats(values):
    a=np.asarray([float(v) for v in values],dtype=np.float64)
    require(np.isfinite(a).all(),'Actual observations must be finite')
    if not len(a):return dict(count=0,mean=None,std=None,min=None,p05=None,p50=None,p95=None,max=None)
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max()),
        **{f'p{p:02d}':float(np.percentile(a,p,method='linear')) for p in (5,50,95)})
def bound_file(completion,path):
    """Accept relocated published files only by unique original basename + SHA."""
    name=Path(path).name
    matches=[(name,completion['outputs'][name])] if name in completion['outputs'] else [(p,h) for p,h in completion['outputs'].items() if Path(p).name==name]
    require(len(matches)==1 and sha(path)==matches[0][1],'File does not match completed output: '+str(path));return matches[0]
def coverage(rows,expected_sources=500,expected_seeds=(2001,2002,2003)):
    keys=set();by_group=defaultdict(list);ids={}
    for r in rows:
        k=(r['method'],int(r['snr_db']),int(r['source_index']),int(r['noise_seed']))
        require(k not in keys,'Duplicate method/source/SNR/seed');keys.add(k);by_group[k[:2]].append(r)
        i=k[2];require(i not in ids or ids[i]==r['source_id'],'Source-index identity differs across methods');ids[i]=r['source_id']
    require(set(ids)==set(range(expected_sources)) and len(set(ids.values()))==expected_sources,'Complete exact source population required')
    for (method,snr),group in by_group.items():
        require({(int(r['source_index']),int(r['noise_seed'])) for r in group}==
            {(i,s) for i in range(expected_sources) for s in expected_seeds},'Every source/noise retained: '+method+':'+str(snr))
    return by_group,ids
def energy_summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],int(r['snr_db'])].append(r)
    result=[]
    for (method,snr),group in sorted(groups.items()):
        energies=[];rhos=[];missing=0;unfit=0;N=set()
        for row in group:
            e=num(first(row,'E_frame','actual_frame_energy'));n=num(first(row,'N','total_symbols','N_allocated','allocated_symbols'))
            if n is None:
                parts=[num(first(row,k)) for k in ('header_symbols','body_symbols','frame_padding_symbols')]
                if all(p is not None for p in parts):n=sum(parts)
            if row.get('failure_state')=='SOURCE_UNFIT':
                unfit+=1;require(e is None,'SOURCE_UNFIT has no waveform; do not record ideal/zero energy')
            if n is not None:require(n>0,'Positive actual allocated N');N.add(n)
            if e is None or n is None:missing+=1;continue
            rho=e/(2*n);observed=num(row.get('rho'))
            require(observed is None or math.isclose(observed,rho,rel_tol=1e-12,abs_tol=1e-12),'Recorded rho differs from E/(2N)')
            energies.append(e);rhos.append(rho)
        es,rs=stats(energies),stats(rhos)
        result.append(dict(method=method,snr_db=snr,N=next(iter(N)) if len(N)==1 else None,source_count=len({r['source_id'] for r in group}),
            noise_count=len({int(r['noise_seed']) for r in group}),frame_count=len(group),actual_energy_frame_count=len(energies),
            missing_energy_frame_count=missing,source_unfit_frame_count=unfit,std_ddof=0,quantile_method='linear',
            **{'E_'+k:v for k,v in es.items() if k!='count'},**{'rho_'+k:v for k,v in rs.items() if k!='count'},
            energy_scope='Actual saved transmitted waveform only; no SOURCE_UNFIT waveform; no ideal substitution'))
    return result
def state_quality(rows,states=None):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],int(r['snr_db'])].append(r)
    results=[];failures=[]
    for (method,snr),group in sorted(groups.items()):
        names=list(states) if states is not None else sorted({r['failure_state'] for r in group})
        for name in ['ALL_FRAMES']+names:
            selected=group if name=='ALL_FRAMES' else [r for r in group if r['failure_state']==name]
            failures.append(dict(method=method,snr_db=snr,failure_state=name,frames=len(selected),source_count=len({r['source_id'] for r in selected}),
                denominator_frames=len(group),mutually_exclusive=name!='ALL_FRAMES'))
            for metric in METRICS:
                source=defaultdict(list);values=[];missing=0
                for row in selected:
                    value=num(row.get(metric))
                    if value is None:missing+=1;continue
                    if metric=='convnext_top1_source_prediction':require(value in (0.,1.),'Agreement is an event, not accuracy/percentage')
                    values.append(value);source[row['source_id']].append(value)
                means=[math.fsum(v)/len(v) for v in source.values()]
                results.append(dict(method=method,snr_db=snr,failure_state=name,metric=metric,source_count=len(source),frame_count=len(selected),
                    metric_frame_count=len(values),missing_metric_frame_count=missing,frame_weighted_mean=math.fsum(values)/len(values) if values else None,
                    source_balanced_conditional_mean=math.fsum(means)/len(means) if means else None,
                    min_selected_noises_per_source=min(map(len,source.values())) if source else 0,
                    max_selected_noises_per_source=max(map(len,source.values())) if source else 0,
                    status='EXISTING_FRAME_ARITHMETIC_ONLY' if values else 'EMPTY_CONDITION' if not selected else 'MISSING_METRICS',
                    ci_low=None,ci_high=None,interval_status='NOT_RESAMPLED',metric_unit='proportion' if metric=='convnext_top1_source_prediction' else 'native'))
    return results,failures
def entropy_rows(path,completion):
    require(completion['status']=='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE' and completion['source_count']==500
        and completion['holdout_used_for_selection'] is False,'Complete scored entropy holdout required')
    bound_file(completion,path)
    from t1_readonly_frame_join import joined_rows
    raw=joined_rows(completion,path);rows=[];missing=[]
    require(len(raw)==completion['frame_count'],'Actual entropy frame count differs')
    for row in raw:
        method=row['family'];require(all(num(row.get(m)) is not None for m in METRICS),'All actual four metrics required')
        # Preserve source parsing failures separately from physical acceptance.
        physical=first(row,'physical_status','receiver_status','status');source=first(row,'source_status')
        if row.get('failure_state'):state=row['failure_state']
        elif boolean(row.get('header_ok'))is False:state='HEADER_REJECT'
        elif boolean(row.get('body_crc_accept'))is False:state='BODY_CRC_REJECT'
        elif boolean(row.get('body_parser_accept'))is False:state='BODY_PARSER_REJECT'
        elif boolean(row.get('source_canonical_accept'))is False:state='ARITHMETIC_SOURCE_REJECT'
        elif boolean(row.get('source_canonical_accept'))is True:state='SOURCE_DECODED'
        else:state=str(source)if source else str(physical)if physical else'MISSING_FAILURE_STATE'
        n=first(row,'N','total_symbols','N_allocated','allocated_symbols')
        if n is None:
            parts=[num(row.get(k)) for k in ('header_symbols','body_symbols','frame_padding_symbols')]
            if all(p is not None for p in parts):n=int(sum(parts))
        normalized=dict(row,method=method,failure_state=state,N=n,E_frame=first(row,'E_frame','actual_frame_energy'),population='holdout')
        rows.append(normalized)
        for name,value in (('failure_state',None if state=='MISSING_FAILURE_STATE' else state),('N',n),('E_frame',normalized['E_frame'])):
            if value in MISSING:missing.append(dict(method=method,snr_db=row['snr_db'],source_index=row['source_index'],noise_seed=row['noise_seed'],field=name))
    coverage(rows,expected_seeds=(6201,6202,6203));return rows,missing
def resource_summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],int(r['snr_db'])].append(r)
    result=[]
    aliases={'target_m':('target_m',),'actual_m':('actual_m',),'K':('K',),'source_token_count':('transmitted_token_count','source_token_count'),
        'q':('q',),'actual_k':('actual_k','k','ldpc_information_bits_k'),'actual_n':('actual_n','n','ldpc_transmitted_bits_n'),
        'header_symbols':('header_symbols','header_allocated_uses'),'body_symbols':('body_symbols','body_allocated_uses'),
        'frame_padding_symbols':('frame_padding_symbols','padding_allocated_uses'),'source_bits':('arithmetic_bits','body_source_bits'),
        'known_information_padding_bits':('known_information_padding_bits','body_known_zero_information_padding_bits')}
    for (method,snr),group in sorted(groups.items()):
        for field,keys in aliases.items():
            values=[num(first(r,*keys)) for r in group];known=[v for v in values if v is not None]
            result.append(dict(method=method,snr_db=snr,field=field,frame_count=len(group),known_frame_count=len(known),missing_frame_count=len(group)-len(known),
                source_count=len({r['source_id'] for r in group}),noise_count=len({r['noise_seed'] for r in group}),**stats(known)))
    return result
def timing_summary(rows):
    measured=[r for r in rows if r['phase']=='measured'];groups=defaultdict(list)
    for row in measured:groups[row['method'],int(row['snr_db'])].append(row)
    result=[]
    for (method,snr),group in sorted(groups.items()):
        require(len(group)==48 and len({r['source_id'] for r in group})==16 and len({(r['source_id'],int(r['repetition'])) for r in group})==48,
            'Each timing method/SNR requires fixed16 x3measured calls')
        for name,condition in [('ALL',lambda r:True),('GRAY_OUTPUT',lambda r:boolean(r['gray']) is True),('NON_GRAY_OUTPUT',lambda r:boolean(r['gray']) is False),
                               ('BODY_CRC_REJECT',lambda r:boolean(r.get('body_crc_accepted')) is False),('HEADER_REJECT',lambda r:boolean(r.get('header_accepted')) is False)]:
            selected=[r for r in group if condition(r)]
            for metric in ('TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds'):
                values=[num(r.get(metric)) for r in selected];require(all(v is not None and v>=0 for v in values),'Missing actual measured timing')
                s=stats(values);result.append(dict(method=method,snr_db=snr,condition=name,metric=metric,sample_count=len(values),
                    source_count=len({r['source_id'] for r in selected}),mean=s['mean'],median=s['p50'],p95=s['p95'],minimum=s['min'],maximum=s['max'],
                    population='development_fixed16_timing',includes_warmup=False,conditional_groups_overlap=True))
    return result
def package_tables(folder,names):
    folder=Path(folder);completion=read(folder/'completion.json');tables={}
    for name in names:
        path=folder/name;bound_file(completion,path);tables[name]=csvread(path)
    return completion,tables
def assemble(args):
    out=Path(args.out);require(not out.exists(),'New summary directory required; old results are immutable')
    inputs={};tables={name:[] for name in ('resource_summary.csv','energy_summary.csv','failure_breakdown.csv','state_quality.csv')};missing=[]
    for scope,directory in [('ORIGINAL_RAW_P_SWIN',args.raw_export),('LATER_ADAPTIVE_BPG',args.bpg_export)]:
        if not directory:missing.append(dict(scope=scope,field='completed_export',reason='NOT_SUPPLIED'));continue
        folder=Path(directory);_,current=package_tables(folder,tables)
        inputs[str(folder/'completion.json')]=sha(folder/'completion.json')
        for name,rows in current.items():
            inputs[str(folder/name)]=sha(folder/name);tables[name].extend(dict(r,evidence_scope=scope,source_table=str(folder/name)) for r in rows)
    if args.entropy_completion:
        completion=read(args.entropy_completion);require(args.entropy_per_frame,'Exact completed entropy per_frame.csv path required')
        rows,gaps=entropy_rows(args.entropy_per_frame,completion);missing.extend(gaps)
        inputs[str(Path(args.entropy_completion))]=sha(args.entropy_completion);inputs[str(Path(args.entropy_per_frame))]=sha(args.entropy_per_frame)
        quality,failure=state_quality(rows)
        for name,data in [('resource_summary.csv',resource_summary(rows)),('energy_summary.csv',energy_summary(rows)),('failure_breakdown.csv',failure),('state_quality.csv',quality)]:
            tables[name].extend(dict(r,evidence_scope='POSTHOC_ENTROPY_AND_RAW_DROP',source_table=str(args.entropy_per_frame)) for r in data)
    else:missing.append(dict(scope='POSTHOC_ENTROPY_AND_RAW_DROP',field='per_frame.csv',reason='NOT_SUPPLIED_NOT_IMPUTED'))
    timing=[];timing_rows=[];history=[]
    if args.timing_completion:
        done=read(args.timing_completion);require(done['status']=='T4_FIXED16_ONLINE_TIMING_COMPLETE' and done['total_frames']==1152
            and done['measured_frames']==576 and done['packet_budget']['unresolved_frames']==0,'Actual complete exclusive timing required')
        require(args.timing_per_call,'Exact completed timing_per_call.csv required');bound_file(done,args.timing_per_call)
        timing_rows=csvread(args.timing_per_call);require(len(timing_rows)==1152,'Complete timing rows including576warmups required');timing=timing_summary(timing_rows)
        inputs[str(Path(args.timing_completion))]=sha(args.timing_completion);inputs[str(Path(args.timing_per_call))]=sha(args.timing_per_call)
        require(done.get('no_online_tokens_or_output_cache')is True,'Online timing cannot consume token/output caches')
        request=read(done['request_path']);require(sha(done['request_path'])==done['request_sha256'],'Actual timing request changed')
        additions=request.get('registered_native_source_additions',{})
        history.append('The completed timing request explicitly registered '+str(len(additions))+' added native-source bindings. Original model weights, numerical flags and prior source hashes were checked before and after timing; the compatibility admission did not bypass those checks.')
        if args.timing_prior_failure:
            failure=read(args.timing_prior_failure)
            require(failure['complete_frames']==0 and failure['packet_budget']is None,'Prior failed admission must contain zero completed frames and no packet budget')
            inputs[str(Path(args.timing_prior_failure))]=sha(args.timing_prior_failure)
            history.append('The prior timing admission attempt exited before any measured frame: complete_frames=0 and packet_budget=null. Model loading in that failed attempt is excluded from timing; only the supplied completed run contributes samples.')
        if args.timing_admission_diagnosis:
            read(args.timing_admission_diagnosis);inputs[str(Path(args.timing_admission_diagnosis))]=sha(args.timing_admission_diagnosis)
            history.append('The separate native-admission diagnosis is SHA-bound as an input; it is an engineering receipt, not a timing measurement.')
    else:missing.append(dict(scope='EXCLUSIVE_T4_TIMING',field='timing_per_call.csv',reason='NOT_SUPPLIED_NOT_IMPUTED'))
    out.mkdir(parents=True)
    for name,rows in tables.items():csvwrite(out/name,rows,fields=None if rows else ['method','snr_db','availability'])
    csvwrite(out/'timing_summary.csv',timing,fields=None if timing else ['method','snr_db','condition','metric','sample_count','mean','median','p95'])
    if timing_rows:shutil.copyfile(args.timing_per_call,out/'timing_per_call.csv')
    csvwrite(out/'missing_fields.csv',missing,fields=None if missing else ['scope','field','reason'])
    report=("# T4 assembled evidence\n\nThis is a read-only assembly of explicitly supplied completed exports. Original raw/P/Swin and later adaptive BPG evidence remain separately identified. Entropy and timing appear only when their completed per-frame outputs are supplied and SHA-verified. Empty timing tables are unavailable results, not zero latency.\n\n"
        "Energy uses actual saved E and rho=E/(2N), with population standard deviation (ddof=0) and linear empirical quantiles. SOURCE_UNFIT has no transmitted waveform and is NA, with explicit counts. No ideal constellation value replaces missing energy.\n\n"
        "All source/noise rows, failed frames, and gray outputs remain in overall quality. Conditional quality is diagnostic; selected noise counts per source can differ. Original published confidence intervals are retained where supplied. No new confidence interval or bootstrap is generated. Prediction agreement remains a proportion in CSV, not classification accuracy. Timing success/CRC conditions overlap; CRC rejection need not imply a gray output. Timing is development16 x3 and does not establish universal latency or precise tails.\n\n"
        f"Missing input/field records: {len(missing)}. See missing_fields.csv.\n\n"+'\n\n'.join(history)+'\n')
    (out/'REPORT_T4.md').write_text(report,encoding='utf-8')
    result=dict(status='T4_READ_ONLY_ASSEMBLY_COMPLETE_WITH_EXPLICIT_AVAILABILITY',input_bindings=inputs,missing_records=len(missing),
        entropy_supplied=bool(args.entropy_completion),timing_supplied=bool(args.timing_completion),new_PHY=0,new_model_calls=0,new_noise_draws=0,new_bootstrap=0,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},script_sha256=sha(__file__))
    save(out/'completion.json',result);return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('raw-export','bpg-export','entropy-completion','entropy-per-frame','timing-completion','timing-per-call',
        'timing-prior-failure','timing-admission-diagnosis'):p.add_argument('--'+n)
    p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(assemble(args),indent=2))
if __name__=='__main__':main()
