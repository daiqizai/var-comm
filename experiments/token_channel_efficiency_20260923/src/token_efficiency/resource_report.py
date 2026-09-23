"""Recompute all B1/B2 tables, compact frame exports and figures from sealed cells."""
import csv,json
from pathlib import Path
import numpy as np
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from latent_enhancement_eval.runner import write_rows
from .common import ROOT,OUT,EXP,read,config
from .statistics import FrameTable,canonical_sha,minimum_tested_uses,paired

DEST=ROOT/'results/token_channel_efficiency_20260923/B1_B2_v1'

def read_grid(folder):
    done=read(folder/'completion.json');reg=read(folder/'registration.json')
    if done['synthetic'] is not False or done['registration_sha256']!=digest(folder/'registration.json'):raise RuntimeError('real completed grid required')
    verify_snapshot(reg['context']['bindings']);verify_snapshot({str(folder/n):v for n,v in done['files'].items()})
    rows=[];timing=[];previews=[]
    for name,sha in done['cell_sha256'].items():
        p=folder/'cells'/name
        if digest(p)!=sha:raise RuntimeError('sealed grid cell changed')
        cell=read(p)
        if cell['registration_sha256']!=done['registration_sha256']:raise RuntimeError('cell registration changed')
        verify_snapshot(cell['previews'])
        previews.extend({'method':cell['method'],'source_id':cell['source_id'],'local_reconstruction':p,'sha256':sha,'public_pixels':False} for p,sha in cell['previews'].items())
        rows.extend(cell['rows']);timing.extend(cell['timing'])
    return rows,timing,reg,done,previews

def export_compact(table,folder):
    methods={m:i for i,m in enumerate(table.methods)};sources={s:i for i,s in enumerate(table.ids)}
    write_json(folder/'index.json',{'table_sha256':table.sha256,'population':table.population,'sources':[{'source_id':s,'preprocessing_id':table.sources[s]} for s in table.ids],'methods':[{'method':m,**table.context[m],**{k:table.frames[(m,table.ids[0],table.snrs[0],table.seeds[0])][k] for k in ('N_header','N_data','N_continuous')}} for m in table.methods],'description':'indices resolve exact source and method context; no pixels, tokens, weights or waveforms'})
    fields=['waveform_sha256','observation_sha256','snr_db','noise_seed','mse','psnr_db','lpips_alex','dino_cosine','E','header_ok','body_crc_ok','source_overflow_erasure','source_complete','m_actual','payload_bits','mother_bits','coded_slots','length_field','overflow_lower_m','decoded_mode','decoded_label','source_error','header_crc_ok','rx_source_overflow_erasure']
    batch=[];chunk=0
    for key in sorted(table.frames):
        r=table.frames[key];batch.append({'source_index':sources[r['source_id']],'method_index':methods[r['method']],**{k:r.get(k,'') for k in fields}})
        if len(batch)==20000:write_rows(folder/f'frames_{chunk:03d}.csv',batch);batch=[];chunk+=1
    if batch:write_rows(folder/f'frames_{chunk:03d}.csv',batch)

def main():
    cfg=config();all_rows=[];all_timing=[];all_previews=[];policies=[];lineage={};contexts={};cal_tables=[];sources=None;decoder=None
    for mcs in ('QPSK','16QAM'):
        base=OUT/'digital_grid_v1'/mcs
        for role in ('calibration','development'):
            rows,timing,reg,done,previews=read_grid(base/role);contexts[reg['context_sha256']]=reg['context']
            if decoder is None:decoder=reg['context']['decoder_sha256']
            if reg['context']['decoder_sha256']!=decoder:raise RuntimeError('decoder context mismatch')
            table=FrameTable(rows,role,reg['sources'],reg['snrs'],reg['seeds'],sorted({r['method'] for r in rows}))
            if table.sha256!=done['table_sha256']:raise RuntimeError('source table receipt mismatch')
            if role=='calibration':cal_tables.append((mcs,table));policies.append(read(base/role/'policy.json'))
            else:
                if sources is not None and sources!=reg['sources']:raise RuntimeError('development population mismatch')
                sources=reg['sources'];all_rows.extend(rows);all_timing.extend(timing)
            all_previews.extend(previews)
            lineage[str(base/role/'completion.json')]=digest(base/role/'completion.json')
    folder=OUT/'continuous_grid_v1';rows,timing,reg,done,previews=read_grid(folder);contexts[reg['context_sha256']]=reg['context']
    if sources!=reg['sources'] or decoder!=reg['context']['decoder_sha256']:raise RuntimeError('continuous/digital data or decoder mismatch')
    all_previews.extend(previews);all_rows.extend(rows);all_timing.extend(timing);lineage[str(folder/'completion.json')]=digest(folder/'completion.json')
    table=FrameTable(all_rows,'development',sources,cfg['snrs_db'],cfg['development_seeds'],sorted({r['method'] for r in all_rows}))
    policy={'status':'FROZEN_CALIBRATION_POLICY','calibration_table_sha256':canonical_sha([p['calibration_table_sha256'] for p in policies]),'choices':[c for p in policies for c in p['choices']]}
    minimum=minimum_tested_uses(table,policy,read(EXP/'quality_targets.json'),['P2048','P3060','P4084']);summary=table.summary()
    comparisons=[paired(table,c['method'],f"P{c['N']}",c['snr_db']) for c in policy['choices']]
    DEST.mkdir(parents=True,exist_ok=True);write_json(DEST/'execution_contexts.json',contexts);write_rows(DEST/'summary.csv',summary);write_json(DEST/'minimum_uses.json',minimum);write_rows(DEST/'minimum_uses.csv',[{k:v for k,v in r.items() if k not in ('digital_evidence','continuous_evidence')} for r in minimum]);write_json(DEST/'paired.json',comparisons);write_json(DEST/'policy.json',policy);write_json(DEST/'lineage.json',lineage);write_rows(DEST/'timing.csv',all_timing)
    write_rows(DEST/'local_reconstruction_index.csv',all_previews)
    export_compact(table,DEST/'development')
    for mcs,cal in cal_tables:export_compact(cal,DEST/'calibration'/mcs)
    failures=[]
    for method in table.methods:
        for snr in table.snrs:
            rows=[table.frames[(method,s,snr,seed)] for s in table.ids for seed in table.seeds]
            failures.append({'method':method,'snr_db':snr,'attempts':len(rows),'header_failures':sum(r['header_ok'] is False for r in rows),'body_failures_with_valid_header':sum(r['header_ok'] is True and r['body_crc_ok'] is False for r in rows),'source_overflow_erasures':sum(bool(r['source_overflow_erasure']) for r in rows),'energy_constraint':table.context[method]['energy_constraint'],'E_mean':float(np.mean([r['E'] for r in rows]))})
    write_rows(DEST/'failure_and_energy.csv',failures)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(r['method'],r['snr_db']):r for r in summary};timing_lookup={}
    for r in all_timing:timing_lookup.setdefault((r['method'],r['snr_db']),[]).append(r['total_ms'])
    for metric,label in [('psnr_db','PSNR (dB)'),('lpips_alex','LPIPS'),('dino_cosine','DINO cosine')]:
        for xaxis in ('N','time'):
            fig,axes=plt.subplots(5,2,figsize=(12,18),squeeze=False)
            for si,snr in enumerate(cfg['snrs_db']):
                for mi,mcs in enumerate(('QPSK','16QAM')):
                    ax=axes[si,mi]
                    for family in ('continuous','raw','arithmetic'):
                        methods=[f'P{N}' for N in cfg['budgets']] if family=='continuous' else [next(c['method'] for c in policy['choices'] if c['family']==family and c['N']==N and c['mcs']==mcs and c['snr_db']==snr) for N in cfg['budgets']]
                        x=[table.context[m]['N'] if xaxis=='N' else float(np.mean(timing_lookup[(m,snr)])) for m in methods];y=[lookup[(m,snr)][metric] for m in methods]
                        ax.plot(x,y,'o-',label=family)
                    ax.set_title(f'{snr} dB / {mcs}'+(' (variable frame energy)' if mcs=='16QAM' else ' (E=2N)'));ax.set_xlabel('Complex channel uses' if xaxis=='N' else 'Full online TX+RX (ms)');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
            fig.tight_layout();fig.savefig(DEST/f'quality_vs_{xaxis}_{metric}.png',dpi=140);plt.close(fig)
    met=sum(r['status']=='MEASURED_GRID_TARGET_MET' for r in minimum)
    report=f'''# Token/channel efficiency B1/B2 actual resource results\n\nFull source A is reported separately. This report uses {len(table.ids)} original\ndevelopment sources, five SNRs and three noise seeds. All failures are retained.\nDigital policies were frozen from1000-source calibration before development.\nThe selected P2048/P3060 checkpoints follow calibration-only continuation\ndecisions; P4084 is the explicitly retained historical10k checkpoint. Different\ntraining opportunities are disclosed, not attributed to bandwidth alone.\n\nOf {len(minimum)} registered target/SNR/family/MCS cells, {met} have both\nfamilies meeting the mean quality target in the measured budget grid. Every\nunreached target and negative saving is retained in minimum_uses.csv/json.\nNo interpolation or continuous-optimum claim is made. Per-frame joint attainment\nis separate from mean attainment.\n\nQPSK/continuous use per-frame E=2N.16QAM is shown separately with fixed\nconstellation scaling and actual varying energy; it is not a strict constant-E\ncomparison. Source bits are not channel uses. Prefix fallback, paid control and\nCRC failures remain in the frame ledger.\n\nMeans, paired differences,10000 source-bootstrap intervals and plots derive from\none validated source/SNR/seed table. These intervals do not include training-seed\nvariation. Timing repeats full CPU RGB -> CPU waveform and CPU observation ->\nCPU RGB execution, with one noise application outside RX timing. Offline quality\nmay reuse source encoding; reported timing never does.\n\nResults: results/token_channel_efficiency_20260923/B1_B2_v1. Compact frame indices\nresolve source ID/preprocessing and complete method/run context via index.json.\nNo weights, raw images, tokens or large tensor caches are published.\n\nThis B1/B2 milestone does not complete merged short-prefix C, its N3060\nconfirmation or training-seed repetitions. New holdout and selectors remain deferred.\n'''
    (ROOT/'reports/token_channel_efficiency_B1_B2_20260924.md').write_text(report)
    write_json(DEST/'completion.json',{'status':'REAL_B1_B2_RESOURCE_REPORT_COMPLETE_C_PENDING','synthetic':False,'new_holdout':False,'table_sha256':table.sha256,'frame_rows':len(all_rows),'minimum_target_cells':len(minimum),'files':{str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and p.name!='completion.json'},'report_sha256':digest(ROOT/'reports/token_channel_efficiency_B1_B2_20260924.md')})
if __name__=='__main__':main()
