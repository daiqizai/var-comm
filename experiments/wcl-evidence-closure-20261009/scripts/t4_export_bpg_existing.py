"""SHA-verified BPG resource/quality CSV export; no scientific pipeline imports."""
import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import t4_summarize as s

METHOD='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'
SNRS=(1,4,7,10,13,19)
STATES=('SOURCE_UNFIT','HEADER_REJECT','BODY_CRC_REJECT','BODY_PARSER_REJECT','BPG_DECODER_REJECT','BPG_SOURCE_FORMAT_REJECT','BPG_DECODED')
def run(args):
    resources=Path(args.resources);metrics=Path(args.metrics);out=Path(args.out)
    s.require(not out.exists(),'Fresh BPG export directory required')
    rd=s.read(resources/'completion.json');md=s.read(metrics/'completion.json')
    s.require(rd['status']=='ADAPTIVE_BPG_READ_ONLY_RESOURCE_EXPORT_COMPLETE_V1' and rd['source_count']==500 and rd['noise_count']==3 and rd['frame_count']==9000,'Completed actual resource export required')
    s.require(md['status']=='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1' and md['source_count']==500 and md['frame_count']==9000,'Completed actual four-metric export required')
    inputs={str(p):s.sha(p) for p in (resources/'completion.json',metrics/'completion.json')};checks=[]
    # Only consumed completed CSV/text outputs; no model/image/feature files read.
    resource_files=[resources/Path(p).name for p in rd['outputs'] if Path(p).suffix=='.csv']
    metric_files=[metrics/name for name in ('rows.json','summary.csv','paired.csv','source_means.csv')]
    for p,done in [(p,rd) for p in resource_files]+[(p,md) for p in metric_files]:
        original,expected=s.bound_file(done,p);inputs[str(p)]=s.sha(p);checks.append(dict(path=str(p),original_path=original,expected_sha256=expected,actual_sha256=inputs[str(p)],match=True))
    frames=s.csvread(resources/'frame_resources.csv');metrics_rows=s.read(metrics/'rows.json')
    lookup={r['frame_id']:r for r in metrics_rows};s.require(len(lookup)==len(metrics_rows)==9000,'Unique9000 original metric frames')
    rows=[]
    for row in frames:
        s.require(row['frame_id'] in lookup,'Missing exact paired resource/metric frame');m=lookup.pop(row['frame_id'])
        for k in ('source_id','status','image_sha256','reference_sha256'):s.require(row[k]==m[k],'Resource/metric identity differs: '+k)
        for k in ('source_index','snr_db','noise_seed','profile_id'):s.require(int(row[k])==m[k],'Resource/metric condition differs: '+k)
        for k in ('psnr_db','actual_frame_energy'):s.require(s.num(row[k])==s.num(m[k]),'Original resource/metric numeric observation differs: '+k)
        e=s.num(row['actual_frame_energy']);N=int(row['N_allocated']);s.require(N==1024,'Only fixedN1024')
        if row['status']=='SOURCE_UNFIT':s.require(e is None and int(row['actual_transmitted_uses'])==0,'SOURCE_UNFIT cannot have a fabricated waveform')
        else:s.require(e is not None and int(row['actual_transmitted_uses'])==N,'Actual transmitted resource record required')
        s.require(all(s.num(m[x]) is not None for x in s.METRICS),'Retain all actual four metrics')
        rows.append(dict(row,method=METHOD,population='holdout',failure_state=row['status'],N=N,E_frame=e,rho=None if e is None else e/(2*N),
            **{k:m[k] for k in s.METRICS if k not in row}))
    s.require(not lookup and len(rows)==9000,'No omitted frame');groups,ids=s.coverage(rows)
    s.require(set(snr for method,snr in groups)==set(SNRS) and [ids[i] for i in range(500)]==md['source_ids'],'Exact500 ordered source/SNR identity')
    failures=s.csvread(resources/'failure_counts.csv')
    for row in failures:
        group=groups[METHOD,int(row['snr_db'])]
        selected=[r for r in group if r['status']==row['event']] if s.boolean(row['mutually_exclusive_final_status']) else [r for r in group if s.boolean(r[row['event']])]
        s.require(len(selected)==int(row['frame_count']) and len({r['source_id'] for r in selected})==int(row['represented_source_count']),'Original failure counts do not reconcile')
    quality,breakdown=s.state_quality(rows,STATES);energy=s.energy_summary(rows);original_summary=s.csvread(metrics/'summary.csv')
    for row in quality:
        if row['failure_state']!='ALL_FRAMES':continue
        matched=[r for r in original_summary if r['point_id']==METHOD+'_SNR_'+str(row['snr_db']) and r['metric']==row['metric']]
        s.require(len(matched)==1 and math.isclose(float(matched[0]['mean']),row['source_balanced_conditional_mean'],rel_tol=1e-12,abs_tol=1e-12),'All-source mean fails published mean reconciliation')
        row.update(ci_low=matched[0]['ci_low'],ci_high=matched[0]['ci_high'],status='PUBLISHED_FULL_FRAME_RESULT_REUSED',
            interval_status='ORIGINAL_PUBLISHED_SOURCE_BOOTSTRAP_REUSED')
    out.mkdir(parents=True);original=out/'original';original.mkdir()
    for p in resource_files:shutil.copyfile(p,original/p.name)
    for p in (resources/'completion.json',metrics/'completion.json'):shutil.copyfile(p,original/('resource_completion.json' if p.parent==resources else 'metrics_completion.json'))
    shutil.copyfile(metrics/'summary.csv',out/'quality_summary_existing.csv')
    s.csvwrite(out/'per_frame.csv',rows);s.csvwrite(out/'energy_summary.csv',energy);s.csvwrite(out/'state_quality.csv',quality)
    s.csvwrite(out/'failure_breakdown.csv',[dict(r,method=METHOD,failure_state=r['event'],frames=r['frame_count']) for r in failures])
    s.csvwrite(out/'resource_summary.csv',[dict(r,method=METHOD) for r in s.csvread(resources/'resource_summary.csv')])
    s.csvwrite(out/'source_verification.csv',checks)
    s.save(out/'source_manifest.json',dict(source_count=500,source_ids=[ids[i] for i in range(500)],snrs=list(SNRS),noise_seeds=[2001,2002,2003]))
    report=("# Existing adaptive BPG resource evidence\n\nAll9000 actual holdout rows (500 sources x6SNR x3noise) were SHA-verified against their original completed resource and metric exports, joined by exact frame ID, and reconciled for source, workpoint, status, image/reference SHA, PSNR and actual waveform energy. All source/noise and failure rows remain. The frozen adaptive source/MCS rule is unchanged; MCS selection used100 calibration sources x3noise, not the holdout.\n\n"
        "original/ contains unchanged source resource/failure CSV copies and the original completion receipts. per_frame.csv preserves raw CSV precision and adds the four existing metrics, E_frame and rho=E_frame/(2N). No model, codec, channel, decoder, metric network or bootstrap ran.\n\n"
        "energy_summary.csv reports actual saved waveform E and rho: mean, population standard deviation(ddof=0), min/max and linear empirical5/50/95percentiles. SOURCE_UNFIT has no waveform; its energy/rho is NA and its count remains explicit. No ideal2N energy is inserted. Zero padding information bits are encoded inside the body and are distinct from physical unused symbols. Existing exact header-energy derivations are retained as such.\n\n"
        "state_quality.csv gives all seven mutually exclusive final statuses, including zero-count states, and ALL_FRAMES. It includes frame counts, source counts, observed metric counts, frame means and source-balanced conditional means. Conditional means first average the selected noise outcomes within each represented source; varying noise counts are reported. These are diagnostic conditions, not fairer replacement test sets. ALL_FRAMES reuses the original published confidence intervals after reconciling the existing means. Conditional intervals remain empty: no resampling. ConvNeXt is source-prediction agreement, stored as a proportion, not classification accuracy. Original overlapping failure diagnostics remain in failure_breakdown.csv and must not be added together.\n\n"
        f"Observed final states: {dict(Counter(r['status'] for r in rows))}. SOURCE_UNFIT rows: {sum(r['status']=='SOURCE_UNFIT' for r in rows)}. No scientific input was modified.\n")
    (out/'README.md').write_text(report,encoding='utf-8')
    result=dict(status='T4_ADAPTIVE_BPG_EXISTING_READ_ONLY_COMPLETE',source_count=500,noise_count=3,frame_count=9000,source_ids=[ids[i] for i in range(500)],
        input_bindings=inputs,source_verification=checks,energy_rows=len(energy),state_quality_rows=len(quality),std_ddof=0,
        new_PHY=0,new_codec_calls=0,new_model_calls=0,new_channel_draws=0,new_metric_calls=0,new_bootstrap=0,original_files_modified=False,
        scripts={str(Path(__file__).resolve()):s.sha(__file__),str(Path(s.__file__).resolve()):s.sha(s.__file__)},
        outputs={str(p.relative_to(out)):s.sha(p) for p in out.rglob('*') if p.is_file()})
    s.save(out/'completion.json',result);return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('resources','metrics','out'):p.add_argument('--'+n,required=True)
    result=run(p.parse_args());print({k:result[k] for k in ('status','frame_count','source_count','noise_count','energy_rows','state_quality_rows')})
if __name__=='__main__':main()
