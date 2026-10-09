"""Plot actual new BPG estimates beside immutable published main estimates.

No inference, PHY, policy selection or bootstrap occurs in this renderer.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

SNRS=(1,4,7,10,13,19)
METRICS=(('psnr_db','PSNR (dB) ↑',1),('lpips_alex','LPIPS-Alex ↓',1),
         ('dinov2_vitl14_cosine','DINOv2-ViT-L/14 cosine ↑',1),
         ('convnext_top1_source_prediction','Source prediction agreement (%) ↑',100))
MAIN_SHA='5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859'
MAIN_COMMIT='252176e041758ecb2d3e81b6fde5b587e7e17bb7'
METHODS=(('RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}','Proposed (raw + partial + VAR)','#0072B2','-','o'),
         ('P1024_SNR_{snr}','P1024','#D55E00','--','s'),
         ('SWIN80K_N1024_SNR_{snr}','SwinJSCC-80k (adapted)','#009E73','-.','^'),
         ('BPG_LDPC_N1024_SNR_{snr}','BPG+LDPC (14-MCS calibration)','#CC79A7',':','D'))


CAPTION=r'''% Actual immutable CSV means and CIs; no scientific recomputation.
% Use \caption{\BPGSupplementCaption}.
\newcommand{\BPGSupplementCaption}{Independent BPG+LDPC (14-MCS calibration)
supplement on the same frozen 500-source holdout at $N=1024$ complex channel symbols.
The three main curves retain their published values and visual identities;
BPG is purple with diamond markers. PSNR, LPIPS-Alex, DINOv2-ViT-L/14 cosine,
and ConvNeXt source-prediction agreement are shown. Higher is better except for
LPIPS; ConvNeXt means and CI endpoints are displayed as percentages, not label
accuracy. Shading shows the supplied pointwise 95\% confidence intervals after
each source's three-noise mean and the existing 10,000-replicate source bootstrap
(seed 2026100701); no new bootstrap is run. All 9000 scheduled BPG rows are included.
Source-unfit, header/CRC/parser/BPG-decode failures retain gray RGB 0.5;
unfit rows make no PHY call. The BPG noise namespace is independent: seed labels
2001--2003 do not imply identical noise arrays or a per-noise paired contrast
with RAW/P. Accepted payload differences from TX are diagnostics only, including
CRC undetected differences; no TX-truth rescue is applied. Only the first 100
original calibration sources select MCS by mean PSNR including failures, rather
than RAW's original 1000-source DINO-L objective. The fixed 14-MCS set excludes
64QAM rate 1/3 available to RAW; this is an adaptation/search-space limitation,
not a demonstrated backend inability or an all-MCS optimum. BPG uses 68 header
and 956 body symbols; complete container bytes must fit the declared body capacity.
Swin's 19 dB point is outside its training/calibration range and is hollow, with
its 13--19 dB segment dashed. Its frozen six-channel/768-body-symbol plus 256-header
adaptation and 80k checkpoint are not asserted optimal or converged. Equal total
$N$ does not imply equal training, tuning or header-protection budgets. PARTIAL
retains the final WHOLE winner and does not force positive partial increments.
No holdout reselection, P-minus-Swin contrast or global-winner claim is added.}
'''

def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(d):
    require({'path','sha256'}<=set(d) and set(d)<={'path','sha256','authority_path'}
            and sha(d['path'])==d['sha256'],'Changed actual report input')
    return Path(d['path'])


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def json_out(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def csv_out(path,rows):
    require(bool(rows),'Empty output table')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def actual(request):
    r=read(request);require(r['schema']=='BPG_ACTUAL_SUPPLEMENT_REPORT_REQUEST_V1','Report request schema differs')
    descriptors={k:checked(r[k]) for k in ('bpg_summary','bpg_statistics_completion','bpg_statistics_exit',
                                        'bpg_holdout_completion','bpg_freeze','bpg_request','published_summary')}
    require(sha(descriptors['published_summary'])==MAIN_SHA,'Published main table changed')
    stats,exit=read(descriptors['bpg_statistics_completion']),read(descriptors['bpg_statistics_exit'])
    require(exit['process_waited'] is True and exit['exit_code']==0,'BPG statistics lacks actual external wait-zero')
    require(stats['outputs'][r['bpg_summary'].get('authority_path',str(descriptors['bpg_summary']))]
            ==r['bpg_summary']['sha256'],'BPG summary is not a normally bound statistics output')
    require(stats['source_count']==500 and stats['frame_count']==9000,'BPG statistics population differs')
    phy=read(descriptors['bpg_holdout_completion']);freeze=read(descriptors['bpg_freeze']);cfg=read(descriptors['bpg_request'])
    require(phy['status']=='BPG_SUPPLEMENT_STAGE_ACTUAL_CHILDREN_WAIT_ZERO_V1' and phy['stage']=='holdout'
        and phy['actual_children_waited'] is True and phy['worker_exit_codes']==[0]*8
        and phy['source_count']==500 and phy['frame_count']==9000 and phy['ledger']['unresolved']==0
        and phy['ledger']['total']<=68456 and phy['root_budget_before']==phy['root_budget_after'],'Actual complete BPG RX normal required')
    require(phy['request_sha256']==sha(descriptors['bpg_request']) and freeze['request_sha256']==sha(descriptors['bpg_request'])
        and freeze['status']=='BPG_POLICIES_FROZEN_ON_CALIBRATION_ONLY_V1'
        and freeze['holdout_used_for_selection'] is False,'BPG calibration-only freeze linkage differs')
    require(cfg['SNRs']==list(SNRS) and cfg['noise_seeds']==[2001,2002,2003]
        and len(cfg['records']['holdout'])==500 and len(cfg['records']['calibration'])==100,'BPG frozen population/noise differs')
    bpg=csv_rows(descriptors['bpg_summary']);main=csv_rows(descriptors['published_summary'])
    selected=[]
    for template,label,color,style,marker in METHODS:
        source=bpg if template.startswith('BPG') else main
        for key,title,scale in METRICS:
            for snr in SNRS:
                rows=[row for row in source if row['point_id']==template.format(snr=snr) and row['metric']==key]
                require(len(rows)==1,'Missing/duplicated actual method/metric/SNR')
                row=dict(rows[0]);mean,low,high=(Decimal(row[k]) for k in ('mean','ci_low','ci_high'))
                require(all(x.is_finite() for x in (mean,low,high)) and low<=mean<=high,'Invalid actual CI')
                require((int(row['source_count']),int(row['noise_count']),int(row['frame_count']))==(500,3,1500),
                        'Source/noise/frame averaging differs')
                require(int(row['bootstrap_replicates'])==10000 and int(row['bootstrap_seed'])==2026100701
                        and row['bootstrap_unit']=='source after original3-noise mean','Source bootstrap identity differs')
                require(int(row['snr_db'])==snr,'Actual SNR metadata differs')
                row.update(label=label,display_scale=scale,published_source_commit=MAIN_COMMIT if not template.startswith('BPG') else '',
                           original_summary_sha256=sha(descriptors['bpg_summary'] if template.startswith('BPG') else descriptors['published_summary']))
                selected.append(row)
    require(len(selected)==96,'Complete 4methods x6SNR x4metric grid required')
    return r,descriptors,stats,phy,freeze,cfg,selected


def failures(r,phy,cfg,stats):
    """Metadata-only mutually exclusive failure counts; actual RGB is never loaded."""
    if 'bpg_failure_breakdown' in r:
        d=r['bpg_failure_breakdown'];path=checked(d)
        require(stats['outputs'].get(d.get('authority_path',str(path)))==d['sha256'],
                'Failure table not in actual normally bound statistics outputs')
        table=csv_rows(path)
        require(len(table)==6 and [int(v['snr_db']) for v in table]==list(SNRS),'Failure table fixed6 SNR grid differs')
        fields=('source_unfit','header_reject','body_CRC_reject','body_parser_reject','BPG_decode_reject','decoded')
        result=[]
        required={'snr_db',*fields,'source_unfit_count','frame_count','total_gray_fraction',
                  'accepted_payload_difference_count','packet_decoder_calls'}
        for raw in table:
            require(set(raw)==required,'Actual BPG failure CSV field schema differs')
            counts={k:int(raw[k]) for k in fields};unfit_sources=int(raw['source_unfit_count'])
            require(int(raw['frame_count'])==1500 and all(0<=v<=1500 for v in counts.values())
                    and sum(counts.values())==1500,'Failure categories do not cover all1500 planned frames')
            require(0<=unfit_sources<=500 and counts['source_unfit']==3*unfit_sources,
                    'Source-unfit source/frame domains differ')
            gray=1500-counts['decoded'];fraction=Decimal(raw['total_gray_fraction'])
            require(fraction.is_finite() and abs(fraction-Decimal(gray)/1500)<=Decimal('1e-15'),
                    'Actual gray fraction differs from complete frame counts')
            require(0<=int(raw['packet_decoder_calls'])<=2*(1500-counts['source_unfit'])
                    and 0<=int(raw['accepted_payload_difference_count'])<=1500-counts['source_unfit'],
                    'Actual packet-call/payload diagnostics differ')
            # Retain every raw value. source_unfit_count is the original500-source count;
            # the six plain category names count scheduled frames, including all3 noises.
            row=dict(raw,source_count=500,noise_count=3,source_unfit_source_count=unfit_sources,
                     source_unfit_frame_count=counts['source_unfit'],total_gray_count=gray,
                     source_unfit_fraction=unfit_sources/500)
            for k,v in counts.items():row[k+'_fraction_of_all_frames']=v/1500
            result.append(row)
        return result
    mirrors=r['source_checkpoint_mirrors']
    require(len(mirrors)==500,'All actual500 source checkpoint mirrors required')
    categories=('source_unfit','header_reject','body_CRC_reject','body_parser_reject','BPG_decode_reject','decoded')
    counts={snr:{k:0 for k in categories} for snr in SNRS};calls={s:0 for s in SNRS}
    unfit={s:set() for s in SNRS};undetected={s:0 for s in SNRS}
    rows=0
    for i,item in enumerate(mirrors):
        p=checked({k:item[k] for k in ('path','sha256')});authority=item['authority_path'];done=read(p)
        require(phy['outputs'].get(authority)==item['sha256'],'Source checkpoint not in actual normal graph')
        require(done['source_index']==i and done['source_id']==cfg['records']['holdout'][i]['source_id']
            and done['status']=='BPG_LDPC_SOURCE_RX_COMPLETE_V1' and len(done['rows'])==18,'Actual checkpoint source/grid differs')
        require([(v['snr_db'],v['noise_seed']) for v in done['rows']]==[(s,n) for s in SNRS for n in (2001,2002,2003)],'Complete fixed18 row order differs')
        for row in done['rows']:
            snr=row['snr_db'];status=row['status'];rows+=1;calls[snr]+=row['packet_decoder_calls']
            if not row['source_encoding_fit']:
                require(status=='SOURCE_UNENCODABLE_WITH_DECLARED_RATE_CONTROL' and row['packet_decoder_calls']==0,'Unfit source accounting differs')
                category='source_unfit';unfit[snr].add(i)
            elif status=='HEADER_REJECT':category='header_reject'
            elif status=='BODY_CRC_REJECT':category='body_CRC_reject'
            elif status=='BODY_PARSER_REJECT':category='body_parser_reject'
            elif status in ('BPG_DECODER_REJECT','BPG_SOURCE_FORMAT_REJECT'):category='BPG_decode_reject'
            else:
                require(status=='BPG_DECODED','Unknown runtime failure cannot enter gray statistics');category='decoded'
            require(row['gray_substitution']==(category!='decoded'),'Gray/source failure label differs')
            counts[snr][category]+=1;undetected[snr]+=row['undetected_payload_difference'] is True
    require(rows==9000,'Actual9000 complete rows required')
    table=[]
    for snr in SNRS:
        c=counts[snr];require(sum(c.values())==1500,'Failure categories do not cover every scheduled frame')
        row=dict(snr_db=snr,source_count=500,noise_count=3,frame_count=1500,source_unfit_source_count=len(unfit[snr]),
                 source_unfit_count=len(unfit[snr]),source_unfit_frame_count=c['source_unfit'],
                 source_unfit_fraction=len(unfit[snr])/500,packet_decoder_calls=calls[snr],
                 accepted_payload_difference_count=undetected[snr],total_gray_count=1500-c['decoded'],
                 total_gray_fraction=(1500-c['decoded'])/1500)
        for key,value in c.items():row[key]=value;row[key+'_fraction_of_all_frames']=value/1500
        table.append(row)
    return table


def figures(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator,ScalarFormatter
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':MAIN_COMMIT,
        'axes.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white','path.simplify':False})
    handles=[Line2D([],[],color=c,ls=ls,marker=m,lw=1.35,ms=4,label=name) for _,name,c,ls,m in METHODS]
    names=('PSNR','LPIPS','DINOv2-L','ConvNeXt')
    slugs=('psnr','lpips','dinov2_l','convnext_agreement');saved=[];ranges={}

    def panel(ax,metric,name,letter=None):
        key,label,scale=metric;lows=[];highs=[]
        for template,method,color,style,marker in METHODS:
            rr=[next(r for r in rows if r['point_id']==template.format(snr=s) and r['metric']==key) for s in SNRS]
            y,lo,hi=[np.asarray([float(r[f])*scale for r in rr]) for f in ('mean','ci_low','ci_high')]
            x=np.asarray(SNRS,dtype=float);lows.extend(lo);highs.extend(hi)
            ax.fill_between(x,lo,hi,color=color,alpha=.13,linewidth=0,zorder=1)
            if template.startswith('SWIN'):
                ax.plot(x[:5],y[:5],color=color,ls=style,marker=marker,lw=1.35,ms=4.4,mew=.85,zorder=3)
                ax.plot(x[4:],y[4:],color=color,ls='--',lw=1.35,zorder=3)
                ax.plot(x[-1:],y[-1:],color=color,ls='none',marker=marker,ms=5.1,mfc='white',mew=1.2,zorder=5)
            else:ax.plot(x,y,color=color,ls=style,marker=marker,lw=1.35,ms=4.4,mew=.85,zorder=4)
        low,high=min(lows),max(highs);span=high-low if high>low else max(abs(high)*.2,.01)
        ax.set_ylim(low-span*.09,high+span*.09);ax.set_xlim(.3,19.7)
        ranges[key]=dict(ci_min=float(low),ci_max=float(high),display_scale=scale,
                         axis_min=float(low-span*.09),axis_max=float(high+span*.09))
        ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)');ax.set_ylabel(label,labelpad=5)
        ax.set_title((f'({letter}) ' if letter else '')+name,loc='left',fontweight='semibold',pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5));fmt=ScalarFormatter(useOffset=False)
        fmt.set_powerlimits((-3,4));ax.yaxis.set_major_formatter(fmt)
        ax.grid(color='#DFE3E6',lw=.55);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        for edge in ('left','bottom'):ax.spines[edge].set_color('#535353')

    def export(fig,stem):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;texts=[]
        for ax in fig.axes:
            texts += [ax.xaxis.label,ax.yaxis.label,ax._left_title,ax.yaxis.get_offset_text(),
                      *ax.get_xticklabels(),*ax.get_yticklabels()]
        boxes=[t.get_window_extent(renderer) for t in texts if t.get_visible() and t.get_text()]
        boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
        require(all(b.x0>=-.5 and b.y0>=-.5 and b.x1<=bounds.x1+.5 and b.y1<=bounds.y1+.5 for b in boxes),
                'Figure text clipped: '+stem)
        for ext in ('pdf','svg','png'):
            path=out/(stem+'.'+ext);kw={'dpi':600} if ext=='png' else {}
            if ext=='pdf':kw['metadata']=dict(Title=stem,Author='VAR_COMM',Subject='Actual BPG supplement; original commit '+MAIN_COMMIT)
            fig.savefig(path,format=ext,**kw);saved.append(path.name)
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(7,5.2),dpi=150)
    fig.subplots_adjust(left=.105,right=.985,bottom=.10,top=.79,wspace=.36,hspace=.57)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,
               frameon=False,handlelength=2.6,columnspacing=1.4,fontsize=8)
    for ax,metric,name,letter in zip(axes.flat,METRICS,names,'abcd'):panel(ax,metric,name,letter)
    export(fig,'bpg_supplement_N1024')
    for metric,name,slug in zip(METRICS,names,slugs):
        fig,ax=plt.subplots(figsize=(3.45,3.35),dpi=150)
        fig.subplots_adjust(left=.235,right=.975,bottom=.16,top=.615)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.995),ncol=1,
                   frameon=False,handlelength=2.6,fontsize=7.6,labelspacing=.35)
        panel(ax,metric,name);export(fig,'bpg_supplement_N1024_'+slug)
    return dict(files=saved,composite_inches=[7,5.2],single_panel_inches=[3.45,3.35],
                png_dpi=600,all_confidence_intervals_in_axis_ranges=True,ranges=ranges)


def render(request,out):
    r,paths,stats,phy,freeze,cfg,rows=actual(request);failure=failures(r,phy,cfg,stats)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    csv_out(out/'comparison_four_metrics.csv',rows);csv_out(out/'BPG_failure_breakdown.csv',failure)
    json_out(out/'BPG_calibration_frozen_policies.json',dict(policies=freeze['policies'],objective=freeze['objective'],
        calibration_means_PSNR=freeze['calibration_means_PSNR'],freeze_sha256=sha(paths['bpg_freeze'])))
    figure_identity=figures(rows,out)
    (out/'caption.tex').write_text(CAPTION,encoding='utf-8')
    notes='''# BPG + LDPC supplement on the frozen common500 sources

The original published result tables and fig02 are unchanged. This separate post-hoc supplement adds official libbpg0.9.8/x265, fixed default compression level8, 420 YCbCr 8bit, and N1024 LDPC transmission.

The complete BPG container must fit the paid body capacity. The header occupies68 symbols and the body956. Body information includes a13bit payload-length field, known zero padding and CRC16. Rate control checks QP51, then a binary QP search and a ±2 local check using actual complete file sizes; it is not claimed globally rate-distortion optimal.

Only the first100 original calibration sources select one of14 fixed MCS per SNR. The fixed supplement catalogue contains QPSK and16QAM at rates1/3,1/2,2/3,3/4,5/6, and64QAM at rates1/2,2/3,3/4,5/6. It excludes64QAM rate1/3, which is available to the original RAW method. This exclusion arose from adapting the older H64 four-rate catalogue, not from a demonstrated backend inability; it is an explicit baseline adaptation/search-space limitation. No all-MCS optimum or equal MCS permissions/search-space claim is made. The objective is mean PSNR across all sources after their three-noise means, including fixed-gray failures; ties use the smallest public profile ID. This objective differs from the original raw method's DINO-L objective. No holdout-optimal MCS or QP-quality search is used. There is no new training.

Every reported point includes500 sources ×3 fixed noises, retaining all9000 scheduled BPG rows including source-unfit gray rows with no PHY call. BPG uses its own noise namespace; labels2001–2003 do not mean the same noise arrays as RAW/P. No per-noise paired comparison is claimed. The supplied confidence intervals use10000 source bootstrap replicates, seed2026100701, with each source first averaging its three noises. These are pointwise95% intervals without multiplicity correction. This renderer does not recompute statistics. ConvNeXt reports source-prediction agreement in percent, not classification accuracy. The DINO field is dinov2_vitl14_cosine.

Source-unfit, header rejection, body CRC rejection, parser rejection and BPG decoding rejection retain a fixed0.5 RGB reconstruction. Unknown program/I/O exceptions terminate the experiment and cannot appear as gray outcomes. The failure table contains mutually exclusive counts over all1500 scheduled frames at each SNR; source-unfit counts also report their500-source denominator. Accepted payload differences from TX, including CRC undetected differences, are diagnostic counts only and are never repaired or rescued using transmitter truth.

Swin19dB is outside the training/calibration range: the marker is hollow and13–19dB is dashed. Swin's six channels/768 body symbols and256 protected-header symbols, fixed80k adaptation, and lack of an optimum/convergence claim remain explicit. EqualN does not imply equal training or tuning budgets. PARTIAL retains the final WHOLE winner and does not force positive partial-residual increments. No P−Swin contrast or global-winner claim is added.

The composite is7in wide,2×2 panels; each of the four single-column panels is also exported independently as PDF/SVG/600dpi PNG. Axis limits cover all CI endpoints, with natural padding. The main three methods preserve the original fig02 colors, line styles, markers and legend labels; BPG is purple/diamond/dotted. [LaTeX caption](caption.tex).

Tables: [four metrics](comparison_four_metrics.csv), [BPG failure breakdown](BPG_failure_breakdown.csv), [calibration-frozen policies](BPG_calibration_frozen_policies.json).
'''
    (out/'REPORT.md').write_text(notes,encoding='utf-8')
    outputs={str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    json_out(out/'completion.json',dict(status='BPG_ACTUAL_SUPPLEMENT_TABLES_AND_FIGURES_RENDERED_V1',
        renderer_sha256=sha(__file__),request_sha256=sha(request),actual_inputs={k:sha(p) for k,p in paths.items()},
        outputs=outputs,main_commit=MAIN_COMMIT,old_main_outputs_modified=False,new_model_calls=0,new_PHY_calls=0,
        new_bootstrap_calls=0,source_count=500,noise_count=3,comparison_rows=96,failure_rows=6,
        figure_identity=figure_identity,
        renderer_external_wait_required=True))
    print(json.dumps(dict(status='BPG_SUPPLEMENT_RENDERER_OUTPUTS_WRITTEN',completion=str(out/'completion.json'),files=len(outputs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();render(a.request,a.out)
