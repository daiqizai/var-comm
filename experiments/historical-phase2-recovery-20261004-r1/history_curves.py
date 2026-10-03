"""Observed quality/resource points from completed metric summaries only.

No reconstruction, selection by score, interpolation, or missing-point repair.
Input summaries and every plotted row retain their SHA and group provenance.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

SNRS = (1, 4, 7, 13, 19)
BUDGETS = (512, 1024, 2048, 3060, 4084)
METRICS = {
    'psnr_db': ('PSNR (dB)', 'higher'),
    'lpips_alex': ('LPIPS AlexNet', 'lower'),
    'dino_cosine': ('DINOv2 ViT-S/14 cosine', 'higher'),
    'dinov2_vitl14_cosine': ('DINOv2 ViT-L/14 cosine', 'higher'),
    'clip_image_cosine': ('CLIP ViT-L/14 image cosine', 'higher'),
    'dists': ('DISTS', 'lower'),
    'resnet50_top1_label': ('ResNet-50 top-1 / true label', 'higher'),
    'resnet50_top1_source_prediction': ('ResNet-50 / source prediction agreement', 'higher'),
}
PANELS = (
    ('U_QPSK', 'Same Dc, no class label; QPSK and continuous, E = 2N'),
    ('U_16QAM', 'Same Dc, no class label; 16QAM actual energy; P has E = 2N'),
    ('C_QPSK', 'Same Dc, paid class label; QPSK, E = 2N; P is unconditional'),
    ('C_16QAM', 'Same Dc, paid class label; 16QAM actual energy; P is unconditional'),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def boolean(value):
    if isinstance(value, bool):return value
    if str(value).lower() in ('true', '1'):return True
    if str(value).lower() in ('false', '0'):return False
    raise ValueError('Explicit Boolean required: ' + str(value))


def number(value):
    try:x = float(value)
    except (ValueError, TypeError):return None
    return x if math.isfinite(x) else None


def canonical_decoder(value):
    return {'frozen_stageA_Dc': 'Dc', 'VAR_D0': 'D0'}.get(str(value), str(value))


def _model_series(row, origin):
    """Fixed study/method admission rules; never inspect a metric value."""
    study, method = row['study'], row['method']
    if study in ('N512', 'N1024'):
        if re.fullmatch(r'P(?:512|1024)', method):return 'P_SELECTED', 'P selected per N'
        if re.fullmatch(r'D_[UC]_(?:QPSK|16QAM)', method):return 'CORE_' + method, method.replace('_', ' ')
        return None
    if study == 'M1':
        names = {'entropy_policy': 'M1 entropy policy', 'raster_policy': 'M1 raster policy'}
        return ('M1_' + method, names[method]) if method in names else None
    if study == 'M2_ACTUAL':
        control = row.get('control') or method
        if control in ('VAR_GUIDED', 'UNGUIDED'):
            projection = row.get('projection', '')
            return 'M2_' + control + '_' + projection, 'M2 ' + control.lower().replace('_', ' ') + ' / ' + projection
        return None
    if study == 'FINAL_P2048_P3060' and method in ('P2048', 'P3060'):
        return 'P_SELECTED', 'P selected per N'
    if study == 'FINAL_P4084_SELECTED_SEEDS':
        if method == 'P4084_N4084_seed2026092304':return 'P_SELECTED', 'P selected per N'
        if method == 'P4084_N4084_seed2026092404':return 'P_SECOND_SEED', 'P4084 seed 2026092404'
        return None
    if study in ('FINAL_DIGITAL_QPSK', 'FINAL_DIGITAL_16QAM'):
        match = re.fullmatch(r'final_(raw|arithmetic)_(QPSK|16QAM)_N(2048|3060|4084)_Dc', method)
        if match:return 'FINAL_DIGITAL_' + match[1], 'Final digital ' + match[1] + ' (paid class)'
        return None
    if study == 'FINAL_PHASE2_N4084_DIGITAL' and method in ('raw_adaptive_m789_Dc', 'arithmetic_adaptive_m789_Dc'):
        return 'PHASE2_' + method, 'Phase 2 ' + method.split('_')[0] + ' (paid class)'
    if study == 'LEGACY_N3060_FINAL' and method in ('raw_adaptive', 'arithmetic_adaptive', 'perceptual_deepjscc'):
        return 'LEGACY_' + method, 'Legacy ' + method.replace('_', ' ')
    if study == 'SELECTED_EXTERNAL_AUTHORS':
        return 'AUTHOR_' + method, method.replace('|', ' / ')
    return None


def normalize(row, origin, path, file_sha, row_index):
    """Normalize the two public summary schemas, preserving original context."""
    original = dict(row)
    if origin == 'six_study':
        row = dict(row, study=row['experiment'], decoder=row['decoder_id'],
                   phy=row['phy_family'], condition='C' if boolean(row['label_conditioned']) else 'U',
                   snr_definition='physical_channel_snr_db', training_seed='', model_id='',
                   reference_only=row['output_role'] in ('reference', 'source_only_reference'),
                   oracle=row['output_role'] in ('oracle', 'paid_oracle_reference'))
    else:
        row = dict(row)
    metric = row['metric'].removeprefix('new_')
    row.update(metric=metric, decoder=canonical_decoder(row['decoder']),
               N=number(row.get('N')), snr_db=number(row.get('snr_db')))
    if metric not in METRICS:return None, 'not_a_requested_curve_metric'
    if origin == 'six_study' and row.get('status') != 'EVALUATED':return None, 'metric_not_evaluated'
    if row['N'] is None or row['N'] <= 0:return None, 'no_paid_resource_budget'
    if row['snr_db'] not in SNRS:return None, 'outside_registered_curve_snr'
    if row.get('snr_definition') not in ('snr_db', 'physical_channel_snr_db'):
        return None, 'different_snr_definition'
    if boolean(row.get('oracle', False)):return None, 'oracle_reference'
    chosen = _model_series(row, origin)
    if chosen is None:return None, 'not_a_registered_main_resource_series'
    base_series, label = chosen
    is_p = base_series in ('P_SELECTED', 'P_SECOND_SEED')
    conditioned = boolean(row['label_conditioned'])
    decoder = row['decoder']
    phy = str(row.get('phy', ''))
    if is_p:
        if decoder != 'Dc' or conditioned:raise ValueError('P baseline has incompatible decoder/class access')
        phy = 'continuous'
    if row['study'].startswith('FINAL_DIGITAL_'):
        expected_phy = row['study'].removeprefix('FINAL_DIGITAL_')
        if phy != expected_phy or not conditioned or decoder != 'Dc':
            raise ValueError('Final digital protocol annotation differs')
    reference = decoder != 'Dc' or row['study'] in ('LEGACY_N3060_FINAL', 'SELECTED_EXTERNAL_AUTHORS')
    if not reference and not is_p and phy not in ('QPSK', '16QAM'):
        return None, 'unsupported_or_unspecified_physical_protocol'
    mean, low, high = [number(row.get(k)) for k in ('mean', 'ci_low', 'ci_high')]
    if any(v is None for v in (mean, low, high)):return None, 'missing_metric_interval'
    if low > high:raise ValueError('Reversed source bootstrap interval')
    # Bootstrap percentile intervals need not include the sample point exactly.
    panels = ['REFERENCE'] if reference else [p[0] for p in PANELS] if is_p else [
        ('C_' if conditioned else 'U_') + phy]
    notes = ('original author assumptions; decoder and side information differ' if row['study']=='SELECTED_EXTERNAL_AUTHORS'
             else 'legacy D0 / external context; not same-Dc ranking' if reference
             else 'classification is descriptive: true class sent to generator' if conditioned
             else 'selected P checkpoint differs by N; seed is retained per point' if is_p
             else 'observed development point only')
    scope = dict(base_series=base_series, decoder=decoder, label_conditioned=conditioned,
                 phy=phy, condition=row.get('condition', ''), projection=row.get('projection', ''),
                 # P's precise checkpoint intentionally varies with N. Other
                 # methods retain their registered protocol and projection.
                 protocol='P_selected_per_budget' if is_p else row['study'],
                 training_role='second_seed' if base_series=='P_SECOND_SEED' else 'primary' if is_p else '')
    # Conditions on P rows are legacy schema vocabulary, not extra model input.
    if is_p:scope['condition']='U'
    # N512 and N1024 are the same two registered baseline families; grouping
    # these does not join any later historical method into a low-N gap.
    if row['study'] in ('N512','N1024') and not is_p:scope['protocol']='core_extreme_bandwidth'
    series = identity(scope)
    return dict(panels=panels, series_id=series, series_label=label, **scope,
        origin=origin, source_path=str(Path(path)), source_sha256=file_sha,
        source_row_index=row_index, original_group_id=row.get('group_id', ''),
        original_summary_json=json.dumps(original, sort_keys=True),
        study=row['study'], method=row['method'], N=row['N'], snr_db=row['snr_db'],
        metric=metric, mean=mean, ci_low=low, ci_high=high,
        n_sources=int(float(row.get('n_sources') or row.get('sources'))),
        frames=int(float(row.get('n_frames') or row.get('frames'))),
        training_seed=row.get('training_seed', ''), model_id=row.get('model_id', ''),
        connect=not reference and base_series!='P_SECOND_SEED',
        energy_protocol='per_frame_2N' if is_p or phy=='QPSK' and not reference else
                        'original_actual_energy_not_assumed_2N' if phy=='16QAM' else 'original_protocol',
        notes=notes), None


def select_points(historical, original, paths, hashes):
    points, excluded = [], []
    for origin, rows in (('historical', historical), ('six_study', original)):
        for i,row in enumerate(rows):
            point, reason = normalize(row, origin, paths[origin], hashes[origin], i)
            if point is None:
                excluded.append(dict(origin=origin, source_row_index=i, source_sha256=hashes[origin],
                    group_id=row.get('group_id',''), method=row.get('method',''),
                    metric=row.get('metric',''), reason=reason))
            else:
                for panel in point.pop('panels'):
                    points.append(dict(point,panel=panel))
    keys = [(p['panel'],p['series_id'],p['metric'],p['snr_db'],p['N']) for p in points]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate resource point within one declared series; do not average distinct checkpoints')
    return points, excluded


def contiguous_segments(points, budgets=BUDGETS):
    """Break a line wherever an explicitly listed intermediate N is unmeasured."""
    ordered = sorted(points,key=lambda p:p['N'])
    if len({p['N'] for p in ordered}) != len(ordered):raise ValueError('Duplicate N in curve series')
    segments=[]
    for p in ordered:
        if not segments or not p['connect'] or not segments[-1][-1]['connect'] or any(
                segments[-1][-1]['N'] < n < p['N'] for n in budgets):
            segments.append([p])
        else:segments[-1].append(p)
    return segments


def resource_ticks(values, budgets=BUDGETS):
    """Share labels for visually adjacent ticks without moving measured data."""
    groups=[]
    for n in sorted(set(budgets)|{p['N'] for p in values}):
        if groups and n/groups[-1][0] < 1.08:groups[-1].append(n)
        else:groups.append([n])
    def label(n):return str(int(n)) if float(n).is_integer() else str(n)
    return [(math.exp(sum(math.log(n) for n in g)/len(g)), '\n'.join(label(n) for n in g)) for g in groups]


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def generate(root, historicalsummary, originalsummary, out):
    """Return paths of generated PNG/PDF figures, exact points and provenance."""
    root,out=Path(root).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths={'historical':Path(historicalsummary).resolve(),'six_study':Path(originalsummary).resolve()}
    hashes={key:sha(value) for key,value in paths.items()}
    points,excluded=select_points(read_csv(paths['historical']),read_csv(paths['six_study']),paths,hashes)
    if not points:raise ValueError('No observed quality/resource points')
    points.sort(key=lambda p:(p['metric'],p['panel'],p['snr_db'],p['series_id'],p['N']))
    files=[]
    p=out/'curve_points.csv';write_csv(p,points,list(points[0]));files.append(p)
    p=out/'curve_exclusions.csv';write_csv(p,excluded,list(excluded[0]) if excluded else
        ['origin','source_row_index','source_sha256','group_id','method','metric','reason']);files.append(p)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    series=sorted({(p['series_id'],p['series_label']) for p in points},key=lambda a:a[1])
    label_order=sorted({label for _,label in series})
    label_colors={label:plt.get_cmap('tab20')(i%20) for i,label in enumerate(label_order)}
    label_markers={label:('o','s','^','D','v','P','X','<','>')[i%9] for i,label in enumerate(label_order)}
    colors={sid:label_colors[label] for sid,label in series}
    markers={sid:label_markers[label] for sid,label in series}
    segments_manifest=[]
    def draw(ax, values, metric, panel, snr_value):
        byseries=defaultdict(list)
        for value in values:byseries[value['series_id']].append(value)
        for sid,rows in sorted(byseries.items()):
            for segment in contiguous_segments(rows):
                x=[p['N'] for p in segment];y=[p['mean'] for p in segment]
                ax.plot(x,y,color=colors[sid],marker=markers[sid],markersize=4,
                        linewidth=1.1,linestyle='-' if len(segment)>1 and segment[0]['connect'] else 'None')
                # Draw interval endpoints directly, including rare percentile
                # intervals whose sample estimate is outside the interval.
                for p0 in segment:
                    ax.vlines(p0['N'],p0['ci_low'],p0['ci_high'],color=colors[sid],alpha=.55,linewidth=.8)
                segments_manifest.append(dict(metric=metric,panel=panel,snr_db=snr_value,
                    series_id=sid,budgets=x,connected=len(segment)>1 and segment[0]['connect']))
        if not values:ax.text(.5,.5,'No measured points',ha='center',va='center',transform=ax.transAxes,color='.45')
        ax.set_xscale('log',base=2)
        ticks=resource_ticks(values)
        ax.set_xticks([x for x,_ in ticks],[label for _,label in ticks],rotation=40,ha='right')
        ax.set_xlim(450,max([4400]+[p['N']*1.12 for p in values]))
        ax.grid(alpha=.18)
        ax.set_xlabel('Paid complex channel uses N')
        if metric.startswith('resnet50_'):ax.set_ylim(0,1)
    for metric,(title,direction) in METRICS.items():
        measured=[p for p in points if p['metric']==metric]
        if not measured:continue
        main=[p for p in measured if p['panel']!='REFERENCE']
        if main:
            fig,axes=plt.subplots(4,5,figsize=(21,13),squeeze=False)
            for ri,(panel,panel_title) in enumerate(PANELS):
                for ci,snr_value in enumerate(SNRS):
                    values=[p for p in main if p['panel']==panel and p['snr_db']==snr_value]
                    draw(axes[ri,ci],values,metric,panel,snr_value)
                    axes[ri,ci].set_title(f'{snr_value} dB')
                    if ci==0:axes[ri,ci].set_ylabel(title)
                axes[ri,0].text(0,1.20,panel_title,transform=axes[ri,0].transAxes,fontsize=10,fontweight='bold')
            labels=sorted({p['series_label'] for p in main})
            handles=[Line2D([],[],color=label_colors[label],marker=label_markers[label],label=label,linewidth=1) for label in labels]
            fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=8,frameon=False)
            fig.suptitle(title+' — '+direction+' is better',fontsize=15,y=.993)
            fig.text(.5,.957,'Observed points only. Lines stop at missing registered budgets. Source-bootstrap 95% intervals; training seeds remain separate.',ha='center',fontsize=9)
            if metric.startswith('resnet50_'):
                fig.text(.5,.940,'Paid-class classification rows are descriptive: the true class was supplied to the generator.',ha='center',fontsize=9)
            fig.subplots_adjust(top=.87,bottom=.16,hspace=.72,wspace=.29)
            for ext in ('png','pdf'):
                p=out/f'quality_resource_{metric}.{ext}';fig.savefig(p,dpi=180);files.append(p)
            plt.close(fig)
        refs=[p for p in measured if p['panel']=='REFERENCE']
        if refs:
            fig,axes=plt.subplots(1,5,figsize=(21,4.8),squeeze=False)
            for ci,snr_value in enumerate(SNRS):
                values=[p for p in refs if p['snr_db']==snr_value]
                draw(axes[0,ci],values,metric,'REFERENCE',snr_value)
                axes[0,ci].set_title(f'{snr_value} dB')
            axes[0,0].set_ylabel(title)
            labels=sorted({p['series_label'] for p in refs})
            handles=[Line2D([],[],color=label_colors[label],marker=label_markers[label],linestyle='None',label=label) for label in labels]
            fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=7,frameon=False)
            fig.suptitle(title+' — historical / author references',fontsize=13)
            fig.text(.5,.905,'Isolated measured points: different decoders and author side-information assumptions. No same-Dc or same-energy ranking implied.',ha='center',fontsize=9)
            fig.subplots_adjust(top=.80,bottom=.34,wspace=.30)
            for ext in ('png','pdf'):
                p=out/f'quality_resource_references_{metric}.{ext}';fig.savefig(p,dpi=180);files.append(p)
            plt.close(fig)
    manifest=out/'curve_provenance.json'
    manifest.write_text(json.dumps(dict(status='OBSERVED_SUMMARY_CURVES_COMPLETE',synthetic=False,
        inputs={str(paths[k]):hashes[k] for k in paths},plot_rows=len(points),excluded_rows=len(excluded),
        resource_grid=list(BUDGETS),snrs_db=list(SNRS),interpolation=False,missing_points_filled=False,
        metric_recomputation=False,experiment_reruns=False,segments=segments_manifest,
        selection='fixed study/method/context whitelist; never metric values',
        notes=['Primary P uses each N-specific frozen selection; it is not one shared checkpoint.',
               'Original N512/N1024 summaries do not record training seed; no seed was invented.',
               'Digital D_U gaps remain gaps; D_C, D0 and author baselines never fill them.',
               '16QAM energy follows its original actual-energy protocol, not assumed 2N.',
               'Adjacent resource ticks may share a stacked label; measured coordinates remain exact.',
               'D0 and external-author points are unconnected and retain protocol labels.'],
        outputs={str(p):sha(p) for p in files}),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    files.append(manifest)
    return [str(p) for p in files]
