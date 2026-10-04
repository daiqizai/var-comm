"""Render the one-condition fixed16 headline using frozen bootstrap and image code."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

RENDER_SHA='dcda404f1ddcc28b1189df86e14e6a86847a6379519200adeaba067a30b7fafb'
M1_LABEL='M1-selected (m8, K=0, 16QAM)'

def load(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=RENDER_SHA:raise RuntimeError('Frozen renderer differs')
    spec=importlib.util.spec_from_file_location('headline_frozen_renderer',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def validate_scope(m,manifest,conditions,assets,indexed):
    m.require(manifest['status']=='HEADLINE_EXACT_CACHE_EXTRACTION_COMPLETE' and conditions==[(1024,13)]
        and len(assets)==80 and len(indexed)==64 and all(a['status']=='AVAILABLE' for a in assets.values()),'Headline scope incomplete')
    m.require(manifest['method_labels']['M1']==M1_LABEL,'M1 must be labeled selected whole-scale K0')
    for row in indexed.values():
        if row['method']=='M1':
            m.require(row['action_id']=='N1024/16QAM/m8/K0/whole' and row['phy_family']=='16QAM'
                and float(row['E'])>0,'M1 action or measured energy missing')

def main_table(m,output,manifest,summaries,indexed):
    metrics=('psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine','dists')
    by={(r['method'],r['metric']):r for r in summaries};labels=dict(m.LABELS,**manifest['method_labels'])
    rows=[];shown=[]
    for method in m.METHODS[1:]:
        values=[float(indexed[1024,13,index,method]['E']) for index in m.FIXED]
        row=dict(N=1024,snr_db=13,method=method,method_label=labels[method],n_sources=16,
            energy_mean=float(np.mean(values)),energy_min=min(values),energy_max=max(values),
            energy_interpretation='actual_fixed_constellation' if method=='M1' else 'continuous_per_frame_2N',
            **{metric:by[method,metric]['mean'] for metric in metrics})
        rows.append(row)
        shown.append([labels[method]]+[f'{row[x]:.2f}' if x=='psnr_db' else f'{row[x]:.3f}' for x in metrics])
    m.write_csv(output/'main_comparison.csv',rows)
    energy=manifest['m1_actual_energy']
    text=['# N1024，13 dB：固定16图五列比较','',
        'Original仅作视觉参照；下表为四种传输方法在相同16张源图、名义噪声种子2001上的均值。该集合用于展示，不代表完整开发集。','']
    text+=m.table(['方法']+[m.METRIC_LABELS[x] for x in metrics],shown)
    text+=['','**M1-selected本工作点为m8、K=0、whole、16QAM；不涉及部分尺度token传输，不能据此证明熵序有效。**','',
        f'总N相同。连续链Swin/HiFi/P采用逐帧E=2048约束；M1保留原16QAM实际能量：均值{energy["mean"]:.6f}，范围[{energy["min"]:.6f}, {energy["max"]:.6f}]。这不是严格同逐帧能量比较。', '',
        'Swin/HiFi共享实际接收波形；P/M1的噪声命名空间不同。配对只在相同源图上进行。Swin为指定80k模型，训练预算截断、未证明收敛。','',
        '[13项指标、95%区间及配对差值](COMPARISON_REPORT.md) · [主表CSV](main_comparison.csv) · [逐帧指标](metrics_per_frame.csv)','',
        '[16图总览](figures/comparison_N1024_SNR13_all16.png) · [总览PDF](figures/comparison_N1024_SNR13_all16.pdf) · [四页近景PDF](figures/comparison_N1024_SNR13_closeups.pdf)','']
    (output/'MAIN_TABLE.md').write_text('\n'.join(text),encoding='utf-8')

def figures(m,output,manifest,conditions,assets,indexed):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    original=Figure.savefig
    def savefig(figure,*args,**kwargs):
        if args and str(args[0]).endswith('_all16.png'):
            kwargs['dpi']=100
        if not getattr(figure,'_headline_disclosure',False):
            figure.text(.5,.43/figure.get_size_inches()[1],
                'M1-selected: m8, K=0, 16QAM; actual E varies. Continuous Swin / HiFi / P: E = 2048. No partial-token claim.',
                ha='center',fontsize=9)
            figure._headline_disclosure=True
        return original(figure,*args,**kwargs)
    display=copy.deepcopy(manifest);display['method_labels']['M1']='M1-selected\nm8, K=0, 16QAM'
    Figure.savefig=savefig
    try:return m.render_figures(output,display,conditions,assets,indexed)
    finally:Figure.savefig=original

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--legacy-dir',required=True,type=Path);p.add_argument('--tables-only',action='store_true');a=p.parse_args()
    m=load(a.legacy_dir/'render_comparison.py');manifest,conditions,assets,indexed=m.load_bundle(a.manifest)
    validate_scope(m,manifest,conditions,assets,indexed)
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    summaries,pairs=m.bootstrap_tables(conditions,indexed)
    main_table(m,output,manifest,summaries,indexed)
    m.write_csv(output/'metrics_per_frame.csv',indexed.values());m.write_csv(output/'metrics_summary.csv',summaries)
    m.write_csv(output/'metrics_paired_intervals.csv',pairs);count=m.export_assets(output,assets)
    result=[] if a.tables_only else figures(m,output/'figures',manifest,conditions,assets,indexed)
    (output/'COMPARISON_REPORT.md').write_text(m.report_text(manifest,conditions,assets,summaries,pairs,result),encoding='utf-8')
    shutil.copyfile(a.manifest,output/'extraction_manifest.json')
    outputs={p.relative_to(output).as_posix():m.sha(p) for p in output.rglob('*') if p.is_file() and p.name!='export_completion.json'}
    m.write_json(output/'export_completion.json',dict(status='TABLES_ONLY' if a.tables_only else 'M1_16QAM13_FIXED16_COMPARISON_COMPLETE',
        N=1024,snr_db=13,source_indices=list(m.FIXED),noise_seed=2001,rows=64,source_pngs=count,
        image_cells=80,summaries=len(summaries),paired_intervals=len(pairs),figures=result,
        m1_action_id='N1024/16QAM/m8/K0/whole',m1_actual_energy=manifest['m1_actual_energy'],
        new_inference=manifest['new_inference'],new_reconstruction_export=True,new_metric_evaluation=False,
        new_training=False,new_calibration=False,exploratory_fixed_subset=True,full_evaluation_resumed=False,
        manifest_sha256=m.sha(a.manifest),script_sha256=m.sha(__file__),frozen_renderer_sha256=RENDER_SHA,outputs=outputs))
    print(json.dumps(dict(status='TABLES_ONLY' if a.tables_only else 'COMPLETE',rows=64,images=count,output=str(output))))

if __name__=='__main__':main()
