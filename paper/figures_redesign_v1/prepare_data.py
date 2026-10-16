"""Compose closed summary rows and verify existing source means; no statistical resampling."""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib, math, argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def method(s):return 'BPG_ADAPTIVE' if s=='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024' else s
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--geometry',type=Path,required=True);a=ap.parse_args()
    R=a.root;out=R/'paper/figures_redesign_v1';d=out/'plot_inputs';d.mkdir(exist_ok=True)
    paths={
      'original_four':'results/entropy_main_common500_20261013/analysis_v1/summary_all_methods.csv',
      'supplement_three':'results/quality_metrics_supplement_20261014_v1/common500/summary.csv',
      't1c':'results/t1c_m10_test_20261016_v1/delivery/summary_19dB.csv',
      'original_source':'results/entropy_main_common500_20261013/analysis_v1/source_means.csv',
      'supplement_source':'results/quality_metrics_supplement_20261014_v1/common500/source_noise_means.csv',
      't1c_source':'results/t1c_m10_test_20261016_v1/delivery/source_means_19dB.csv',
      't1c_endpoints':'results/t1c_m10_test_20261016_v1/delivery/actual_fallback_counts.csv',
      't1c_lengths':'results/t1c_m10_test_20261016_v1/delivery/m10_source_lengths_test600.csv',
      't1c_paired':'results/t1c_m10_test_20261016_v1/delivery/paired_19dB.csv',
      'status':'RESEARCH_STATUS.md',
      'manuscript':'paper/manuscripts/nestcom_wcl_v5_20261012/nestcom_wcl_draft_v5.tex'}
    inventory=[dict(role=k,path=v,bytes=(R/v).stat().st_size,sha256=sha(R/v))for k,v in paths.items()]
    tables={k:read(R/v) for k,v in paths.items()if v.endswith('.csv')}
    table={}; source_groups=defaultdict(list)
    for role in ('original_four','supplement_three'):
      for n,r in enumerate(tables[role],2):
        k=(method(r['method']),int(float(r['snr_db'])),r['metric']);assert k not in table
        table[k]=dict(population='common500',method_id=k[0],snr_db=k[1],metric=k[2],mean=r['mean'],ci_low=r.get('ci_low',r.get('CI95_low')),ci_high=r.get('ci_high',r.get('CI95_high')),sources=500,noises_per_source=3,logical_frames=1500,input_role=role,input_path=paths[role],input_csv_line=n,interval='pointwise 95% source-bootstrap CI; existing bounds copied',bootstrap_replicates=10000,bootstrap_seed=2026100701)
    replacements=[]
    for n,r in enumerate(tables['t1c'],2):
      if r['population']!='common500':continue
      k=('EC_VAR_'+r['family'],19,r['metric']);old=table[k].copy()
      assert abs(float(old['mean'])-float(r['old_mean']))<1e-12
      table[k].update(mean=r['mean'],ci_low=r['ci_low'],ci_high=r['ci_high'],input_role='t1c',input_path=paths['t1c'],input_csv_line=n)
      replacements.append(dict(method_id=k[0],snr_db=19,metric=k[2],old_mean=old['mean'],new_mean=r['mean'],old_ci_low=old['ci_low'],old_ci_high=old['ci_high'],new_ci_low=r['ci_low'],new_ci_high=r['ci_high']))
    assert len(replacements)==14
    for role,col in [('original_source','mean'),('supplement_source','source_mean')]:
      for r in tables[role]:
        k=(method(r['method']),int(float(r['snr_db'])),r['metric']);source_groups[k].append((int(r['source_index']),float(r[col])))
    for k in list(source_groups):
      if k[0] in ('EC_VAR_WHOLE','EC_VAR_PARTIAL')and k[1]==19:source_groups[k]=[]
    for r in tables['t1c_source']:
      if r['population']=='common500':source_groups['EC_VAR_'+r['family'],19,r['metric']].append((int(r['source_index']),float(r['new_mean'])))
    verification=[]
    for k,r in table.items():
      values=source_groups[k];assert len(values)==500 and {i for i,v in values}==set(range(500)),k
      observed=math.fsum(v for i,v in values)/500
      assert abs(observed-float(r['mean']))<1e-10,(k,observed,r['mean'])
      assert float(r['ci_low'])<=float(r['mean'])<=float(r['ci_high'])
      verification.append(dict(method_id=k[0],snr_db=k[1],metric=k[2],source_count=len(values),mean_difference=observed-float(r['mean'])))
    methods=json.loads((out/'methods.json').read_text());style=json.loads((out/'style.json').read_text());mainrows=[]
    for spec in style['metrics']:
      for m in methods:
        for snr in [4,10,19]:
          r=table[m['id'],snr,spec['id']].copy();r.update(display_label=m['label'],display_scale=spec['scale']);mainrows.append(r)
    assert len(mainrows)==60
    # Include all seven available metrics for six methods, and old raw-whole's four.
    write(d/'common500_all_metrics_versioned.csv',list(table.values()))
    write(d/'fig2_plot_data.csv',mainrows);write(d/'t1c_replacement_log.csv',replacements)
    write(d/'source_mean_verification.csv',verification)
    geom=json.loads(a.geometry.read_bytes());(out/'template_geometry.json').write_text(json.dumps(geom,indent=2)+'\n')
    record=dict(status='EXISTING_INPUTS_VERIFIED',head=geom['head'],inputs=inventory,main_numeric_positions=60,updated_entropy_family_positions=14,updated_fig2_positions=4,missing_fig2_data=[],other_gaps=[dict(method='RAW_WHOLE',metrics=['dists','dreamsim','ms_ssim'],action='not present; no scores fabricated')],new_scoring_calls=0,new_bootstrap_calls=0,new_experiments=0,transformations=['select common500 only','replace both entropy families at19dB with exact T1-C summary bounds','alias BPG identifier','agreement displayed as percent','verify existing means against existing per-source means; no new intervals'],geometry=geom)
    (out/'input_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],positions=len(mainrows),replacements=len(replacements),sha256=sha(d/'fig2_plot_data.csv'))))
if __name__=='__main__':main()
