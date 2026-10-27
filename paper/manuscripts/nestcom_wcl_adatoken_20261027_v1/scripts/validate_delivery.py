"""Read-only data/identity checks. No metrics, models, PHY or bootstrap."""
from pathlib import Path
import csv,hashlib,json,re
P=Path(__file__).resolve().parents[1]
def rows(p):return list(csv.DictReader(p.open(encoding='utf8')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'input_manifest.json').read_text())
for r in manifest['inputs']:
 p=P/r['local_path'];assert len(p.read_bytes())==r['bytes'] and sha(p)==r['sha256']
alias={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
source=rows(P/'inputs/common500_summary.csv')
def key(r):return(alias.get(r['method'],r['method']),r['snr_db'],r['metric'])
bank={key(r):r for r in source};assert len(bank)==len(source)==72
for r in rows(P/'figures/figure2_data.csv'):
 a=bank[key(r)]
 for f in ['mean','ci_low','ci_high']:assert r[f]==a[f]
assert len(rows(P/'figures/figure2_data.csv'))==36
for r in rows(P/'tables/table_ii.csv'):
 for metric,prefix in [('lpips_alex','lpips'),('clip_image_cosine','clip')]:
  a=bank[(r['method'],r['snr_db'],metric)]
  assert r[metric]==a['mean'] and r[prefix+'_ci_low']==a['ci_low'] and r[prefix+'_ci_high']==a['ci_high']
assert len(rows(P/'tables/table_ii.csv'))==18
t1=rows(P/'tables/table_i.csv');assert len(t1)==10
for r in t1:
 if r['block']=='A':
  expected=500 if r['population']=='common500' else 24;assert int(r['source_count'])==expected
  inputs=rows(P/('inputs/internal_'+r['population']+'_paired.csv'))
  a=[a for a in inputs if a['snr_db']==r['snr_db'] and a['metric']=='psnr_db' and(a.get('comparison')=='PARTIAL_minus_WHOLE' or a.get('method')=='NEST_PARTIAL' and a.get('reference')=='NEST_WHOLE')];assert len(a)==1
 else:
  assert r['source_count']=='100'and r['snr_db']=='10';a=[a for a in rows(P/'inputs/a_summary.csv')if a['arm']==r['grid']and a['snr_db']=='10'and a['metric']=='psnr_db'];assert len(a)==1
 for field,old in [('psnr_mean','mean'),('ci_low','ci_low'),('ci_high','ci_high')]:assert r[field]==a[0][old]
images=rows(P/'figure3_metrics.csv');assert len(images)==8
assert {(r['source_index'],r['method'])for r in images}=={(i,m)for i in ('95','21')for m in ['NEST','ADA_PREDCLASS_FIXEDFRAME','P100000','ARPC']}
for r in images:
 assert r['N']=='1024'and float(r['snr_db'])==10
 assert sha(P/r['image_path'])==r['image_png_sha256']
 assert r['native_file_sha256']and r['source_id']and r['noise_seed']
text='\n'.join(p.read_text()for p in(P/'sections').glob('*.tex'))
refs=(P/'references.tex').read_text();cited={k for g in re.findall(r'\\cite\{([^}]+)\}',text)for k in g.split(',')}
assert cited==set(re.findall(r'\\bibitem\{([^}]+)\}',refs))
a=(P/'sections/abstract.tex').read_text().split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
a=a.replace('~',' ').replace('\\%','%');count=len(a.split());assert 75<=count<=100,count
check=dict(status='PASS',baseline_commit=manifest['base_commit'],pinned_inputs=len(manifest['inputs']),frozen_summary_mean_ci_rows=72,figure2_exact_mean_ci_rows=36,table_ii_exact_mean_ci_rows=36,table_i_main_rows=6,table_i_diagnostic_rows=4,figure3_reconstructions=8,figure3_sources=[95,21],abstract_words_whitespace_convention=count,cited_references=len(cited),new_bootstrap=False,new_main_grid=False)
(P/'data_validation.json').write_text(json.dumps(check,indent=2)+'\n',newline='\n')
print(json.dumps(check))
