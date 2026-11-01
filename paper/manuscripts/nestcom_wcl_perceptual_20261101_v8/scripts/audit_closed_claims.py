"""Audit closed-result claims without models, resampling or experiment calls."""
from pathlib import Path
import csv,hashlib,json,re
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def one(vs,**kw):
    a=[v for v in vs if all(str(v[k])==str(x) for k,x in kw.items())]
    assert len(a)==1,(kw,len(a));return a[0]
def favorable(v):
    return float(v['ci95_low'])>0 if v['metric'] in ['psnr_db','clip_image_cosine'] else float(v['ci95_high'])<0
def interval(v):return {k:v[k] for k in ['mean_delta','ci95_low','ci95_high']}
e1=rows(P/'inputs/e1/paired_deltas.csv');s1=rows(P/'inputs/e1/summary.csv')
e3=rows(P/'inputs/e3/paired_deltas.csv');e4=rows(P/'inputs/e4/paired_deltas.csv')
assert len(e1)==220 and len(s1)==220 and len(e3)==30 and len(e4)==30
assert all(v['n_sources']=='500' for v in e1+s1)
assert all(v['n_sources']=='100' for v in e3+e4)
core=[v for v in e1 if v['candidate']=='final_NeST' and v['reference']=='WHOLE_final_receiver']
assert len(core)==35 and all(map(favorable,core))
quarter=[v for v in e1 if v['candidate']=='final_NeST' and v['reference']=='original_NeST' and v['metric'] in ['dists','dreamsim']]
assert len(quarter)==14 and sum(map(favorable,quarter))==12
null_quarter=[v for v in quarter if not favorable(v)]
assert all(v['N']=='1024' and v['snr_db']=='7' and float(v['ci95_low'])<=0<=float(v['ci95_high']) for v in null_quarter)
mixed=[]
for N,snr,m in [(1024,10,'dists'),(1024,10,'dreamsim'),(1024,19,'dists'),(512,19,'dists'),(512,19,'dreamsim')]:
    v=one(e1,candidate='final_NeST',reference='Ada_frozen',N=N,snr_db=snr,metric=m)
    assert float(v['ci95_low'])>0
    mixed.append(dict(N=N,snr_db=snr,metric=m,interpretation='Ada favored',**interval(v)))
for N,snr,m in [(1024,7,'dreamsim'),(1024,19,'dreamsim'),(512,10,'clip_image_cosine'),(512,19,'clip_image_cosine')]:
    v=one(e1,candidate='final_NeST',reference='Ada_frozen',N=N,snr_db=snr,metric=m)
    assert float(v['ci95_low'])<=0<=float(v['ci95_high'])
    mixed.append(dict(N=N,snr_db=snr,metric=m,interpretation='interval spans zero',**interval(v)))
abstract=(P/'sections/abstract.tex').read_text(encoding='utf8').split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
words=re.findall(r'\S+',abstract)
assert 75<=len(words)<=100
cls=re.search(r'\\documentclass\[([^]]+)\]\{IEEEtran\}',(P/'main.tex').read_text(encoding='utf8'))
assert cls and {'journal','10pt'}<=set(cls[1].split(',')) and 'onecolumn' not in cls[1].split(',')
checks={
 'status':'NUMERIC_CLAIMS_PASS_VISUAL_REVIEW_SEPARATE',
 'closed_evidence':['E1','E2','E3','E4','E5','E6'],'pending_evidence':[],
 'base_commit':'231c3e4b5b8a975330a4baf6fd3ad6116c6d7623',
 'abstract_word_count_whitespace':len(words),
 'E1_summary_rows':len(s1),'E1_direct_paired_rows':len(e1),
 'NeST_minus_WHOLE_all_7_cells_all_5_metrics_directional_CI':True,
 'NeST_minus_Quarter_DISTS_DreamSim_6_of_7_cells':True,
 'unfavorable_and_nonsignificant_Ada_findings':mixed,
 'E3_mean_rows':len(rows(P/'inputs/e3/summary.csv')),
 'E3_paired_rows':len(e3),'E4_mean_rows':len(rows(P/'inputs/e4/summary.csv')),'E4_paired_rows':len(e4),
 'numeric_inputs':{p.relative_to(P).as_posix():sha(p) for p in sorted((P/'inputs').rglob('*.csv'))},
 'no_models_no_PHY_no_bootstrap_rerun':True,
 'not_checked_here':['final rendered pages','repository checks','publication']}
e2=rows(P/'inputs/e2/paired_deltas.csv');e6=rows(P/'inputs/e6/summary.csv');e5=rows(P/'inputs/e5/timing_summary.csv')
assert len(e2)==90 and len(e6)==15 and len(e5)==48
e2_internal=[v for v in e2 if v['reference']=='WHOLE']
assert len(e2_internal)==30 and all(map(favorable,e2_internal))
for stage,filename in [('e2','summary.csv'),('e5','timing_summary.csv'),('e6','summary.csv')]:
 c=json.loads((P/'inputs'/stage/'execution_closure.json').read_text())
 assert c['actual_wait_returncode']==0 and c['owner_gone'] and c['child_gone']
 assert c['summary_sha256']==sha(P/'inputs'/stage/filename)
for v in e6:
 if float(v['snr_db'])==8:assert float(v['frame_rejection_rate'])==1
 if float(v['snr_db'])==10:assert float(v['frame_rejection_rate'])==0
checks.update(E2_paired_rows=90,E5_summary_rows=48,E5_groups=16,E6_summary_rows=15,
 E2_all_30_NeST_WHOLE_intervals_favorable=True,E6_all_reject_8dB_zero_observed_10dB=True)
body='\n'.join(p.read_text()for p in (P/'sections').glob('*.tex'))
assert 'Draft completion gate'not in body and 'awaits actual measurements'not in body
(P/'validation/claim_audit.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
(P/'validation/closed_claim_audit.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:checks[k] for k in ['status','abstract_word_count_whitespace','E1_summary_rows','E1_direct_paired_rows']}))
