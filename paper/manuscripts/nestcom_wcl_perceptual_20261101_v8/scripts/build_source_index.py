"""Record exact figure/table input identities without recomputing results."""
from pathlib import Path
import hashlib,json,re
P=Path(__file__).resolve().parents[1]
BASE='231c3e4b5b8a975330a4baf6fd3ad6116c6d7623'
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM'
D='results/nestcom_perceptual_followup_20261030_v2'
def pin(p):return {'path':p.relative_to(P).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def caption(path):
    s=path.read_text(encoding='utf8');start=s.index(r'\caption{')+len(r'\caption{');n=1
    for j in range(start,len(s)):
        if s[j]=='{' and (j==0 or s[j-1]!='\\'):n+=1
        if s[j]=='}' and (j==0 or s[j-1]!='\\'):n-=1
        if not n:return s[start:j]
    raise ValueError(path)
items=[
 ('Fig. 1','sections/method.tex',['inputs/figure1_v7_editable.pptx','inputs/e1/frozen_policy_metadata.json'],
  'Native PPTX; same geometry; schematic capacity bars and paid 12-bit profile metadata. No empirical utilization bars.',None),
 ('Fig. 2','sections/figure2_float.tex',['inputs/e1/summary.csv','style.json'],
  'N=1024; SNR=4,7,10,19; final_NeST/Ada_frozen/ARPC/SwinJSCC/Latent_JSCC/BPG_LDPC; metrics dists,dreamsim; mean/ci95_low/ci95_high.',500),
 ('Fig. 3','sections/figure3_float.tex',['inputs/f1/fig3_metrics.csv','inputs/f1/selected_native_mapping.json','inputs/f1/selected_examples.json'],
  'Sources84/143; N1024/7dB; original channel_repeat0; Source/WHOLE_final_receiver/Ada_frozen/final_NeST; displayed DISTS/DreamSim, not source-average quality.',2),
 ('Table I','tables/table_i.tex',['tables/table_i.csv','inputs/e1/paired_deltas.csv','inputs/e1/capacity_summary.csv','inputs/e3/summary.csv','inputs/e3/paired_deltas.csv','inputs/e3/efficiency.csv','inputs/e4/paired_deltas.csv'],
  'A:500sources,1024all4SNR,final_NeST-WHOLE; B:100sources,1024/10,fixed8,126,q4r2/3; C:100sources,10dB,B0-A0/C0-B0.',None),
 ('Table II','tables/table_ii.tex',['tables/table_ii.csv','inputs/e1/summary.csv','inputs/e5/timing_summary.csv','inputs/e5/execution_closure.json'],
  'Quality:common500,N1024/10,PSNR/LPIPS/CLIP. Cost:timing32,same N/SNR,actual TX/RX equal-source means in seconds converted to ms; group map is explicit in build_evidence_tables.py.',500),
 ('Table III','tables/table_iii.tex',['tables/table_iii.csv','inputs/e1/summary.csv','inputs/e1/paired_deltas.csv'],
  'common500,N512,4/10/19,WHOLE/final_NeST/Ada; PSNR/DISTS/DreamSim; direct final_NeST-WHOLE PSNR CIs.',500)]
index=[];caps=['# Captions — V8','',
 'Extracted from the current TeX. Figure numbering and supplement section numbering are distinct. Artifact inspection is recorded separately.','']
for title,tex,ins,condition,n in items:
    c=caption(P/tex)
    index.append(dict(label=title,tex=pin(P/tex),caption=c,inputs=[pin(P/f) for f in ins],filter=condition,n_sources=n,
       uncertainty=('Quality uses source-bootstrap intervals; timing gives empirical mean/median/p95, without CI.' if title=='Table II' else 'Direct source-paired percentile intervals for differences; pointwise source-bootstrap intervals for means; 10,000 registered resamples; Fig.3 has realization values, not intervals.')))
    caps.extend(['## '+title,'',c,''])
index.append({'label':'Supplementary Fig. S3','inputs':[pin(P/f) for f in ['inputs/f1/selected_native_mapping.json','inputs/f1/external_native_mapping.json','inputs/f1/external_display_metrics.csv']],
 'filter':'Same two full source IDs,7dB,original repeat0; all9columns including source,oldQuarter and 4external systems.',
 'native_outputs':'figures/individual/*; exact exported PNG bytes pinned by native mapping; metrics from original native arrays, never PNG requantization.'})
for label,stage,note in [('Supplementary Fig. S1','e2','common500,6 total information budgets;5 metric means and direct pairs; original10000x500bank'),('Supplementary Fig. S2','e6','control100,5 SNR,20 noise repeats; source-clustered overall and successful-only quality')]:
    index.append(dict(label=label,inputs=[pin(P/'inputs'/stage/'summary.csv'),pin(P/'inputs'/stage/'execution_closure.json')],filter=note))
record={'status':'COMPLETE_INPUT_INDEX_VISUAL_REVIEW_SEPARATE','repository_root':R,'base_commit':BASE,'new_commit':'See Git history and final publication receipt',
 'experiment_result_directory':D,'original_results_unchanged':True,
 'manuscript_configuration':'original_receiver_plus_D_token',
 'artifacts':index,'exact_input_files':[pin(p) for p in sorted((P/'inputs').rglob('*')) if p.is_file()],
 'artifact_sources':[pin(p) for p in sorted((P/'scripts').glob('*')) if p.is_file()],
 'figure_outputs':[pin(p) for p in sorted((P/'figures').glob('*')) if p.is_file()],
 'figure1_editability':'PPTX native editable text/shapes. PDF/SVG exported; SVG fonts are paths.',
 'figure2_3_editability':'Matplotlib source plus SVG text; native reconstruction pixels preserved.',
 'pending':[], 'visual_acceptance':'validation/artifact_review.json; index alone is not visual PASS'}
(P/'figure_table_sources.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
(P/'captions.md').write_text('\n'.join(caps),encoding='utf8')
print(json.dumps({'indexed_main_items':len(items),'input_files':len(record['exact_input_files']),'status':record['status']}))
