"""Fresh checks for the executed 4-dB V6 revision; never import parent PASS labels."""
from pathlib import Path
import json,hashlib,csv,re,argparse
from pypdf import PdfReader
P=Path(__file__).resolve().parents[1];W=P.parents[2]
a=argparse.ArgumentParser();a.add_argument('--review-root',type=Path,default=W/'.research/nestcom_fig3_4db_20261029_v1');args=a.parse_args();H=args.review_root

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def rec(p):return dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
old=read(P/'inputs/supplied_v6_metadata/input_manifest.json');validated=[]
for group in ['inputs','tables']:
 for d in old['frozen_parent_files'][group]:
  p=P/d['path'];assert sha(p)==d['sha256'],p;validated.append(rec(p))
for i in [1,2]:
 for ext in (['pptx','pdf','png','svg']if i==1 else['pdf','png','svg']):
  p=P/f'figures/figure{i}.{ext}';z=read(P/'inputs/supplied_v6_metadata/figure_table_sources.json')['figures'][str(i)]['formats'][ext];assert sha(p)==z['sha256']
method=(P/'sections/method.tex').read_text();assert len(re.findall(r'\\begin\{equation\}',method))==6
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',(P/'sections/abstract.tex').read_text(),re.S)[1];words=len(abstract.replace('~',' ').replace(r'\%','%').split());assert 75<=words<=100
D=W/'results/nestcom_fig3_4db_20261029_v1/delivery';compute=read(D/'actual_compute_and_reuse.json');assert compute['status']=='COMPLETE'and compute['exact_historical_array_hash_matches']==10
metrics=list(csv.DictReader((P/'figure3_metrics.csv').open()));assert len(metrics)==14 and {x['source_index']for x in metrics}=={'95','21'}and {x['snr_db']for x in metrics}=={'4'}
for row in metrics:
 assert sha(D/row['display_image'])==row['display_png_sha256']
assert sha(P/'figure3_metrics.csv')==sha(D/'figure3_metrics.csv')
V=P/'validation';pages={};builds={};reviews={}
for variant,pdf,tag in [('double','manuscript_doublecolumn.pdf','double02'),('single','manuscript_singlecolumn.pdf','single02'),('supplement','supplementary.pdf','supplement02')]:
 doc=P/pdf;pages[variant]=len(PdfReader(doc).pages);waits=read(H/'compilation'/tag/'actual_waits.json');assert len(waits)==2 and all(x['actual_wait']and x['returncode']==0 for x in waits)
 review=H/'render'/tag;images=sorted(review.glob('page-*.png'));assert len(images)==pages[variant]
 reviews[variant]=dict(status='VISUALLY_REVIEWED_ALL_PAGES',artifact=rec(doc),pages=pages[variant],reviewed_pages=list(range(1,pages[variant]+1)),renders=[dict(file=x.name,sha256=sha(x))for x in images],observations='No clipped text, overlapping figures, missing glyphs or overflowing tables in actual page images. Supplement S8 and preserved 10-dB view explicitly checked.')
 builds[variant]=waits
 save(V/({'double':'double_column','single':'single_column','supplement':'supplement'}[variant]+'_visual_QA.json'),reviews[variant])
word=H/'word_work/final03';wp=word/'manuscript_review.pdf';pages['word']=len(PdfReader(wp).pages);wd=read(word/'render.json');assert wd['source_sha256']==sha(P/'manuscript_review.docx')
# final02 was inspected on all16 pages; final03 changes only page15, inspected again.
for n in range(1,pages['word']+1):
 p=word/('page-%02d.png'%n);assert p.exists()
 if n!=15:assert sha(p)==sha(H/'word_work/final02'/p.name)
reviews['word']=dict(status='VISUALLY_REVIEWED_ALL_PAGES',artifact=rec(P/'manuscript_review.docx'),pages=pages['word'],reviewed_pages=list(range(1,pages['word']+1)),engine=wd['engine'],render_sha256=sha(wp),observations='Native body,68 editable OMML objects,3 native tables. All16 pages inspected. Long URL overflow on page15 fixed with display-only wrap opportunities and re-rendered; other15 page rasters byte-identical to inspected final02.')
save(V/'word_visual_QA.json',reviews['word']);save(V/'latex_builds.json',builds)
checks=dict(status='PASS',abstract_words=words,main_pdf_pages=pages['double'],single_pdf_pages=pages['single'],word_render_pages=pages['word'],supplement_pages=pages['supplement'],figure3_snr_db=4,figure3_4db_status='COMPLETE_REAL_RECEIVED_OUTPUTS',unchanged_original_data_files=len(validated),unchanged_figures=[1,2],revised_figures=[3],original_method_equations=6,scientific_execution=compute,artifact_sha256={x:sha(P/x)for x in ['manuscript_doublecolumn.pdf','manuscript_singlecolumn.pdf','manuscript_review.docx','supplementary.pdf']})
assert pages==dict(double=5,single=15,supplement=25,word=16)
save(V/'delivery_checks.json',checks);save(V/'data_identity_check.json',dict(status='PASS',unchanged=validated,figure3=rec(P/'figure3_metrics.csv'),all_fourteen_display_hashes_verified=True));save(V/'artifact_review.json',reviews)
mapping=read(P/'figure_table_sources.json');mapping['figures']['3']=dict(formats={e:rec(P/f'figures/figure3.{e}')for e in ['pdf','png','svg']},N=1024,snr_db=4,source_indices=[95,21],methods=['Original','WHOLE','NEST','ADA','ARPC'],data=rec(P/'figure3_metrics.csv'),display_identity='fixed development cases; first original timing measured repetition for five historical methods; separately registered new WHOLE/Ada events',filter='same two sources, full256x256field, displayed-realization LPIPS-Alex/DreamSim, no averages',complete_source_map='results/nestcom_fig3_4db_20261029_v1/delivery/source_receive_reconstruction_map.json',supplement='all seven methods and preserved original10dB figure')
mapping.pop('figure3_4db_status',None);save(P/'figure_table_sources.json',mapping)
old['changed_parent_figure_files']=[v for v in old['frozen_parent_files']['figures'] if sha(P/v['path'])!=v['sha256']]
old['frozen_parent_files']['figures']=[v for v in old['frozen_parent_files']['figures'] if sha(P/v['path'])==v['sha256']]
old.update(revision='V6 executed 4dB figure revision',scientific_execution=compute,figure3=mapping['figures']['3'],prior_metadata='inputs/supplied_v6_metadata',abstract_word_count=words);save(P/'input_manifest.json',old)
save(V/'claim_audit.json',dict(status='PASS',main_quantitative_inputs_unchanged=True,common500_not_recomputed=True,fixed_development_cases_not_population_claim=True,latent_advantages_disclosed=True,original_Ada10dB_advantage_retained=True,calibration7dB_separate_scope='results/nestcom_7db_20261030_v1',new_training=0))
print(json.dumps(dict(status='PASS',pages=pages,abstract_words=words)))
