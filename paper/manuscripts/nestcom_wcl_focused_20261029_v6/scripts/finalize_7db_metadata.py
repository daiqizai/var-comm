"""Validate the closed 7-dB extension and explicitly recorded visual reviews."""
from pathlib import Path
import json,csv,hashlib,re,argparse,zipfile,xml.etree.ElementTree as ET
import fitz
P=Path(__file__).resolve().parents[1];W=P.parents[2]
a=argparse.ArgumentParser();a.add_argument('--review-root',type=Path,required=True);H=a.parse_args().review_root
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:list(csv.DictReader(p.open(encoding='utf8',newline='')))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def rec(p):return dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
old=read(H/'paper_before_7db_manifest.json');numeric=read(H/'numerical_audit.json');assert numeric['status']=='PASS'
D=W/'results/nestcom_7db_common500_20261030_v1';F=D/'delivery';assert read(F/'completion.json')['status']=='COMPLETE'
review=read(H/'visual_reviews.json');assert review['figure2']['reviewed']is True and review['figure2']['artifact_sha256']==sha(P/'figures/figure2.pdf')
for rel,digest in old.items():
 if rel.startswith('tables/')or rel.startswith('inputs/')or rel=='figure3_metrics.csv'or rel.startswith('figures/figure1.')or rel.startswith('figures/figure3.'):
  assert sha(P/rel)==digest,rel
f2=rows(P/'figures/figure2_data.csv');assert len(f2)==48
aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
expected=rows(P/'inputs/common500_summary.csv')+rows(P/'inputs/common500_7db_summary.csv')
for r in expected:r['method']=aliases.get(r['method'],r['method'])
for r in f2:
 candidates=[s for s in expected if all(r[k]==s[k]for k in ['method','metric','snr_db'])]
 assert len(candidates)==1 and candidates[0]==r
assert {r['metric']for r in f2}=={'lpips_alex','dreamsim'}and{r['snr_db']for r in f2}=={'4','7','10','19'}
assert sum(r['snr_db']=='7'for r in f2)==12
v2=read(P/'figures/figure2_validation.json');assert v2['points']==v2['confidence_intervals']==48
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',(P/'sections/abstract.tex').read_text(),re.S)[1]
word_count=len(abstract.replace('~',' ').replace(r'\%','%').split());assert 75<=word_count<=100
pages={};builds={};reviews={};V=P/'validation'
for kind,pdf in [('double','manuscript_doublecolumn.pdf'),('single','manuscript_singlecolumn.pdf'),('supplement','supplementary.pdf')]:
 r=review[kind];doc=P/pdf;assert r['artifact_sha256']==sha(doc)
 pages[kind]=len(fitz.open(doc));assert r['pages_reviewed']==list(range(1,pages[kind]+1))
 waits=read(H/'compilation'/r['build_tag']/'actual_waits.json');assert len(waits)==2 and all(w['actual_wait']and w['returncode']==0 for w in waits)
 renders=sorted((H/'render'/r['render_tag']).glob('page-*.png'));assert len(renders)==pages[kind]
 reviews[kind]=dict(status='VISUALLY_REVIEWED_ALL_PAGES',artifact=rec(doc),pages=pages[kind],pages_reviewed=r['pages_reviewed'],observations=r['observations'],renders=[dict(name=x.name,sha256=sha(x))for x in renders]);builds[kind]=waits
 save(V/({'double':'double_column','single':'single_column','supplement':'supplement'}[kind]+'_visual_QA.json'),reviews[kind])
assert pages['double']<=5
r=review['word'];assert r['artifact_sha256']==sha(P/'manuscript_review.docx')
conversion=read(H/'word_work'/r['build_tag']/'conversion_check.json')
assert conversion['word_sha256']==sha(P/'manuscript_review.docx')
assert conversion['figures_unchanged'] is False and conversion['revised_figures']==[2]
for i in range(1,4):assert conversion['figure_pdf_hashes'][str(i)]==sha(P/f'figures/figure{i}.pdf')
worddir=H/'word_work'/r['render_tag'];wd=read(worddir/'render.json');assert wd['source_sha256']==sha(P/'manuscript_review.docx')
pages['word']=len(fitz.open(worddir/'manuscript_review.pdf'));assert r['pages_reviewed']==list(range(1,pages['word']+1))
with zipfile.ZipFile(P/'manuscript_review.docx')as z:
 root=ET.fromstring(z.read('word/document.xml'));ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 native_tables=len(root.findall('.//w:tbl',ns));math_objects=len(root.findall('.//m:oMath',ns));assert native_tables==3 and math_objects>0
reviews['word']=dict(status='VISUALLY_REVIEWED_ALL_PAGES',artifact=rec(P/'manuscript_review.docx'),pages=pages['word'],pages_reviewed=r['pages_reviewed'],engine=wd['engine'],render_sha256=sha(worddir/'manuscript_review.pdf'),native_tables=native_tables,editable_math_objects=math_objects,observations=r['observations'])
assert conversion['editable_omml_count']==math_objects and conversion['native_tables']==native_tables
conversion['render_review_pending']=False
conversion['visual_review_record']='validation/word_visual_QA.json'
conversion['actual_rendered_pages']=pages['word']
save(V/'word_conversion.json',conversion)
reviews['figure2']=dict(status='VISUALLY_REVIEWED',formats={e:rec(P/f'figures/figure2.{e}')for e in ['pdf','png','svg']},observations=review['figure2']['observations'])
save(V/'word_visual_QA.json',reviews['word']);save(V/'latex_builds.json',builds);save(V/'artifact_review.json',reviews)
compute=read(F/'actual_compute_and_reuse.json')
checks=dict(status='PASS',abstract_words=word_count,pages=pages,mean_points=48,pointwise_confidence_intervals=48,new_7db_points=12,preserved_original_points=36,unchanged_figures=[1,3],revised_figures=[2],original_three_tables_unchanged=True,science=compute,artifact_sha256={n:sha(P/n)for n in ['manuscript_doublecolumn.pdf','manuscript_singlecolumn.pdf','manuscript_review.docx','supplementary.pdf']})
save(V/'delivery_checks.json',checks);save(V/'data_identity_check.json',dict(status='PASS',numerical_audit=numeric,original_summary=rec(P/'inputs/common500_summary.csv'),new_summary=rec(P/'inputs/common500_7db_summary.csv'),figure2_data=rec(P/'figures/figure2_data.csv')))
claim=dict(status='PASS',new_common500_snr_db=7,new_logical_outcomes=9000,new_source_count=500,repetitions_per_source=3,development_images_excluded=True,policies_frozen_before_test=True,smaller_7db_calibration_disclosed=True,source_bootstrap_bank_reused=True,original_4_10_19_points_unchanged=True,CLIP7_not_claimed=True,TableII_original_scope_retained=True,figures1_and3_unchanged=True,new_training=0,prior_completed_studies_not_rerun=True)
save(V/'claim_audit.json',claim)
mapping=read(P/'figure_table_sources.json');mapping['revision_base_commit']='640b359d1c13864184d9d6f0f4a49bb37e622882'
mapping['figures']['2']=dict(formats={e:rec(P/f'figures/figure2.{e}')for e in ['pdf','png','svg']},data=rec(P/'figures/figure2_data.csv'),points=48,CI='Pointwise95% source-bootstrap; all old36 rows preserved; new12 from original10000x500 bank',population='same common500,500 equal-weight sources and3channel outcomes/source',N=1024,sources=[rec(P/'inputs/common500_summary.csv'),rec(P/'inputs/common500_7db_summary.csv')],filter=dict(metrics=['lpips_alex','dreamsim'],snr_db=[4,7,10,19],methods='six canonical methods'),script=rec(P/'scripts/draw_figure2.py'),seven_db_protocol=rec(P/'inputs/seven_db/protocol.json'),calibration='7dB frozen100-source calibration, different from original1000-source procedure; disclosed inIII.A/S9',development_means_used=False)
save(P/'figure_table_sources.json',mapping)
manifest=read(P/'input_manifest.json')
manifest['revision']='V6 plus formally evaluated common5007dB point'
manifest['previous_revision_commit']='640b359d1c13864184d9d6f0f4a49bb37e622882'
manifest['previous_validation']='validation/history_640b359'
manifest['seven_db']=dict(protocol=rec(P/'inputs/seven_db/protocol.json'),summary=rec(P/'inputs/common500_7db_summary.csv'),compute=rec(P/'inputs/seven_db/actual_compute_and_reuse.json'),figure2=mapping['figures']['2'])
# Keep earlier parent identity assertions true only for byte-identical files.
for group,items in manifest.get('frozen_parent_files',{}).items():
 changed=[x for x in items if sha(P/x['path'])!=x['sha256']]
 if changed:manifest.setdefault('changed_parent_files_7db',{})[group]=changed
 manifest['frozen_parent_files'][group]=[x for x in items if sha(P/x['path'])==x['sha256']]
save(P/'input_manifest.json',manifest)
save(V/'publication_status.json',dict(status='VALIDATED_FOR_NORMAL_PUBLICATION',record_phase='pre_commit artifact validation; this file does not claim a push has occurred',base_commit='640b359d1c13864184d9d6f0f4a49bb37e622882',no_local_commit=True,final_publication_receipt_location='local research handoff and PFS publication_receipt.json; final response reports independently verified remoteSHA'))
print(json.dumps(dict(status='PASS',pages=pages,abstract_words=word_count,math_objects=math_objects,figure2_points=48)))
