"""Verify and describe figure identities, without neural/PHY/scoring work."""
from pathlib import Path
import csv,hashlib,json,zipfile,xml.etree.ElementTree as ET
import pymupdf
P=Path(__file__).resolve().parents[1];old=P.parent/'nestcom_wcl_focused_20261029_v2'
if (P/'inputs/common500_7db_summary.csv').exists():
 raise RuntimeError('Historical figure verifier is inapplicable; use verify_figures_7db.py for the current revision.')
V=P/'validation/figures';V.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref='e2ed72c95ddeab15cc4ca42b1dd7f2e0e5236c8c'
fig2={}
for p in (P/'figures').glob('figure2*'):
 fig2[p.name]=sha(p)
r2=list(csv.DictReader((P/'figures/figure2_data.csv').open()));assert len(r2)==36
assert len({r['method'] for r in r2})==6
assert all(r['ci_low'] and r['ci_high'] for r in r2)
assert {r['metric']for r in r2}=={'lpips_alex','dreamsim'}
source=list(csv.DictReader((P/'inputs/common500_summary.csv').open()))
aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
for r in source:r['method']=aliases.get(r['method'],r['method'])
for r in r2:
 match=[x for x in source if all(x[k]==r[k]for k in ('method','snr_db','metric'))]
 assert len(match)==1 and match[0]==r
for n in (1,3):
 for ext in ('.pdf','.svg','.png'):
  assert (P/f'figures/figure{n}{ext}').read_bytes()==(old/f'figures/figure{n}{ext}').read_bytes()
assert (P/'figures/figure1.pptx').read_bytes()==(old/'figures/figure1.pptx').read_bytes()
rows=list(csv.DictReader((P/'figure3_metrics.csv').open()));original=list(csv.DictReader((old/'figure3_metrics.csv').open()))
assert len(rows)==len(original)==8
for a,b in zip(rows,original):
 assert {k:v for k,v in a.items()if k!='image_path'}=={k:v for k,v in b.items()if k!='image_path'}
 assert sha(P/a['image_path'])==a['image_png_sha256']
 assert int(a['N'])==1024 and float(a['snr_db'])==10
assert {r['source_id']for r in rows}=={'n07860988/ILSVRC2012_val_00010848_n07860988','n02105641/ILSVRC2012_val_00018762_n02105641'}
assert {r['method']for r in rows}=={'NEST','ADA_PREDCLASS_FIXEDFRAME','P100000','ARPC'}
with zipfile.ZipFile(old/'figures/figure1.pptx')as a,zipfile.ZipFile(P/'figures/figure1.pptx')as b:
 oldmedia={hashlib.sha256(a.read(n)).hexdigest() for n in a.namelist()if n.startswith('ppt/media/')}
 newmedia={hashlib.sha256(b.read(n)).hexdigest() for n in b.namelist()if n.startswith('ppt/media/')}
 assert oldmedia<=newmedia
 root=ET.fromstring(b.read('ppt/slides/slide1.xml'));ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
 native=dict(native_shapes=len(root.findall('.//p:sp',ns)),native_connectors=len(root.findall('.//p:cxnSp',ns)),embedded_pictures=len(root.findall('.//p:pic',ns)),source_bitmap_hashes_preserved=sorted(oldmedia),not_single_screenshot=True)
assert native['native_shapes']>1000 and native['embedded_pictures']==2
figs={}
for number in (1,2,3):
 with pymupdf.open(P/f'figures/figure{number}.pdf')as d:
  assert len(d)==1;dims=[d[0].rect.width/72,d[0].rect.height/72]
 figs[f'figure{number}']=dict(size_inches=dims,assets={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in (P/'figures').glob(f'figure{number}.*') if p.suffix in ('.pdf','.png','.svg','.pptx')})
result=dict(source_commit=ref,figure1=dict(source='paper/manuscripts/nestcom_wcl_focused_20261029_v2/figures/figure1.pptx',source_sha256=sha(old/'figures/figure1.pptx'),native_editability=native,editable_text_source='figures/figure1.pptx',svg_text_outlined=True,source_images_unchanged=True),figure2=dict(mean_count=36,interval_count=36,method_count=6,all_asset_bytes_unchanged=False,metrics=['lpips_alex','dreamsim'],exact_frozen_input_rows=True,assets=fig2),figure3=dict(population='two existing development examples; not common500',N=1024,snr_db=10,rows=2,columns=5,source_order=['Pastries','Dog'],methods=['Original','NEST','ADA_PREDCLASS_FIXEDFRAME','P100000','ARPC'],metric_order=['lpips_alex','dreamsim'],full_precision_metrics_csv='figure3_metrics.csv',metrics_preserved_exactly=True,source_metrics_sha256=sha(old/'figure3_metrics.csv'),display_metric_decimals=3,native_arrays_not_reopened_this_revision=True,existing_native_hashes_retained_in_csv=True,no_shared_AWGN_claim=True,full_field=True,no_ROI=True,no_new_scientific_calls=True,inputs=rows),preserved_supplement=dict(lamp_roi='supplemental/N512_lamp_roi.pdf',lamp_roi_sha256=sha(P/'supplemental/N512_lamp_roi.pdf'),same_as_V2_supplement=True,previous_N512_two_cases='supplemental/N512_previous_figure3.pdf'),exports=figs)
assert (P/'supplemental/N512_lamp_roi.pdf').read_bytes()==(old/'supplemental/N512_lamp_roi.pdf').read_bytes()
(V/'figure_sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(figure2_count=len(r2),figure3_rows=len(rows),native=native,dimensions={k:v['size_inches']for k,v in figs.items()}),ensure_ascii=False))
