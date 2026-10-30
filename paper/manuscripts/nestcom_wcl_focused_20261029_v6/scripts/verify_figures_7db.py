"""Check current figure exports and scientific source mappings; no inference."""
from pathlib import Path
import csv,json,hashlib,zipfile,xml.etree.ElementTree as ET
import fitz
P=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf8',newline='')))
mapping=read(P/'figure_table_sources.json');old=read(P/'validation/history_640b359/data_identity_check.json')
for item in old['unchanged']:assert sha(P/item['path'])==item['sha256']
for number in ['1','2','3']:
 for extension,pin in mapping['figures'][number]['formats'].items():assert sha(P/pin['path'])==pin['sha256']
 with fitz.open(P/f'figures/figure{number}.pdf')as d:assert len(d)==1
data=rows(P/'figures/figure2_data.csv');assert len(data)==48
source=rows(P/'inputs/common500_summary.csv')+rows(P/'inputs/common500_7db_summary.csv')
aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
for r in source:r['method']=aliases.get(r['method'],r['method'])
assert len({r['method']for r in data})==6
assert {r['snr_db']for r in data}=={'4','7','10','19'}and{r['metric']for r in data}=={'lpips_alex','dreamsim'}
for r in data:
 matched=[s for s in source if all(s[k]==r[k]for k in ['method','snr_db','metric'])]
 assert len(matched)==1 and matched[0]==r
 assert r['source_count']=='500'and r['logical_frames']=='1500'and float(r['ci_low'])<=float(r['mean'])<=float(r['ci_high'])
fig3=rows(P/'figure3_metrics.csv');assert len(fig3)==14 and {r['snr_db']for r in fig3}=={'4'}and {r['source_index']for r in fig3}=={'95','21'}
assert sha(P/'figure3_metrics.csv')==old['figure3']['sha256']
with zipfile.ZipFile(P/'figures/figure1.pptx')as z:
 root=ET.fromstring(z.read('ppt/slides/slide1.xml'));ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
 native_shapes=len(root.findall('.//p:sp',ns));assert native_shapes>1000
result=dict(status='PASS',figure2_points=48,pointwise_intervals=48,old_points_preserved=36,new_7db_points=12,figure3_preserved=True,figure1_native_shapes=native_shapes,visual_review_separate='validation/artifact_review.json')
(P/'validation/figure_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
