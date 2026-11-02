"""Review frozen Fig.3 candidates. No model, metric or bootstrap execution.

Inputs: inventory.json (original CSV/protocol), mapped_frames.json (identity-checked
repeat-zero records), and display_tiles.zip (hash-pinned saved PNG blocks).
Outputs: complete eligible galleries, numerical audit, PDFs and local HTML.
"""
from pathlib import Path
import json,csv,io,zipfile,hashlib,html,math,sys,argparse
from PIL import Image
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
H=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=H)
parser.add_argument('--tiles-zip',type=Path,default=H/'display_tiles.zip')
parser.add_argument('--output-dir',type=Path,default=Path('C:/Users/11946/Downloads/NeST_Com_Fig3_Candidates_20261102'))
args=parser.parse_args();INPUT=args.input_dir;OUT=args.output_dir
OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir();(OUT/'preview').mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def csvout(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
inventory=json.loads((INPUT/'inventory.json').read_text());mapped=json.loads((INPUT/'mapped_frames.json').read_text());frames=mapped['frames']
assert inventory['head']==mapped['head']=='f03a295ed956b2a6dd376a032f088a5eaa7936f9'
indices=list(csv.DictReader(io.StringIO(inventory['files']['candidate_index.csv'])))
data={(v['snr_db'],v['source_id'],v['method']):v for v in frames};assert len(data)==3000
with zipfile.ZipFile(args.tiles_zip)as z:
 assert z.testzip()is None
 for n in z.namelist():
  p=OUT/n;assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(z.read(n))
extraction=json.loads((OUT/'extraction_identity.json').read_text());assert extraction['images']==664
for r in extraction['records']:assert sha(OUT/'tiles'/r['filename'])==r['sha256']
methods=[('WHOLE','WHOLE_final_receiver'),('Ada','Ada_frozen'),('NeST','final_NeST')]
metrics=[('dists',-1,'DISTS'),('dreamsim',-1,'DreamSim'),('psnr_db',1,'PSNR'),('lpips_alex',-1,'LPIPS'),('clip_image_cosine',1,'CLIP-I')]
allrows=[];cards=[]
for r in indices:
 s,idx=int(r['snr_db']),int(r['source_index']);g={label:data[s,r['source_id'],method]for label,method in methods}
 win={k:all((g['NeST'][k]-g[m][k])*direction>0 for m in ['WHOLE','Ada'])for k,direction,_ in metrics}
 leads=win['dists']and win['dreamsim'];assert leads==(r['both_metrics_both_comparators']=='True')
 gray=any(g[m]['image']is None for m in g)
 row=dict(snr_db=s,source_index=idx,source_id=r['source_id'],repeat_index=0,DISTS_DreamSim_lead=leads,all_five_lead=all(win.values()),has_gray=gray,original_candidate_rank=int(r['rank']))
 for label,f in g.items():
  assert f['source_pixels_sha256']==r['source_pixels_sha256']and f['reference_sha256']==r['reference_sha256']
  row[label+'_status']=f['receiver_status'];row[label+'_noise_seed']=f['noise_seed']
  for k,_,_ in metrics:row[label+'_'+k]=f[k]
 for k,_,_ in metrics:
  for m in ['WHOLE','Ada']:row['NeST_minus_'+m+'_'+k]=g['NeST'][k]-g[m][k]
 allrows.append(row)
 if leads:cards.append(dict(**row,frames=g,source_pixels_sha256=r['source_pixels_sha256'],reference_sha256=r['reference_sha256']))
cards.sort(key=lambda v:(v['snr_db'],v['has_gray'],not v['all_five_lead'],v['original_candidate_rank']))
for s in [7,10]:
 for i,v in enumerate([c for c in cards if c['snr_db']==s],1):v['candidate_id']=f'C{s}-{i:03d}'
counts={str(s):dict(total=500,two_metrics=sum(v['snr_db']==s for v in cards),five_metrics=sum(v['snr_db']==s and v['all_five_lead']for v in cards),five_metrics_no_gray=sum(v['snr_db']==s and v['all_five_lead']and not v['has_gray']for v in cards),gray_comparator=sum(v['snr_db']==s and v['has_gray']for v in cards))for s in [7,10]}
assert counts['7']['two_metrics']==143 and counts['10']['two_metrics']==23
csvout(OUT/'all_1000_candidates_audit.csv',allrows)
selected=[]
for v in cards:selected.append(dict(candidate_id=v['candidate_id'],**{k:v[k]for k in allrows[0]}))
csvout(OUT/'all_166_leading_candidates.csv',selected)
save(OUT/'candidate_metadata.json',cards)
save(OUT/'inputs/mapped_frames.json',dict(head=mapped['head'],source_input=mapped['source_input'],frames=frames))
save(OUT/'inputs/inventory.json',dict(head=inventory['head'],files=inventory['files']))
save(OUT/'selection_summary.json',dict(counts=counts,comparison='NeST versus WHOLE and Ada-TokenCom',primary_metrics='Both DISTS and DreamSim strictly lower, unrounded native scores',five_metric_badge='Also PSNR and CLIP-I strictly higher and LPIPS strictly lower than both comparators',ranking='non-gray first; all-five-leading first; then pre-existing illustration rank',population='The complete previously registered N1024/7 and N1024/10 candidate pool: 500 sources per SNR, first prescribed realization only',noise='WHOLE/NeST seed 6201; Ada seed 2001; historical independent namespaces retained',zero_new_model_PHY_metrics_bootstrap=True,selection_use='Illustrative advantage selection; aggregate common500 results unchanged',base_commit=mapped['head']))
# Native 256x256 image tiles are reused without enhancement or new cropping.
font=Path('C:/Windows/Fonts/arial.ttf');pdfmetrics.registerFont(TTFont('ReviewArial',str(font)))
page_manifest=[]
for s in [7,10]:
 group=[v for v in cards if v['snr_db']==s];path=OUT/f'Fig3_{s}dB_all_leading_candidates.pdf';c=canvas.Canvas(str(path),pagesize=A4);c.setTitle(f'NeST-Com Fig3: all qualifying {s} dB candidates')
 W,HH=A4;left=12*mm;side=44*mm;gap=3.3*mm;rowh=62*mm;firsttop=267*mm
 total=math.ceil(len(group)/4)
 for start in range(0,len(group),4):
  c.setFont('ReviewArial',13);c.drawString(left,HH-15*mm,f'Fig.3 candidates | N=1024, {s} dB | {start//4+1}/{total}')
  c.setFont('ReviewArial',8);c.drawString(left,HH-21*mm,'NeST strictly leads WHOLE and Ada on DISTS / DreamSim. FIVE = all five metrics lead.')
  c.setFont('ReviewArial',7.5);c.drawString(left,HH-26*mm,'Fixed first realization; native full field of view. Ordered by FIVE status, then original advantage rank.')
  ids=[]
  for row,v in enumerate(group[start:start+4]):
   top=firsttop-row*rowh;ids.append(v['candidate_id'])
   c.setFillColorRGB(.08,.23,.38);c.setFont('ReviewArial',9)
   badge='FIVE'if v['all_five_lead']else 'DISTS+DreamSim';gray=' | GRAY COMPARATOR'if v['has_gray']else ''
   c.drawString(left,top,f"{v['candidate_id']} | source {v['source_index']} | {badge}{gray}")
   c.setFillColorRGB(0,0,0)
   for col,label in enumerate(['Source','WHOLE','Ada','NeST']):
    x=left+col*(side+gap);y=top-49*mm;p=OUT/'tiles'/f"s{s:02d}_i{v['source_index']:03d}_{label}.png"
    c.setFont('ReviewArial',8);c.drawCentredString(x+side/2,top-4*mm,{'Ada':'Ada-TokenCom','NeST':'NeST-Com'}.get(label,label))
    c.drawImage(str(p),x,y,width=side,height=side)
    if label!='Source':
     f=v['frames'][label];text=f"{f['dists']:.5f} / {f['dreamsim']:.5f}"
    else:text=f"ID {v['source_index']} | repeat 0"
    c.setFont('ReviewArial',8);c.drawCentredString(x+side/2,y-3.7*mm,text)
   c.setFont('ReviewArial',6.6);c.drawString(left,top-57*mm,v['source_id'])
  c.setFont('ReviewArial',7);c.drawString(left,9*mm,'Lower DISTS / DreamSim is better. Full five-metric values and status flags are in the CSV and HTML.')
  c.showPage();page_manifest.append(dict(snr=s,page=start//4+1,ids=ids))
 c.save()
 doc=fitz.open(path);assert len(doc)==total
 for i,page in enumerate(doc):
  dest=OUT/'preview'/f'{s}db_page_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(dest)
  text=page.get_text();assert all(x in text for x in page_manifest[-total+i]['ids'])
save(OUT/'page_manifest.json',page_manifest)
body=[]
for v in cards:
 imgs=[]
 for label in ['Source','WHOLE','Ada','NeST']:
  filename=f"tiles/s{v['snr_db']:02d}_i{v['source_index']:03d}_{label}.png"
  metric=''if label=='Source'else f"{v['frames'][label]['dists']:.5f} / {v['frames'][label]['dreamsim']:.5f}"
  name={'Ada':'Ada-TokenCom','NeST':'NeST-Com'}.get(label,label)
  imgs.append(f'<figure><figcaption>{name}</figcaption><a href="{filename}" target="_blank"><img src="{filename}" loading="lazy" width="256" height="256"></a><p>{metric}</p></figure>')
 table='<table><tr><th>方法</th>'+''.join(f'<th>{name}{"↑"if direction==1 else "↓"}</th>'for _,direction,name in metrics)+'<th>接收状态</th></tr>'
 for label,f in v['frames'].items():table+='<tr><td>'+label+'</td>'+''.join(f'<td>{f[k]:.6f}</td>'for k,_,_ in metrics)+f'<td>{html.escape(f["receiver_status"])}</td></tr>'
 table+='</table>'
 badge='五项均领先'if v['all_five_lead']else 'DISTS / DreamSim 均领先';gray=' · 含对照灰图'if v['has_gray']else ''
 body.append(f'<article data-snr="{v["snr_db"]}" data-five="{int(v["all_five_lead"])}" data-gray="{int(v["has_gray"])}" data-search="{v["candidate_id"]} {v["source_index"]} {html.escape(v["source_id"])}"><h2>{v["candidate_id"]} · 源图 {v["source_index"]} <small>{badge}{gray}</small></h2><div class="images">'+''.join(imgs)+f'</div><details><summary>完整五指标、身份和接收状态</summary><p>{html.escape(v["source_id"])}</p>'+table+'</details></article>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeST-Com Fig.3 全部领先候选</title><style>
body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#162b40;margin:0}header{padding:24px;max-width:1100px;margin:auto}h1{font-size:26px;margin:0 0 12px}p{line-height:1.6}.controls{position:sticky;top:0;background:#fff;padding:12px 20px;box-shadow:0 1px 8px #ccd5df;z-index:2;display:flex;gap:14px;align-items:center;flex-wrap:wrap}select,input{font:inherit;padding:6px}main{max-width:1100px;margin:auto;padding:16px}article{background:#fff;border:1px solid #dbe2e9;border-radius:8px;padding:16px;margin-bottom:24px}h2{font-size:19px;margin-top:0}small{font-size:13px;color:#2c6d43;margin-left:15px}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;text-align:center}img{width:100%;height:auto;aspect-ratio:1}figcaption{font-weight:600;margin-bottom:8px}figure p{font:14px ui-monospace,monospace;margin:6px 0}summary{cursor:pointer;margin-top:12px;color:#1f67aa}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}article[hidden]{display:none}a{color:#1f67aa}.meta{color:#526273;font-size:14px}#count{font-weight:bold}@media(max-width:700px){main{padding:6px}article{padding:8px}.images{gap:5px}figure p{font-size:10px}figcaption{font-size:12px}table{font-size:10px}td,th{padding:3px}}
</style><header><h1>Fig. 3：全部指标领先候选</h1><p>固定 N=1024、原第一信道重复。比较 NeST-Com、WHOLE、Ada-TokenCom（predicted-class, fixed-frame adaptation）。图下数字为 <b>DISTS↓ / DreamSim↓</b>，来自该显示输出的原生数组评分。</p><p>7 dB：143 张双指标领先，60 张五项领先。10 dB：23 张双指标领先，11 张五项领先。默认显示 7 dB、五项均领先、无对照灰图。可切换筛选查看全部 166 个候选工作点。</p><p class="meta">五项：DISTS、DreamSim、PSNR、LPIPS、CLIP-I；严格比较未舍入数值。固定原重复，不挑种子。候选排序仅用于优势例图选择，正式总体统计保持原样。</p><p><a href="Fig3_7dB_all_leading_candidates.pdf">7 dB 全候选 PDF</a> · <a href="Fig3_10dB_all_leading_candidates.pdf">10 dB 全候选 PDF</a> · <a href="all_166_leading_candidates.csv">166 候选数据</a> · <a href="all_1000_candidates_audit.csv">完整 1000 工作点审计</a></p></header>
<div class="controls"><label>SNR <select id="snr"><option value="7">7 dB</option><option value="10">10 dB</option><option value="all">全部</option></select></label><label>条件 <select id="metric"><option value="five">五项均领先</option><option value="two">DISTS / DreamSim 均领先</option></select></label><label><input id="gray" type="checkbox">包含对照灰图</label><input id="search" placeholder="搜索编号或 source ID"><span id="count"></span></div><main>'''+''.join(body)+'''</main><script>
const cards=[...document.querySelectorAll('article')],snr=document.querySelector('#snr'),metric=document.querySelector('#metric'),gray=document.querySelector('#gray'),search=document.querySelector('#search');function filter(){let n=0;for(const c of cards){const yes=(snr.value==='all'||c.dataset.snr===snr.value)&&(metric.value==='two'||c.dataset.five==='1')&&(gray.checked||c.dataset.gray==='0')&&c.dataset.search.toLowerCase().includes(search.value.trim().toLowerCase());c.hidden=!yes;if(yes)n++;}document.querySelector('#count').textContent=n+' 个候选';}for(const e of[snr,metric,gray,search])e.addEventListener('input',filter);filter();
</script></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8',newline='\n')
# Compact overview uses the first four five-metric-leading 7 dB cases.
first=[v for v in cards if v['snr_db']==7 and v['all_five_lead']and not v['has_gray']][:4]
assert [x['candidate_id']for x in first]==page_manifest[0]['ids']
Image.open(OUT/'preview/7db_page_01.png').save(OUT/'preview_first_four.png')
report=f'''# Fig.3 全部领先候选：2026-11-02

基准 main `{mapped['head']}`。当前稿 Fig.3 为 N1024、7 dB、Source / WHOLE / Ada / NeST。当前电视源图 84 满足五项均领先。

本轮读取既有完整候选池（7/10 dB 各 500 源），保留原第一信道重复。筛选条件为 NeST 的 DISTS、DreamSim 均严格小于 WHOLE 与 Ada；另标记五项指标全部领先。

7 dB：143 个双指标候选，60 个五指标候选，{counts['7']['five_metrics_no_gray']} 个五指标且无灰图候选。3 个双指标候选含对照灰图，完整保留并显式标注。
10 dB：23 个双指标候选，11 个五指标候选，均无灰图。

默认 HTML 展示 7 dB 五项均领先且无灰图；切换筛选即可看完整 166 个候选。PDF 包含全部双指标候选，7 dB 36 页、10 dB 6 页；五项领先、无灰图优先，随后保留原优势排序。编号关联完整 source ID；可据编号选图。

所有图像保持完整 256×256 视野，精确提取自既有哈希固定的 contact sheets；22 页原接触图复用、664 张显示图片、零新模型/PHY/指标/统计调用。已通过源图像素身份、完整源标识、参考 hash、方法、重复与实际输出身份连接；原电视/皮革两例的提取 RGB 与既有展示图片逐字节相同。显示 PNG 不替代原生数组指标输入。

这里是按优势筛选的插图候选；正式 common500 均值、CI、失败分母及当前稿件未改动。五项领先是当前显示实现逐图数值比较，不是逐图统计显著性判定。

文件：`index.html` 为可筛选图库；两个 PDF 为全部候选；`all_166_leading_candidates.csv` 为候选数值；`all_1000_candidates_audit.csv` 同时保留未入选工作点；`candidate_metadata.json` / `extraction_identity.json` 保存来源身份。
'''
(OUT/'README_zh.md').write_text(report,encoding='utf-8',newline='\n')
save(OUT/'artifact_checks.json',dict(status='GENERATED_TECHNICAL_CHECKS_PASS_VISUAL_REVIEW_PENDING',counts=counts,PDF_pages={'7dB':36,'10dB':6},candidate_ids_unique=len({v['candidate_id']for v in cards})==166,all_tile_hashes_verified=664,page_membership_verified=166,input_frames_sha256=sha(INPUT/'mapped_frames.json'),input_candidate_index_sha256=hashlib.sha256(inventory['files']['candidate_index.csv'].encode()).hexdigest(),figure3_unchanged=True,new_scientific_calls=0))
print(json.dumps(dict(output=str(OUT),counts=counts,pages=42,first_four=[dict(id=v['candidate_id'],source=v['source_index'])for v in first]),ensure_ascii=False))
