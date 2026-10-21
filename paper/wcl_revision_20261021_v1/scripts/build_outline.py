"""Deterministic Word outline; no inference, scoring or statistical resampling."""
from pathlib import Path
import csv,json
from docx import Document
from docx.shared import Inches,Mm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
H=Path(__file__).resolve().parent
P=H.parent
sections=dict(json.loads((P/'outline_sections.json').read_text(encoding='utf-8')))
d=Document();sec=d.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Mm(13);sec.left_margin=sec.right_margin=Mm(14)
sec.header_distance=sec.footer_distance=Mm(6)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 st=d.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
 st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Microsoft YaHei')
d.styles['Normal'].font.size=Pt(10.5)
d.styles['Normal'].paragraph_format.line_spacing=1.12
d.styles['Normal'].paragraph_format.space_after=Pt(5)
d.styles['Title'].font.size=Pt(19);d.styles['Title'].paragraph_format.space_after=Pt(7)
d.styles['Heading 1'].font.size=Pt(13)
d.styles['Heading 1'].paragraph_format.space_before=Pt(9)
d.styles['Heading 1'].paragraph_format.space_after=Pt(6)
d.styles['Heading 2'].font.size=Pt(11.5)
d.styles['Caption'].font.size=Pt(8.5)
d.styles['Caption'].font.bold=False
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run('NeST-Com · WCL revision · ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
for r in footer.runs:r.font.size=Pt(8);r.font.color.rgb=RGBColor(0,0,0)
def para(t):
 return d.add_paragraph(t)
def section(name,indices=None,heading=True):
 if heading:d.add_heading(name,1)
 items=sections[name]
 for i,(title,text) in enumerate(items):
  if indices is not None and i not in indices:continue
  p=d.add_paragraph();p.add_run(title+'。').bold=True;p.add_run(text)
def fig(n,caption):
 p=d.add_paragraph();p.paragraph_format.space_after=Pt(2);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 p.add_run().add_picture(str(P/f'figure{n}.png'),width=Mm(181.864))
 p=d.add_paragraph(caption,'Caption');p.paragraph_format.space_after=Pt(7)
def table(title,headers,data,widths,notes):
 d.add_paragraph(title,'Heading 2')
 t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for c,w in zip(t.columns,widths):c.width=Mm(w)
 for c,s in zip(t.rows[0].cells,headers):c.text=s
 for row in data:
  cells=t.add_row().cells
  for c,s in zip(cells,row):c.text=str(s)
 for ri,row in enumerate(t.rows):
  for j,c in enumerate(row.cells):
   c.width=Mm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tcPr=c._tc.get_or_add_tcPr()
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    node=OxmlElement('w:'+side);node.set(qn('w:val'),'single');node.set(qn('w:sz'),'4');node.set(qn('w:color'),'D9D9D9');borders.append(node)
   tcPr.append(borders)
   margins=OxmlElement('w:tcMar')
   for side in ['top','left','bottom','right']:
    node=OxmlElement('w:'+side);node.set(qn('w:w'),'80');node.set(qn('w:type'),'dxa');margins.append(node)
   tcPr.append(margins)
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E9EDF2');tcPr.append(shade)
   for p in c.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.05
    for r in p.runs:r.font.size=Pt(9);r.bold=(ri==0)
  trPr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trPr.append(cant)
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
 d.add_paragraph(notes,'Caption')
def read(n):return list(csv.DictReader((P/n).open(encoding='utf-8')))
def interval(r,k,n):
 return f"{float(r[k+'_mean']):.{n}f}\n[{float(r[k+'_ci_low']):.{n}f}, {float(r[k+'_ci_high']):.{n}f}]"
d.add_paragraph('NeST-Com: Fixed-Budget\nNext-Scale Token Communication','Title')
para('WCL 修订大纲与实际图表｜2026年10月21日')
para('本稿给出段落任务和内容要点，尚未扩写完整论文。主证据沿用已公开冻结版本；ARPC 解码敏感性已经完成；四张13 dB候选另行交付，正文保留10 dB全图。其结果单列，未替换 common500 主曲线。')
section('I. Introduction');section('II.A. System Model and Multiscale Representation')
para('修订结构：II 为完整方法；III.B 用五方法评价系统，III.C 用 WHOLE/PARTIAL 检验机制，III.D 与 III.E 分别说明预算扩展和处理成本。')
d.add_page_break()
d.add_heading('II. Proposed NeST-Com Framework',1)
fig(1,'Fig.1. 从用户确认的系统图源文件重排。主链路保留 AWGN；接收端区分已传 token 的熵解码和未传表示的生成。利用率条已移至 Table I；图中保留原图像和多尺度网格。')
section('II.B. Budget-Adaptive Within-Scale Prefix Transmission')
section('II.C. Prefix-Constrained Reconstruction')
d.add_page_break()
d.add_heading('III. Experimental Results',1)
section('III.A. Experimental Setup')
fig(2,'Fig.2. common500、N1024、4/10/19 dB 的五方法主结果。每源先平均三次原有噪声，误差棒为来源级逐工作点95% CI。LPIPS为AlexNet；CLIP-I为ViT-L/14图像—图像余弦。')
section('III.B. Comparison With Existing Methods',[0])
d.add_page_break()
d.add_heading('III.B. Qualitative comparison and decoder sensitivity',1)
fig(3,'Fig.3. 既有 development 展示：Pastries95 与 Shopping cart75，N1024、10 dB。两行均保留完整视野，无局部放大；标注为整幅 PSNR (dB) / LPIPS-Alex，使用实际该次重建的原生浮点分数。')
section('III.B. Comparison With Existing Methods',[1,2],False)
d.add_page_break()
section('III.C. Effect of Within-Scale Endpoints')
table('Table I  N1024 within-scale endpoints',
 ['SNR (dB)','WHOLE\nutil. (%)','PARTIAL\nutil. (%)','ΔPSNR (dB)\n[95% CI]','ΔDreamSim\n[95% CI]'],
 [[r['snr_db'],f"{100*float(r['whole_utilization']):.2f}",f"{100*float(r['partial_utilization']):.2f}",interval(r,'psnr',3),interval(r,'dreamsim',4)] for r in read('table_i.csv')],
 [16,28,28,53,53],'Δ=PARTIAL−WHOLE，common500；直接来源配对区间。利用率=Lterm/(k−13−16)。')
section('III.D. Performance With a Larger Symbol Budget')
table('Table II  N2048 paired comparison',
 ['Condition','ΔPSNR (dB)\n[95% CI]','ΔDreamSim\n[95% CI]'],
 [[r['condition'],interval(r,'psnr',3),interval(r,'dreamsim',4)] for r in read('table_ii.csv')],
 [62,58,58],'Δ=NeST−ARPC，matched100，每源三噪声；每方法每条件300/300送达。三行配置不同。正 ΔDISTS 表示 ARPC 更好。')
d.add_page_break()
section('III.E. Computational Cost')
mapping={'NEST':'NeST-Com','P100000':'Latent JSCC','Swin200000':'SwinJSCC','BPG':'BPG + LDPC','ARPC':'ARPC + LDPC'}
table('Table III  Software processing latency at 10 dB',
 ['Method','TX (ms)','RX (ms)','Total (ms)'],
 [[mapping[r['method']]]+[f"{float(r[k+'_ms_mean']):.2f}" for k in ['tx','rx','total']] for r in read('table_iii.csv')],
 [67,37,37,37],'RTX PRO 6000 Blackwell GPU1；同6个CPU核心；batch1；16 development，每源3次预热+3次测量。Total=TX+RX，不含空口；保留原框架、精度和共享负载记录。')
section('IV. Conclusion')
d.add_heading('交付边界与核验入口',1)
para('已完成：三图三表、中文大纲、Word与LaTeX骨架。此前 ARPC 敏感性已闭合，本轮未重跑。另行完成四张既有 development 图的13 dB检查：20个逻辑事件，12张数字方法重建精确复用，8张连续方法重建新增；未训练或新增bootstrap。正文仍使用10 dB两张全图。')
para('不在本轮：完整论文正文、N2048 CLIP、main common500 策略或检查点替换。所有主图60个均值和CI、Fig.3图片与分数、表格原始精度均有逐项追溯。')
para('详细敏感性：sensitivity/REPORT.md；全部原始精度表格：CSV；完整英文图注：captions.md。公开基线 a5f8ceb0233b0e1f599ee8bb902c1a094d47119d。')
d.add_heading('相关工作核对来源',1)
para('Ada-TokenCom（率自适应 token 编码与生成）：https://arxiv.org/abs/2608.28086\nTONIC（任务导向 token 通信）：https://arxiv.org/abs/2605.21553')
for root in [d.element,d.styles.element]:
 for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
d.save(P/'outline.docx')
print(P/'outline.docx')
