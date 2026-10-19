"""Inspect already exported report files; rasterize existing PDFs for viewing only."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader

W=Path('C:/Users/11946/Documents/ChatGPT/comm')
H=W/'.research/manuscript_completion_20261019/b2b3_report_visual_closure_v1'
D=W/'results/arpc_n2048_adaptive_20261019_v1/b2b3_report_v1/figures'
Q=H/'paper_size_review';Q.mkdir()
POP='C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
sizes={'adaptive_quality':(180,102),'direct_paired':(180,102),'calibration':(180,64)}
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
metadata=[];previews=[]
for name,size in sizes.items():
    p=D/(name+'.pdf');r=PdfReader(p);assert len(r.pages)==1;page=r.pages[0]
    w,h=float(page.mediabox.width)*25.4/72,float(page.mediabox.height)*25.4/72
    assert abs(w-size[0])<.001 and abs(h-size[1])<.001 and not page.images
    fonts=[]
    for ref in page['/Resources']['/Font'].get_object().values():
        f=ref.get_object();assert f['/Subtype']=='/Type0'
        descendants=f['/DescendantFonts'].get_object();df=descendants[0].get_object()['/FontDescriptor'].get_object()
        assert any(k in df for k in ('/FontFile','/FontFile2','/FontFile3'))
        fonts.append(dict(subtype=str(f['/Subtype']),base_font=str(f['/BaseFont']),embedded=True))
    points=[]
    page.extract_text(visitor_text=lambda txt,cm,tm,font,size:points.append(size) if txt.strip() else None)
    metadata.append(dict(**pin(p),pages=1,size_mm=[w,h],image_xobjects=0,fonts=fonts,text_sizes_pt=sorted(set(points))))
    svg=D/(name+'.svg');tree=ET.fromstring(svg.read_bytes());ns='{http://www.w3.org/2000/svg}'
    assert not list(tree.iter(ns+'image'));texts=list(tree.iter(ns+'text'));assert texts
    metadata.append(dict(**pin(svg),width=tree.attrib['width'],height=tree.attrib['height'],viewBox=tree.attrib['viewBox'],editable_text_nodes=len(texts),image_nodes=0))
    for gray in (False,True):
        png=D/(name+('_gray'if gray else '')+'.png');density=300 if gray else 600
        with Image.open(png)as im:
            dims=im.size;dpi=im.info['dpi'];assert all(abs(d-density)<.02 for d in dpi)
            assert all(abs(v/density*25.4-s)<25.4/density+.001 for v,s in zip(dims,size))
            if im.mode=='RGBA':assert im.getchannel('A').getextrema()==(255,255)
            metadata.append(dict(**pin(png),pixels=list(dims),dpi=list(dpi),opaque=True))
    prefix=Q/(name+'_pdf_96dpi')
    p=subprocess.run([POP,'-singlefile','-r','96','-png',str(D/(name+'.pdf')),str(prefix)],capture_output=True,timeout=60)
    (Q/(name+'.stdout')).write_bytes(p.stdout);(Q/(name+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,p.stderr
    previews.append(pin(prefix.with_suffix('.png')))
result=dict(status='PASS_EXPORT_METADATA',pdf_count=3,svg_count=3,png_count=6,files=metadata,
    paper_size_pdf_rasters_96dpi=previews,scientific_calls=0,report_render_calls=0,
    operation='Rasterization of existing PDF for visual review only',actual_visual_review_pending=True)
(Q/'export_metadata.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(metadata=pin(Q/'export_metadata.json'),paper_previews=previews,font_sizes={Path(x['path']).name:x['text_sizes_pt']for x in metadata if'text_sizes_pt'in x})))
