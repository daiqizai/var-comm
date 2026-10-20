from pathlib import Path
import zipfile,xml.etree.ElementTree as E,json
import os
H=Path(os.environ['NEST_FIGURE_WORK_DIR'])
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
with zipfile.ZipFile(H/'figure1_draft.pptx') as z:
 data={n:z.read(n) for n in z.namelist()}
 pres=E.fromstring(data['ppt/presentation.xml']);size=pres.find('p:sldSz',ns)
 oldcx=int(size.attrib['cx']);ratio=6547104/oldcx
 # The native layout uses 1774 CSS pixels in width and 970 in height.
 size.set('cx','6547104');size.set('cy',str(round(6547104*970/1774)))
 data['ppt/presentation.xml']=E.tostring(pres,encoding='utf-8',xml_declaration=True)
 root=E.fromstring(data['ppt/slides/slide1.xml'])
 for node in root.iter():
  tag=node.tag.rsplit('}',1)[-1]
  if tag in ['off','ext','chOff','chExt']:
   for attr in ['x','y','cx','cy']:
    if attr in node.attrib:node.set(attr,str(round(int(node.attrib[attr])*ratio)))
  for attr in ['sz']:
   if attr in node.attrib:node.set(attr,str(max(700,round(int(node.attrib[attr])*ratio))))
  if tag=='ln' and 'w' in node.attrib:node.set('w',str(round(int(node.attrib['w'])*ratio)))
  if tag=='bodyPr':
   for attr in ['lIns','rIns','tIns','bIns']:
    if attr in node.attrib:node.set(attr,str(round(int(node.attrib[attr])*ratio)))
  if tag in ['latin','ea','cs']:node.set('typeface','Arial')
  if tag in ['rPr','defRPr','endParaRPr']:
   for fonttag in ['latin','ea','cs']:
    font=node.find('a:'+fonttag,ns)
    if font is None:font=E.SubElement(node,'{'+ns['a']+'}'+fonttag)
    font.set('typeface','Arial')
 data['ppt/slides/slide1.xml']=E.tostring(root,encoding='utf-8',xml_declaration=True)
 with zipfile.ZipFile(H/'figure1_normalized.pptx','w',zipfile.ZIP_DEFLATED) as z:
  for n,b in data.items():z.writestr(n,b)
 fonts=[int(n.attrib['sz'])/100 for n in root.iter() if 'sz' in n.attrib]
 (H/'figure1_normalization.json').write_text(json.dumps(dict(width_inches=7.16,height_inches=7.16*970/1774,ratio=ratio,min_font_pt=min(fonts),max_font_pt=max(fonts),expected_size_emu=f"6547104,{size.attrib['cy']}"),indent=2))
 print(min(fonts),max(fonts),size.attrib)
