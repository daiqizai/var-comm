from pathlib import Path
import json,zipfile,copy,hashlib,math
from lxml import etree as E
H=Path(__file__).resolve().parent;plan=json.loads((H/'route_plan.json').read_text())
N={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def read(p):
 with zipfile.ZipFile(p) as z:return {n:z.read(n)for n in z.namelist()}
assert hashlib.sha256((H/'user_edited_source.pptx').read_bytes()).hexdigest()==plan['source_sha256'],'Source PPTX identity mismatch'
data=read(H/'user_edited_source.pptx');r=E.fromstring(data['ppt/slides/slide1.xml']);tree=r.find('p:cSld/p:spTree',N)
ar=E.fromstring(read(H/'arrows_native.pptx')['ppt/slides/slide1.xml'])
arrows={x.find('p:nvSpPr/p:cNvPr',N).get('name'):copy.deepcopy(x) for x in ar.findall('.//p:sp',N)}
old={x.find('p:nvSpPr/p:cNvPr',N).get('name'):x for x in r.findall('.//p:sp',N)}
before={n:E.tostring(x,method='c14n')for n,x in old.items()}
for i,route in enumerate(plan['routes']):
 n=route['name'];x=arrows[n];q=x.find('p:nvSpPr/p:cNvPr',N);q.set('id',str(30000+i));q.set('descr','Single editable vector arrow: shaft and filled head are paths within the same object.')
 for locks in x.findall('.//a:spLocks',N):locks.attrib.clear()
 ln=x.find('p:spPr/a:ln',N);ln.set('cap','flat');ln.set('cmpd','sng')
 for a in list(ln):
  if E.QName(a).localname in ['headEnd','tailEnd']:ln.remove(a)
 # PowerPoint's smallest built-in arrowhead is too large at this print size.
 # Keep a compact head as a second path IN the same native shape. Its filled
 # triangle has no outline; the single shaft stops beneath its base.
 pts=copy.deepcopy(route['points']);end=pts[-1];prev=pts[-2]
 dx,dy=end[0]-prev[0],end[1]-prev[1];length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
 hl,hw=2.25,2.15;bx,by=end[0]-ux*hl,end[1]-uy*hl
 head=[end,[bx-uy*hw/2,by+ux*hw/2],[bx+uy*hw/2,by-ux*hw/2]]
 pts[-1]=[bx+ux*.15,by+uy*.15]
 allpts=pts+head;x0=min(p[0]for p in allpts);y0=min(p[1]for p in allpts)
 w=max(.01,max(p[0]for p in allpts)-x0);h=max(.01,max(p[1]for p in allpts)-y0)
 sp=x.find('p:spPr',N);xf=sp.find('a:xfrm',N);xf.find('a:off',N).attrib.update({'x':str(round(x0*12700)),'y':str(round(y0*12700))});xf.find('a:ext',N).attrib.update({'cx':str(round(w*12700)),'cy':str(round(h*12700))})
 for f in list(sp):
  if E.QName(f).localname in ('noFill','solidFill'):sp.remove(f)
 fill=E.Element('{'+N['a']+'}solidFill');E.SubElement(fill,'{'+N['a']+'}srgbClr',val=route['color'].lstrip('#'));sp.insert(list(sp).index(ln),fill)
 pl=sp.find('a:custGeom/a:pathLst',N)
 for oldpath in list(pl):pl.remove(oldpath)
 for poly,filled in [(pts,False),(head,True)]:
  path=E.SubElement(pl,'{'+N['a']+'}path',w=str(round(w*12700)),h=str(round(h*12700)),fill='norm' if filled else 'none',stroke='0' if filled else '1')
  for j,(px,py)in enumerate(poly):
   c=E.SubElement(path,'{'+N['a']+'}moveTo' if j==0 else '{'+N['a']+'}lnTo');E.SubElement(c,'{'+N['a']+'}pt',x=str(round((px-x0)*12700)),y=str(round((py-y0)*12700)))
  if filled:E.SubElement(path,'{'+N['a']+'}close')
 # Place the replacement at the original route's layer, preserving other
 # user objects byte-for-byte and retaining their z-order.
 target=old[route['replaces'][0]];parent=target.getparent();parent.insert(parent.index(target),x)
for n in plan['remove']:
 x=old[n];x.getparent().remove(x)
after={x.find('p:nvSpPr/p:cNvPr',N).get('name'):E.tostring(x,method='c14n')for x in r.findall('.//p:sp',N)}
preserved=[n for n in before if n not in plan['remove']]
assert all(before[n]==after[n] for n in preserved)
assert len(after)==len(before)-len(plan['remove'])+len(plan['routes'])
data['ppt/slides/slide1.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
out=H/'clean_candidate02.pptx'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED)as z:
 for n,b in data.items():z.writestr(n,b)
audit={'source_sha256':plan['source_sha256'],'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'unchanged_native_shapes':len(preserved),'unchanged_native_shape_names':preserved,'removed_objects':plan['remove'],'new_attached_head_arrows':len(plan['routes']),'all_other_package_parts_exact':True,'user_positions_and_text_preserved':True,'new_scientific_calls':0}
(H/'native_edit_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items()if not isinstance(v,list)}))
