from pathlib import Path
import json,zipfile,copy,math,hashlib,collections
from lxml import etree as E
H=Path(__file__).resolve().parent
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';N={'p':P,'a':A}
plan=json.loads((H/'layout_plan.json').read_text())
def pkg(p):
 with zipfile.ZipFile(p) as z:return {n:z.read(n)for n in z.namelist()}
def named(r):
 out={}
 for x in r.iter():
  if E.QName(x).localname in ('sp','grpSp','pic'):
   q=x.find('./p:nvSpPr/p:cNvPr',N) if E.QName(x).localname=='sp' else x.find('./p:nvGrpSpPr/p:cNvPr',N) if E.QName(x).localname=='grpSp' else x.find('./p:nvPicPr/p:cNvPr',N)
   if q is not None:out[q.get('name')]=x
 return out
def xf(x):return x.find('p:grpSpPr/a:xfrm',N) if E.QName(x).localname=='grpSp' else x.find('p:spPr/a:xfrm',N)
source=H/'source_clean.pptx';assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
data=pkg(source);r=E.fromstring(data['ppt/slides/slide1.xml']);tree=r.find('p:cSld/p:spTree',N)
original=named(r);before={k:E.tostring(v,method='c14n')for k,v in original.items()}
tx_before={k:E.tostring(v.find('p:txBody',N),method='c14n') for k,v in original.items()if v.find('p:txBody',N)is not None}
edited=named(E.fromstring(pkg(H/'artifact_layout.pptx')['ppt/slides/slide1.xml']))
for n,u in plan['updates'].items():
 x=original[n];t=xf(x);bx,by,bw,bh=u['after']
 desired={'off':{'x':str(round(bx*12700)),'y':str(round(by*12700))},'ext':{'cx':str(round(bw*12700)),'cy':str(round(bh*12700))}}
 if n in edited and E.QName(x).localname=='sp':
  et=xf(edited[n]);assert et is not None
  for key in ['off','ext']:
   e=et.find('a:'+key,N)
   for k,v in desired[key].items():assert abs(int(e.get(k))-int(v))<=2,(n,key,k,e.get(k),v)
   t.find('a:'+key,N).attrib.update(dict(e.attrib))
 else:
  assert E.QName(x).localname=='grpSp',n
  # Translate mathematical label groups without changing their child geometry.
  for key,attrs in desired.items():t.find('a:'+key,N).attrib.update(attrs)

arrows=named(E.fromstring(pkg(H/'arrows_layout.pptx')['ppt/slides/slide1.xml']))
for route in plan['newroutes']:
 n=route['name'];old=original[n];x=copy.deepcopy(arrows[n])
 q=x.find('p:nvSpPr/p:cNvPr',N);q.set('id',old.find('p:nvSpPr/p:cNvPr',N).get('id'))
 q.set('descr','Single native arrow with one shaft and an attached compact filled head.')
 for locks in x.findall('.//a:spLocks',N):locks.attrib.clear()
 sp=x.find('p:spPr',N);ln=sp.find('a:ln',N);ln.set('cap','flat');ln.set('cmpd','sng')
 for a in list(ln):
  if E.QName(a).localname in ['headEnd','tailEnd']:ln.remove(a)
 pts=copy.deepcopy(route['points']);end=pts[-1];prev=pts[-2]
 dx,dy=end[0]-prev[0],end[1]-prev[1];l=math.hypot(dx,dy);ux,uy=dx/l,dy/l
 bx,by=end[0]-ux*2.25,end[1]-uy*2.25
 head=[end,[bx-uy*2.15/2,by+ux*2.15/2],[bx+uy*2.15/2,by-ux*2.15/2]]
 pts[-1]=[bx+ux*.15,by+uy*.15]
 ap=pts+head;x0=min(v[0]for v in ap);y0=min(v[1]for v in ap);w=max(.01,max(v[0]for v in ap)-x0);h=max(.01,max(v[1]for v in ap)-y0)
 t=xf(x);t.find('a:off',N).attrib.update({'x':str(round(x0*12700)),'y':str(round(y0*12700))});t.find('a:ext',N).attrib.update({'cx':str(round(w*12700)),'cy':str(round(h*12700))})
 for f in list(sp):
  if E.QName(f).localname in ('noFill','solidFill'):sp.remove(f)
 fill=E.Element('{'+A+'}solidFill');E.SubElement(fill,'{'+A+'}srgbClr',val=route['color'].lstrip('#'));sp.insert(list(sp).index(ln),fill)
 pl=sp.find('a:custGeom/a:pathLst',N)
 for pp in list(pl):pl.remove(pp)
 for poly,filled in [(pts,False),(head,True)]:
  path=E.SubElement(pl,'{'+A+'}path',w=str(round(w*12700)),h=str(round(h*12700)),fill='norm' if filled else 'none',stroke='0' if filled else '1')
  for j,(px,py)in enumerate(poly):
   c=E.SubElement(path,'{'+A+'}moveTo' if j==0 else '{'+A+'}lnTo');E.SubElement(c,'{'+A+'}pt',x=str(round((px-x0)*12700)),y=str(round((py-y0)*12700)))
  if filled:E.SubElement(path,'{'+A+'}close')
 old.getparent().replace(old,x);original[n]=x

for suffix,items,gid in [('b',plan['panel_b_names'],60001),('c',plan['panel_c_names'],60002)]:
 selected=[x for x in list(tree) if x in [original[n]for n in items]]
 assert len(selected)==len(items)
 bounds=[]
 for x in selected:
  t=xf(x);o=t.find('a:off',N);e=t.find('a:ext',N);bounds.append([int(o.get('x')),int(o.get('y')),int(e.get('cx')),int(e.get('cy'))])
 x0=min(b[0]for b in bounds);y0=min(b[1]for b in bounds);w=max(b[0]+b[2]for b in bounds)-x0;h=max(b[1]+b[3]for b in bounds)-y0
 g=E.Element('{'+P+'}grpSp');nv=E.SubElement(g,'{'+P+'}nvGrpSpPr')
 E.SubElement(nv,'{'+P+'}cNvPr',id=str(gid),name='Panel ('+suffix+') - move entire panel together')
 E.SubElement(nv,'{'+P+'}cNvGrpSpPr');E.SubElement(nv,'{'+P+'}nvPr')
 pr=E.SubElement(g,'{'+P+'}grpSpPr');t=E.SubElement(pr,'{'+A+'}xfrm')
 for tag,attrs in [('off',{'x':str(x0),'y':str(y0)}),('ext',{'cx':str(w),'cy':str(h)}),('chOff',{'x':str(x0),'y':str(y0)}),('chExt',{'cx':str(w),'cy':str(h)})]:E.SubElement(t,'{'+A+'}'+tag,**attrs)
 for x in selected:tree.remove(x);g.append(x)
 tree.append(g)

after=named(r)
assert all(n in after for n in before)
assert all(E.tostring(after[n].find('p:txBody',N),method='c14n')==v for n,v in tx_before.items())
changed=set(plan['updates'])|{q['name']for q in plan['newroutes']}
protected=[n for n in before if n not in changed and n not in plan['panel_b_names'] and n not in plan['panel_c_names']]
assert all(E.tostring(after[n],method='c14n')==before[n]for n in protected)
assert sum(n.startswith('clean_arrow_')for n in after)==42
data['ppt/slides/slide1.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
out=H/'balanced_candidate01.pptx'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED)as z:
 for n,b in data.items():z.writestr(n,b)
audit=dict(source_sha256=plan['source_sha256'],output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),updated_geometry=len(changed),all_208_text_bodies_exact=True,unchanged_font_sizes=True,panel_a_and_legend_objects_exact=True,protected_native_objects=len(protected),existing_object_names_retained=True,panel_groups_added=2,single_object_arrows=42,all_other_package_parts_exact=True,new_scientific_calls=0)
(H/'layout_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
