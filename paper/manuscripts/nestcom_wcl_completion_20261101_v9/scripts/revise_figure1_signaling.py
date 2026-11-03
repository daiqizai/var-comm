"""Local native-figure edit: preconfigured MCS and endpoint-only profile ID.

Usage: python revise_figure1_signaling.py original.pptx revised.pptx audit.json
Only panel (a)'s specified signal labels/arrows change. Export the revised
native source through export_figure1_native.ps1, then export_figure1_views.py.
"""
from pathlib import Path
from lxml import etree as E
import argparse, zipfile, copy, json, hashlib, math
N={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
   'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def tag(k): return '{'+N['a']+'}'+k
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('audit',type=Path);args=ap.parse_args()
 assert not args.output.exists()
 with zipfile.ZipFile(args.source) as z:data={n:z.read(n) for n in z.namelist()}
 r=E.fromstring(data['ppt/slides/slide1.xml']);tree=r.find('p:cSld/p:spTree',N)
 shapes={s.find('p:nvSpPr/p:cNvPr',N).get('name'):s for s in r.findall('.//p:sp',N)}
 before={n:E.tostring(s,method='c14n') for n,s in shapes.items()}
 next_id=max(int(q.get('id')) for q in r.findall('.//p:cNvPr',N))+1
 def new_identity(s,name):
  nonlocal next_id
  n=s.find('p:nvSpPr/p:cNvPr',N);n.attrib.clear();n.set('id',str(next_id));next_id+=1;n.set('name',name)
  for c in list(n):n.remove(c)
 def set_pos(s,x,y,w,h):
  xf=s.find('p:spPr/a:xfrm',N)
  for k,v in [('off',dict(x=x,y=y)),('ext',dict(cx=w,cy=h))]:
   q=xf.find('a:'+k,N)
   for a,b in v.items():q.set(a,str(round(b*12700)))
 def text(name,value,x,y,w,h,size):
  s=copy.deepcopy(shapes['text_002_0']);new_identity(s,name);set_pos(s,x,y,w,h)
  s.find('.//a:t',N).text=value
  for q in s.findall('.//a:rPr',N)+s.findall('.//a:defRPr',N):
   q.set('sz',str(round(size*100)));q.find('a:solidFill/a:srgbClr',N).set('val','8B66A5')
  if name=='shared_policy_configuration':
   paragraph=s.find('p:txBody/a:p',N);run=paragraph.find('a:r',N);paragraph.remove(run)
   for word,italic in [('Shared preconfigured policy: MCS ',False),('a',True),(' and profile directory',False)]:
    segment=copy.deepcopy(run);segment.find('a:t',N).text=word;segment.find('a:rPr',N).set('i','1'if italic else'0');paragraph.append(segment)
  tree.append(s)
 def arrow(name,pts):
  s=copy.deepcopy(shapes['clean_arrow_17']);new_identity(s,name)
  end=pts[-1];prev=pts[-2];dx=end[0]-prev[0];dy=end[1]-prev[1];norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
  hlen,hw=2.2,1.05;back=(end[0]-hlen*ux,end[1]-hlen*uy)
  head=[end,(back[0]-hw*uy,back[1]+hw*ux),(back[0]+hw*uy,back[1]-hw*ux)]
  points=pts+head;x0=min(p[0]for p in points);y0=min(p[1]for p in points);w=max(p[0]for p in points)-x0;h=max(p[1]for p in points)-y0
  set_pos(s,x0,y0,w,h);pl=s.find('p:spPr/a:custGeom/a:pathLst',N)
  for p in list(pl):pl.remove(p)
  for poly,filled in [(pts[:-1]+[back],False),(head,True)]:
   path=E.SubElement(pl,tag('path'),w=str(round(w*12700)),h=str(round(h*12700)),fill='norm'if filled else'none',stroke='0'if filled else'1')
   for j,(x,y)in enumerate(poly):
    cmd=E.SubElement(path,tag('moveTo'if j==0 else'lnTo'));E.SubElement(cmd,tag('pt'),x=str(round((x-x0)*12700)),y=str(round((y-y0)*12700)))
   if filled:E.SubElement(path,tag('close'))
  tree.append(s)
 remove=['text_093_0','text_094_0','text_095_0','text_096_0','text_097_0',
         'text_098_0','text_109_0','clean_arrow_15','clean_arrow_17']
 assert ''.join(shapes[n].find('.//a:t',N).text for n in remove[:5])=='(e,a)'
 text('profile_id_label','Profile ID',183.0,94.0,26.0,9.3,5.8)
 text('shared_policy_configuration','Shared preconfigured policy: MCS a and profile directory',132.0,8.0,268.0,11.2,6.8)
 arrow('preconfigured_mcs_tx',[(204.8,20.0),(204.8,67.0),(209.0,67.0)])
 arrow('preconfigured_mcs_rx',[(313.8,20.0),(313.8,67.0),(309.5,67.0)])
 for n in remove:
  s=shapes[n];s.getparent().remove(s)
 after={s.find('p:nvSpPr/p:cNvPr',N).get('name'):E.tostring(s,method='c14n') for s in r.findall('.//p:sp',N)}
 preserved=[n for n in before if n not in remove]
 assert all(before[n]==after[n]for n in preserved)
 assert 'clean_arrow_18'in after and before['clean_arrow_18']==after['clean_arrow_18']
 data['ppt/slides/slide1.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED)as z:
  for n,b in data.items():z.writestr(n,b)
 audit=dict(source_sha256=sha(args.source.read_bytes()),output_sha256=sha(args.output.read_bytes()),unchanged_native_shapes=len(preserved),removed_objects=remove,added_objects=[n for n in after if n not in before],other_package_parts_preserved=True,panel_b_c_preserved=True,profile_decode_endpoint_arrow_preserved=True,profile_field='Profile ID',mcs='Shared preconfigured policy, supplied to TX and RX body processing',new_scientific_calls=0)
 args.audit.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit))
if __name__=='__main__':main()
