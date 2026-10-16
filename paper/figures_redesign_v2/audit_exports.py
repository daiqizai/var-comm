"""Verify vector containers/page sizes and create separate visual review images."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
from PIL import Image,ImageOps
from pypdf import PdfReader
O=Path(__file__).resolve().parent;Q=O/'review';Q.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ex=json.loads((O/'export_manifest.json').read_bytes());records=[]
 for r in ex['exports']:
  p=O/r['path'];assert sha(p)==r['sha256']and p.stat().st_size==r['bytes'];v=dict(path=r['path'],sha256=sha(p))
  if p.suffix=='.pdf':
   pdf=PdfReader(p);assert len(pdf.pages)==1;page=pdf.pages[0]
   mm=[float(page.mediabox.width)*25.4/72,float(page.mediabox.height)*25.4/72]
   assert abs(mm[0]-r['width_mm'])<1e-5 and abs(mm[1]-r['height_mm'])<1e-5
   assert len(page.images)==0 and 'SNR' in page.extract_text()
   fonts=[dict(name=str(f.get_object()['/BaseFont']),type=str(f.get_object()['/Subtype']))for f in page['/Resources']['/Font'].values()]
   assert all(f['type']!='/Type3'for f in fonts);v.update(page_mm=mm,image_objects=0,searchable_text=True,fonts=fonts)
  elif p.suffix=='.svg':
   t=ET.parse(p).getroot();tags=[e.tag.split('}')[-1]for e in t.iter()]
   assert 'image'not in tags and 'text'in tags and 'path'in tags
   v.update(image_nodes=0,editable_text_nodes=tags.count('text'),vector_paths=tags.count('path'))
  else:
   im=Image.open(p);assert all(abs(d-600)<.1 for d in im.info['dpi']);assert abs(im.width-r['width_mm']/25.4*600)<2 and abs(im.height-r['height_mm']/25.4*600)<2
   assert im.getchannel('A').getextrema()==(255,255);v.update(pixels=list(im.size),dpi=im.info['dpi'],opaque_white_canvas=True)
   if '/'not in r['path']:
    screen=im.convert('RGB');screen.thumbnail((1800,1000),Image.Resampling.LANCZOS);screen.save(Q/(p.stem+'_screen.png'))
    small=im.convert('RGB').resize((round(r['width_mm']/25.4*96),round(r['height_mm']/25.4*96)),Image.Resampling.LANCZOS)
    small.save(Q/(p.stem+'_96dpi_size.png'));ImageOps.grayscale(small).save(Q/(p.stem+'_96dpi_gray.png'))
    ImageOps.grayscale(im).save(Q/(p.stem+'_gray_600dpi.png'),dpi=(600,600))
  records.append(v)
 sheet=Image.new('RGB',(1500,1000),'white')
 pngs=[r for r in ex['exports']if r['path'].startswith('panels/')and r['path'].endswith('.png')]
 for i,r in enumerate(pngs):
  im=Image.open(O/r['path']).convert('RGB');im.thumbnail((500,500),Image.Resampling.LANCZOS);sheet.paste(im,((i%3)*500,(i//3)*500))
 sheet.save(Q/'six_panels_review.png')
 (Q/'technical_audit.json').write_text(json.dumps(dict(status='VECTOR_DIMENSIONS_AND_TEXT_PASS',records=records,warning='96dpi images approximate paper size on a96dpi display; monitor calibration and viewer zoom are unknown. PDF page sizes and point sizes are exact. Short CIs can be narrower than markers; caps remain at their true positions.'),indent=2)+'\n', encoding='utf8', newline='\n')
 print('PASS',len(records))
if __name__=='__main__':main()
