"""Check the delivered PDF/SVG/PNG containers and make separate review views."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,math
from PIL import Image,ImageOps,ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def luminance(c):
    rgb=[int(c[i:i+2],16)/255 for i in (1,3,5)]
    return sum(w*(x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4)for w,x in zip((.2126,.7152,.0722),rgb))
def main():
    P=ROOT/'previews';Q=ROOT/'review';Q.mkdir(exist_ok=True)
    ex=json.loads((P/'export_manifest.json').read_bytes());files=[]
    for r in ex['exports']:
      p=P/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
      result=dict(file=r['path'],sha256=sha(p),width_mm=r['width_mm'],height_mm=r['height_mm'])
      if p.suffix=='.pdf':
        reader=PdfReader(p);assert len(reader.pages)==1;page=reader.pages[0]
        assert abs(float(page.mediabox.width)/72*25.4-r['width_mm'])<.001
        assert abs(float(page.mediabox.height)/72*25.4-r['height_mm'])<.001
        assert len(page.images)==0
        text=page.extract_text();assert 'SNR' in text and len(text)>30
        fonts=[]
        for k,obj in page['/Resources']['/Font'].items():
          f=obj.get_object();fonts.append(dict(basefont=str(f['/BaseFont']),subtype=str(f['/Subtype'])))
          assert str(f['/Subtype'])!='/Type3'
        result.update(raster_images=0,searchable_text=True,fonts=fonts)
      elif p.suffix=='.svg':
        t=ET.parse(p).getroot();tags=[e.tag.split('}')[-1]for e in t.iter()]
        assert 'image'not in tags and 'text'in tags and 'path'in tags
        result.update(raster_images=0,editable_text_count=tags.count('text'),vector_paths=tags.count('path'))
      else:
        im=Image.open(p);dpi=im.info['dpi'];assert all(abs(x-600)<.1 for x in dpi)
        assert abs(im.width-r['width_mm']/25.4*600)<2 and abs(im.height-r['height_mm']/25.4*600)<2
        if im.mode=='RGBA':assert im.getchannel('A').getextrema()==(255,255)
        result.update(pixels=list(im.size),dpi=list(dpi),opaque=True)
      files.append(result)
    for style in ('A','B'):
      source=P/('fig2_'+style+'.png');im=Image.open(source).convert('RGB')
      im.thumbnail((1500,1400),Image.Resampling.LANCZOS);im.save(Q/('fig2_'+style+'_screen.png'))
      original=Image.open(source).convert('RGB');dim=(round(original.width*96/600),round(original.height*96/600))
      small=original.resize(dim,Image.Resampling.LANCZOS);small.save(Q/('fig2_'+style+'_96dpi_size_check.png'))
      ImageOps.grayscale(small).save(Q/('fig2_'+style+'_96dpi_grayscale.png'))
      panels=[]
      for metric in ['psnr','lpips','dreamsim','prediction_agreement']:
        im=Image.open(P/('fig2_'+style+'_'+metric+'.png')).convert('RGB');im.thumbnail((600,500),Image.Resampling.LANCZOS);panels.append(im)
      sheet=Image.new('RGB',(1200,1000),'white')
      for i,im in enumerate(panels):sheet.paste(im,((i%2)*600,(i//2)*500))
      sheet.save(Q/('fig2_'+style+'_individual_panels_review.png'))
    methods=json.loads((ROOT/'methods.json').read_text())
    palette=[dict(method=x['label'],color=x['color'],contrast_on_white=1.05/(luminance(x['color'])+.05),redundancy=[x['marker'],x['linestyle'],x['fill']])for x in methods]
    report=dict(status='CONTAINER_AND_DIMENSION_CHECKS_PASS_MANUAL_REVIEW_REQUIRED',files=files,palette=palette,disclaimers=['96 dpi views approximate physical size on a96dpi display; monitor calibration and viewer zoom are unknown. PDF physical dimensions and point sizes are exact.','Grayscale inspection does not establish complete color-vision/accessibility compliance.','Short CIs can be narrower than markers and overlapping CIs are intentionally not separated by changing their coordinates.'],new_metric_calls=0,new_bootstrap_calls=0)
    (Q/'technical_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],files=len(files),palette=palette)))
if __name__=='__main__':main()
