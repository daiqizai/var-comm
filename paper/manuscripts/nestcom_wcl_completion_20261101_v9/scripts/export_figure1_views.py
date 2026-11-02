"""Derive vector/600 dpi/print previews from the native PowerPoint PDF export.

Usage: python scripts/export_figure1_views.py exported/figure1.pdf new_output_dir
SVG glyphs are paths. The PPTX remains the editable text/diagram source.
"""
from pathlib import Path
import argparse,fitz,json,hashlib
from PIL import Image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('pdf',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    doc=fitz.open(args.pdf);assert len(doc)==1
    p=doc[0]
    (args.output/'figure1.svg').write_text(p.get_svg_image(text_as_path=True),encoding='utf8')
    for name,dpi in [('figure1.png',600),('figure1_print_size.png',150)]:
        p.get_pixmap(dpi=dpi).save(args.output/name)
    Image.open(args.output/'figure1_print_size.png').convert('L').save(args.output/'figure1_grayscale.png')
    report=dict(pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),pdf_points=list(p.rect),pdf_fonts=p.get_fonts(),svg_text_as_paths=True)
    (args.output/'figure1_export.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
