"""Convert the native PowerPoint PDF to vector SVG with outlined labels; native text remains editable in PPTX; no inference."""
from pathlib import Path
import pymupdf
P=Path(__file__).resolve().parents[1]
doc=pymupdf.open(P/'figures/figure1.pdf')
assert len(doc)==1
(P/'figures/figure1.svg').write_text(doc[0].get_svg_image(text_as_path=True),encoding='utf8')
doc[0].get_pixmap(dpi=600).save(P/'figures/figure1.png')
