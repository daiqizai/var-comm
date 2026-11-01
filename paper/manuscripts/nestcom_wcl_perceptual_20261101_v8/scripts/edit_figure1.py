"""Reproduce the three native text edits, preserving the original slide geometry."""
from pathlib import Path
import hashlib,io,json,zipfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]
src=P/'inputs/figure1_v7_editable.pptx';dst=P/'figures/figure1.pptx'
changes={'651 endpoints; schematic':'Schematic bars'}
buffer=io.BytesIO()
with zipfile.ZipFile(src)as z,zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)as out:
    for item in z.infolist():
        b=z.read(item.filename)
        if item.filename=='ppt/slides/slide1.xml':
            s=b.decode();counts={k:s.count(k)for k in changes}
            assert set(counts.values())=={1},counts
            for a,b2 in changes.items():s=s.replace(a,b2)
            b=s.encode();ET.fromstring(b)
        out.writestr(item,b)
data=buffer.getvalue()
dst.write_bytes(data)
record=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(data).hexdigest(),changes=changes,
            geometry_and_images_preserved=True,native_editable_text=True)
(P/'validation/figure1_edit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
