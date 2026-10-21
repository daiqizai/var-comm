"""Refresh only the Chinese Markdown outline from the reviewed section data.
Captions, the Word file and existing LaTeX source remain independently editable.
"""
from pathlib import Path
import json
P=Path(__file__).resolve().parent.parent
sections=json.loads((P/'outline_sections.json').read_text(encoding='utf8'))
lines=['# NeST-Com: Fixed-Budget Next-Scale Token Communication','']
for title,points in sections:
    lines += ['## '+title,'']
    for heading,body in points:
        lines += ['**'+heading+'。** '+body,'']
(P/'outline_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
