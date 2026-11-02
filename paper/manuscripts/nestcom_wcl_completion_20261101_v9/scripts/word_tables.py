"""Native Word tables from the frozen table CSVs, independent of Pandoc tabular parsing."""
from pathlib import Path
import csv
import hashlib
import re

from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


def braced_argument(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != '{':
        raise ValueError('Expected a braced TeX argument')
    depth = 1
    end = start + 1
    while depth and end < len(text):
        if text[end] == '\\':
            end += 2
            continue
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth:
        raise ValueError('Unclosed TeX argument')
    return text[start + 1:end - 1], end


def plain_caption(text):
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    text = text.replace('---', '—').replace('--', '–')
    replacements = {
        r'\%': '%', r'\uparrow': '↑', r'\downarrow': '↓',
        r'\Delta': 'Δ', r'\to': '→', r'\footnotesize': '', r'\scriptsize': '',
        r'\smallskip': '', r'\par': ' ', r'\,': ' ', r'\;': ' ',
        '~': ' ', '$': '',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r'\s+', ' ', text).strip()
    if '\\' in text or '{' in text or '}' in text:
        raise ValueError('Unconverted TeX in table caption or note: ' + text)
    return text


def table_spec(tex, number):
    marker = 'NESTNATIVETABLE' + ('ONE', 'TWO', 'THREE')[number - 1]
    cap_start = tex.index(r'\caption') + len(r'\caption')
    caption, _ = braced_argument(tex, cap_start)
    note = ''
    if r'\parbox' in tex:
        start = tex.index(r'\parbox') + len(r'\parbox')
        _, end = braced_argument(tex, start)
        note, _ = braced_argument(tex, end)
    elif r'\begin{minipage}' in tex:
        start = tex.index(r'\begin{minipage}') + len(r'\begin{minipage}')
        _, end = braced_argument(tex, start)
        note = tex[end:tex.index(r'\end{minipage}', end)]
    return dict(number=number, marker=marker,
                caption=plain_caption(caption), note=plain_caption(note))


def read_csv(path):
    with path.open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def unique(rows, **keys):
    match = [row for row in rows if all(row[key] == str(value) for key, value in keys.items())]
    if len(match) != 1:
        raise ValueError((keys, len(match)))
    return match[0]


def paired_cell(row, multiline=False):
    mean = f"{float(row['mean']):+.4f}"
    bounds = f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"
    return mean + ('\n' if multiline else ' ') + bounds


def data_rows(number, rows):
    def pair(mean,low,high,n=4):
        return f"{float(mean):+.{n}f}\n[{float(low):.{n}f}, {float(high):.{n}f}]"
    labels={'final_NeST':'NeST-Com','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada-TokenCom','Latent_JSCC':'Latent JSCC','SwinJSCC':'SwinJSCC','BPG_LDPC':'BPG + LDPC','ARPC':'ARPC + LDPC'}
    if number==1:
        assert len(rows)==10
        yield 'group',['(A) 500 sources, N = 1024, family-specific policies']
        yield 'header',['SNR','Util. W→N (%)','ΔDISTS [95% CI]','ΔDreamSim [95% CI]']
        for s in [4,7,10,19]:
            r=unique(rows,block='A',snr_db=s)
            yield 'data',[str(s),f"{100*float(r['whole_utilization']):.2f} → {100*float(r['nest_utilization']):.2f}",pair(r['dists_delta'],r['dists_low'],r['dists_high']),pair(r['dreamsim_delta'],r['dreamsim_low'],r['dreamsim_high'])]
        yield 'group',['(B) 100 quality sources; 32 timing sources; ceiling (8,126)']
        yield 'header',['Grid','Utilization (%)','DISTS','DreamSim','TX (ms)']
        for m in ['whole','quarter','eighth','token']:
            r=unique(rows,block='B',grid=m)
            yield 'data',[m.title(),f"{100*float(r['utilization']):.2f}",f"{float(r['dists']):.4f}",f"{float(r['dreamsim']):.4f}",f"{float(r['tx_ms']):.1f}"]
        yield 'group',['(C) 100 sources, 10 dB, same received prefixes']
        yield 'header',['Intervention','ΔPSNR (dB)','ΔDISTS','ΔDreamSim']
        for c,b in [('B0','A0'),('C0','B0')]:
            r=unique(rows,block='C',candidate=c,reference=b)
            yield 'data',[c+'−'+b]+[pair(r[m+'_mean_delta'],r[m+'_ci95_low'],r[m+'_ci95_high'],3 if m=='psnr_db'else 4)for m in ['psnr_db','dists','dreamsim']]
    elif number==2:
        assert len(rows)==7
        yield 'header',['Method','PSNR↑ (dB)','LPIPS↓','CLIP-I↑','TX (ms)','RX (ms)']
        for r in rows:
            # Preserve the seven-row input; report the ARPC adaptation in the supplement.
            if r['method']=='ARPC':
                continue
            yield 'data',[labels[r['method']],f"{float(r['psnr_db']):.2f}",f"{float(r['lpips_alex']):.3f}",f"{float(r['clip_image_cosine']):.3f}"]+[f"{float(r[k]):.1f}"if r[k]else'—'for k in ['tx_ms','rx_ms']]
    elif number==3:
        assert len(rows)==9
        yield 'header',['SNR / method','PSNR↑ (dB)','DISTS↓','DreamSim↓']
        for s in [4,10,19]:
            for m in ['WHOLE_final_receiver','final_NeST','Ada_frozen']:
                r=unique(rows,snr_db=s,method=m)
                yield 'data',[str(s)+' / '+labels[m],f"{float(r['psnr_db']):.2f}",f"{float(r['dists']):.4f}",f"{float(r['dreamsim']):.4f}"]
            # Same direct paired source as the TeX row, not a marginal subtraction.
            q=unique(read_csv(Path(__file__).resolve().parents[1]/'inputs/e1/paired_deltas.csv'),N=512,snr_db=s,candidate='final_NeST',reference='WHOLE_final_receiver',metric='psnr_db')
            yield 'group',[f"NeST−WHOLE ΔPSNR: {float(q['mean_delta']):+.3f} [{float(q['ci95_low']):.3f}, {float(q['ci95_high']):.3f}] dB"]
    else:raise ValueError(number)


def paragraph_format(paragraph, *, bold=False, left=False, keep_next=True):
    fmt = paragraph.paragraph_format
    fmt.line_spacing = 1.0
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.keep_together = True
    fmt.keep_with_next = keep_next
    fmt.widow_control = True
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
    for run in paragraph.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.bold = bold


def fill_cell(cell, text, *, kind, left=False):
    cell.text = text
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for paragraph in cell.paragraphs:
        paragraph_format(paragraph, bold=kind != 'data', left=left)
    props = cell._tc.get_or_add_tcPr()
    margins = OxmlElement('w:tcMar')
    for edge, amount in [('top', 40), ('bottom', 40), ('left', 65), ('right', 65)]:
        item = OxmlElement('w:' + edge)
        item.set(qn('w:w'), str(amount))
        item.set(qn('w:type'), 'dxa')
        margins.append(item)
    props.append(margins)
    if kind in ('group', 'header'):
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), 'EDEDED' if kind == 'group' else 'F5F5F5')
        shading.set(qn('w:val'), 'clear')
        props.append(shading)


def insert_csv_tables(document, manuscript, specs):
    records = []
    widths_by_number = {1: [0.85, 1.35, 1.6, 1.7, 1.0], 2: [1.5, 1, 1, 1, 1, 1], 3: [1.9, 1.3, 1.65, 1.65]}
    assert len(specs) == 3
    for spec in specs:
        number = spec['number']
        source = Path(manuscript) / 'tables' / ('table_' + ('i', 'ii', 'iii')[number - 1] + '.csv')
        values = read_csv(source)
        content = list(data_rows(number, values))
        if number==1: content=[(kind, texts+['']*(5-len(texts)) if kind!='group' else texts) for kind,texts in content]
        markers = [p for p in document.paragraphs if p.text.strip() == spec['marker']]
        if len(markers) != 1:
            raise ValueError(('Table placeholder count', spec['marker'], len(markers)))
        anchor = markers[0]._p
        caption = document.add_paragraph(spec['caption'], style='Caption')
        paragraph_format(caption, left=True)
        caption.paragraph_format.space_before = Pt(6)
        caption.paragraph_format.space_after = Pt(4)
        anchor.addprevious(caption._p)

        widths = widths_by_number[number]
        assert sum(widths) == 6.5
        table = document.add_table(rows=0, cols=len(widths))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        props = table._tbl.tblPr
        width = props.find(qn('w:tblW'))
        width.set(qn('w:w'), '9360')
        width.set(qn('w:type'), 'dxa')
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
            item = OxmlElement('w:' + edge)
            item.set(qn('w:val'), 'single')
            item.set(qn('w:sz'), '4')
            item.set(qn('w:color'), 'D9D9D9')
            borders.append(item)
        props.append(borders)
        for kind, texts in content:
            row = table.add_row()
            row_props = row._tr.get_or_add_trPr()
            row_props.append(OxmlElement('w:cantSplit'))
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
            if kind == 'group':
                cell = row.cells[0].merge(row.cells[-1])
                fill_cell(cell, texts[0], kind=kind, left=True)
            else:
                assert len(texts) == len(widths)
                for index, (cell, text) in enumerate(zip(row.cells, texts)):
                    fill_cell(cell, text, kind=kind, left=number == 2 and index == 0)
                if number==1 and texts[-1]=='':
                    row.cells[3].merge(row.cells[4])
        # Keep each self-labelled block together, but permit a page break before
        # the next block instead of forcing the whole multi-block table forward.
        if number == 1:
            for index in range(len(content) - 1):
                if content[index + 1][0] == 'group':
                    for cell in table.rows[index].cells:
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.keep_with_next = False
        if number == 3:
            for index, (kind, _) in enumerate(content):
                if kind == 'group':
                    for cell in table.rows[index].cells:
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.keep_with_next = False
        anchor.addprevious(table._tbl)
        if spec['note']:
            note = document.add_paragraph(spec['note'])
            paragraph_format(note, left=True, keep_next=False)
            note.paragraph_format.space_before = Pt(4)
            note.paragraph_format.space_after = Pt(8)
            anchor.addprevious(note._p)
        else:
            for cell in table.rows[-1].cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = False
            following = anchor.getnext()
            if following is not None and following.tag == qn('w:p'):
                Paragraph(following, document).paragraph_format.space_before = Pt(6)
        anchor.getparent().remove(anchor)
        records.append(dict(number=number, source_csv=str(source.relative_to(manuscript)),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            source_records=len(values), word_rows=len(table.rows),
                            word_columns=len(table.columns), width_inches=sum(widths),
                            caption=spec['caption'], note=spec['note'],
                            cells=[dict(kind=kind, text=texts) for kind, texts in content]))
    assert len(document.tables) == 3
    return records
