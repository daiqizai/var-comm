"""Lossless publication packaging of an already completed read-only audit.

No experiment, source decoding, metrics or resampling runs here. The original
CSV is retained outside Git; the two budget-specific CSVs satisfy the existing
10 MB repository limit. Run after read_only_audit.py, on its completed output.
"""
from pathlib import Path
import csv, hashlib, io, json, sys

def sha(data):
    return hashlib.sha256(data).hexdigest()

def split(source, destination):
    source=Path(source);destination=Path(destination)
    raw=source.read_bytes()
    with source.open(encoding='utf-8',newline='')as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    # The schema is taken from the completed audit rather than inferred by order.
    budget_field=next(k for k in fields if k.lower() in ('n','budget','n_symbols'))
    parts=[]
    destination.mkdir(parents=True,exist_ok=True)
    for budget in ('512','1024'):
        indices=[i for i,r in enumerate(rows)if r[budget_field]==budget]
        subset=[rows[i]for i in indices]
        assert len(subset)=={'512':13500,'1024':9000}[budget]
        name=f'formal_event_mapping_n{budget}.csv';p=destination/name
        buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(subset);data=buffer.getvalue().encode('utf-8')
        assert len(data)<=10_000_000
        with p.open('xb')as f:f.write(data)
        with p.open(encoding='utf-8',newline='')as f:assert list(csv.DictReader(f))==subset
        parts.append(dict(path=name,budget=int(budget),rows=len(subset),bytes=len(data),sha256=sha(data),original_zero_based_row_indices=indices))
    assert sum(p['rows']for p in parts)==len(rows)==22500
    index=dict(operation='lossless CSV row partition by N; original order preserved within each part',original=dict(name=source.name,bytes=len(raw),sha256=sha(raw),rows=len(rows)),budget_field=budget_field,columns=fields,parts=parts,scientific_calls=0,metrics_recomputed=False)
    with(destination/'formal_event_mapping_index.json').open('x',encoding='utf-8',newline='\n')as f:json.dump(index,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(rows),parts=[{k:v for k,v in p.items()if k!='original_zero_based_row_indices'}for p in parts])))

if __name__=='__main__':split(sys.argv[1],sys.argv[2])
