"""Publish lossless plain-text tables within the repository's file-size limit."""
from pathlib import Path
import csv
import gzip
import io
import json
import shutil
from uep_common import sha,write,require

LIMIT=5_000_000

def copy_light(source,target,limit=LIMIT):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    compressed=source.suffix=='.gz'
    if target.suffix=='.gz':target=target.with_suffix('')
    if not compressed and source.stat().st_size<=limit:
        shutil.copyfile(source,target);return [target]
    opener=gzip.open if compressed else open
    with opener(source,'rt',encoding='utf-8-sig',newline='') as f:
        if target.suffix=='.json':
            data=json.load(f);require(isinstance(data,list),'Large JSON must be a record array')
            header='[';footer=']\n';records=(json.dumps(x,ensure_ascii=False,separators=(',',':')) for x in data)
            separator=',';kind='json_record_array'
        elif target.suffix=='.csv':
            reader=csv.reader(f);fields=next(reader)
            def line(values):
                s=io.StringIO(newline='');csv.writer(s,lineterminator='\n').writerow(values);return s.getvalue()
            header=line(fields);footer='';separator='';records=(line(values) for values in reader);kind='csv_same_header_each_part'
        else:raise RuntimeError('Unsupported large publication table: '+str(source))
        parts=[];chunk=[];used=len((header+footer).encode());count=0
        def flush():
            if not chunk:return
            path=target.with_name(target.stem+'.part%03d'%len(parts)+target.suffix)
            path.write_text(header+separator.join(chunk)+footer,encoding='utf-8',newline='')
            require(path.stat().st_size<=limit,'Chunk exceeds repository limit')
            parts.append(dict(path=path.name,rows=len(chunk),sha256=sha(path),bytes=path.stat().st_size))
        for record in records:
            size=len(record.encode())+(len(separator) if chunk else 0)
            require(size+len((header+footer).encode())<=limit,'One record exceeds publication limit')
            if chunk and used+size>limit:flush();chunk=[];used=len((header+footer).encode());size=len(record.encode())
            chunk.append(record);used+=size;count+=1
        flush()
    index=target.with_name(target.stem+'.index.json')
    write(index,dict(format=kind,original_path=str(source),original_sha256=sha(source),original_compressed=compressed,
        rows=count,parts=parts,lossless_records=True,interpolation=False))
    return [target.parent/p['path'] for p in parts]+[index]
