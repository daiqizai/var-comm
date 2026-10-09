"""Restore three byte-identical scientific tables from ordered text parts.

Uses Python's standard library only. Existing identical files are reused;
different existing files are never overwritten. No scientific computation runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,relative):
    candidate=(root/relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Path leaves requested directory: '+relative)
    return candidate


def restore(parts_dir,output_root):
    parts_dir=Path(parts_dir).resolve();output_root=Path(output_root).resolve()
    manifest=json.loads((parts_dir/'manifest.json').read_text(encoding='utf8'))
    if manifest['schema']!='LOSSLESS_LARGE_SCIENTIFIC_TABLE_PARTS_V1':
        raise ValueError('Unexpected manifest schema')
    results=[]
    for item in manifest['files']:
        target=safe_path(output_root,item['original_relative_path'])
        if target.exists():
            if not target.is_file()or target.stat().st_size!=item['original_bytes']or digest(target)!=item['original_sha256']:
                raise FileExistsError('Refuse to replace different existing file: '+str(target))
            results.append(dict(path=str(target),status='REUSED_IDENTICAL',sha256=item['original_sha256']))
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            total=0;h=hashlib.sha256()
            with tempfile.NamedTemporaryFile(mode='wb',dir=target.parent,prefix=target.name+'.restoring-',delete=False)as stream:
                temporary=Path(stream.name)
                for number,part in enumerate(item['parts'],1):
                    if part['order']!=number:raise ValueError('Part order differs')
                    path=safe_path(parts_dir,part['path']);data=path.read_bytes()
                    if len(data)!=part['bytes']or hashlib.sha256(data).hexdigest()!=part['sha256']:
                        raise ValueError('Part length/hash differs: '+str(path))
                    data.decode('utf8')
                    stream.write(data);h.update(data);total+=len(data)
                stream.flush();os.fsync(stream.fileno())
            if total!=item['original_bytes']or h.hexdigest()!=item['original_sha256']:
                raise ValueError('Restored original hash differs: '+str(target))
            # Preserve an independently created destination, even if it appeared
            # after the first existence check.
            if target.exists():raise FileExistsError('Destination appeared during restore: '+str(target))
            temporary.rename(target);temporary=None
            results.append(dict(path=str(target),status='RESTORED',sha256=h.hexdigest(),bytes=total))
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[3],
                        help='Repository root or fresh temporary destination')
    args=parser.parse_args()
    result=restore(Path(__file__).resolve().parent,args.output_root)
    print(json.dumps(dict(status='COMPLETE',files=result),indent=2))


if __name__=='__main__':main()
