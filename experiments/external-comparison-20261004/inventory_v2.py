"""Read-only discovery of local external baselines; never read holdout images."""
from pathlib import Path
import collections
import datetime
import json
import os
import re
import subprocess
import time

ROOT = Path('/home/liulu/projects/VAR_COMM')
OUT = ROOT/'outputs/EXTERNAL-COMPARISON-20261004/inventory_v2'
PATTERNS = {
    'SwinJSCC': r'swin[-_ ]?jscc', 'SGD-JSCC': r'sgd[-_ ]?jscc',
    'ADJSCC': r'adjscc', 'DeepJSCC': r'deep[-_ ]?jscc',
    'DiffCom': r'diffcom', 'NTSCC': r'ntscc', 'WITT': r'\bwitt\b',
    'MambaJSCC': r'mamba[-_ ]?jscc', 'DiffJSCC': r'diff[-_ ]?jscc',
    'DiT-JSCC': r'dit[-_ ]?jscc',
}
REGEX = {k: re.compile(v, re.I) for k,v in PATTERNS.items()}
ANY_METHOD = re.compile('|'.join('(?:'+v+')' for v in PATTERNS.values()), re.I)
WEIGHTS = {'.pt','.pth','.ckpt','.safetensors','.h5','.hdf5','.model','.onnx'}
TEXT = {'.py','.md','.yaml','.yml','.json','.toml','.ini','.cfg','.txt','.log','.sh'}
OMIT = {'.git','__pycache__','site-packages','node_modules','.venv','venv','environment','.pytest_cache'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roots = [Path('/home/liulu/projects')]
    for name in ['/home/liulu/.cache/huggingface/hub','/home/liulu/.cache/torch/hub/checkpoints',
                 '/home/liulu/models','/home/liulu/weights','/home/liulu/checkpoints']:
        if Path(name).is_dir(): roots.append(Path(name))
    started = time.time()
    counts=collections.Counter(); matches=collections.Counter(); errors=[]; omitted=[]; repos=[]; symlinks=[]
    def error(e): errors.append(str(e))
    with (OUT/'files.jsonl').open('w') as stream:
        def emit(record): stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        for scan_root in roots:
            for directory, dirs, names in os.walk(scan_root,followlinks=False,onerror=error):
                parent=Path(directory)
                if '.git' in dirs or '.git' in names: repos.append(str(parent))
                kept=[]
                for name in dirs:
                    p=parent/name
                    if 'holdout' in name.lower() or name in OMIT:
                        omitted.append(str(p)); continue
                    if p.is_symlink(): symlinks.append(dict(path=str(p),target=str(p.resolve()))); continue
                    kept.append(name)
                dirs[:]=kept
                for name in names:
                    if 'holdout' in name.lower(): continue
                    p=parent/name; counts['files_enumerated']+=1
                    try:
                        suffix=p.suffix.lower(); stat=p.stat()
                        method_names={k for k,v in REGEX.items() if v.search(str(p))}
                        is_weight=suffix in WEIGHTS or name.endswith(('.pth.tar','.pt.tar')) or (name.endswith('.bin') and any(s in str(p).lower() for s in ('checkpoint','model','weight','huggingface')))
                        if is_weight:
                            counts['weight_files']+=1
                            emit(dict(kind='weight',path=str(p),size=stat.st_size,methods=sorted(method_names),symlink=p.is_symlink()))
                        is_result=suffix in {'.csv','.json','.npz','.png','.jpg','.webp'} and any(s in str(p).lower() for s in ('result','output','recon'))
                        if is_result and method_names:
                            counts['named_results']+=1
                            emit(dict(kind='named_result',path=str(p),size=stat.st_size,methods=sorted(method_names)))
                        if suffix not in TEXT or stat.st_size>8_000_000: continue
                        if 'EXTERNAL-COMPARISON-20261004' in str(p): continue
                        if 'source_checkpoints' in p.parts: continue
                        # Scan source, config and lightweight evidence. Never open RGB/data archives.
                        data=p.read_bytes(); counts['text_files_read']+=1; counts['text_bytes_read']+=len(data)
                        if b'\x00' in data[:4096]: continue
                        text=data.decode('utf-8',errors='replace')
                        content_names={k for k,v in REGEX.items() if v.search(text)} if ANY_METHOD.search(text) else set()
                        found=method_names|content_names
                        if found:
                            counts['matched_text_files']+=1; matches.update(found)
                            emit(dict(kind='text',path=str(p),size=stat.st_size,methods=sorted(found),content_matches=sorted(content_names)))
                    except (OSError,ValueError) as e: errors.append(str(e))
                if counts['files_enumerated']%1000==0:
                    stream.flush()
    versions=[]
    for repo in repos:
        def git(*args):
            q=subprocess.run(['git','-C',repo,*args],capture_output=True,text=True,timeout=10)
            return q.stdout.strip() if q.returncode==0 else None
        try: versions.append(dict(path=repo,commit=git('rev-parse','HEAD'),origin=git('remote','get-url','origin')))
        except subprocess.TimeoutExpired: errors.append('git timeout '+repo)
    result=dict(status='DISCOVERY_COMPLETE',observed_at=datetime.datetime.now().astimezone().isoformat(),
        elapsed_seconds=time.time()-started,roots=list(map(str,roots)),counts=dict(counts),methods=dict(matches),
        excluded_directories=omitted,symlink_directories=symlinks,repositories=versions,errors=errors,
        scope='All project filenames; code/config/evidence text <=8MB excluding per-source scoring checkpoints; all weight metadata including .model/.pth.tar; no holdout directory or holdout-named file opened',
        assets_hashed=False,scientific_result=False)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts','methods','elapsed_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
