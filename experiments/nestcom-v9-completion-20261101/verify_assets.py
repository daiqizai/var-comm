"""Read-only external asset resolution and SHA256 verification; never downloads."""
from pathlib import Path
import argparse,hashlib,json,os

def pins(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and len(value['sha256'])==64:
            yield value
        for x in value.values():yield from pins(x)
    elif isinstance(value,list):
        for x in value:yield from pins(x)

def verify(manifest,root=None,overrides=None,only=None):
    mapping=manifest['reader_asset_mapping'];allpins=list(pins(manifest));out=[]
    names=only or list(mapping)
    for name in names:
        if name not in mapping:raise ValueError('Unknown asset '+name)
        expected={p['sha256'] for p in allpins if Path(p['path']).name==name}
        if len(expected)!=1:raise ValueError('Missing/ambiguous file hash for '+name)
        path=Path((overrides or {}).get(name,str(Path(root)/name) if root else mapping[name]))
        if not path.is_file():raise FileNotFoundError(path)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        actual=digest.hexdigest()
        if actual!=next(iter(expected)):raise ValueError('File SHA256 mismatch for '+name)
        out.append(dict(name=name,path=str(path.resolve()),bytes=path.stat().st_size,sha256=actual))
    return dict(status='PASS',scope='file identity only; no tensor loading or inference',files=out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--asset-root',default=os.environ.get('NESTCOM_ASSET_ROOT'))
    p.add_argument('--overrides',default=os.environ.get('NESTCOM_ASSETS_JSON'))
    p.add_argument('--only',nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());ov=json.loads(Path(a.overrides).read_text()) if a.overrides else {}
    result=verify(m,a.asset_root,ov,a.only)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],verified_files=len(result['files']))))
if __name__=='__main__':main()
