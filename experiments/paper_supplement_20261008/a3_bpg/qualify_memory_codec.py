"""20 original calibration images, QP29/51: real CLI vs memory byte/pixel parity."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from memory_codec import MemoryBPG, png_bytes

REQUEST_SHAS={'7670f623c093bf62e8865d2618eb35c0ca43ed5ee9af75a77a3d210f41300e30',
              'd4fc512cb0970e5826c79287e7e64bbf3da5d4355fa7cc94951343b0ec4fb833'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(v,message):
    if not v:raise ValueError(message)

def prepare(args):
    original,build=Path(args.original_request).resolve(),Path(args.build_receipt).resolve()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    require(sha(original)in REQUEST_SHAS,'Not an original frozen BPG request')
    cfg,b=read(original),read(build)
    require(sha(b['library'])==b['library_sha256'],'Changed memory library')
    records=cfg['records']['calibration'][:20]
    require(len(records)==20 and len({x['source_id'] for x in records})==20,'Need20 original calibration sources')
    bindings={str(original):sha(original),str(build):sha(build),b['library']:b['library_sha256'],
        str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('memory_codec.py').resolve()):sha(Path(__file__).with_name('memory_codec.py'))}
    for key in ['bpgenc','bpgdec']:
        require(sha(cfg[key]['path'])==cfg[key]['sha256'],'Original CLI binary changed')
        bindings[cfg[key]['path']]=cfg[key]['sha256']
    for rec in records:
        for key in ['archive','checkpoint']:
            require(sha(rec[key])==rec[key+'_sha256'],'Original calibration source changed')
            bindings[rec[key]]=rec[key+'_sha256']
    q=dict(status='PREPARED_NOT_RUN',version='A3-BPG-MEMORY-PARITY-20261009-V1',output=str(out),
        records=records,source_count=20,source_population='original calibration first20',qps=[29,51],
        bpgenc=cfg['bpgenc'],bpgdec=cfg['bpgdec'],library=b['library'],bindings=bindings,
        actual_memory_encodes_planned=40,actual_original_CLI_encodes_planned=40,
        require_complete_byte_identity=True,require_exact_uint8_pixel_identity=True,
        paid_phy_decodes=0,new_channel_draws=0,training_updates=0,holdout_pixels_read=False,
        timing_use='Qualification elapsed is not deployment latency. CLI file I/O is used only here for parity.')
    target=out/'request.json'
    if target.exists():require(read(target)==q,'Changed qualification request')
    else:write(target,q)
    print(json.dumps({'status':'PREPARED','request':str(target)}),flush=True)

def run(args):
    q=read(args.request);out=Path(q['output'])
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in q['bindings'].items():require(sha(p)==h,'Qualification input changed: '+p)
    if (out/'completion.json').exists():
        done=read(out/'completion.json')
        require(done['request_sha256']==sha(args.request),'Completed qualification changed')
        for p,h in done['outputs'].items():require(sha(p)==h,'Completed qualification output changed')
        print(json.dumps({'status':'REUSE_COMPLETE','source_count':20}));return
    require(not (out/'failure.json').exists(),'Previous parity failure needs diagnosis; no silent retry')
    codec=MemoryBPG(q['library']);rows=[];began=time.time()
    try:
        for i,rec in enumerate(q['records']):
            with np.load(rec['archive'],allow_pickle=False) as z:pixels=z['pixels'].copy()
            require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Original calibration RGB changed')
            require(hashlib.sha256(pixels.tobytes()).hexdigest()==rec['preprocessing_id'],'Original preprocessing changed')
            rgb=np.ascontiguousarray(pixels.transpose(1,2,0))
            directory=out/f'source{i:04d}';directory.mkdir(exist_ok=True)
            png=directory/'source.png';png.write_bytes(png_bytes(rgb))
            for qp in q['qps']:
                target,ppm=directory/f'qp{qp}.bpg',directory/f'qp{qp}.ppm'
                enc=[q['bpgenc']['path'],'-e','x265','-m','8','-f','420','-c','ycbcr','-b','8','-q',str(qp),'-o',str(target),str(png)]
                dec=[q['bpgdec']['path'],'-o',str(ppm),str(target)]
                with (directory/f'qp{qp}_cli.log').open('wb')as log:
                    subprocess.run(enc,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                    subprocess.run(dec,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                stream=codec.encode(rgb,qp)
                decoded=codec.decode(stream)
                with Image.open(ppm)as im:cli_rgb=np.asarray(im).copy()
                row=dict(source_index=i,source_id=rec['source_id'],qp=qp,
                    complete_bytes=len(stream),cli_bytes=target.stat().st_size,
                    complete_byte_identity=stream==target.read_bytes(),
                    exact_uint8_pixel_identity=decoded is not None and np.array_equal(decoded,cli_rgb),
                    memory_stream_sha256=hashlib.sha256(stream).hexdigest(),cli_stream_sha256=sha(target),
                    memory_rgb_sha256=hashlib.sha256(decoded.tobytes()).hexdigest() if decoded is not None else None,
                    cli_rgb_sha256=hashlib.sha256(cli_rgb.tobytes()).hexdigest(),
                    original_cli_encoder_argv=enc,original_cli_decoder_argv=dec)
                rows.append(row);write(directory/f'qp{qp}_parity.json',row)
                require(row['complete_byte_identity'] and row['exact_uint8_pixel_identity'],'Actual BPG memory/CLI parity mismatch')
            print(json.dumps({'sources_completed':i+1,'source_count':20,'status':'PASS'}),flush=True)
        fields=[k for k in rows[0] if not k.endswith('_argv')]
        with (out/'parity.csv').open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
        outputs={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['run.lock','completion.json']}
        write(out/'completion.json',dict(status='COMPLETE',source_count=20,case_count=40,qps=[29,51],
            actual_memory_encodes=40,actual_CLI_encodes=40,actual_memory_decodes=40,actual_CLI_decodes=40,
            complete_byte_identity_all=True,exact_uint8_pixel_identity_all=True,request_sha256=sha(args.request),
            library=q['library'],library_sha256=sha(q['library']),outputs=outputs,
            paid_phy_decodes=0,new_channel_draws=0,training_updates=0,holdout_pixels_read=False,
            qualification_elapsed_seconds=time.time()-began,deployment_timing_measured=False))
        print(json.dumps({'status':'COMPLETE','source_count':20,'cases':40,'paid_phy_decodes':0}),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(status='FAILED',error=repr(e),finished_cases=len(rows),paid_phy_decodes=0))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    z=s.add_parser('prepare');z.add_argument('--original-request',required=True);z.add_argument('--build-receipt',required=True);z.add_argument('--output',required=True)
    z=s.add_parser('run');z.add_argument('--request',required=True)
    a=p.parse_args();(prepare if a.command=='prepare' else run)(a)
