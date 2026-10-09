"""Build separate PIC objects from unchanged0.9.8 source and original x265 archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

DECODER = ('libavcodec/hevc_cabac libavcodec/hevc_filter libavcodec/hevc libavcodec/hevcpred '
    'libavcodec/hevc_refs libavcodec/hevcdsp libavcodec/hevc_mvs libavcodec/hevc_ps libavcodec/hevc_sei '
    'libavcodec/utils libavcodec/cabac libavcodec/golomb libavcodec/videodsp libavutil/mem '
    'libavutil/buffer libavutil/log2_tab libavutil/frame libavutil/pixdesc libavutil/md5 libbpg').split()
ENCODER_SHA='dd3fe49a61e0731c41ee32288911ecd6976f979c2c2f0ad38f66dd00be367362'
GLUE_SHA='999f7f5f1e75376929cdb6ab7ad39035038b62449603d60af2797f79a4527eef'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(args):
    vendor,out=Path(args.vendor).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'build.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    wrapper=Path(__file__).with_name('bpg_memory.c').resolve()
    assert sha(vendor/'bpgenc.c')==ENCODER_SHA and sha(vendor/'x265_glue.c')==GLUE_SHA
    archives=[vendor/f'x265.out/{depth}bit/libx265.a' for depth in [8,10,12]]
    inputs=[wrapper,Path(__file__).resolve(),vendor/'bpgenc.c',vendor/'x265_glue.c',vendor/'Makefile',
        *[vendor/(name+'.c') for name in DECODER],*archives,*vendor.glob('*.h'),
        *[p for dirname in ['libavcodec','libavutil','libavformat','libswscale'] for p in (vendor/dirname).glob('*.h')],
        vendor/'x265/source/x265.h',vendor/'x265.out/8bit/x265_config.h']
    bindings={str(p):sha(p) for p in inputs}
    done=out/'build_receipt.json'
    if done.exists():
        old=json.loads(done.read_text())
        assert old['inputs']==bindings and sha(old['library'])==old['library_sha256']
        print(json.dumps({'status':'REUSE_COMPLETE','library':old['library']}));return
    flags=['-Os','-Wall','-fPIC','-fno-asynchronous-unwind-tables','-fdata-sections','-ffunction-sections',
        '-fno-math-errno','-fno-signed-zeros','-fno-tree-vectorize','-fomit-frame-pointer','-g',
        '-D_FILE_OFFSET_BITS=64','-D_LARGEFILE_SOURCE','-D_REENTRANT','-I'+str(vendor),'-DCONFIG_BPG_VERSION="0.9.8"']
    commands=[];objects=[]
    for name in DECODER:
        obj=out/(name.replace('/','_')+'.o');objects.append(obj)
        commands.append(['gcc',*flags,'-D_ISOC99_SOURCE','-D_POSIX_C_SOURCE=200112','-D_XOPEN_SOURCE=600',
            '-DHAVE_AV_CONFIG_H','-std=c99','-D_GNU_SOURCE=1','-DUSE_VAR_BIT_DEPTH','-DUSE_PRED',
            '-c',str(vendor/(name+'.c')),'-o',str(obj)])
    enc=out/'bpg_memory.o';glue=out/'x265_glue.o';objects.extend([enc,glue])
    commands.append(['gcc',*flags,'-DUSE_X265','-Wno-unused-but-set-variable','-c',str(wrapper),'-o',str(enc)])
    commands.append(['gcc',*flags,'-I'+str(vendor/'x265/source'),'-I'+str(vendor/'x265.out/8bit'),
        '-c',str(vendor/'x265_glue.c'),'-o',str(glue)])
    def call(pair):
        i,command=pair
        with (out/f'compile_{i:02d}.log').open('wb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(call,enumerate(commands)))
    library=out/'libvarcomm_bpg_memory.so'
    link=['g++','-shared','-Wl,--gc-sections','-Wl,-Bsymbolic','-o',str(library),*[str(p) for p in objects],
        '-Wl,--start-group',*[str(p) for p in archives],'-Wl,--end-group','-lpng','-ljpeg','-lz','-lrt','-lm','-lpthread']
    with (out/'link.log').open('wb') as log:
        subprocess.run(link,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=dict(status='BUILT_NOT_YET_QUALIFIED',library=str(library),library_sha256=sha(library),
        inputs=bindings,commands=commands+[link],original_vendor_modified=False,
        compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
        use_original_x265_static_archives=True,codec_parameters='x265 / m8 / 420 / ycbcr / 8bit',
        boundary='In-process source PNG bytes, fmemopen and memory write callback; no disk files in codec calls.')
    done.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'library':str(library)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--vendor',required=True);p.add_argument('--output',required=True)
    build(p.parse_args())
