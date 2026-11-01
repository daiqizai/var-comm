"""Two fixed synthetic codec checks; no source images, PHY or quality calls."""
from followup_common import *
import subprocess,traceback

def main(request,expected):
 assert sha(request)==expected
 r=read(request);[checked(x) for x in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False)
 os.sched_setaffinity(0,set(r['cpu_affinity']));assert len(os.sched_getaffinity(0))==2
 results=[];calls=[]
 try:
  source=checked(r['shim_source']);old=checked(r['old_shim_source'])
  import re
  strip=lambda text:re.sub(r'/\*.*?\*/','',text,flags=re.S).strip()
  assert strip(source.read_text())==strip(old.read_text()).replace('CPU_COUNT(&set) != 6','CPU_COUNT(&set) != 2')
  so=out/'affinity_cpu_count.so'
  build=subprocess.run(['gcc','-shared','-fPIC','-O2','-o',str(so),str(source),'-ldl'],capture_output=True,timeout=60)
  save(out/'build.json',dict(returncode=build.returncode,stderr=build.stderr.decode()));assert build.returncode==0
  cfg=cr(r['config'])['bpg_config'];enc=checked(cfg['bpgenc']);dec=checked(cfg['bpgdec'])
  env=dict(os.environ,LD_PRELOAD=str(so))
  # Check the exact adapter branch, under the already registered affinity.
  probe=subprocess.run(['/usr/bin/getconf','_NPROCESSORS_ONLN'],env=env,capture_output=True,timeout=10)
  assert probe.returncode==0 and probe.stdout.strip()==b'2'
  from PIL import Image
  for side in [32,256]:
   yy,xx=np.indices((side,side));pixels=np.stack([(3*xx+5*yy)%256,(xx*7)%256,(yy*11)%256],axis=-1).astype('uint8')
   src=out/f'synthetic{side}.png';Image.fromarray(pixels).save(src)
   bitstream=out/f'synthetic{side}.bpg';decoded=out/f'synthetic{side}.ppm'
   argv=[str(enc),'-e','x265','-m','8','-f','420','-c','ycbcr','-b','8','-q','0','-o',str(bitstream),str(src)]
   calls.append(dict(kind='encode',side=side,status='STARTED'));save(out/'calls.json',calls)
   start=time.monotonic();p=subprocess.run(argv,env=env,capture_output=True,timeout=150)
   calls[-1].update(status='COMPLETE' if p.returncode==0 else 'FAILED',returncode=p.returncode,seconds=time.monotonic()-start)
   save(out/'calls.json',calls);(out/f'encode{side}.stderr').write_bytes(p.stderr);assert p.returncode==0
   calls.append(dict(kind='decode',side=side,status='STARTED'));save(out/'calls.json',calls)
   p=subprocess.run([str(dec),'-o',str(decoded),str(bitstream)],capture_output=True,timeout=30)
   calls[-1].update(status='COMPLETE' if p.returncode==0 else 'FAILED',returncode=p.returncode);save(out/'calls.json',calls)
   assert p.returncode==0 and Image.open(decoded).size==(side,side)
   results.append(dict(side=side,source=pin(src),stream=pin(bitstream),decoded=pin(decoded),encoder_seconds=calls[-2]['seconds'],thread_messages=[s for s in (out/f'encode{side}.stderr').read_text().splitlines() if 'thread' in s.lower()]))
  save(out/'completion.json',dict(status='COMPLETE',synthetic_only=True,engineering_encode_calls=2,engineering_decode_calls=2,
   real_source_calls=0,PHY_calls=0,quality_calls=0,adapter=pin(so),source=pin(source),results=results,
   CPU_affinity=sorted(os.sched_getaffinity(0)),codec_binaries_unchanged=True,automatic_retry=False))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),calls=calls,results=results,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
