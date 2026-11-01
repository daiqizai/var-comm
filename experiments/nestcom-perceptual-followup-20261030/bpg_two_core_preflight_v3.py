"""Finish only missing codec checks; reuse successful synthetic32 encode."""
from followup_common import *
import subprocess,traceback

def main(request,expected):
 assert sha(request)==expected
 r=read(request);[checked(x) for x in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 calls=[];results=[]
 def record(v):
  calls.append(v);save(out/f'call_{len(calls):02d}.json',v)
 try:
  prior=cr(r['preserved_failure']);assert len(prior['calls'])==1
  assert prior['calls'][0]['kind']=='encode' and prior['calls'][0]['side']==32 and prior['calls'][0]['status']=='COMPLETE' and prior['calls'][0]['returncode']==0
  assert 'FileExistsError' in prior['traceback'] and 'calls.json' in prior['traceback']
  so=checked(r['adapter']);source=checked(r['shim_source'])
  oldsrc=checked(r['reuse_source']);oldstream=checked(r['reuse_stream'])
  cfg=cr(r['config'])['bpg_config'];enc=checked(cfg['bpgenc']);dec=checked(cfg['bpgdec'])
  env=dict(os.environ,LD_PRELOAD=str(so))
  probe=subprocess.run(['/usr/bin/getconf','_NPROCESSORS_ONLN'],env=env,capture_output=True,timeout=10)
  assert probe.returncode==0 and probe.stdout.strip()==b'2'
  from PIL import Image
  for side in [32,256]:
   yy,xx=np.indices((side,side));pixels=np.stack([(3*xx+5*yy)%256,(xx*7)%256,(yy*11)%256],axis=-1).astype('uint8')
   src=out/f'synthetic{side}.png';Image.fromarray(pixels).save(src)
   decoded=out/f'synthetic{side}.ppm';seconds=None
   if side==32:
    assert sha(src)==sha(oldsrc)
    bitstream=oldstream;seconds=prior['calls'][0]['seconds']
    record(dict(kind='encode',side=side,status='REUSED_COMPLETE',stream=pin(bitstream),prior_failure=r['preserved_failure'],new_calls=0))
   else:
    bitstream=out/f'synthetic{side}.bpg'
    argv=[str(enc),'-e','x265','-m','8','-f','420','-c','ycbcr','-b','8','-q','0','-o',str(bitstream),str(src)]
    record(dict(kind='encode',side=side,status='STARTED'))
    start=time.monotonic();p=subprocess.run(argv,env=env,capture_output=True,timeout=150);seconds=time.monotonic()-start
    record(dict(kind='encode',side=side,status='COMPLETE' if p.returncode==0 else 'FAILED',returncode=p.returncode,seconds=seconds))
    (out/f'encode{side}.stderr').write_bytes(p.stderr);assert p.returncode==0
   record(dict(kind='decode',side=side,status='STARTED'))
   p=subprocess.run([str(dec),'-o',str(decoded),str(bitstream)],capture_output=True,timeout=30)
   record(dict(kind='decode',side=side,status='COMPLETE' if p.returncode==0 else 'FAILED',returncode=p.returncode))
   assert p.returncode==0 and Image.open(decoded).size==(side,side)
   results.append(dict(side=side,source=pin(src),stream=pin(bitstream),decoded=pin(decoded),encoder_seconds=seconds,reused_encode=side==32))
  save(out/'completion.json',dict(status='COMPLETE',synthetic_only=True,engineering_encode_calls=2,engineering_decode_calls=2,
   new_encode_calls=1,reused_encode_calls=1,new_decode_calls=2,real_source_calls=0,PHY_calls=0,quality_calls=0,
   adapter=pin(so),source=pin(source),results=results,CPU_affinity=sorted(os.sched_getaffinity(0)),codec_binaries_unchanged=True,automatic_retry=False))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),calls=calls,results=results,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
