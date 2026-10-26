"""Bounded official weights download. No torch/model/scientific imports."""
from common import *
import urllib.request,traceback,argparse
def main(path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/'c_assets';out.mkdir(parents=True,exist_ok=False);beg=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv))
 try:
  items=[]
  for w in r['weights']:
   assert time.monotonic()-beg<7200 and w['bytes']<2_000_000_000
   dest=out/w['name'];tmp=out/(w['name']+'.partial')
   q=urllib.request.Request(w['url'],headers={'User-Agent':'NeST-Com-research/1'})
   h=hashlib.sha256();n=0
   with urllib.request.urlopen(q,timeout=60)as resp,tmp.open('xb')as f:
    while True:
     assert time.monotonic()-beg<7200
     b=resp.read(8<<20)
     if not b:break
     n+=len(b);assert n<=w['bytes'];f.write(b);h.update(b)
     if n%(128<<20)<8<<20:print(w['name'],n,w['bytes'],flush=True)
    f.flush();os.fsync(f.fileno())
   assert n==w['bytes']and h.hexdigest()==w['sha256'],(w['name'],n,h.hexdigest())
   os.link(tmp,dest);items.append(dict(path=str(dest),bytes=n,sha256=h.hexdigest(),url=w['url']))
   print('VERIFIED',w['name'],n,h.hexdigest(),flush=True)
  save(out/'completion.json',dict(status='OFFICIAL_ASSETS_SHA_VERIFIED',files=items,registration_sha256=expected,model_calls=0,seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),model_calls=0,automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.registration,a.expected_registration)
