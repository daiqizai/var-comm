"""Own original scheduling locks during the user-requested receiver probe.
SIGTERM is handled by the existing safe-checkpoint supervisor; no hard kill.
This lease never launches GPU work or resumes a queue by itself.
"""
import fcntl,json,os,signal,shutil,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929'
OLD=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def identity(pid):
 try:
  q=Path('/proc')/str(pid);v=(q/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,start_ticks=v[19],state=v[0],cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 except (FileNotFoundError,ProcessLookupError):return None
def same(a,b):return a and a['state']!='Z' and all(a[k]==b[k] for k in ('pid','start_ticks','cmdline'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 own=(OUT/'lease.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'pause_request.json').exists():raise RuntimeError('existing pause; inspect before retry')
 archive=OUT/'original_queue_before_pause';archive.mkdir()
 main=OLD/'delivery_chain_v1'
 r=read(main/'launch_receipt.json');expected=r['process']
 assert same(identity(expected['pid']),expected)
 status=read(main/'status.json');assert status['pid']==expected['pid'] and time.time()-status['time']<40
 for name in ['delivery_chain_v1','reference_common_metrics_v1','n4084_common_replay_v1','author_native_timing_v1','n3060_native_timing_v1']:
  dest=archive/name;dest.mkdir()
  for leaf in ['launch_receipt.json','status.json','worker_status.json','registration.json','worker_registration.json']:
   p=OLD/name/leaf
   if p.exists():shutil.copyfile(p,dest/leaf)
 for p,h in read(main/'registration.json')['bindings'].items():assert sha(p)==h
 write(OUT/'pause_request.json',dict(process=expected,time=time.time(),reason='User execution sheet step 0: safe pause for RX posterior diagnostic',signal='SIGTERM_TO_EXISTING_SAFE_STOP_HANDLER'))
 assert same(identity(expected['pid']),expected)
 os.kill(expected['pid'],signal.SIGTERM)
 while same(identity(expected['pid']),expected):
  write(OUT/'lease_status.json',dict(status='WAITING_FOR_ORIGINAL_SAFE_CHECKPOINT_STOP',process=identity(os.getpid()),time=time.time()))
  time.sleep(2)
 stopped=read(main/'status.json');assert stopped['status']=='STOPPED_BY_REQUEST',stopped
 handles=[]
 for p in [main/'chain.lock',OLD/'C_followups/followups.lock',ROOT/'outputs/SHORT-PREFIX-20260923/controller.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
 train=ROOT/'outputs/SHORT-PREFIX-20260923/training/m8_seed2026092504'
 latest=read(train/'latest.json');assert sha(latest['path'])==latest['sha256']
 write(OUT/'pause_complete.json',dict(status='ORIGINAL_QUEUE_SAFELY_PAUSED_LOCKS_HELD',stopped=stopped,latest=latest,process=identity(os.getpid()),time=time.time(),original_registration_sha256=sha(main/'registration.json')))
 while not (OUT/'release_lease.json').exists():
  write(OUT/'lease_status.json',dict(status='HOLDING_ORIGINAL_LOCKS_FOR_USER_RX_PROBE',process=identity(os.getpid()),time=time.time()))
  time.sleep(10)
 for h in reversed(handles):h.close()
 write(OUT/'lease_released.json',dict(status='LOCKS_RELEASED_NO_IMPLICIT_RESTART',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:
  write(OUT/'pause_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
