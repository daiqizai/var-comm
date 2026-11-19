import json, os, sys, time, subprocess
from pathlib import Path
p=Path(sys.argv[1]); limit=int(sys.argv[2])
def record(name, data):
 with (p/name).open("x") as f: json.dump(data,f,indent=2)
def ident(pid):
 return {"pid":pid, "start_ticks":int(Path("/proc",str(pid),"stat").read_text().rsplit(")",1)[1].split()[19])}
start=time.time()
worker=subprocess.Popen(sys.argv[3:], stdin=subprocess.DEVNULL)
record("child.json",{"worker":ident(worker.pid),"owner":ident(os.getpid()),"started":start,"argv":sys.argv[3:]})
timed_out=False
try: rc=worker.wait(timeout=limit)
except subprocess.TimeoutExpired:
 timed_out=True
 worker.terminate()
 try: rc=worker.wait(timeout=30)
 except subprocess.TimeoutExpired:
  worker.kill();rc=worker.wait()
record("actual_wait.json",{"returncode":rc,"timed_out":timed_out,"started":start,"ended":time.time(),"pid":worker.pid})
sys.exit(rc if rc>=0 else 128-rc)
