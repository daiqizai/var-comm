from pathlib import Path
import hashlib,json,os,subprocess,time
H=Path(__file__).resolve().parent;D=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/quality_metrics_supplement_20261014_v1')
C=D/'bridge_resume_transport_control_v1';A=D/'digital_input_bridge_v2.tar.gz'
size=1422655488;prefix_sha='0065b9530f9794b2446a4e803c08f846d38a74f889a7c729f1273413d4aa9188'
assert not C.exists() and A.stat().st_size==2377544488
h=hashlib.sha256()
with A.open('rb') as f:
    remaining=size
    while remaining:
        b=f.read(min(8<<20,remaining));assert b;h.update(b);remaining-=len(b)
assert h.hexdigest()==prefix_sha,'Existing remote prefix is not exact local bytes'
C.mkdir()
def save(n,v):
    with (C/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
save('preflight.json',dict(existing_bytes=size,prefix_sha256=prefix_sha,prefix_exact_PASS=True,source_expected_bytes=2377544488,source_expected_SHA='93e1b11d9b214827baf15ec1fd10126b67c9fedd5cbb005d22ff4e5552b8ad7a',prior_scp_returncode=1,prior_scp_error='Connection reset; Broken pipe',new_scientific_calls=0))
batch=C/'batch.txt';batch.write_text('reput "'+A.as_posix()+'" "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1/digital_input_bridge_transport_v2/digital_input_bridge_v2.tar.gz"\n',encoding='ascii')
argv=['sftp','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=40','-b',str(batch),'DevMachineModel-H100-4C-P1660-01'];start=time.monotonic()
with (C/'stdout.txt').open('xb') as out,(C/'stderr.txt').open('xb') as err:
    child=subprocess.Popen(argv,stdout=out,stderr=err);save('launch.json',dict(argv=argv,pid=child.pid,started_unix=time.time(),finite_seconds=900,resume_only_verified_original_bytes=True))
    timed_out=False
    try:rc=child.wait(timeout=900)
    except subprocess.TimeoutExpired:timed_out=True;child.terminate();rc=child.wait(timeout=15)
save('actual_wait.json',dict(actual_wait=True,returncode=rc,pid=child.pid,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt'),scientific_calls=0,remote_full_SHA_not_yet_checked=True))
print(json.dumps(dict(returncode=rc,control=str(C),timed_out=timed_out,stdout=(C/'stdout.txt').read_text()[-1000:],stderr=(C/'stderr.txt').read_text()[-1000:])),flush=True)
if rc:raise SystemExit(rc)
