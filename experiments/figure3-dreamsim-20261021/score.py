"""Score exactly ten existing Fig.3 native-float pairs with frozen DreamSim."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, sqlite3, subprocess
import sys, tempfile, time, traceback

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
O = R/'results/figure3_dreamsim_20261021_v1'
L = B/'figure3_dreamsim_20261021_v1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def desc(p): return dict(path=str(p), bytes=Path(p).stat().st_size, sha256=sha(p))
def checked(d):
    p = Path(d['path'])
    assert not p.is_symlink() and sha(p) == d['sha256'], p
    if 'bytes' in d: assert p.stat().st_size == d['bytes'], p
    return p
def save(p, v):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(v, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def proc(pid):
    v = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(v[19]))

def worker(reg, expected):
    out = L/'score'; out.mkdir()
    save(out/'launch.json', dict(process=proc(os.getpid()), parent_process=proc(os.getppid()), argv=sys.argv))
    cache = out/'runtime_cache'; (cache/'tmp').mkdir(parents=True)
    tempfile.tempdir = str(cache/'tmp')
    os.environ.update(TMPDIR=str(cache/'tmp'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        TRITON_CACHE_DIR=str(cache/'triton'), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.sched_setaffinity(0, set(range(6)))
    db = sqlite3.connect(out/'ledger.sqlite')
    db.execute('CREATE TABLE calls(kind TEXT,id TEXT,status TEXT,PRIMARY KEY(kind,id))'); db.commit()
    start = time.monotonic()
    try:
        import numpy as np, torch, ctypes
        import torch.nn.functional as F
        torch.set_num_threads(6); torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        assert torch.__version__ == '2.11.0+cu128'
        torch.cuda.set_device(0)
        lib = ctypes.CDLL('libcuda.so.1'); device = ctypes.c_int()
        assert lib.cuDeviceGet(ctypes.byref(device), 0) == 0
        bus = ctypes.create_string_buffer(32)
        assert lib.cuDeviceGetPCIBusId(bus, 32, device) == 0
        assert bus.value.decode().lower() == reg['device']['cuda_bus']
        assert torch.cuda.mem_get_info()[0] >= 8 << 30
        torch.cuda.set_per_process_memory_fraction((6 << 30)/torch.cuda.get_device_properties(0).total_memory)
        def call(kind, key, fn):
            assert time.monotonic()-start < 600 and torch.cuda.mem_get_info()[0] >= 2 << 30
            assert db.execute('SELECT COUNT(*) FROM calls WHERE kind=?', (kind,)).fetchone()[0] < reg['limits'][kind]
            db.execute('INSERT INTO calls VALUES(?,?,?)', (kind, key, 'RESERVED')); db.commit()
            try: value = fn()
            except BaseException:
                db.execute('UPDATE calls SET status="FAILED" WHERE kind=? AND id=?', (kind, key)); db.commit(); raise
            db.execute('UPDATE calls SET status="COMPLETE" WHERE kind=? AND id=?', (kind, key)); db.commit()
            return value
        cfg = reg['metric_config']
        overlay = read(checked(cfg['supplement']['dependency_overlay']))
        for d in overlay['files']: checked(d)
        sys.path.insert(0, cfg['config']['supplemental_overlay_root'])
        spec = importlib.util.spec_from_file_location('_frozen_figure3_metrics', checked(cfg['supplement']['loader']))
        met = importlib.util.module_from_spec(spec); spec.loader.exec_module(met)
        assets = read(checked(cfg['supplement']['metric_assets']))
        torch.manual_seed(2026101401)
        with met.no_network():
            dream = call('model_load', 'ensemble', lambda: met.load_dreamsim(assets['models']['dreamsim'], torch.device('cuda:0')))
        assert all(p.dtype == torch.float32 and not p.requires_grad for p in dream.parameters())
        identity = dict(registration_sha256=expected, torch=torch.__version__, cuda=torch.version.cuda,
            device=reg['device'], actual_bus=bus.value.decode(), device_name=torch.cuda.get_device_name(0),
            fp32=True, amp=False, tf32=False, batch_size=1, threads=6, interop_threads=2,
            deterministic=True, loader=cfg['supplement']['loader'], assets=assets['models']['dreamsim'],
            protocol=cfg['supplement']['protocol']['dreamsim'], preprocessing='Native float BCHW; 224x224 bicubic antialias tensor resize; no crop, PIL, or quantization',
            definition='1 - cosine_similarity(ensemble.embed(source), ensemble.embed(reconstruction))')
        save(out/'metric_identity.json', identity)
        (out/'pairs').mkdir(); (out/'features').mkdir(); rows=[]
        with torch.inference_mode():
            for s in reg['sources']:
                with np.load(checked(s['archive']), allow_pickle=False) as z: raw = z['pixels'].copy()
                assert raw.dtype == np.uint8 and raw.shape == (3,256,256)
                x = raw.astype(np.float32)/np.float32(255)
                assert hashlib.sha256(x.tobytes()).hexdigest() == s['float_sha256']
                xt = torch.from_numpy(x[None]).cuda()
                fx = call('source_embed', s['source_id'], lambda: dream.embed(met.preprocess_dreamsim(xt)).float())
                np.save(out/'features'/f"source_{s['source_index']}.npy", fx.cpu().numpy(), allow_pickle=False)
                for j in [j for j in reg['jobs'] if int(j['source_index']) == s['source_index']]:
                    key = j['timing_case']+'_'+j['method']
                    with np.load(checked(j['input']), allow_pickle=False) as z: y = z['image'].copy()
                    assert y.dtype == np.float32 and y.shape == (3,256,256) and np.isfinite(y).all() and 0 <= y.min() <= y.max() <= 1
                    y = np.ascontiguousarray(y)
                    assert hashlib.sha256(str(y.dtype).encode()+str(y.shape).encode()+y.tobytes()).hexdigest() == j['timing_float_array_sha256']
                    yt = torch.from_numpy(y[None]).cuda()
                    fy = call('image_embed', key, lambda: dream.embed(met.preprocess_dreamsim(yt)).float())
                    np.save(out/'features'/(key+'.npy'), fy.cpu().numpy(), allow_pickle=False)
                    value = call('distance_pair', key, lambda: float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item()))
                    assert np.isfinite(value)
                    row={k:v for k,v in j.items() if k != 'input'}
                    row.update(dreamsim=value, source_archive_path=s['archive']['path'],
                        source_archive_sha256=s['archive']['sha256'], source_float_sha256=s['float_sha256'],
                        metric_registration_sha256=expected)
                    save(out/'pairs'/(key+'.json'), row); rows.append(row)
                    print(key, repr(value), flush=True)
        counts = db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall()
        assert len(rows) == 10 and all(st == 'COMPLETE' and n == reg['limits'][k] for k,st,n in counts)
        with (out/'figure3_metrics.csv').open('x',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        save(out/'completion.json',dict(status='COMPLETE',rows=10,counts=counts,unresolved=0,
            registration_sha256=expected, metrics=desc(out/'figure3_metrics.csv'), identity=desc(out/'metric_identity.json'),
            elapsed_seconds=time.monotonic()-start,new_lpips_calls=0,new_reconstructions=0,new_channel_events=0,new_bootstrap=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally: db.close()

def main():
    a=argparse.ArgumentParser();a.add_argument('role',choices=['dispatch','worker']);a.add_argument('--registration',required=True);a=a.parse_args()
    assert sha(O/'registration.json') == a.registration
    reg=read(O/'registration.json')
    for d in reg['code']+[reg['display_manifest'],reg['source_registration'],reg['prior_metric_registration']]: checked(d)
    if a.role == 'worker': return worker(reg,a.registration)
    L.mkdir(); control=L/'control';control.mkdir()
    save(control/'supervisor.json',dict(process=proc(os.getpid()),parent_process=proc(os.getppid()),argv=sys.argv))
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--registration',a.registration]
    start=time.monotonic();timed_out=False
    with (control/'stdout.txt').open('xb') as so,(control/'stderr.txt').open('xb') as se:
        child=subprocess.Popen(argv,stdout=so,stderr=se,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        ident=proc(child.pid);save(control/'launch.json',dict(process=ident,parent_process=proc(os.getpid()),argv=argv))
        try: rc=child.wait(timeout=660)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();rc=child.wait()
    save(control/'actual_wait.json',dict(process=ident,returncode=rc,elapsed_seconds=time.monotonic()-start,
        actual_wait=True,timed_out=timed_out,killed=timed_out,terminated=False,metadata_error=None))
    if rc or timed_out: raise RuntimeError('Score child failed; preserve all evidence, no automatic retry')
    assert read(L/'score/completion.json')['status'] == 'COMPLETE'
    print('TEN_FIXED_PAIRS_COMPLETE',flush=True)

if __name__ == '__main__': main()
