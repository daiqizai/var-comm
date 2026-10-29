"""Two received-only replays per method; compare historical hashes in owner."""
import argparse,contextlib,os,sys,tempfile,traceback,time
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
EOLD=R/'experiments/five-method-timing-20261020'
sys.path.insert(0,str(EOLD))
from contract import checked,read,save,pin,sha,process,B
import native
L=B/'nestcom_fig3_4db_20261029_v1'

def main(reg_path,expected,method):
    assert sha(reg_path)==expected
    reg=read(reg_path)
    for d in reg['code']:checked(d)
    assert method in reg['methods'] and reg['source_indices']==[95,21]
    r=read(checked(reg['timing_config']))
    rows=[{k:c[k] for k in ('key','source_index','counter','received_input')} for c in reg['cases'] if c['method']==method]
    assert len(rows)==2
    del reg
    out=L/('recovery_'+method);out.mkdir()
    os.sched_setaffinity(0,set(range(6)))
    cache=out/'runtime_cache';(cache/'tmp').mkdir(parents=True)
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    tempfile.tempdir=str(cache/'tmp')
    save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),registration_sha256=expected))
    done=[]
    try:
        for row in rows:checked(row['received_input'])
        sys.path.insert(0,str(EOLD/'library_recovery_v1'))
        from adapter import adapted_worker
        w=adapted_worker()
        blocked=w.firewall(r,method,out)
        import numpy as np
        t=None;ctx=None;calls=native.Calls()
        if method in ('P100000','Swin200000'):
            ctx,calls=native.continuous(r,out,method);t=ctx.t
        elif method=='NEST':
            ctx=native.nest_codec(r,out);t,calls=ctx[0].torch,ctx[3]
        elif method=='ARPC':
            ctx=native.arpc(r,out,'rx');t=ctx.torch
        else:
            import bpg
            bpg.workspace=contextlib.contextmanager(native.replace_function(bpg,'workspace',[("T/'BPG/failed_codec_evidence'/path.name","Path("+repr(str(out/'failed_codec_evidence'))+")/path.name")]))
            bpg.legacy();r['_tmpfs']='/dev/shm/nestcom_fig3_4db_20261029_v1_BPG';Path(r['_tmpfs']).mkdir()
        save(out/'runtime.json',dict(python=sys.version,torch=None if t is None else t.__version__,cuda=None if t is None else t.version.cuda,threads=None if t is None else t.get_num_threads(),affinity=sorted(os.sched_getaffinity(0))))
        (out/'cases').mkdir()
        for row in rows:
            case=out/'cases'/row['key'];case.mkdir()
            assert not (L/'STOP').exists()
            save(case/'reservation.json',dict(row,process=process(os.getpid())))
            meter=w.Clock(t)
            if method in ('P100000','Swin200000'):
                with np.load(row['received_input']['path'],allow_pickle=False)as z:observed=z['observed'].copy()
                image,actual=ctx.receive(observed,4,row['counter'],meter,row['key']);status=actual['status']
            else:
                image,status=w.native_rx(read(row['received_input']['path']),r,method,ctx,meter)
            assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and 0<=image.min()<=image.max()<=1
            with (case/'image.npz').open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
            result=dict(key=row['key'],source_index=row['source_index'],method=method,status=status,image=pin(case/'image.npz'),image_sha256=w.array_sha(image),received_input=row['received_input'],model_calls=calls.counts.copy(),source_truth_used=False)
            save(case/'complete.json',result);done.append(result)
            print('RECOVERED',method,row['key'],flush=True)
        assert len(done)==2 and not blocked
        save(out/'completion.json',dict(status='COMPLETE',results=done,blocked_reads=blocked,tx_calls=0,channel_events=0,source_truth_used=False))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed=len(done),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--registration',required=True);p.add_argument('--expected',required=True);a=p.parse_args();main(a.registration,a.expected,a.method)
