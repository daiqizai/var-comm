"""Twenty predeclared development examples at 13 dB; no policy fitting."""
import argparse, os, sys, time, traceback, subprocess, signal, tempfile, fcntl
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OLD_E=R/'experiments/five-method-timing-20261020'
sys.path.insert(0,str(OLD_E))
from contract import read,save,pin,sha,checked,process,binding,B,RT,PY,CPY
import native
import worker as w
E=R/'experiments/selected-examples-13db-20261021'
O=R/'results/selected_examples_13db_20261021_v1'
T=B/'selected_examples_13db_20261021_v1'
OLD_T=B/'five_method_timing_20261020_v1'
PIX=B/'selected_examples_10db_20261020_v1'
OLD='d4e7a6bbad058fff98d0c061515c11c562af5d8cc59be5c9e249351363964f31'
INDICES=[21,92,95,75]
METHODS=['NEST','P100000','Swin200000','BPG','ARPC']
def bind(expected):
    assert sha(O/'registration.json')==expected
    reg=read(O/'registration.json')
    assert reg['indices']==INDICES and reg['snr_db']==13
    for d in reg['code']:checked(d)
    return reg,binding(OLD)
def setup(out):
    os.sched_setaffinity(0,set(range(6)))
    c=out/'runtime_cache';(c/'tmp').mkdir(parents=True)
    os.environ.update(TMPDIR=str(c/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(c/'inductor'),TRITON_CACHE_DIR=str(c/'triton'),
        HF_HOME=str(c/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    tempfile.tempdir=str(c/'tmp')
def npz(path,**values):
    import numpy as np
    with path.open('xb') as f:np.savez(f,**values);f.flush();os.fsync(f.fileno())
def stage(expected,method,role):
    reg,r=bind(expected);out=T/method/role;out.mkdir(parents=True,exist_ok=False);setup(out)
    save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv))
    rows=[c for c in reg['cases'] if c['method']==method]
    assert len(rows)==4
    done=[]
    try:
        import numpy as np
        calls=native.Calls();ctx=None;t=None;blocked=[]
        if role=='rx':
            # No source paths, old expected hashes or TX records are opened after this boundary.
            rows=[{k:c[k] for k in ('key','source_index','counter')} for c in rows]
            del reg
            fw=native.replace_function(w,'firewall',[("allowed = [p.resolve() for p in allowed]",
                "allowed += [Path("+repr(str(T/method/'rx_inputs'))+"), Path("+repr(str(RT/'assets/frozen_namespace_v1/external/usr/lib/python3/dist-packages'))+")]\n    allowed = [p.resolve() for p in allowed]")])
            blocked=fw(r,method,out)
        if role=='phy':ctx=w.phy_runtime(r,method,out)
        else:
            save(out/'resource_before.json',w.resource(r,initial=True))
            ctx,calls=native.continuous(r,out,method);t=ctx.t
            save(out/'runtime.json',dict(torch=t.__version__,cuda=t.version.cuda,precision='FP32, TF32 off',threads=t.get_num_threads(),interop=t.get_num_interop_threads(),device=str(t.cuda.get_device_properties(0)),cpu_affinity=sorted(os.sched_getaffinity(0))))
        if role in ('phy','tx'):(T/method/'rx_inputs').mkdir()
        (out/'cases').mkdir()
        for c in rows:
            d=out/'cases'/c['key'];d.mkdir()
            save(d/'reserved.json',dict(source_index=c['source_index'],process=process(os.getpid())))
            clock=w.Clock(t)
            if role=='phy':
                data=read(checked(c['tx_payload']))
                actual,evidence=w.phy_call(data,r,method,c,ctx,calls,clock)
                save(T/method/'rx_inputs'/(c['key']+'.json'),actual)
                result=dict(actual=actual,evidence=evidence,received_input=pin(T/method/'rx_inputs'/(c['key']+'.json')))
            elif role=='tx':
                s=next(s for s in r['sources'] if s['source_index']==c['source_index'])
                with np.load(checked(s['archive']),allow_pickle=False) as z:pixels=z['pixels'].copy()
                wave,_=ctx.encode(pixels,13,c['counter'],clock)
                observed=ctx.noise(wave,c['source_id'],13)
                assert wave.shape==observed.shape==(1024,2) and np.isfinite(observed).all()
                npz(T/method/'rx_inputs'/(c['key']+'.npz'),observed=observed)
                result=dict(wave_sha256=w.array_sha(wave),observation_sha256=w.array_sha(observed),received_input=pin(T/method/'rx_inputs'/(c['key']+'.npz')))
            else:
                with np.load(T/method/'rx_inputs'/(c['key']+'.npz'),allow_pickle=False) as z:observed=z['observed'].copy()
                image,actual=ctx.receive(observed,13,c['counter'],clock,c['key'])
                assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all() and image.min()>=0 and image.max()<=1
                npz(d/'image.npz',image=image)
                result=dict(image=pin(d/'image.npz'),image_sha256=w.array_sha(image),status=actual['status'],source_truth_used=False)
            save(d/'complete.json',dict(result,source_index=c['source_index'],key=c['key'],model_calls=calls.counts.copy()))
            done.append(result);print(method,role,len(done),flush=True)
        assert not blocked
        save(out/'completion.json',dict(status='COMPLETE',results=done,blocked_reads=blocked,model_calls=calls.counts.copy()))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed=len(done),automatic_retry=False));raise
def assemble(expected):
    """Reuse requires complete actual received-information equality, never TX truth."""
    reg,r=bind(expected);out=T/'assembly';out.mkdir();done=[]
    for c in reg['cases']:
        m=c['method'];key=c['key']
        if m in ('P100000','Swin200000'):
            result=read(T/m/'rx/cases'/key/'complete.json');reuse=False
        else:
            actual=read(T/m/'rx_inputs'/(key+'.json'))
            prior=read(checked(c['previous_rx_input']))
            # Fields include full accepted profile and received payload; parser/status too.
            assert actual==prior, ('NEW_DIGITAL_RX_INPUT_REQUIRES_SEPARATE_RECONSTRUCTION',m,key)
            result=read(checked(c['previous_reconstruction']))
            checked(result['image']);reuse=True
        row=dict(method=m,source_index=c['source_index'],source_id=c['source_id'],key=key,snr_db=13,
            image=result['image'],image_sha256=result['image_sha256'],status=result['status'],
            rx_reused=reuse,noise_seed=c['noise_seed'],noise_namespace=c['noise_namespace'],counter=c['counter'])
        if reuse:
            v=read(checked(c['previous_metrics']))
            assert v['image_sha256']==row['image_sha256'] and v['source_id']==row['source_id']
            row.update(psnr_db=v['psnr_db'],lpips_alex=v['lpips_alex'],metrics_reused=True)
        done.append(row)
    save(out/'completion.json',dict(status='COMPLETE',rows=done,new_channel_events=20,digital_rx_reused=12,new_neural_rx=8))
def score(expected):
    reg,r=bind(expected);out=T/'metrics';out.mkdir();setup(out)
    rows=read(T/'assembly/completion.json')['rows'];plan=read(checked(reg['metric_registration']))
    done=[]
    try:
        import numpy as np, torch, lpips, math
        native.flags(torch);torch.cuda.set_device(0)
        save(out/'resource_before.json',w.resource(r,initial=True))
        metric=native.load(checked(plan['metric_math']),'_selected_13_metrics')
        psnr,proof=metric.psnr_projection(checked(plan['psnr_source']))
        loader,pair,proof2=metric.lpips_projection(checked(plan['lpips_source']),dict(Path=Path,lpips=lpips,torch=torch))
        linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth';assert sha(linear)==metric.WEIGHT_SHA['lpips']
        save(out/'model_reserved.json',dict(alexnet=plan['alexnet']))
        model=loader(dict(alexnet_checkpoint=str(checked(plan['alexnet']))),'cuda:0')
        save(out/'model_complete.json',dict(psnr=proof,lpips=proof2,linear=pin(linear)))
        (out/'pairs').mkdir()
        for row in rows:
            if row['metrics_reused'] if 'metrics_reused' in row else False:
                done.append(row);continue
            d=out/'pairs'/(row['method']+'_'+str(row['source_index']));d.mkdir()
            s=next(s for s in r['sources'] if s['source_index']==row['source_index'])
            with np.load(checked(s['archive']),allow_pickle=False) as z:target=z['pixels'].astype(np.float32)/np.float32(255)
            with np.load(checked(row['image']),allow_pickle=False) as z:image=z['image'].copy()
            assert w.array_sha(image)==row['image_sha256']
            save(d/'reserved.json',dict(image=row['image'],source_id=row['source_id']))
            with torch.inference_mode():value=float(pair(model,torch.from_numpy(image[None]).cuda(),torch.from_numpy(target[None]).cuda())[0].item())
            values=dict(psnr_db=psnr(target,image),lpips_alex=value);assert all(math.isfinite(v) for v in values.values())
            result=dict(row,**values,metrics_reused=False);save(d/'complete.json',result);done.append(result)
        assert len(done)==20 and sum(not x['metrics_reused'] for x in done)==8
        save(out/'completion.json',dict(status='COMPLETE',rows=done,new_pairs=8,reused_pairs=12,metrics=['PSNR','LPIPS-Alex'],bootstrap=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed=len(done),automatic_retry=False));raise
def owner(expected):
    reg,r=bind(expected);d=T/'control';d.mkdir()
    lock=(OLD_T/'timing.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(d/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv))
    stages=[(m,'phy',PY) for m in ['NEST','ARPC','BPG']]+[(m,role,PY) for m in ['P100000','Swin200000'] for role in ['tx','rx']]+[('ALL','assemble',CPY),('ALL','score',PY)]
    try:
        for m,role,py in stages:
            q=d/(m+'_'+role);q.mkdir();argv=[py,'-B',str(E/'execute.py'),role,'--registration',expected,'--method',m]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if role in ('phy','assemble') else '1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6');env.pop('PYTHONPATH',None)
            start=time.monotonic();timed_out=killed=False
            with (q/'stdout.log').open('xb') as out,(q/'stderr.log').open('xb') as err:
                p=subprocess.Popen(argv,env=env,cwd=E,stdout=out,stderr=err,start_new_session=True);ident=process(p.pid)
                save(q/'launch.json',dict(process=ident,parent_process=process(os.getpid()),argv=argv))
                try:code=p.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=30)
                    except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
            save(q/'actual_wait.json',dict(returncode=code,child=ident,seconds=time.monotonic()-start,timed_out=timed_out,killed=killed,child_gone=process(p.pid) is None))
            assert code==0 and not timed_out,(m,role,code)
        save(d/'completion.json',dict(status='COMPLETE',stages=len(stages)))
    except BaseException:save(d/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def supervise(expected):
    bind(expected);d=T/'dispatch';d.mkdir();start=time.monotonic()
    argv=[CPY,'-B',str(E/'execute.py'),'owner','--registration',expected]
    with (d/'stdout.log').open('xb') as out,(d/'stderr.log').open('xb') as err:
        p=subprocess.Popen(argv,cwd=E,stdout=out,stderr=err,start_new_session=True);ident=process(p.pid)
        save(d/'launch.json',dict(process=ident,parent_process=process(os.getpid()),argv=argv))
        code=p.wait(timeout=16500)
    save(d/'actual_wait.json',dict(returncode=code,child=ident,seconds=time.monotonic()-start,child_gone=process(p.pid) is None))
    assert code==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--method',default='ALL');p.add_argument('--registration',required=True);a=p.parse_args()
    if a.role in ('phy','tx','rx'):stage(a.registration,a.method,a.role)
    else:globals()[a.role](a.registration)
