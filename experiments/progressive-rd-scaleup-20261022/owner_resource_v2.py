"""Single bounded owner for two validation shards then six matched decoder jobs."""
from rdscale_common import *
import argparse,subprocess,signal,concurrent.futures,traceback,csv,collections
from rdscale_resource_v2 import admitted

def close_validation(reg,regsha):
    out=L/'validation_delivery_v1';out.mkdir(exist_ok=False)
    sources=[];rows=[];counts=collections.Counter();receipts=[]
    for i in range(2):
        d=L/'dispatch_v1'/('validation_%d'%i);s=L/'validation_cache_v1'/('shard_%02d'%i)
        w=read(d/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and not w['timed_out'] and process(w['process']['pid'])is None
        assert not(s/'failure.json').exists() and not(d/'failure.json').exists()
        c=read(s/'completion.json');assert c['registration_sha256']==regsha and c['sources']==100 and c['endpoint_rows']==800
        events=[json.loads(x)for x in checked(c['ledger']).read_text().splitlines()]
        begin=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='STARTED')
        end=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='COMPLETE')
        assert begin==end and all(v==1 for v in begin.values()) and all(x['status']in('STARTED','COMPLETE')for x in events)
        counts.update(k[0]for k in end)
        for row in reg['validation_sources'][i::2]:
            p=s/('validation_i%05d'%row['source_index'])/'completion.json';v=read(p)
            for k in ('wire','received','latents'):checked(v[k])
            assert len(v['rows'])==8 and [(x['m'],x['K'])for x in v['rows']]==[tuple(e)for e in reg['endpoints']]
            assert all(x['source_id']==row['source_id'] and x['source_index']==row['source_index'] and x['status']=='LOSSLESS_SOURCE_RX_COMPLETE'for x in v['rows'])
            sources.append(dict(**row,old_latent_shard=row['latent_shard'],latents=v['latents'],source_pixels_sha256=v['rows'][0]['source_pixels_sha256'],endpoint_rows=v['rows'],source_completion=pin(p)))
            rows.extend(v['rows'])
        receipts.append(dict(shard=i,wait=pin(d/'actual_wait.json'),completion=pin(s/'completion.json'),ledger=c['ledger']))
    assert len(sources)==200 and len({x['source_id']for x in sources})==200 and len(rows)==1600
    assert counts['tx_probability_pass']==200 and all(counts[k]==1600 for k in ['independent_source_decode','predict_overwrite_advance','lpips_distance','ms_ssim'])
    lookup={x['source_id']:x for x in sources};sources=[lookup[x['source_id']]for x in reg['validation_sources']]
    save(out/'index.json',dict(sources=sources,registration_sha256=regsha))
    with(out/'baseline.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'closure.json',dict(status='PASS',registration_sha256=regsha,index=pin(out/'index.json'),baseline=pin(out/'baseline.csv'),receipts=receipts,counts=dict(counts),new_sources=200,rows=1600,PHY_calls=0,tokenizer_calls=0))

def main(regsha):
    assert sha(O/'registration.json')==regsha;reg=read(O/'registration.json')
    for d in reg['sources']:checked(d)
    q=read(O/'qualification_resource_v2.json');assert q['returncode']==0 and q['cuda_initialized']is False
    out=L/'dispatch_v1';out.mkdir(exist_ok=False)
    assert not(L/'training_v1').exists() and not(L/'validation_cache_v1').exists()
    active={};stop=[False]
    def stop_owned(*_):
        stop[0]=True
        for pid,ident in list(active.items()):
            cur=process(pid)
            if cur and cur['start_ticks']==ident['start_ticks']:
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop_owned);signal.signal(signal.SIGINT,stop_owned)
    def run_job(name,resource,script,arguments,wall):
        d=out/name;d.mkdir(exist_ok=False)
        try:
            assert not stop[0] and time.time()+wall+180<reg['absolute_deadline_unix'],'Insufficient remaining bounded window'
            fields=[x.strip()for x in subprocess.check_output(['nvidia-smi','-i',str(resource['index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
            assert fields[0]==resource['uuid']and admitted(int(fields[1]),int(fields[2])),fields
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(resource['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
            argv=['taskset','-c',','.join(map(str,resource['cpu_affinity'])),str(PY),'-B',str(E/script),'--registration-sha256',regsha,*arguments]
            t=time.perf_counter()
            with(d/'stdout.log').open('xb')as so,(d/'stderr.log').open('xb')as se:
                p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True);ident=process(p.pid);assert ident;active[p.pid]=ident
                save(d/'launch.json',dict(argv=argv,process=ident,parent_process=process(os.getpid()),resource=resource,started_unix=time.time()))
                term=False;killed=False
                try:rc=p.wait(timeout=wall+180)
                except subprocess.TimeoutExpired:
                    cur=process(p.pid);assert cur and cur['start_ticks']==ident['start_ticks'];os.killpg(p.pid,signal.SIGTERM);term=True
                    try:rc=p.wait(timeout=90)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
                active.pop(p.pid,None)
            v=dict(returncode=rc,actual_wait=True,process=ident,elapsed_seconds=time.perf_counter()-t,terminated=term,killed=killed,timed_out=term,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'))
            save(d/'actual_wait.json',v);return v
        except BaseException as e:
            save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));return dict(error=repr(e))
    with concurrent.futures.ThreadPoolExecutor(2)as pool:
        results=list(pool.map(lambda i:run_job('validation_%d'%i,reg['resources'][i+6],'validation_worker.py',['--shard',str(i)],reg['validation_walltime_seconds']),range(2)))
    assert all(x.get('returncode')==0 for x in results),'Validation failed; no decoder launch'
    close_validation(reg,regsha)
    with concurrent.futures.ThreadPoolExecutor(6)as pool:
        jobs=list(reg['jobs']);results=list(pool.map(lambda n:run_job(n,reg['jobs'][n]['resource'],'train.py',['--job',n],reg['training_walltime_seconds']),jobs))
    save(out/'completion.json',dict(all_exit0=all(x.get('returncode')==0 for x in results),results=results,parent_process=process(os.getpid()),registration_sha256=regsha))
    assert all(x.get('returncode')==0 for x in results),'A bounded training job failed; preserve and diagnose'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args()
    try:main(a.registration_sha256)
    except BaseException as e:
        d=L/'dispatch_v1';d.mkdir(parents=True,exist_ok=True)
        if not(d/'failure.json').exists():save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
