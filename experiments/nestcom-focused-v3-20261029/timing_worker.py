"""Finite real-weight TX timing, same-route stream gate, no RX or quality calls."""
import sys,os,json,time,base64,hashlib,subprocess,traceback
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import pin,checked,cr,save,ident,Ledger,asha,runtime_cache
from timed_codec import encode,select,endpoints
import numpy as np

class NativeCounts:
    def __init__(self):self.counts={}
    def call(self,kind,key,fn):
        assert kind=='prior_scale',kind
        self.counts[kind]=self.counts.get(kind,0)+1
        return fn()

def gpu(r):
    v=subprocess.check_output(['nvidia-smi','-i',r['gpu_uuid'],'--query-gpu=uuid,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert v[0]==r['gpu_uuid'];return dict(unix=time.time(),uuid=v[0],free_MiB=int(v[1]),util=int(v[2]),temperature=int(v[3]),power_W=float(v[4]),load=list(os.getloadavg()))

def main(regpath,expected):
    rp=Path(regpath);assert pin(rp)['sha256']==expected;r=json.loads(rp.read_bytes());out=Path(r['output']);out.mkdir(parents=True,exist_ok=False)
    for p in r['code']:checked(p)
    runtime_cache(out);os.sched_setaffinity(0,set(r['gpu_affinity']));start=time.monotonic()
    ledger=Ledger(out/'calls.sqlite',{'complete_tx':610,'model_load':1});native_counts=NativeCounts();server=None
    save(out/'started.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),registration_sha256=expected))
    def boundary():
        assert time.monotonic()-start<r['max_seconds'] and not(out.parent/'STOP').exists()
    try:
        status=gpu(r);assert status['free_MiB']>=40000 and status['util']<=15;save(out/'resource_initial.json',status)
        sys.path[:0]=[str(R/'experiments'/p) for p in ['kodak-current-20261026/nest_v1','t1b-m10-calibration-20261016','entropy-partial-mismatch-20261010/scripts','wcl-evidence-closure-20261009/scripts']]
        from kn_native import load,tokenizer
        from ep_source_codec import PartialSourceCodec
        n,_=ledger.call('model_load','once',lambda:load(cr(r['model_routes']),native_counts,boundary,out,r['gpu_pci']))
        codec=PartialSourceCodec(R,native=n);torch=n.torch
        items=[]
        for s in r['sources']:
            with np.load(checked(s['archive']),allow_pickle=False) as z:
                pix=z['pixels'].copy();tokens=z['tokens'].copy()
            assert asha(pix)==s['pixels_sha256'];assert hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
            with np.load(checked(s['streams']),allow_pickle=False) as z:streams={k:z[k].copy() for k in z.files}
            items.append((pix,tokens,streams,cr(s['trace'])))
        (out/'calls').mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        se=(out/'phy_stderr.log').open('xb')
        server=subprocess.Popen([r['cpu_python'],'-B',str(Path(__file__).with_name('tx_phy_server.py')),str(rp),str(out/'phy')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=se,text=True,bufsize=1,env=env)
        ready=json.loads(server.stdout.readline());assert ready['ready'];save(out/'phy_ready.json',ready)
        profiles=cr(r['catalogue'])['profiles'];records=[]
        def one(phase,pos,grid,rep):
            boundary();s=r['sources'][pos];pixels,original_tokens,reference_streams,reference_trace=items[pos]
            job=s['jobs'][str(grid)][str(6201+rep)];key=f'{phase}/{pos:03d}/{grid}/{rep}'
            server.stdin.write(json.dumps(dict(key=key,profile_id=job['profile_id'],counter=job['public_frame_counter']))+'\n');server.stdin.flush()
            assert json.loads(server.stdout.readline())==dict(prepared=key)
            before=native_counts.counts.get('prior_scale',0)
            def tx():
                torch.cuda.synchronize();total_begin=time.perf_counter()
                begin=time.perf_counter();tokens=tokenizer(n,pixels);torch.cuda.synchronize();tokenizer_ms=(time.perf_counter()-begin)*1000
                begin=time.perf_counter();streams,cdfs,parts,counts=encode(codec,tokens,grid,torch.cuda.synchronize);source_ms=(time.perf_counter()-begin)*1000
                begin=time.perf_counter();p,bits,checks=select(streams,grid,profiles);lookup_ms=(time.perf_counter()-begin)*1000
                # No disk I/O or audit reservation inside this full TX interval.
                begin=time.perf_counter();server.stdin.write(json.dumps(dict(bits=base64.b64encode(bits.tobytes()).decode()))+'\n');server.stdin.flush()
                response=json.loads(server.stdout.readline());wave=np.frombuffer(base64.b64decode(response['wave']),dtype=response['dtype']).reshape(1024,2).copy()
                ipc_phy_ms=(time.perf_counter()-begin)*1000;torch.cuda.synchronize();total_ms=(time.perf_counter()-total_begin)*1000
                # Identity checks below are excluded from all timed regions.
                valid_tokens=np.array_equal(tokens,original_tokens)
                trace=[asha(cdf) for cdf in cdfs]
                valid_cdf=trace==[v['cdf_sha256'] for v in reference_trace]
                valid_streams=all(np.array_equal(v,reference_streams['m%d_K%d'%e]) for e,v in streams.items())
                valid_endpoint=(p['m'],p['K'],checks,len(bits),asha(bits))==(job['m'],job['K'],job['length_checks'],job['arithmetic_bits'],job['payload_sha256'])
                valid_wave=asha(wave)==job['waveform_sha256']
                row=dict(key=key,phase=phase,source_position=pos,source_id=s['source_id'],source_index=s['source_index'],grid=grid,repeat=rep+1,tokenizer_ms=tokenizer_ms,source_traversal_ms=source_ms,**parts,length_lookup_ms=lookup_ms,packing_ldpc_qam_ms=response['packing_ldpc_qam_ms'],phy_with_ipc_ms=ipc_phy_ms,total_tx_ms=total_ms,**counts,actual_prior_scale_calls=native_counts.counts.get('prior_scale',0)-before,length_checks=checks,endpoint_m=p['m'],endpoint_K=p['K'],payload_bits=len(bits),tokens_exact=bool(valid_tokens),cdfs_exact=bool(valid_cdf),streams_exact=bool(valid_streams),endpoint_exact=bool(valid_endpoint),wave_exact=bool(valid_wave),wave_sha256=asha(wave))
                save(out/'calls'/f'{len(records):04d}.json',row);records.append(row)
                assert all([valid_tokens,valid_cdf,valid_streams,valid_endpoint,valid_wave]),('QUALIFICATION_IDENTITY_MISMATCH',row)
                assert counts['finish_calls']==len(endpoints(grid)) and row['actual_prior_scale_calls']==10
                return row
            result=ledger.call('complete_tx',key,tx)
            print('TX_COMPLETE',len(records),key,round(result['total_tx_ms'],3),flush=True)
        for pos in (0,1):
            for grid in (4,8):one('qualification',pos,grid,0)
        save(out/'qualification.json',dict(status='PASS',calls=4,all_streams_cdfs_tokens_endpoints_wave_exact=True))
        for rep in range(3):
            for grid in ((4,8) if rep%2==0 else (8,4)):one('warmup',0,grid,rep)
        for rep in range(3):
            for pos in range(100):
                for grid in ((4,8) if (pos+rep)%2==0 else (8,4)):one('measured',pos,grid,rep)
                if pos%10==0:save(out/f'resource_{rep}_{pos:03d}.json',gpu(r))
        server.stdin.write(json.dumps(dict(quit=True))+'\n');server.stdin.flush();server.stdin.close();rc=server.wait(timeout=60)
        save(out/'phy_actual_wait.json',dict(returncode=rc,actual_wait=True,process=ready['process']));assert rc==0
        save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,calls=len(records),measured=600,qualification=4,warmup=6,ledger=ledger.summary(),native_counts=native_counts.counts,seconds=time.monotonic()-start,new_RX=0,new_metrics=0,new_noise=0,new_training=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),native_counts=native_counts.counts,automatic_retry=False))
        if server and server.poll() is None:
            server.stdin.close();rc=server.wait(timeout=60);save(out/'phy_failure_actual_wait.json',dict(returncode=rc,actual_wait=True))
        raise

if __name__=='__main__':main(*sys.argv[1:])
