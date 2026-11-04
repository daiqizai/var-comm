"""Single finite P1 source-codec chain. Launch each registered role at most once."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback,hashlib
from owned_process import run_role,proc_identity

def main():
    request=Path(sys.argv[1]);r=json.loads(request.read_text());L=Path(r['root']);S=Path(__file__).resolve().parent;O=L/'p1_source_chain_v1';O.mkdir(exist_ok=False)
    (O/'owner.json').write_text(json.dumps(dict(**proc_identity(os.getpid()),started=time.time())))
    for p,h in r['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert r['reserved_gpu_hours']<=64 and json.loads((L/'p0_gate.json').read_text())['passed']
    procs=[]
    try:
        for item in r['shards']:
            req=Path(item['request']);sr=json.loads(req.read_text());assert not Path(sr['output_root']).exists()
            p=subprocess.Popen([sys.executable,'-u',str(S/'p1_shard_owner.py'),str(req)],stdin=subprocess.DEVNULL,stdout=(O/(item['name']+'_stdout.log')).open('xb'),stderr=(O/(item['name']+'_stderr.log')).open('xb'))
            procs.append((item,p));(O/(item['name']+'_owner.json')).write_text(json.dumps(dict(**proc_identity(p.pid),started=time.time())))
        results=[]
        for item,p in procs:
            rc=p.wait();results.append(dict(name=item['name'],returncode=rc,child_gone=not Path('/proc',str(p.pid)).exists()))
            (O/'shard_actual_waits.json').write_text(json.dumps(results,indent=2))
        assert all(v['returncode']==0 and v['child_gone'] for v in results),'Independent failure retained; successful shards not repeated'
        for item in r['shards']:
            sr=json.loads(Path(item['request']).read_text());assert json.loads((Path(sr['output_root'])/'complete.json').read_text())['passed']
        # The selected GPU is released by its shard before starting metrics.
        gpu=r['post_gpu'];uuid=gpu['uuid'];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert uuid not in apps
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S),TORCH_EXTENSIONS_DIR=str(L/'extensions_headers_v2'),CPATH=str(L/'dependencies/python310_headers/Include')+':'+str(L/'dependencies/python310_headers'))
        ledger=[]
        for role,script,req,seconds,cuda in [('p1_r0_rate_proxy_v1','r0_rate_proxy.py',r['proxy_request'],1200,True),('p1_source_metrics_v1','score_prefixes.py',r['metrics_request'],14400,True),('p1_source_analysis_v1','analyze_r0.py',r['analysis_request'],1800,False)]:
            e=env.copy()
            if not cuda:e['CUDA_VISIBLE_DEVICES']=''
            ledger.append(run_role(L,role,[sys.executable,'-u',str(S/script),'--request',req],e,gpu['cpus'],[uuid] if cuda else [],seconds,r['bindings']))
            (O/'post_resource_ledger.json').write_text(json.dumps(ledger,indent=2))
        result=json.loads((L/'p1_source_analysis_v1/complete.json').read_text());assert result['passed'] and result['all_budgets_feasible']
        charge=sum(x['charged_gpu_device_hours'] for x in ledger)
        for item in r['shards']:
            sr=json.loads(Path(item['request']).read_text());charge+=json.loads((Path(sr['output_root'])/'complete.json').read_text())['gpu_hours']
        assert charge<=r['reserved_gpu_hours']<=64
        gate=dict(passed=True,source_codec_curves_closed=True,wireless_reliability_pending=True,gpu_hours=charge,source_results=result,finished=time.time())
        (L/'p1_source_gate.json').write_text(json.dumps(gate,indent=2));(O/'complete.json').write_text(json.dumps(gate,indent=2))
    except BaseException:
        # Do not terminate unrelated jobs or re-run a successful scientific call.
        (O/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
