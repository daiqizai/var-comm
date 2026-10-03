"""Exclusive real-calibration A/B test of raw telemetry caching; no science writes."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

NAME='M2-TELEMETRY-20261002'
PROTOCOL=dict(source_indices=[0],noise_snr_db=7,noise_seeds=[4101,4102,4103],
    projections='every frozen resource-feasible projection',warmup_complete_sources=1,
    repeated_orders=[['original','cached'],['cached','original']],minimum_speedup=1.10,
    exact_tokens=True,exact_latent=True,exact_diagnostics=True,exact_rgb=True,exact_old_metrics=True,
    training_updates=0,policy_selection_updates=0,metric_batch_size_unchanged=True)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)
def verify(bindings):
    if not bindings:raise RuntimeError('Nonempty engineering bindings required')
    for path,digest in bindings.items():
        if sha(path)!=digest:raise RuntimeError('Benchmark input changed: '+path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def tensor_sha(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()


def numpy_sha(array):
    return hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()


def fingerprint(output):
    return dict(base_latent_sha256=tensor_sha(output['base']),base_rgb_sha256=numpy_sha(output['base_rgb']),
        base_metrics=output['base_metrics'],cases=[dict(case=row['case'],measurement=row['measurement'],
        token_sha256=[tensor_sha(t) for t in row['inferred']['tokens']],
        fhat_sha256=tensor_sha(row['inferred']['fhat']),diagnostics=row['inferred']['diagnostics'],
        inference=row['inferred']['inference'],rgb_sha256=numpy_sha(row['rgb']),metrics=row['metrics']) for row in output['cases']])


def compare(torch,np,first,second):
    if len(first['cases'])!=len(second['cases']) or not first['cases']:raise RuntimeError('Incomplete case comparison')
    if not torch.equal(first['base'],second['base']) or not np.array_equal(first['base_rgb'],second['base_rgb']):
        raise RuntimeError('Telemetry changed original unguided latent/RGB')
    if identity(first['base_metrics'])!=identity(second['base_metrics']):raise RuntimeError('Telemetry changed baseline metrics')
    for a,b in zip(first['cases'],second['cases']):
        if a['case']!=b['case'] or identity(a['measurement'])!=identity(b['measurement']):raise RuntimeError('Measurement/case changed')
        aa,bb=a['inferred'],b['inferred']
        if len(aa['tokens'])!=len(bb['tokens']) or any(not torch.equal(x,y) for x,y in zip(aa['tokens'],bb['tokens'])):
            raise RuntimeError('Telemetry changed a token')
        if not torch.equal(aa['fhat'],bb['fhat']):raise RuntimeError('Telemetry changed the latent')
        if identity(aa['diagnostics'])!=identity(bb['diagnostics']) or aa['inference']!=bb['inference']:
            raise RuntimeError('Telemetry changed receiver diagnostics')
        if not np.array_equal(a['rgb'],b['rgb']) or identity(a['metrics'])!=identity(b['metrics']):
            raise RuntimeError('Telemetry changed RGB or an original metric')
    return True


def compare_history(output, historical, registration_sha, replay):
    if historical.get('registration_sha256')!=registration_sha:
        raise RuntimeError('Historical gate source registration differs')
    rows=historical.get('rows',[])
    by_key={(r['projection'],int(r['noise_seed']),r['control']):r for r in rows}
    expected={(case['case']['projection'],case['case']['noise_seed'],control)
              for case in output['cases'] for control in ('UNGUIDED','VAR_GUIDED')}
    if len(rows)!=len(expected) or set(by_key)!=expected:
        raise RuntimeError('Historical full gate source coverage differs')
    comparisons=[]
    for case in output['cases']:
        for control in ('UNGUIDED','VAR_GUIDED'):
            key=(case['case']['projection'],case['case']['noise_seed'],control);row=by_key[key]
            if int(row['source_index'])!=0 or row['stage']!='full_calibration_gate_reduced_noisy':
                raise RuntimeError('Historical gate source/stage differs')
            computed=dict(output['base_metrics'] if control=='UNGUIDED' else case['metrics'],
                **case['measurement'],projection=key[0],control=control,
                **{'lambda':0. if control=='UNGUIDED' else case['case']['lambda_value']})
            comparisons.append(dict(projection=key[0],noise_seed=key[1],control=control,
                                    parity=replay.parity_check(row,computed,required_quality=True)))
    return dict(status='ORIGINAL_GATE_PARITY_PASS',rows=len(rows),comparisons=comparisons,
                original_tolerances=replay.PARITY_TOLERANCES,candidate_vs_baseline_requires_exact_equality=True)


def decision(rows):
    if len(rows)!=4 or [(r['repeat'],r['implementation']) for r in rows]!=[(0,'original'),(0,'cached'),(1,'cached'),(1,'original')]:
        raise RuntimeError('Complete predeclared interleaved timings required')
    if any(not math.isfinite(r['seconds']) or r['seconds']<=0 or r.get('exact') is not True for r in rows):
        raise RuntimeError('Every complete timing must have exact scientific equality')
    totals={name:sum(r['seconds'] for r in rows if r['implementation']==name) for name in ('original','cached')}
    speedup=totals['original']/totals['cached'];passed=speedup>=PROTOCOL['minimum_speedup']
    return dict(status='QUALIFIED' if passed else 'USE_ORIGINAL',selected_implementation='cached' if passed else 'original',
        speedup=speedup,original_seconds=totals['original'],cached_seconds=totals['cached'],strict_equality_passed=True)


@contextmanager
def safety_counter(common):
    safety=common.assets.old.b.SAFETY
    if safety is None:raise RuntimeError('Original Safety instance missing')
    original=safety.check;stats={'calls':0,'seconds':0.}
    def check():
        tick=time.perf_counter()
        try:return original()
        finally:stats['calls']+=1;stats['seconds']+=time.perf_counter()-tick
    safety.check=check
    try:yield stats
    finally:safety.check=original


def rng_state(torch,np):
    return dict(torch=torch.get_rng_state().clone(),cuda=[v.clone() for v in torch.cuda.get_rng_state_all()],
                numpy=np.random.get_state(),python=random.getstate())
def restore_rng(state,torch,np):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
def same_rng(a,b,torch,np):
    return (torch.equal(a['torch'],b['torch']) and len(a['cuda'])==len(b['cuda'])
        and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))
        and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1])
        and a['numpy'][2:]==b['numpy'][2:] and a['python']==b['python'])


def runtime_flags(torch):
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())


def approved_self():
    raw=Path('/proc/self/stat').read_text()
    fields=raw[raw.rfind(') ')+2:].split()
    if len(fields)<20 or not fields[19].isdigit():raise RuntimeError('Cannot identify benchmark process')
    return {os.getpid():fields[19]}


def kv_closed(var):
    for block in var.blocks:
        if block.attn.caching or block.attn.cached_k is not None or block.attn.cached_v is not None:
            raise RuntimeError('VAR cache remained active after benchmark receiver')


def source_pass(common,rx,runner,loaded,data,state,policy,names):
    torch=rx.torch;projections,statistics,operators,static=state;index=0
    common.check();prefix=[t[index:index+1].to(loaded['device']) for t in runner.split_batch(data['T'])[:4]]
    pre=rx.prefix_latent(loaded['vae'].quantize,prefix);clean=data['F'][index:index+1].to(loaded['device'])
    base=runner.greedy(loaded,prefix);base_metrics,base_rgb=runner.score(loaded,data,index,base,images=True)
    cases=[]
    for name in sorted(names):
        projection=projections[name];lam=policy['var'][name]['selected_lambda']
        for seed in common.CAL_SEEDS:
            observation,variance,info=runner.measure(clean,pre,projection,data['records'][index],7,seed)
            inferred=rx.infer(loaded['vae'],loaded['var'],prefix,observation,projection,statistics[name],
                noise_variance=variance,prior='VAR',lam=lam,static=static,operators=operators[name],check=common.check)
            metrics,rgb=runner.score(loaded,data,index,inferred['fhat'],images=True);kv_closed(loaded['var'])
            cases.append(dict(case=dict(source_index=index,projection=name,noise_seed=seed,snr_db=7,lambda_value=lam),
                measurement=info,inferred=inferred,metrics=metrics,rgb=rgb))
    return dict(base=base,base_metrics=base_metrics,base_rgb=base_rgb,cases=cases)


def run(root):
    root=Path(root).resolve();out=root/'outputs'/NAME;out.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    if here!=out/'runtime':raise RuntimeError('Benchmark must run from its immutable separate runtime')
    import fcntl
    lock=(out/'benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own={str(p):sha(p) for p in sorted(here.iterdir()) if p.suffix in ('.py','.md')}
    import telemetry
    if Path(telemetry.__file__).resolve().parent!=here:raise RuntimeError('Unexpected telemetry implementation origin')
    wrapper_path=root/'experiments/metric-speed-20261002/wrapper.py'
    w=load('_telemetry_benchmark_original_wrapper',wrapper_path)
    if w.ROOT.resolve()!=root:raise RuntimeError('Original wrapper root differs')
    speed_q=root/'outputs/METRIC-SPEED-20261002/m2_speed_qualification.json'
    speed=read(speed_q)
    if speed.get('status')!='USE_ORIGINAL' or speed.get('selected_implementation')!='original':
        raise RuntimeError('Benchmark requires the actual unchanged original receiver implementation')
    parent=w.require_m1(root);w.verify(speed['source_bindings']);w.verify(speed['base_source_bindings'])
    common,rx,runner=w.load_original(root);torch=rx.torch
    import numpy as np
    parent_dir=root/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
    regpath=parent_dir/'m2_calibration_registration.json';registration=read(regpath)
    policy_path=parent_dir/'m2/calibration/policy.json';policy=read(policy_path)
    statistics_path=parent_dir/'m2/statistics_complete.json';statistics_receipt=read(statistics_path)
    screen_path=parent_dir/'m2/calibration/screen_per_frame.csv'
    gate_path=parent_dir/'m2/calibration/gate/source_0000.json';gate_source=read(gate_path)
    parity_path=root/'experiments/unified-metrics-20261002/replay.py'
    parity_registration_path=root/'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json'
    if read(parity_registration_path)['source_bindings'].get(str(parity_path))!=sha(parity_path):
        raise RuntimeError('Original registered replay parity source differs')
    replay=load('_telemetry_benchmark_original_parity',parity_path)
    if (policy.get('status')!='M2_ORACLE_POLICY_FROZEN' or policy.get('development_read') is not False
            or policy.get('training_updates')!=0 or policy.get('registration_sha256')!=sha(regpath)
            or policy.get('screen_sha256')!=sha(screen_path)
            or policy.get('statistics_receipt_sha256')!=sha(statistics_path)
            or list(common.CAL_SEEDS)!=PROTOCOL['noise_seeds']):raise RuntimeError('Frozen policy scope/evidence differs')
    bindings={**own,**parent,**registration['source_bindings'],**policy['source_bindings'],**statistics_receipt['files'],
        str(wrapper_path):sha(wrapper_path),str(speed_q):sha(speed_q),str(regpath):sha(regpath),
        str(policy_path):sha(policy_path),str(statistics_path):sha(statistics_path),str(screen_path):sha(screen_path),
        str(gate_path):sha(gate_path),str(parity_path):sha(parity_path),str(parity_registration_path):sha(parity_registration_path)}
    verify(bindings)
    receipt_path=out/'telemetry_qualification.json'
    if receipt_path.exists():
        completed=read(receipt_path)
        if (completed.get('status') not in ('QUALIFIED','USE_ORIGINAL') or completed.get('inputs')!=bindings
                or completed.get('protocol')!=PROTOCOL
                or completed.get('payload_sha256')!=identity({k:v for k,v in completed.items() if k!='payload_sha256'})):
            raise RuntimeError('Existing telemetry qualification is incomplete or has different bindings')
        return completed
    receipt=dict(status='RUNNING',protocol=PROTOCOL,inputs=bindings,source_bindings=own,scientific_result=False,
        engineering_fixture=True,real_model_weights=True,calibration_sources=[0],development_read=False,
        training_updates=0,policy_selection_updates=0,original_scientific_outputs_written=False,latency=[],
        raw_telemetry_max_age_seconds=1.,unknown_gpu_detection_may_be_delayed_seconds=1.)
    write(receipt_path,receipt);start=time.monotonic();saved_rng=None
    try:
        common.check();loaded=common.setup();common.assert_frozen(loaded)
        if loaded['identity']!=registration['identity']:raise RuntimeError('Frozen calibration visual model identity differs')
        if w.numerical_backend(torch,loaded['device'])!=speed['numerical_backend']:
            raise RuntimeError('Actual production numerical backend differs')
        receipt['model_identity']=loaded['identity'];flags=runtime_flags(torch);receipt['numerical_runtime']=flags
        data=common.assets.load_calibration()
        if (len(data['records'])!=1000 or [r['image_id'] for r in data['records']]!=registration['source_ids']
                or [r['preprocessing_id'] for r in data['records']]!=registration['preprocessing_ids']):
            raise RuntimeError('Original complete calibration population/source differs')
        if data['bindings']!=registration['data_bindings']:raise RuntimeError('Calibration data bindings differ')
        data['reference']={}
        with torch.no_grad():
            # The original score(source0) accesses only source0 and its fixed mismatch source1.
            for index in (0,1):
                pixels=torch.as_tensor(data['records'][index]['pixels'][None],dtype=torch.float32,device=loaded['device'])/255.
                data['reference'][index]=common.dino_features(loaded['dino'],pixels)[0].cpu()
            state=runner.load_state(loaded)
            names={f'g{g}_c{ch}' for g,ch in rx.PROJECTIONS for N in (512,1024) for phy in ('QPSK','16QAM')
                   if rx.resource_record(N,g,ch,phy)['feasible']}
            if len(names)!=5 or any(name not in policy['var'] for name in names):raise RuntimeError('Frozen gate projection inventory differs')
            receipt['projection_names']=sorted(names);receipt['expected_cases_per_pass']=len(names)*len(common.CAL_SEEDS)
            receipt['fixture']=dict(source_id=data['records'][0]['image_id'],preprocessing_id=data['records'][0]['preprocessing_id'],
                pixels_sha256=numpy_sha(data['records'][0]['pixels']),F_sha256=tensor_sha(data['F'][0]),T_sha256=tensor_sha(data['T'][0]),
                selected_lambdas={name:policy['var'][name]['selected_lambda'] for name in sorted(names)})
            saved_rng=rng_state(torch,np)
            with telemetry.QueryCache(approved_pids=approved_self,enabled=False):
                warm=source_pass(common,rx,runner,loaded,data,state,policy,names)
            receipt['historical_gate_parity']=compare_history(warm,gate_source,sha(regpath),replay)
            kv_closed(loaded['var']);warm_fingerprint=fingerprint(warm);del warm
            baseline=None;baseline_rng=None
            for repeat,order in enumerate(PROTOCOL['repeated_orders']):
                for implementation in order:
                    restore_rng(saved_rng,torch,np)
                    if runtime_flags(torch)!=flags:raise RuntimeError('Numerical runtime changed')
                    torch.cuda.synchronize(loaded['device'])
                    with telemetry.QueryCache(approved_pids=approved_self,enabled=implementation=='cached') as query:
                        with safety_counter(common) as safety:
                            tick=time.perf_counter()
                            value=source_pass(common,rx,runner,loaded,data,state,policy,names)
                            torch.cuda.synchronize(loaded['device']);seconds=time.perf_counter()-tick
                        query_stats=query.stats()
                    after=rng_state(torch,np);kv_closed(loaded['var'])
                    if baseline is None:baseline=value;baseline_rng=after
                    exact=compare(torch,np,baseline,value)
                    if not same_rng(baseline_rng,after,torch,np):raise RuntimeError('Telemetry changed RNG state')
                    proof=fingerprint(value)
                    if identity(proof)!=identity(warm_fingerprint):raise RuntimeError('Warm/real benchmark outputs differ')
                    receipt['latency'].append(dict(repeat=repeat,implementation=implementation,seconds=seconds,exact=exact,
                        safety=dict(safety),query=query_stats,scientific_fingerprint_sha256=identity(proof)))
                    write(out/'benchmark_status.json',dict(status='RUNNING',passes=len(receipt['latency']),total=4,
                        elapsed_seconds=time.monotonic()-start,last=receipt['latency'][-1]))
                    if value is not baseline:del value
            common.assert_frozen(loaded);verify(bindings)
            if runtime_flags(torch)!=flags:raise RuntimeError('Numerical runtime changed')
            receipt.update(decision(receipt['latency']));receipt['scientific_fingerprint']=fingerprint(baseline)
        receipt.update(rng_state_restored=True,elapsed_seconds=time.monotonic()-start)
        receipt['payload_sha256']=identity(receipt);write(receipt_path,receipt);return receipt
    except Exception as error:
        receipt.update(status='FAILED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write(out/'telemetry_qualification_failed.json',receipt);raise
    finally:
        if saved_rng is not None:restore_rng(saved_rng,torch,np)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    answer=run(parser.parse_args().root)
    print(json.dumps({k:answer[k] for k in ('status','selected_implementation','speedup')}))
