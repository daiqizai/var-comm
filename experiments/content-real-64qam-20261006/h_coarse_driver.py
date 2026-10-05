"""Registered H coarse BLER table. Separate from all frozen H implementations.

Workers measure disjoint halves of the eight qualified physical layouts.
No image, source population, source-quality table, or development data is read.
Every actual decoder call uses the existing independent H budget ledger.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

BLOCKS = 2048
SNRS = (13,16,19)
SEED = 2026100602
WORKERS = 2
STOP = False


def require(ok,message):
    if not ok: raise RuntimeError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def stop(*_):
    global STOP
    STOP=True


def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def csv_write(path,rows):
    require(bool(rows),'Empty coarse CSV')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    os.replace(tmp,p)


def verify(bindings):
    for path,checksum in bindings.items(): require(sha(path)==checksum,'Changed registered file: '+path)


def generator(protocol,layout_id,snr,index,stream):
    import numpy as np
    require(stream in ('payload','noise'),'Unregistered random stream')
    seed=int.from_bytes(hashlib.sha256(canonical([protocol,'coarse',SEED,str(layout_id),int(snr),int(index),stream]).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed))


def public_counter(layout_index,snr,index):
    require(type(layout_index) is int and 0<=layout_index<8 and snr in SNRS and 0<=index<BLOCKS,'Invalid public coarse counter')
    return SEED*100000+layout_index*len(SNRS)*BLOCKS+SNRS.index(snr)*BLOCKS+index


def worker_layout_indices(index,workers=WORKERS):
    require(type(index) is int and index in (0,1) and workers==2,'Exactly two registered workers')
    return list(range(index*4,(index+1)*4))


def wilson(k,n):
    require(type(k) is int and type(n) is int and 0<=k<=n and n>0,'Invalid binomial counts')
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-width),min(1.,center+width)]


def energy_summary(values):
    import numpy as np
    x=np.asarray(values,dtype=np.float64)
    require(x.ndim==1 and len(x)>0 and np.isfinite(x).all(),'Invalid measured body energies')
    return dict(count=len(x),mean=float(x.mean()),sd_population=float(x.std()),
                minimum=float(x.min()),p05=float(np.quantile(x,.05)),p50=float(np.quantile(x,.5)),
                p95=float(np.quantile(x,.95)),maximum=float(x.max()))


def summarize_rows(rows):
    require(bool(rows),'Empty physical cell')
    counts={name:sum(int(r[name]) for r in rows) for name in
            ('correct','rejected','undetected','crc_accepted','parser_invalid')}
    n=len(rows)
    require(counts['correct']+counts['rejected']+counts['undetected']==n,'C/R/U do not partition measured packets')
    require(counts['crc_accepted']==counts['correct']+counts['undetected'],'CRC accepted/correct distinction changed')
    require(counts['parser_invalid']<=counts['undetected'],'Parse-invalid outside wrong accepted set')
    return dict(trials=n,**counts,empirical_BLER=1-counts['correct']/n,
        probabilities={name:counts[name]/n for name in ('correct','rejected','undetected')},
        wilson95={name:wilson(counts[name],n) for name in ('correct','rejected','undetected')},
        body_energy=energy_summary([r['body_energy'] for r in rows]),
        paid_frame_energy_with_constant_QPSK_header=energy_summary([r['body_energy']+136 for r in rows]),
        header_energy_is_fixed_algebraic_addition=True,header_decode_calls=0)


def measure_cell(backend,ledger,bucket,layout_index,snr,boundary,blocks=BLOCKS):
    """One independent actual packet per index; small counts are test injection only."""
    import numpy as np
    from h64_catalog import PROTOCOL,make_profile
    from h64_phy import transmit_body,receive_body,pack_body,array_sha
    require(0<blocks<=BLOCKS,'Test/production block count invalid')
    p=make_profile(bucket,6,0,'arithmetic',['COARSE_RANDOM_MESSAGE_PHY_ONLY'])
    lid=bucket['layout']['layout_id'];capacity=p['source_capacity'];rows=[]
    for index in range(blocks):
        boundary()
        payload=generator(PROTOCOL,lid,snr,index,'payload').integers(0,2,capacity,dtype=np.uint8)
        counter=public_counter(layout_index,snr,index)
        wave,meta=transmit_body(backend,payload,p,counter,'H_COARSE')
        noise=generator(PROTOCOL,lid,snr,index,'noise').standard_normal(wave.shape).astype(np.float32)
        received=(wave+noise*(10**(-float(snr)/20))).astype(np.float32)
        event_id=f'HCOARSE:{lid}:{snr}:{index}'
        event=receive_body(backend,received,p,snr,counter,ledger,'coarse',event_id,'H_COARSE')
        actual=np.asarray(event['decoded_bits'],dtype=np.uint8)
        sent=pack_body(payload,p)
        accepted=bool(event['crc_accepted'])
        correct=bool(accepted and np.array_equal(actual,sent))
        rows.append(dict(index=index,event_id=event_id,public_frame_counter=counter,
            correct=int(correct),rejected=int(not accepted),undetected=int(accepted and not correct),
            crc_accepted=int(accepted),parser_invalid=int(accepted and not event['parser_accepted']),
            body_energy=meta['actual_energy'],noise_sha256=array_sha(noise),
            received_sha256=array_sha(received),decoded_sha256=array_sha(actual)))
    summary=summarize_rows(rows)
    return rows,dict(layout_id=lid,layout_index=layout_index,phy_profile_key=p['profile_key'],
        resource_id=bucket['resource_id'],q=bucket['q'],nominal_rate=bucket['nominal_rate'],
        k=bucket['k'],n=bucket['n'],body_symbols=956,snr_db=snr,
        random_message='Uniform independent source_capacity bits plus actual13bit length and CRC16; no source-model parsing',
        source_capacity=capacity,seed=SEED,**summary)


def load_registered(config_path):
    cfg=read(config_path);reg=read(cfg['registration']);protocol=read(cfg['protocol'])
    require(reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H','New H execution revision missing')
    verify(reg['source_bindings']);verify(reg['input_bindings'])
    bound=dict(reg['source_bindings'],**reg['input_bindings'])
    require(bound.get(str(Path(config_path).absolute()))==sha(config_path),'Coarse config not source-bound')
    for name in ('h_coarse_driver.py','h64_catalog.py','h64_backend.py','h64_phy.py'):
        path=str(Path(__file__).absolute().with_name(name))
        require(bound.get(path)==sha(path),'Missing/changed executing module binding: '+path)
    for key in ('protocol','budget_registration','qualification_completion','catalogue','ledger_module','reference_qualification'):
        require(bound.get(cfg[key])==sha(cfg[key]),'Required input/module not bound: '+key)
    oldbackend=str(Path(cfg['legacy_runtime'])/'ldpc_backend.py')
    require(bound.get(oldbackend)==sha(oldbackend),'Old adapter source binding missing')
    require(protocol['status']=='FROZEN_BEFORE_DATA' and protocol['N']==1024 and protocol['body_symbols']==956
            and protocol['header_symbols']==68 and protocol['population']['coarse_seed']==SEED
            and protocol['modulation_bits']==[4,6] and protocol['rates']==['1/2','2/3','3/4','5/6'],
            'Frozen H protocol differs')
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
            and budget['phase_limits']['coarse']==49152,'Wrong fixed coarse budget')
    qual=read(cfg['qualification_completion']);verify(qual['outputs'])
    require(qual['status']=='H_PHY_QUALIFICATION_PASS' and qual['budget_registration_sha256']==sha(cfg['budget_registration']),
            'Qualification not passed on this independent budget')
    require(qual['outputs'].get(cfg['catalogue'])==sha(cfg['catalogue']),'Catalogue not sealed by qualification')
    verify(qual['source_bindings'])
    cat=read(cfg['catalogue']);buckets=cat['buckets']
    require(len(buckets)==8 and all(b['admission']=='ADMITTED' for b in buckets), 'Coarse expects all eight already admitted layouts')
    require([(b['q'],b['nominal_rate']) for b in buckets]==[(q,r) for q in (4,6) for r in ('1/2','2/3','3/4','5/6')],
            'Qualified public layout ordering changed')
    require(len({b['layout']['layout_id'] for b in buckets})==8,'Duplicate physical layouts')
    affinity=cfg['cpu_affinity']
    require(len(affinity)==2 and all(len(a)==2 and len(set(a))==2 for a in affinity)
            and not(set(affinity[0])&set(affinity[1])),'Two workers need separate pairs of CPU cores')
    return cfg,reg,protocol,budget,qual,buckets


def cpu_configure(cores,with_torch):
    require(sys.platform.startswith('linux'),'Actual coarse execution requires registered Linux worker')
    require(set(cores).issubset(os.sched_getaffinity(0)),'Registered CPU cores unavailable')
    os.sched_setaffinity(0,set(cores));os.setpriority(os.PRIO_PROCESS,0,15)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    if with_torch:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(2)


def make_ledger(cfg,budget):
    spec=importlib.util.spec_from_file_location('h_registered_coarse_ledger',cfg['ledger_module'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.BudgetLedger(cfg['ledger'],sha(cfg['budget_registration']),'H',budget['phase_limits'])


@contextlib.contextmanager
def exclusive(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def audit_worker_ledger(path,buckets,indices,blocks=BLOCKS):
    """Read-only exact identity audit; another live worker may have a reservation."""
    expected={f'HCOARSE:{buckets[i]["layout"]["layout_id"]}:{snr}:{j}' for i in indices for snr in SNRS for j in range(blocks)}
    prefixes=tuple(f'HCOARSE:{buckets[i]["layout"]["layout_id"]}:' for i in indices)
    own=set()
    with contextlib.closing(sqlite3.connect(str(path),timeout=60)) as c:
        # Stream rows: never retain hundreds of MB of actual decoded bit vectors.
        for r in c.execute("SELECT event_id,phase,kind,status,result,result_sha FROM events WHERE phase='coarse'"):
            if not r[0].startswith(prefixes):continue
            require(r[0] not in own,'Duplicate actual event identity')
            own.add(r[0])
            require(r[1:4]==('coarse','body','COMPLETE'),'Worker has unresolved or wrong-category event')
            require(digest(json.loads(r[4]))==r[5],'Actual decoded evidence hash changed')
    require(own==expected,'Worker ledger coverage differs from exact registered events')
    return dict(event_count=len(own),event_ids_sha256=digest(sorted(own)),all_complete=True,
                other_worker_may_still_have_live_reservations=True)


def run_worker(config_path,worker_index):
    cfg,reg,protocol,budget,qual,buckets=load_registered(config_path)
    indices=worker_layout_indices(worker_index);out=Path(cfg['out'])/f'worker_{worker_index}'
    out.mkdir(parents=True,exist_ok=True);regsha=sha(cfg['registration']);start=time.monotonic()
    require(not(out/'failure.json').exists(),'Existing coarse failure requires registered recovery')
    with exclusive(out/'worker.lock'):
        try:
            if (out/'completion.json').exists():
                done=read(out/'completion.json');verify(done['outputs'])
                require(done['status']=='H_COARSE_WORKER_COMPLETE' and done['registration_sha256']==regsha,'Completed worker identity differs')
                return done
            cpu_configure(cfg['cpu_affinity'][worker_index],True)
            ledger=make_ledger(cfg,budget)
            from h64_backend import H64Backend
            backend=H64Backend(reference_qualification=read(cfg['reference_qualification']),legacy_backend_path=Path(cfg['legacy_runtime'])/'ldpc_backend.py')
            require(backend.identity==qual['backend_identity'],'Current backend no longer matches passed H qualification')
            for i in indices:
                b=buckets[i]
                require(backend.plan(b['k'],b['n'],b['q'])==b['layout'],'Current exact layout differs from qualified layout')
            cells=[];outputs={}
            def boundary():
                require(not STOP and not Path(cfg['stop_file']).exists() and not(out/'STOP').exists(),'Safe packet-boundary STOP requested')
                require(time.monotonic()-start<cfg['max_worker_seconds'],'Registered coarse worker time exhausted')
            for i in indices:
                for snr in SNRS:
                    boundary()
                    cell_dir=out/'cells'/f'layout{i}_snr{snr}';cp=cell_dir/'completion.json'
                    if cp.exists():
                        done=read(cp);verify(done['outputs'])
                        require(done['status']=='H_COARSE_CELL_COMPLETE' and done['registration_sha256']==regsha
                                and done['layout_index']==i and done['snr_db']==snr and done['trials']==BLOCKS,
                                'Completed cell differs from registered identity')
                        verify(done['source_bindings'])
                    else:
                        rows,summary=measure_cell(backend,ledger,buckets[i],i,snr,boundary)
                        csv_write(cell_dir/'packets.csv',rows);atomic(cell_dir/'summary.json',summary)
                        verify(reg['source_bindings'])
                        done=dict(status='H_COARSE_CELL_COMPLETE',registration_sha256=regsha,
                            budget_registration_sha256=sha(cfg['budget_registration']),worker_index=worker_index,
                            layout_index=i,snr_db=snr,trials=BLOCKS,source_bindings=reg['source_bindings'],
                            outputs={str(cell_dir/name):sha(cell_dir/name) for name in ('packets.csv','summary.json')})
                        atomic(cp,done)
                    outputs.update(done['outputs']);outputs[str(cp)]=sha(cp)
                    cells.append(read(cell_dir/'summary.json'))
                    atomic(out/'status.json',dict(status='RUNNING',completed_cells=len(cells),total_cells=12,
                        worker_index=worker_index,completed_packets=len(cells)*BLOCKS,
                        elapsed_seconds=time.monotonic()-start,registration_sha256=regsha))
            proof=audit_worker_ledger(cfg['ledger'],buckets,indices)
            atomic(out/'table.json',cells);outputs[str(out/'table.json')]=sha(out/'table.json')
            verify(reg['source_bindings']);verify(reg['input_bindings'])
            done=dict(status='H_COARSE_WORKER_COMPLETE',worker_index=worker_index,workers=2,
                registration_sha256=regsha,budget_registration_sha256=sha(cfg['budget_registration']),
                qualification_completion_sha256=sha(cfg['qualification_completion']),layout_indices=indices,
                cells=12,body_decode_events=24576,header_decode_events=0,ledger_audit=proof,
                elapsed_seconds=time.monotonic()-start,GPU_used=False,source_images_read=False,
                development_used=False,outputs=outputs,source_bindings=reg['source_bindings'])
            atomic(out/'completion.json',done);atomic(out/'status.json',done);return done
        except BaseException as exc:
            if not(out/'failure.json').exists():
                atomic(out/'failure.json',dict(status='FAILED',registration_sha256=regsha,worker_index=worker_index,
                    error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,
                    existing_measurements_preserved=True,automatic_retry=False))
            raise


def run_merge(config_path):
    cfg,reg,protocol,budget,qual,buckets=load_registered(config_path)
    out=Path(cfg['out']);regsha=sha(cfg['registration'])
    with exclusive(out/'merge.lock'):
        require(not(out/'merge_failure.json').exists(),'Existing merge failure requires diagnosis')
        try:
            cpu_configure(cfg['cpu_affinity'][0],False)
            cells=[];outputs={};input_receipts={}
            for worker in (0,1):
                directory=out/f'worker_{worker}'
                require(not(directory/'failure.json').exists(),'Coarse worker has failure evidence')
                path=directory/'completion.json';done=read(path);verify(done['outputs'])
                require(done['status']=='H_COARSE_WORKER_COMPLETE' and done['registration_sha256']==regsha
                        and done['worker_index']==worker and done['layout_indices']==worker_layout_indices(worker),
                        'Coarse worker completion identity differs')
                audit_worker_ledger(cfg['ledger'],buckets,worker_layout_indices(worker))
                cells.extend(read(directory/'table.json'));outputs.update(done['outputs'])
                outputs[str(path)]=sha(path);input_receipts[str(path)]=sha(path)
            require(len(cells)==24 and {(x['layout_index'],x['snr_db']) for x in cells}=={(i,s) for i in range(8) for s in SNRS},
                    'Complete coarse Cartesian table missing or duplicated')
            require(all(x['trials']==BLOCKS and x['correct']+x['rejected']+x['undetected']==BLOCKS for x in cells),
                    'Coarse cell counts differ')
            ledger=make_ledger(cfg,budget);snapshot=ledger.assert_quiescent()
            require(snapshot['phase_charged']['coarse']==49152,'Coarse ledger total is not exact49152')
            cells.sort(key=lambda x:(x['layout_index'],x['snr_db']))
            table=out/'bler_counts.json';atomic(table,cells);outputs[str(table)]=sha(table)
            flat=[]
            for x in cells:
                row={k:x[k] for k in ('layout_id','layout_index','q','nominal_rate','k','n','snr_db','trials','correct','rejected','undetected','crc_accepted','parser_invalid','empirical_BLER')}
                for label in ('correct','rejected','undetected'):
                    row[label+'_wilson95_low'],row[label+'_wilson95_high']=x['wilson95'][label]
                row.update(body_energy_mean=x['body_energy']['mean'],body_energy_sd=x['body_energy']['sd_population'])
                flat.append(row)
            csv_write(out/'bler_counts.csv',flat);outputs[str(out/'bler_counts.csv')]=sha(out/'bler_counts.csv')
            verify(reg['source_bindings']);verify(reg['input_bindings'])
            done=dict(status='H_COARSE_COMPLETE',registration_sha256=regsha,
                budget_registration_sha256=sha(cfg['budget_registration']),qualification_completion_sha256=sha(cfg['qualification_completion']),
                cells=24,trials_per_cell=2048,body_decode_events=49152,header_decode_events=0,
                GPU_used=False,source_images_read=False,development_used=False,model_probabilities_updated=False,
                ledger=snapshot,outputs=outputs,input_receipts=input_receipts,source_bindings=reg['source_bindings'],
                BLER_theory='Measured coded AWGN table; no closed-form LDPC theory claim')
            atomic(out/'completion.json',done);return done
        except BaseException as exc:
            if not(out/'merge_failure.json').exists():
                atomic(out/'merge_failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
                    registration_sha256=regsha,automatic_retry=False))
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('worker','merge'),required=True)
    parser.add_argument('--worker-index',type=int)
    args=parser.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage=='worker':
        require(args.worker_index in (0,1),'Worker index0/1 required')
        result=run_worker(args.config,args.worker_index)
    else:
        require(args.worker_index is None,'Merge has no worker index')
        result=run_merge(args.config)
    print(json.dumps({k:result[k] for k in ('status','registration_sha256')},sort_keys=True))


if __name__=='__main__':main()
