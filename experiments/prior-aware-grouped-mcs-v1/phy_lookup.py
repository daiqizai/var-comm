"""Actual CRC/LDPC/header Monte Carlo lookup, with deterministic resumable blocks.

Random raw payloads are an explicit approximation pending image-payload checks.
This runner never samples a Bernoulli receive state or consults image quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
from uep_common import require,identity,read,write,seal,sha,rng,csv_write

SNRS=(1,4,7,10,13,19)
COARSE=256
MAX_BLOCKS=20000
ERROR_STOP=100
VERSION='UEP-ACTUAL-LDPC-HEADER-BLER-V1'
HERE=Path(__file__).resolve().parent


class StopRequested(Exception):pass


def wilson(successes,n,z=1.959963984540054):
    require(type(n)is int and n>0 and type(successes)is int and 0<=successes<=n,'Invalid binomial count')
    p=successes/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def physical_catalog(profiles):
    require(isinstance(profiles,list) and profiles,'Expected N1024 candidate list')
    require(all(p['N']==1024 and p.get('encoder_qualified') is True for p in profiles),'Only qualified N1024 resource candidates may be measured')
    catalog={};headers=set()
    for profile in profiles:
        headers.add(profile['header_phy_key'])
        for group in profile['groups']:
            spec={k:group[k] for k in ('phy_key','source_bits','crc_bits','tail_bits','information_bits','transmitted_bits','modulation','layout')}
            spec.update(kind='body',q={'QPSK':2,'16QAM':4}[group['modulation']])
            require(spec['crc_bits']==16 and spec['tail_bits']==0 and spec['information_bits']==spec['source_bits']+16,'Body CRC/tail accounting differs')
            require(spec['transmitted_bits']%spec['q']==0,'Modulation not aligned')
            key=spec['phy_key'];require(key not in catalog or catalog[key]==spec,'One physical key names different actual layouts')
            catalog[key]=spec
    require(len(headers)==1,'All profiles must share the same paid header chain')
    header=next(iter(headers));catalog[header]=dict(phy_key=header,kind='header',source_bits=12,crc_bits=16,tail_bits=6,
        symbols=68,transmitted_bits=136,q=2,modulation='QPSK',profile_id_range=[0,4095])
    return catalog


def shard_points(catalog,points,shard_index=0,shard_count=1):
    """One physical configuration (all SNRs) per shard; paid header only shard0.

    Assignment does not change frame counters, payloads, scrambling or noise.
    Keep configurations together to avoid repeated LDPC codec construction.
    """
    require(type(shard_count)is int and shard_count>0 and type(shard_index)is int and 0<=shard_index<shard_count,'Invalid stable shard identity')
    body=sorted(key for key,value in catalog.items() if value['kind']=='body')
    owners={key:index%shard_count for index,key in enumerate(body)}
    owners.update({key:0 for key,value in catalog.items() if value['kind']=='header'})
    require(all(key in owners and snr in SNRS for key,snr in points),'Unknown physical point in shard request')
    return sorted((key,snr) for key,snr in points if owners[key]==shard_index)


def counters_for(phy_key,snr,start,count):
    # Public protocol/config/SNR/frame index; no transmitted content enters it.
    return [int.from_bytes(hashlib.sha256(json.dumps([VERSION,phy_key,int(snr),int(i)],separators=(',',':')).encode()).digest()[:8],'little') & ((1<<63)-1)
            for i in range(start,start+count)]


def random_payloads(spec,snr,start,count):
    counters=counters_for(spec['phy_key'],snr,start,count)
    payload=np.stack([rng('BLER_random_raw_payload',spec['phy_key'],int(snr),int(i)).integers(0,2,spec['source_bits'],dtype=np.uint8)
        for i in range(start,start+count)])
    return payload,counters


def simulate_header(header,spec,snr,start,count):
    counters=counters_for(spec['phy_key'],snr,start,count);correct=[];rejected=[];undetected=[];energy=[]
    for i,counter in zip(range(start,start+count),counters):
        pid=int(rng('BLER_header_profile_id',int(snr),int(i)).integers(0,4096))
        wave=np.asarray(header.transmit(pid),dtype=np.float32)
        require(wave.shape==(68,2),'Header transmitter changed paid symbol count')
        noise=rng('BLER_actual_header_AWGN',int(snr),counter).standard_normal((68,2)).astype(np.float32)
        decision=header.receive(wave+noise*(10**(-float(snr)/20)),snr,codebook=None)
        accepted=bool(decision['header_ok']);exact=accepted and decision['profile_id']==pid
        correct.append(exact);rejected.append(not accepted);undetected.append(accepted and not exact)
        energy.append(float(np.square(wave,dtype=np.float64).sum()))
    return dict(correct=np.asarray(correct),rejected=np.asarray(rejected),undetected=np.asarray(undetected),actual_E=energy)


def actual_batch(backend,header,spec,snr,start,count):
    if spec['kind']=='header':return simulate_header(header,spec,snr,start,count)
    from uep_phy import simulate_packets
    payload,counters=random_payloads(spec,snr,start,count)
    return simulate_packets(backend,payload,spec['transmitted_bits'],spec['q'],snr,counters,
        group=0,session='UEP_BLER_public_counter_v1',noise_namespace='UEP_BLER_AWGN/'+spec['phy_key'])


def batch_receipt(meta,start,count):
    events=[]
    for name in ('correct','rejected','undetected'):
        value=np.asarray(meta[name]);require(value.shape==(count,) and value.dtype==np.bool_,'Actual CRC event vector required')
        events.append(value)
    require(np.all(np.sum(np.stack(events).astype(np.int64),axis=0)==1),'CRC outcome categories are not a disjoint partition')
    energy=np.asarray(meta['actual_E'],dtype=np.float64)
    require(energy.shape==(count,) and np.isfinite(energy).all() and (energy>0).all(),'Actual transmitted energy missing')
    return dict(start=start,n_blocks=count,n_correct=int(events[0].sum()),n_reject=int(events[1].sum()),n_undetected=int(events[2].sum()),
        event_bits_sha256=hashlib.sha256(np.packbits(np.stack(events),axis=1,bitorder='big').tobytes()).hexdigest(),
        energy_sum=float(energy.sum()),energy_sum_squares=float(np.square(energy).sum()),energy_min=float(energy.min()),energy_max=float(energy.max()))


def verify_checkpoint(cp,binding,point):
    require(cp['binding']==binding and cp['point']==point and cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Lookup checkpoint identity changed')
    cursor=0;counts={k:0 for k in ('n_blocks','n_correct','n_reject','n_undetected')}
    for batch in cp['batches']:
        require(batch['start']==cursor and batch['n_blocks']>0,'Missing/overlapping deterministic Monte Carlo counters')
        require(batch['n_correct']+batch['n_reject']+batch['n_undetected']==batch['n_blocks'],'Stored outcome counts differ')
        cursor+=batch['n_blocks']
        for k in counts:counts[k]+=batch[k]
    require(counts==cp['counts'] and cursor<=MAX_BLOCKS,'Stored Monte Carlo aggregate differs')
    return cp


def lookup_row(cp,spec):
    counts=cp['counts'];n=counts['n_blocks'];require(n>0,'Empty lookup point')
    energy_sum=math.fsum(b['energy_sum'] for b in cp['batches']);energy_sq=math.fsum(b['energy_sum_squares'] for b in cp['batches'])
    average=energy_sum/n;symbols=68 if spec['kind']=='header' else spec['transmitted_bits']//spec['q']
    return dict(phy_key=spec['phy_key'],snr_db=cp['point']['snr_db'],source_id='*',kind=spec['kind'],
        applicability='iid_scrambled_payload_approximation' if spec['kind']=='body' else 'source_independent_verified',
        **counts,p_correct=counts['n_correct']/n,p_reject=counts['n_reject']/n,p_undetected=counts['n_undetected']/n,
        p_correct_ci=wilson(counts['n_correct'],n),p_reject_ci=wilson(counts['n_reject'],n),p_undetected_ci=wilson(counts['n_undetected'],n),
        interval='Wilson two-sided95% binomial',block_errors=n-counts['n_correct'],zero_observed_errors_is_not_true_BLER_zero=True,
        actual_energy_mean=average,actual_energy_sd=math.sqrt(max(0.,energy_sq/n-average*average)),
        actual_energy_min=min(b['energy_min'] for b in cp['batches']),actual_energy_max=max(b['energy_max'] for b in cp['batches']),
        observed_average_energy_per_complex_symbol=average/symbols,power_protocol='fixed_constellation_average_Es2',
        code_impl=spec.get('layout',{}).get('implementation',{'implementation':'original_header_convolutional_68'}),
        physical_configuration=spec,point_checkpoint_sha256=cp['payload_sha256'],
        real_image_payload_probability_validation_complete=False)


def synchronize(backend):
    if str(backend.device).startswith('cuda'):backend.torch.cuda.synchronize()


def benchmark(backend,header,catalog,output,bindings,stop=lambda:False):
    """Engineering-only fixed random blocks, no image or development selection."""
    representatives=[]
    for mod in ('QPSK','16QAM'):
        group=sorted((x for x in catalog.values() if x['kind']=='body' and x['modulation']==mod),key=lambda x:(x['transmitted_bits'],x['phy_key']))
        if group:representatives.append(group[len(group)//2])
    require(representatives,'No body configuration to benchmark')
    reference={};results=[]
    for batch in (32,64,128):
        for spec in representatives:
            if stop():raise StopRequested('Benchmark stop requested')
            actual_batch(backend,header,spec,7,700000,64);synchronize(backend)
        timings=[];signature=[]
        for repeat in range(3):
            began=time.perf_counter();events=[]
            for spec in representatives:
                events_for_spec=[]
                for start in range(0,256,batch):
                    if stop():raise StopRequested('Benchmark stop requested')
                    meta=actual_batch(backend,header,spec,7,800000+start,batch)
                    events_for_spec.append(np.stack([np.asarray(meta[k]) for k in ('correct','rejected','undetected')]))
                joined=np.concatenate(events_for_spec,axis=1);digest=hashlib.sha256(joined.tobytes()).hexdigest()
                key=spec['phy_key'];reference.setdefault(key,digest);require(reference[key]==digest,'Batch shape/repeated timing changed fixed CRC outcomes')
                events.append(digest)
            synchronize(backend);timings.append(time.perf_counter()-began);signature.append(events)
        results.append(dict(batch_size=batch,seconds_per_block_median=float(np.median(timings))/(256*len(representatives)),
            timed_seconds=timings,crc_outcome_signature=signature[0]))
    chosen=min(results,key=lambda r:(r['seconds_per_block_median'],r['batch_size']))['batch_size']
    if stop():raise StopRequested('Benchmark stop requested')
    value=dict(status='ACTUAL_PHY_BATCH_BENCHMARK_PASS',synthetic_payloads=True,scientific_quality_result=False,
        image_or_development_data_used=False,backend_identity=backend.identity,device=backend.device,
        chosen_batch_size=chosen,candidates=results,representatives=[x['phy_key'] for x in representatives],source_bindings=bindings)
    seal(output,value);return value


class LookupRunner:
    def __init__(self,catalog,out,binding,backend,header,batch_size,stop=lambda:False,simulate=None,synthetic=False):
        require(batch_size in (32,64,128),'Only benchmarked32/64/128 batches are allowed')
        require(simulate is None or synthetic,'Injected test events cannot enter scientific lookup')
        self.catalog=catalog;self.out=Path(out);self.binding=binding;self.backend=backend;self.header=header
        self.batch=batch_size;self.stop=stop;self.simulate=simulate or actual_batch;self.synthetic=synthetic
        self.out.mkdir(parents=True,exist_ok=True)

    def check(self):
        if self.stop() or (self.out/'STOP').exists():raise StopRequested('Safe stop requested between actual physical batches')

    def run_point(self,key,snr,refine=False):
        self.check();spec=self.catalog[key];point=dict(phy_key=key,snr_db=int(snr));folder=self.out/'points'/identity(point)
        path=folder/'checkpoint.json';snapshot=folder/('refined.json' if refine else 'coarse.json')
        if snapshot.exists():
            saved=verify_checkpoint(read(snapshot),self.binding,point)
            n=saved['counts']['n_blocks'];errors=n-saved['counts']['n_correct']
            require(n>=COARSE and (not refine or errors>=ERROR_STOP or n>=MAX_BLOCKS),'Premature completed point snapshot')
            return saved
        if path.exists():cp=verify_checkpoint(read(path),self.binding,point)
        else:cp=dict(binding=self.binding,point=point,batches=[],counts=dict(n_blocks=0,n_correct=0,n_reject=0,n_undetected=0),synthetic=self.synthetic)
        while True:
            counts=cp['counts'];n=counts['n_blocks'];errors=n-counts['n_correct']
            if (n>=COARSE and not refine) or (refine and n>=COARSE and (errors>=ERROR_STOP or n>=MAX_BLOCKS)):break
            self.check();limit=MAX_BLOCKS if refine and n>=COARSE else COARSE;count=min(self.batch,limit-n)
            meta=self.simulate(self.backend,self.header,spec,snr,n,count)
            record=batch_receipt(meta,n,count);cp['batches'].append(record)
            for field in counts:counts[field]+=record[field]
            cp['payload_sha256']=identity({k:v for k,v in cp.items() if k!='payload_sha256'});write(path,cp)
            write(self.out/'status.json',dict(status='MEASURING_REAL_PHY' if not self.synthetic else 'SYNTHETIC_FIXTURE',
                pid=os.getpid(),phase='refine' if refine else 'coarse',phy_key=key,snr_db=snr,n_blocks=counts['n_blocks'],
                block_errors=counts['n_blocks']-counts['n_correct'],updated=time.time()))
        cp['payload_sha256']=identity({k:v for k,v in cp.items() if k!='payload_sha256'});verify_checkpoint(cp,self.binding,point)
        seal(snapshot,cp);self.check();return cp

    def export(self,status):
        rows=[];bindings={}
        for path in sorted((self.out/'points').glob('*/checkpoint.json')):
            cp=read(path);point=cp['point'];verify_checkpoint(cp,self.binding,point)
            require(point['phy_key'] in self.catalog and point['snr_db'] in SNRS,'Unregistered measured point')
            rows.append(lookup_row(cp,self.catalog[point['phy_key']]));bindings[str(path)]=sha(path)
        bundle=dict(version=VERSION,status=status,synthetic=self.synthetic,conditional_group_independence=True,
            source_probability_mode='random raw payload and public coding scrambler; image payload applicability not yet verified',
            real_image_payload_probability_validation_complete=False,registration_sha256=self.binding,rows=rows,checkpoint_bindings=bindings)
        write(self.out/'bler_lookup.json',bundle)
        if rows:csv_write(self.out/'bler_lookup.csv',rows);csv_write(self.out/'bler_counts.csv',[{k:r[k] for k in ('phy_key','snr_db','kind','n_blocks','n_correct','n_reject','n_undetected','block_errors')} for r in rows])
        return bundle


def source_bindings():
    return {str(HERE/name):sha(HERE/name) for name in ('phy_lookup.py','test_phy_lookup.py','uep_phy.py','uep_common.py','ldpc_backend.py','profiles.py')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--profiles',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--qualification',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--stage',choices=('benchmark','coarse','refine'),required=True);parser.add_argument('--refine',help='Frozen optimizer selection JSON')
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--benchmark-from',help='Directory containing a shared matching batch_qualification.json')
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    require(not (out/'failure.json').exists(),'Previous actual lookup failure requires review before retry')
    import fcntl
    lock=(out/'lookup.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    from ldpc_backend import SionnaBackend
    from uep_phy import Header
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    backend=SionnaBackend(args.device,args.qualification);require(backend.qualified,'Real pinned backend qualification required')
    qualification=read(args.qualification)
    for path,digest in qualification['source_bindings'].items():require(sha(path)==digest,'Qualified PHY implementation changed: '+path)
    profiles=read(args.profiles)
    require(all(row.get('backend_id')==identity(backend.identity) for row in profiles),'Resource enumeration used a different actual encoder backend')
    header=Header(args.root);catalog=physical_catalog(profiles);bindings=source_bindings()
    bindings[str(Path(header.s.__file__).resolve())]=sha(header.s.__file__)
    all_points=[(key,snr) for key in sorted(catalog) for snr in SNRS]
    owned_points=shard_points(catalog,all_points,args.shard_index,args.shard_count)
    benchmark_path=(Path(args.benchmark_from).resolve() if args.benchmark_from else out)/'batch_qualification.json'
    if args.stage=='benchmark':
        require(not args.benchmark_from,'Benchmark writes only its own output directory')
        require(not (out/'registration.json').exists(),'Cannot benchmark a different batch after measurement registration')
        try:
            result=benchmark(backend,header,catalog,benchmark_path,bindings,stop=lambda:stop[0] or (out/'STOP').exists());print(json.dumps(result));return
        except StopRequested as error:
            write(out/'status.json',dict(status='PAUSED',pid=os.getpid(),reason=str(error)));raise SystemExit(75)
        except Exception as error:
            write(out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False));raise
    measured=read(benchmark_path)
    require(measured['status']=='ACTUAL_PHY_BATCH_BENCHMARK_PASS' and measured['backend_identity']==backend.identity
        and measured['device']==backend.device and measured['source_bindings']==bindings,'Benchmark/runtime identity differs')
    registration=dict(version=VERSION,N=1024,snrs=list(SNRS),profiles_sha256=sha(args.profiles),backend_identity=backend.identity,
        qualification_sha256=sha(args.qualification),batch_qualification_sha256=sha(benchmark_path),device=backend.device,
        batch_size=measured['chosen_batch_size'],source_bindings=bindings,catalog_sha256=identity(catalog),
        coarse_blocks=COARSE,refine_max_blocks=MAX_BLOCKS,refine_block_errors=ERROR_STOP,
        shard_index=args.shard_index,shard_count=args.shard_count,
        shard_rule='sorted body phy_key ordinal modulo shard_count; all header SNRs on shard0; counters unchanged',
        full_catalog_point_count=len(all_points),owned_points=[list(point) for point in owned_points],
        payload='independent uniform raw bits per public configuration/SNR/block index',
        random_generator='PCG64 from uep_common.rng, protocol-domain SHA256 seed',
        counters='SHA256(version,phy_key,SNR,block_index), low63bits; independent of payload',
        synthetic=False,development_read=False,quality_read=False,header_profile_distribution='uniform integer0..4095; actual CRC and original68-symbol chain')
    seal(out/'registration.json',registration);binding=identity(registration)
    runner=LookupRunner(catalog,out,binding,backend,header,measured['chosen_batch_size'],stop=lambda:stop[0])
    if args.stage=='refine':
        require(args.refine,'Refinement needs a frozen optimizer candidate list');selection=read(args.refine)
        require(read(out/'coarse_completion.json')['status']=='COARSE_COMPLETE','Complete coarse lookup required before refinement')
        require(selection.get('synthetic') is False and selection.get('development_read') is False,'Real calibration selection required')
        points=sorted({(r['phy_key'],int(r['snr_db'])) for r in selection['refinement_requirements']})
        require(points and all(k in catalog and s in SNRS for k,s in points),'Refinement requested an unregistered/N2048 physical point')
        points=shard_points(catalog,points,args.shard_index,args.shard_count)
        seal(out/'refinement_requests'/(sha(args.refine)+'.json'),dict(selection_path=str(Path(args.refine).resolve()),selection_sha256=sha(args.refine),points=[list(point) for point in points]))
    else:points=owned_points
    try:
        for key,snr in points:runner.run_point(key,snr,refine=args.stage=='refine')
        for path,digest in bindings.items():require(sha(path)==digest,'Scientific source changed during lookup')
        result=runner.export('REFINEMENT_COMPLETE' if args.stage=='refine' else 'COARSE_COMPLETE')
        completion=dict(status=result['status'],phase=args.stage,points=len(points),registration_sha256=binding,
            lookup_sha256=sha(out/'bler_lookup.json'),synthetic=False,real_image_payload_probability_validation_complete=False,
            shard_index=args.shard_index,shard_count=args.shard_count,full_catalog_point_count=len(all_points),
            phase_points=[list(point) for point in points],owned_points=[list(point) for point in owned_points])
        name='coarse_completion.json' if args.stage=='coarse' else 'refine_'+sha(args.refine)+'_completion.json'
        seal(out/name,completion);write(out/'status.json',completion);print(json.dumps(completion))
    except StopRequested as error:
        runner.export('PAUSED_PARTIAL');write(out/'status.json',dict(status='PAUSED',pid=os.getpid(),reason=str(error)));raise SystemExit(75)
    except Exception as error:
        write(out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False));raise


if __name__=='__main__':main()
