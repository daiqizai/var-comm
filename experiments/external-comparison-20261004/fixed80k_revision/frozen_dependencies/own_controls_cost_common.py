"""CPU contracts for a fixed, same-population five-receiver cost benchmark."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import subprocess
import numpy as np
import external_eval_common as c

HERE=Path(__file__).resolve().parent
VERSION='FIVE-RECEIVER-CAL4-COST-20261004-R1'
METHODS={1024:('P1024','D_U_QPSK','M1_entropy_N1024_frozen',*c.METHODS),
         2048:('P2048','D_U_whole_N2048_v1','M1_entropy_N2048_v1',*c.METHODS)}
PROTOCOL=dict(version=VERSION,source_role='original_calibration',source_indices=[0,1,2,3],
    budgets=[1024,2048],snrs=[1,7,13],noise_seed=4101,timed_repeats=1,batch_size=1,
    warmup='One complete receive per method and N at cal source0, SNR13, seed4101 on every process invocation; never included in timing rows',
    receiver_boundary='received_full_waveform_to_RGB',cuda_synchronized=True,includes_header_decode=True,
    includes_TX=False,includes_model_loading=False,includes_metric_scoring=False,
    sampler_step_limit=None,posterior_sampling_seed=23,same_population_all5=True,
    selection_uses_development=False,development_images_read=False,holdout_access=False,training_updates=0,
    policy_selection_updates=0,synthetic=False)
FILES=('own_controls_cost.py','own_controls_cost_common.py','own_controls_cost_native.py','own_controls_cost_tests.py')
STATUS={'own':'OWN_RECEIVER_BENCHMARK_COMPLETE','external':'EXTERNAL_RECEIVER_BENCHMARK_COMPLETE'}

def require(condition,message):
    if not condition:raise RuntimeError(message)

def folder(root):return Path(root)/'outputs/EXTERNAL-COMPARISON-20261004/receiver_cost'

def code_bindings():return {str(HERE/name):c.sha(HERE/name) for name in FILES}

def stage_methods(stage,n):return METHODS[n][:3] if stage=='own' else METHODS[n][3:]

def key(row):return (int(row['source_index']),int(row['N']),int(row['snr_db']),int(row['noise_seed']),row['method'])

def specs(stage):
    return [dict(source_index=i,N=n,snr_db=s,noise_seed=4101,method=m)
            for i in range(4) for n in (1024,2048) for s in (1,7,13) for m in stage_methods(stage,n)]

def gpu_identity(torch):
    import os
    require(os.environ.get('CUDA_VISIBLE_DEVICES','0')=='0','Benchmark requires the registered physical GPU0')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
    values=[v.strip() for v in text.splitlines()[0].split(',')]
    require(len(values)==4 and values[0]=='0' and values[1].startswith('GPU-'),'GPU UUID identity unavailable')
    return dict(index=0,uuid=values[1],name=values[2],driver=values[3],torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)

def source_record(index,record,rgb):
    target=c.pixels(rgb)
    require(hashlib.sha256(np.rint(target*255).astype(np.uint8).tobytes()).hexdigest()==record['preprocessing_id'],
            'Calibration original uint8 preprocessing differs')
    return dict(source_index=index,image_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
                class_index=int(record['class_index']),rgb_sha256=c.rgb_sha(target))

def validate_population(reg,archive):
    require(reg.get('protocol')==PROTOCOL and reg.get('source_indices')==[0,1,2,3], 'Benchmark population/protocol changed')
    require(c.sha(archive)==reg['archive_sha256'],'Calibration population cache changed')
    with np.load(archive,allow_pickle=False) as data:images=data['source_rgb'].copy()
    require(images.shape==(4,3,256,256) and len(reg['sources'])==4,'Calibration benchmark requires exactly four originals')
    result=[]
    for i,(r,image) in enumerate(zip(reg['sources'],images)):
        require(source_record(i,r,image)==r,'Calibration population identity changed')
        result.append(dict(r,rgb=image))
    return result

def validate_row(row,source,runtime):
    n=int(row['N']);method=row['method']
    require(n in METHODS and method in METHODS[n] and row['source_index']==source['source_index']
        and row['source_id']==source['image_id'] and row['preprocessing_id']==source['preprocessing_id']
        and row['reference_sha256']==source['rgb_sha256'] and row['true_class_index']==source['class_index'],
        'Receiver timing source or method identity differs')
    require(row['snr_db'] in (1,7,13) and row['noise_seed']==4101 and row['runtime']==runtime
        and row['source_population']=='original_calibration_indices_0_1_2_3' and row['batch_size']==1
        and row['timed_repeats']==1 and row['warmup_per_method_N_per_invocation']==1,
        'Receiver timing protocol/runtime differs')
    require(math.isfinite(row['RX_seconds']) and row['RX_seconds']>0
        and abs(row['actual_energy']-2*n)<.02 and row['E']==2*n,'Invalid actual timing or full-frame energy')
    require(row['header_accepted'] in (True,False,'not_applicable') and row['synthetic'] is False,
        'Actual header outcome required')
    if method==c.METHODS[1]:
        require((row['header_accepted'] is True and row['NFE']==row['t_start'] and row['NFE']>=2
                 and row['complete_posterior_schedule'] is True)
                or (row['header_accepted'] is False and row['NFE']==0),'Truncated HiFi cannot be a receiver benchmark')

def validate_frame(value,binding,spec,source,runtime):
    require(value.get('payload_sha256')==c.identity({k:v for k,v in value.items() if k!='payload_sha256'})
        and value['binding']==binding and value['spec']==spec and value['runtime']==runtime,
        'Receiver frame registration/payload changed')
    row=value['row'];require(key(row)==key(spec),'Receiver timing frame grid changed')
    validate_row(row,source,runtime)
    require(c.sha(value['archive'])==value['archive_sha256'],'Receiver float/waveform cache changed')
    with np.load(value['archive'],allow_pickle=False) as archive:
        image=c.pixels(archive['image']);wave=archive['transmitted'];observed=archive['observed']
    require(wave.shape==observed.shape==(spec['N'],2) and np.isfinite(wave).all() and np.isfinite(observed).all(),
        'Invalid actual physical waveform')
    require(c.rgb_sha(image)==row['image_sha256'] and c.array_sha(wave)==row['transmitted_sha256']
        and c.array_sha(observed)==row['observed_sha256']
        and abs(float(np.square(wave.astype(np.float64)).sum())-row['actual_energy'])<.02,
        'Measured waveform or RGB identity changed')
    if spec['method']==c.METHODS[1]:c.validate_full_sampler(value['hifi_receipt'],row['header_accepted'])
    return value

def summarize(rows):
    require(len(rows)==120 and {key(r) for r in rows}=={key(s) for g in STATUS for s in specs(g)},
        'Cost report needs all 120 unique measurements')
    groups=defaultdict(list)
    for row in rows:groups[row['N'],row['snr_db'],row['method']].append(row)
    output=[]
    for (n,s,m),items in sorted(groups.items()):
        require(len(items)==4 and {r['source_index'] for r in items}==set(range(4)),'Unpaired calibration cost group')
        require(all(r['runtime']==items[0]['runtime'] for r in items),'Runtime changed within timing group')
        seconds=np.array([r['RX_seconds'] for r in items],np.float64)
        output.append(dict(N=n,snr_db=s,method=m,frames=4,n_frames=4,
            RX_mean_seconds=float(seconds.mean()),RX_median_seconds=float(np.median(seconds)),
            RX_p95_seconds=float(np.percentile(seconds,95)),
            header_accepted=sum(r['header_accepted'] is True for r in items),
            header_failed=sum(r['header_accepted'] is False for r in items),
            header_not_applicable=sum(r['header_accepted']=='not_applicable' for r in items),
            NFE_total=sum(r.get('NFE',0) for r in items),batch_size=1,timed_repeats=1,
            warmup_per_method_N_per_invocation=1,source_population='original_calibration_indices_0_1_2_3',
            noise_seed=4101,receiver_boundary=PROTOCOL['receiver_boundary'],same_population_all5=True,
            GPU_uuid=items[0]['runtime']['uuid'],GPU_name=items[0]['runtime']['name'],
            torch=items[0]['runtime']['torch'],cuda=items[0]['runtime']['cuda']))
    return output

def load_stage(out,stage):
    completion_path=Path(out)/(stage+'_completion.json');done=c.read(completion_path);rp=Path(out)/(stage+'_registration.json');reg=c.read(rp)
    require(done.get('status')==STATUS[stage] and done.get('rows')==len(specs(stage))
        and done['registration_sha256']==c.sha(rp) and done['binding']==c.identity(reg)
        and reg['protocol']==PROTOCOL and done['synthetic'] is False,'Incomplete cost stage')
    c.verify(done['bindings']);c.verify(done['outputs']);c.verify(reg['bindings'])
    population_path=Path(out)/'population.json';population=c.read(population_path)
    require(reg['bindings'].get(str(population_path))==c.sha(population_path),'Unbound calibration population')
    sources=validate_population(population,population['archive'])
    expected=specs(stage);require(len(done['frames'])==len(expected),'Cost stage frame count differs')
    rows=[]
    def native_proof(frame,spec,source):
        if stage!='external':return
        path=frame['native_receipt'];require(done['bindings'].get(path)==frame['native_receipt_sha256']==c.sha(path),
            'External receiver native measurement is not bound')
        raw=c.validate_frame_receipt(c.read(path),done['binding'],{kk:vv for kk,vv in spec.items() if kk!='method'},source)
        require(done['bindings'].get(raw['archive'])==raw['archive_sha256'],'Native waveform proof is not bound')
        rr=next(r for r in raw['rows'] if r['method']==spec['method']);row=frame['row']
        for field in ('RX_seconds','header_accepted','NFE','complete_posterior_schedule','image_sha256','observed_sha256','transmitted_sha256'):
            require(row[field]==rr[field],'Cost row differs from the actual native timed receiver: '+field)
    for spec,path in zip(expected,done['frames']):
        require(done['outputs'].get(path)==c.sha(path),'Unbound receiver frame')
        frame=validate_frame(c.read(path),done['binding'],spec,sources[spec['source_index']],reg['runtime'])
        require(done['outputs'].get(frame['archive'])==frame['archive_sha256'],'Unbound receiver float cache')
        link_path=Path(path).parent/'warmup.json';link=c.read(link_path)
        require(done['outputs'].get(str(link_path))==c.sha(link_path)
                and done['outputs'].get(link['path'])==link['sha256']==c.sha(link['path']), 'Unbound warmup proof')
        warm_spec=dict(spec,source_index=0,snr_db=13)
        warm=validate_frame(c.read(link['path']),done['binding'],warm_spec,sources[0],reg['runtime'])
        require(done['outputs'].get(warm['archive'])==warm['archive_sha256'],'Unbound warmup waveform')
        native_proof(frame,spec,sources[spec['source_index']]);native_proof(warm,warm_spec,sources[0])
        rows.append(frame['row'])
    return rows,reg,{str(completion_path):c.sha(completion_path),str(rp):c.sha(rp),**done['bindings'],**done['outputs']}

def verify_report(path):
    path=Path(path);done=c.read(path)
    require(done.get('status')=='OWN_CONTROLS_RECEIVER_COST_COMPLETE' and done.get('rows')==120
        and done.get('groups')==30 and done.get('same_population_all5') is True
        and done.get('protocol')==PROTOCOL and done.get('synthetic') is False,'Incomplete all-five receiver cost report')
    c.verify(done['bindings']);c.verify(done['outputs'])
    rows=[];runtimes=[];bindings={str(path):c.sha(path)}
    for stage in STATUS:
        rr,reg,bb=load_stage(path.parent,stage);rows+=rr;runtimes.append(reg['runtime']);bindings.update(bb)
    require(all(r['uuid']==runtimes[0]['uuid'] and r['name']==runtimes[0]['name'] and r['driver']==runtimes[0]['driver']
                for r in runtimes),'Five receivers were timed on different physical GPUs or drivers')
    summary=summarize(rows);require(summary==done['summary'],'Cost summary does not reproduce its actual frame receipts')
    return summary,{**bindings,**done['bindings'],**done['outputs']}
