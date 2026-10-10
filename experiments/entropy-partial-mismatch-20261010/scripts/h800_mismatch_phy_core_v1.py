"""Original raw433 TX/KEEP RX composition for the frozen one-bin diagnostic.

The receiver is unchanged and receives only noisy samples, true SNR and the
original counter. Fresh paid calls use a separate finite ledger. No images,
neural models, metrics, new policies or historical event admission occur here.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import mismatch_plan as plan

require=plan.require
digest=plan.digest
PREFIX=(0,1,5,14,30,55,91,155,255,424,680)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return dict(path=str(path),sha256=sha(path))
def readpin(value):
    path=Path(value['path']);require(not path.is_symlink() and sha(path)==value['sha256'],'Changed bound file: '+str(path))
    return json.loads(path.read_bytes())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(plan.canonical(value)+'\n')
    return pin(path)
def load(path,name):
    path=Path(path).absolute()
    require(name not in sys.modules,'Fresh scientific module namespace required: '+name)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def source_scales(record):
    import numpy as np
    cp=readpin(record['actual_checkpoint']);archive=record['actual_archive']
    require(cp['status']=='COMMON500_FROZEN_SOURCE_ASSET_READY_V1' and cp['source_index']==record['source_index'] and
        cp['source_id']==record['source_id'] and cp['tokens_sha256']==record['tokens_sha256'] and
        cp['archive']==record['archive'] and cp['outputs'][record['archive']]==archive['sha256'] and
        cp['encoder_tokens_verified'] is True and cp['asset_readback_exact'] is True,'Frozen source checkpoint differs')
    require(sha(archive['path'])==archive['sha256'],'Frozen source archive differs')
    with np.load(archive['path'],allow_pickle=False) as z:
        require(set(z.files)=={'tokens','pixels'},'Original common500 archive schema differs');tokens=z['tokens']
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),'Frozen680 raw tokens required')
    require(hashlib.sha256(b'int64:680\0'+tokens.astype('<i8',copy=False).tobytes()).hexdigest()==record['tokens_sha256'],
        'Original token hash differs')
    return [tokens[PREFIX[i]:PREFIX[i+1]] for i in range(10)]


def create_runtime(binding,ledger,guard):
    """Compose original pinned modules; constructor/layout checks do not decode."""
    import os
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only receiver required')
    for row in binding['files']:
        require(sha(row['path'])==row['sha256'],'Original raw dependency changed: '+row['path'])
    paths=binding['roles'];root=Path(binding['project_root'])
    common=load(paths['original_common'],'uep_common');load(paths['original_planner'],'profiles')
    legacy=load(paths['original_phy'],'uep_phy')
    oldmodule=load(paths['original_backend'],'_mismatch_original_ldpc')
    p=load(paths['plan_module'],'main_raw64_plan_only')
    packet=load(paths['adapter_module'],'main_raw64_packet_adapter')
    load(paths['action_module'],'main_action_space')
    original=load(paths['original_receiver_module'],'_mismatch_original_keep_receiver')
    rx=load(paths['receiver_module'],'_mismatch_mixed_keep_receiver')
    noise=load(paths['noise_module'],'_mismatch_original_common500_noise')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name)=='2','Frozen original two-thread PHY environment required')
    old=oldmodule.SionnaBackend('cpu',paths['legacy_qualification']);packet.runtime_flags(old.torch)
    require(not old.torch.cuda.is_initialized(),'Raw PHY initialized CUDA')
    backend=packet.Backend(old);header=legacy.Header(root)
    cat=rx.MixedPublicCatalogue(json.loads(Path(paths['catalogue']).read_bytes()),
        json.loads(Path(paths['profiles']).read_bytes()),json.loads(Path(paths['aliases']).read_bytes()),
        legacy_backend_identity=old.identity,original_receiver=original)
    require(cat.digest==binding['catalogue_sha256'] and cat.aliases_digest==binding['aliases_sha256'],
        'Full433 effective receive domain changed')
    def charge(event,request,callback):
        guard();require(event['phase']=='raw_holdout' and event['kind'] in ('header','body'),'Wrong paid receive phase')
        return ledger.call(event,request,callback)
    receiver=rx.Receiver(cat,backend=backend,legacy_phy=legacy,header=header,charge=charge)
    return SimpleNamespace(common=common,legacy=legacy,packet=packet,original=original,rx=rx,noise=noise,
        backend=backend,header=header,cat=cat,receiver=receiver,torch=old.torch)


def observed_identity(row,tx,payload_sha,noise_sha,received_sha,runtime_identity):
    c=row['channel'];source=row['source']
    require(c==plan.channel_identity(source,row['actual_snr_db'],c['noise_seed']),'Original channel identity changed')
    return dict(source_index=source['source_index'],source_id=source['source_id'],source_archive_sha256=source['archive_sha256'],
        tokens_sha256=source['tokens_sha256'],preprocessing_id=source['preprocessing_id'],channel=c,
        TX_profile_id=row['profile_id'],TX_wire_key=row['wire_key'],payload_sha256=payload_sha,
        waveform_sha256=tx['waveform_sha256'],noise_sha256=noise_sha,received_sha256=received_sha,
        runtime_identity=runtime_identity,receiver_phase='raw_holdout',body_session='actual-body',body_group=0)


def normalized(actual):
    value=dict(header_ok=actual['header_ok'],body_attempted=actual['body_decode_complete'],
        gray=actual['gray'],source_decode_complete=actual['source_decode_complete'],rule=actual['rule'],
        execution_status='complete',rx_profile_id=actual['rx_profile_id'],source_status=actual['source_status'],
        body_crc_accept=actual['body_crc_accept'],packet_call_count=actual['logical_packet_calls'],
        rx_profile_legal=actual['header_ok'])
    plan.validate_receiver_summary(value);return value


def verify_paid(actual,ledger):
    """Bind every real request/result; an incomplete call cannot become gray."""
    require(actual['packet_event_ids']==[x['event_id'] for x in actual['packet_events']], 'Paid event IDs differ')
    for entry in actual['packet_events']:
        saved=ledger.db.execute('SELECT event,request,result,status FROM events WHERE event_id=?',(entry['event_id'],)).fetchone()
        require(saved is not None and saved[3]=='COMPLETE','Missing closed paid receive')
        event,request,result=(json.loads(x) for x in saved[:3])
        require(event['event_id']==entry['event_id'] and event['kind']==entry['kind'] and
            event['phase']=='raw_holdout' and event['phy_key']==entry['phy_key'] and
            digest(request)==entry['request_sha256'] and digest(result)==entry['result_sha256'] and
            result==actual[entry['kind']],'Paid event/result differs')
        require(request['received_sha256']==actual['received_sha256'] and
            request['public_frame_counter']==actual['public_frame_counter'] and
            request['codebook_sha256']==actual['codebook_sha256'] and request['aliases_sha256']==actual['aliases_sha256'],
            'Paid request is not the actual full-domain frame')


def execute(runtime,rows,records,ledger,boundary,out,runtime_identity):
    import numpy as np
    require(len(rows)==5400 and len({plan.physical_key(r) for r in rows})==4500,'Frozen5400/4500 grid required')
    out=Path(out);require(not out.exists(),'Fresh frame output required');out.mkdir()
    source_cache={};physical={};logical=[];scheduled={};calls=0
    # Group by original source solely to limit open cached assets, never reorder identities.
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['source']['source_index'],i))
    for index in order:
        boundary();row=rows[index];source=row['source'];i=source['source_index'];c=row['channel']
        require({k:records[i][k] for k in plan.SOURCE_KEYS}==source,'Wrong fixed source record')
        if i not in source_cache:source_cache={i:source_scales(records[i])}
        counter=runtime.noise.frame_counter(i,row['actual_snr_db'],c['noise_seed'])
        require(counter==c['public_counter'],'Original500-stride counter changed')
        profile=runtime.cat.entry(row['profile_id'])
        require(profile['wire_key']==row['wire_key'] and profile['m']==row['m'] and profile['K']==row['K'] and
            row['candidate_id'] in profile['alias_candidate_ids'],'Frozen TX policy/alias mismatch')
        payload=runtime.original.serialize_raw(source_cache[i],profile)
        wave,tx=runtime.rx.transmit_frame(runtime.cat,runtime.backend,runtime.legacy,runtime.header,row['profile_id'],payload,counter)
        require(wave.dtype==np.float64 and wave.shape==(1024,2) and np.isfinite(wave).all() and
            tx['waveform_sha256']==runtime.original.array_sha(wave) and tx['E']==float(np.square(wave).sum()),
            'Original unnormalized full1024 waveform required')
        noise=runtime.noise.standard_noise(source['source_id'],row['actual_snr_db'],c['noise_seed'])
        require(noise.dtype==np.float64 and noise.shape==(1024,2),'Original common float64 noise required')
        received=wave+noise*10**(-float(row['actual_snr_db'])/20)
        identity=observed_identity(row,tx,runtime.original.array_sha(payload),runtime.original.array_sha(noise),
            runtime.original.array_sha(received),runtime_identity);key=digest(identity);scheduled_key=plan.physical_key(row)
        require(scheduled_key not in scheduled or scheduled[scheduled_key]==key,'Same scheduled frame changed actual observations')
        scheduled[scheduled_key]=key
        if key not in physical:
            require(len(physical)<4500,'Physical frame cap exhausted')
            actual=runtime.receiver.receive(received,float(row['actual_snr_db']),counter,
                phase='raw_holdout',event_prefix='H800_RAW_MISMATCH_V1/'+key)
            require(actual['received_sha256']==identity['received_sha256'] and actual['public_frame_counter']==counter and
                actual['codebook_sha256']==runtime.cat.digest and actual['aliases_sha256']==runtime.cat.aliases_digest,
                'Receiver did not consume the registered observations')
            # Reparse actual hard information; TX/source tokens never repair it.
            require(all(actual[k]==v for k,v in runtime.original.present_actual(actual['header'],actual['body'],runtime.cat).items()),
                'Independent actual hard-token presentation differs')
            summary=normalized(actual);verify_paid(actual,ledger)
            frame=dict(schema='H800_RAW_MISMATCH_ACTUAL_RX_V1',physical_key=key,state='complete',mode='new',
                observation_identity=identity,actual_RX=actual,receiver=summary,transmission=tx,
                actual_new_packet_calls=summary['packet_call_count'],historical_reuse_admitted=False)
            physical[key]=save(out/'physical'/f'{key}.json',frame);calls+=summary['packet_call_count']
        physical_row=readpin(physical[key])
        require(physical_row['observation_identity']==identity,'Cached actual observation mismatch')
        logical.append(dict(logical_index=index,logical_frame_id=digest(row),physical_frame_id=key,
            source_index=i,source_id=source['source_id'],actual_snr_db=row['actual_snr_db'],config_snr_db=row['config_snr_db'],
            family=row['family'],noise_seed=c['noise_seed'],public_counter=counter,tx_profile_id=row['profile_id'],
            tx_candidate_id=row['candidate_id'],tx_m=row['m'],tx_K=row['K'],actual_RX=physical[key]))
    snapshot=ledger.snapshot();require(len(physical)==4500 and len(logical)==5400 and
        snapshot==dict(total=calls,unresolved=0,cap=9000),'Full actual mismatch budget did not close')
    accounting=plan.call_accounting([dict(readpin(p),physical_key=k) for k,p in physical.items()])
    require(accounting['fresh_physical_frames']==4500 and accounting['declared_historical_reuse_frames']==0 and
        accounting['actual_new_packet_calls']==calls,'Actual accounting differs')
    return dict(logical_frames=save(out/'logical_frames.json',sorted(logical,key=lambda r:r['logical_index'])),
        physical_frames=[dict(physical_key=k,actual_RX=v) for k,v in sorted(physical.items())],logical_count=5400,
        physical_count=4500,packet_ledger=snapshot,accounting=accounting,historical_reuse_admitted=0,
        neural_model_calls=0,image_reconstructions=0,quality_scores=0,new_policy_selection=False)
