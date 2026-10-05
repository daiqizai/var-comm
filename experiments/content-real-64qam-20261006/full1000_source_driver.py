"""Separately registered frozen H source encoding for calibration1000.

Reuses exactly200 previously validated source bitstreams by source ID. Encodes
the remaining800 with the original independent TX/RX arithmetic contexts. No
PHY, image rendering, clean-state scoring, selection or development is run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np

STOP=False
DEADLINE=1791564605.9549868
DONE='H_FULL1000_SOURCE_CODEC_COMPLETE'
ASSETS_DONE='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
MODES=(6,7,8,9)
SIZES=(1,2,3,4,5,6,8,10,13,16)


def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def verify(bindings):
    for path,digest in bindings.items():require(Path(path).is_absolute() and sha(path)==digest,'Changed bound file: '+path)

def merge(*maps):
    result={}
    for values in maps:
        for path,digest in values.items():
            require(path not in result or result[path]==digest,'Conflicting SHA: '+path);result[path]=digest
    return result

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def status(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)

def claim_source_output(out,regsha):
    out=Path(out)
    if out.exists():
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Existing source attempt requires independent recovery')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_FULL1000_SOURCE_CODEC_ATTEMPT',registration_sha256=regsha,
        pid=os.getpid(),started_at=time.time(),new_packet_decodes=0))

def stop(*_):
    global STOP
    STOP=True

def verify_frozen_budget(before,reg):
    require(before.get('created') is True and before.get('unresolved')==0 and before==reg.get('budget_before'),
            'Packet budget changed since source-only execution registration')


def validate_lengths(arrays,lengths):
    """Validate the same canonical m6..m9 bitstream contract for both paths."""
    require(set(arrays)=={f'm{m}_bits' for m in MODES},'Only four arithmetic bit arrays are allowed')
    require(len(lengths)==4 and [r['m'] for r in lengths]==list(MODES),'Whole-prefix coverage differs')
    result={};rows=[]
    for r in lengths:
        m=r['m'];a=np.asarray(arrays[f'm{m}_bits'])
        require(a.dtype==np.uint8 and a.ndim==1 and len(a)>=2 and np.isin(a,(0,1)).all(),
                'Invalid encoded arithmetic bits')
        require(type(r['arithmetic_bits']) is int and r['arithmetic_bits']==len(a)
            and r['raw_bits']==12*sum(s*s for s in SIZES[:m]) and r['zero_extension_reads']==30
            and type(r['flush_bits']) is int and 2<=r['flush_bits']<=len(a),'Arithmetic length/canonical evidence differs')
        result[f'm{m}_bits']=a.copy()
        rows.append({k:r[k] for k in ('m','raw_bits','arithmetic_bits','flush_bits','zero_extension_reads')})
    return result,rows


def encode_new(flat,source_api,codec_api,factory,primitives):
    """Only bits+public m go into the independent RX; truth checks follow decode."""
    scales=source_api.split_tokens(flat)
    encoded=codec_api.encode_prefixes(scales,factory,primitives)
    require(set(encoded)==set(MODES),'Encoder did not return the four registered prefixes')
    arrays={};rows=[]
    for m in MODES:
        rec=encoded[m];bits=np.asarray(rec['bits']).copy()
        decoded=codec_api.decode_prefix(bits.copy(),m,factory,primitives)
        require(decoded['canonical'] is True and decoded['status']=='SOURCE_DECODED'
                and decoded['payload_bits']==len(bits) and decoded['zero_extension_reads']==30,
                'Independent canonical source decode incomplete')
        require(len(decoded['scales'])==m and all(np.array_equal(scales[i],decoded['scales'][i]) for i in range(m)),
                'Independent source roundtrip differs from source tokens')
        arrays[f'm{m}_bits']=bits
        rows.append(dict(m=m,raw_bits=rec['raw_bits'],arithmetic_bits=rec['arithmetic_bits'],
                         flush_bits=rec['flush_bits'],zero_extension_reads=decoded['zero_extension_reads']))
    return validate_lengths(arrays,rows)


def validate_manifest(manifest,completion,calibration):
    require(manifest['status']==completion['status']==ASSETS_DONE and manifest['source_count']==completion['source_count']==1000,
            'Complete CPU asset preparation required')
    ids=calibration['source_ids'];require(calibration['stage']==calibration['calibration_or_development']=='m1_calibration'
            and len(ids)==len(set(ids))==1000 and manifest['source_ids']==ids,'Original full calibration ordering differs')
    require(manifest['visual_identity']==calibration['identity'] and manifest['original_data_bindings']==calibration['data_bindings'],
            'Original visual/population identity differs')
    for value in (manifest,completion):
        require(value['reused_codec_sources']==200 and value['pending_source_encoding_count']==800
            and value['source1000_codec_complete'] is False and value['new_packet_decodes']==0
            and value['development_used'] is False and value['holdout_used'] is False,'Asset scope/encoding state differs')
    records=manifest['records'];require(len(records)==1000,'Missing full calibration source records')
    reuse_indices=[]
    for i,r in enumerate(records):
        require(r['source_index']==i and r['source_id']==ids[i] and r['preprocessing_id']==calibration['preprocessing_ids'][i]
            and completion['outputs'].get(r['checkpoint'])==r['checkpoint_sha256'],'Asset record mapping not sealed')
        legacy=r['legacy_source200_index']
        if legacy is None:
            require(r['codec_status']=='NEEDS_FROZEN_SOURCE_ENCODING' and r['codec_archive'] is None,'Unencoded source misclassified')
        else:
            require(type(legacy) is int and r['codec_status']=='SOURCE_CODEC_REUSED_BY_EXACT_ID' and r['codec_archive'],
                    'Reused source identity incomplete');reuse_indices.append(legacy)
    require(sorted(reuse_indices)==list(range(200)),'Source200 reuse is duplicated/missing')
    return records


def load_source(ctx,index):
    r=ctx['records'][index];cp_path=Path(r['checkpoint']);done=ctx['assets_done'];api=ctx['assets_api']
    require(done['outputs'].get(str(cp_path))==sha(cp_path)==r['checkpoint_sha256'],'Source asset checkpoint changed')
    cp=read(cp_path)
    for key in ('source_index','source_id','preprocessing_id','tokens_sha256','archive','codec_status','codec_archive','legacy_source200_index'):
        require(cp[key]==r[key],'Source asset record differs: '+key)
    require(cp['status']=='H_FULL1000_SOURCE_ASSET_READY' and cp['registration_sha256']==sha(ctx['cfg']['assets_registration'])
        and cp['original_calibration_index']==index and cp['new_source_encoding'] is False,'CPU source asset provenance differs')
    verify(cp['outputs'])
    require(all(done['outputs'].get(p)==s for p,s in cp['outputs'].items()),'Asset source outputs unsealed')
    with np.load(cp['archive'],allow_pickle=False) as z:
        require(set(z.files)=={'tokens','pixels'},'Unexpected source asset array')
        tokens,pixels=api.validate_arrays(z['tokens'],z['pixels'],cp['preprocessing_id'])
    require(api.token_sha(tokens)==cp['tokens_sha256'],'Source token identity changed')
    expected={cp['archive']};cached=None
    if cp['legacy_source200_index'] is not None:
        expected.add(cp['codec_archive']);proof=cp['reuse_proof']
        require(proof['status']=='SOURCE_CODEC_REUSED_BY_EXACT_ID' and proof['source_id']==cp['source_id']
            and proof['full_source_index']==index and proof['legacy_source200_index']==cp['legacy_source200_index']
            and proof['tokens_sha256']==cp['tokens_sha256'] and proof['preprocessing_id']==cp['preprocessing_id']
            and proof['independent_roundtrip_reused'] is True and proof['new_canonical_decode'] is False
            and proof['clean_image_keys_read'] is False,'Original exact-ID reuse proof differs')
        verify(proof['original_input_bindings'])
        require(all(done['input_bindings'].get(p)==s for p,s in proof['original_input_bindings'].items()),'Old source evidence unbound')
        with np.load(cp['codec_archive'],allow_pickle=False) as z:
            cached=validate_lengths({k:z[k] for k in z.files},proof['original_lengths'])
    require(set(cp['outputs'])==expected,'Unexpected per-source asset output')
    return cp,tokens,cached


def load_registered(config_path):
    path=Path(config_path).resolve();cfg=read(path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_FULL1000_SOURCE_CODEC_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['branch']=='H' and reg['allowed_stage_ids']==['source'] and reg['source_stage_scope']=='FULL1000_SOURCE_CODEC_ONLY',
        'Independent source-only GPU execution registration required')
    bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    required=('protocol','calibration_registration','assets_module','assets_config','assets_registration','assets_completion','assets_manifest',
        'assets_owner_config','assets_owner_launch','source_owner_config','owner_module','wait_module','source_driver_module',
        'static_closure_module','numerical_reference','budget_registration')
    for name in required:require(bound.get(cfg[name])==sha(cfg[name]),'Required source/input unbound: '+name)
    require(bound.get(str(path))==sha(path) and bound.get(str(Path(__file__).resolve()))==sha(__file__),'Source driver/config unbound')
    hout=Path(cfg['H_out']);out=Path(cfg['out'])
    require(out.is_absolute() and hout in out.parents and cfg['stop_file']==str(hout/'STOP')
        and cfg['overall_deadline_unix']==DEADLINE and 0<cfg['max_seconds']<=86400,'Source output/deadline scope differs')
    protocol=read(cfg['protocol']);require(protocol['schema']=='H_CODEC_PROTOCOL_V1' and protocol['status']=='FROZEN_BEFORE_DATA'
        and protocol['source']['whole_targets']==list(MODES) and protocol['source']['sizes']==list(SIZES)
        and protocol['source']['ordering']=='public raster' and protocol['source']['new_training']==0,'Frozen source protocol differs')
    for directory,names in ((Path(cfg['runtime_dir']),('h64_catalog.py','h64_source.py')),
                           (Path(cfg['uep_runtime']),('quality_driver.py','source_quality.py')),
                           (Path(cfg['root'])/'src/var_comm',('whole_entropy.py','entropy.py'))):
        for name in names:require(bound.get(str(directory/name))==sha(directory/name),'Frozen source dependency unbound: '+name)
    native_files=list(Path(cfg['native_runtime']).glob('m1_*.py'))
    require({'m1_common.py','m1_native.py','m1_phy.py','m1_performance.py'}<={p.name for p in native_files},'Native runtime incomplete')
    for p in native_files:require(bound.get(str(p))==sha(p),'Native dependency unbound: '+str(p))
    closure=module(cfg['static_closure_module'],'full1000_static_visual_closure')
    closure_bindings=closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
    require(closure.compare_bindings(closure_bindings,bound)['status']=='EXACT_SOURCE_CLOSURE_MATCH',
            'Static full visual source closure differs; register before loading models')
    ac=read(cfg['assets_config']);ar=read(cfg['assets_registration']);done=read(cfg['assets_completion']);manifest=read(cfg['assets_manifest'])
    require(ar['status']=='H_EXECUTION_REVISION_REGISTERED' and ar['source_stage_scope']=='FULL1000_CPU_ASSETS_ONLY'
        and ac['registration']==cfg['assets_registration'] and ac['root']==cfg['root']
        and Path(ac['out'])/'completion.json'==Path(cfg['assets_completion']) and Path(ac['out'])/'manifest.json'==Path(cfg['assets_manifest'])
        and done['registration_sha256']==sha(cfg['assets_registration']) and done['outputs'].get(cfg['assets_manifest'])==sha(cfg['assets_manifest']),
        'CPU asset registration/output identity differs')
    require(ar['input_bindings'].get(cfg['assets_config'])==sha(cfg['assets_config'])
        and ar['source_bindings'].get(cfg['source_driver_module'])==sha(cfg['source_driver_module']),
        'CPU assets did not bind this config/original source implementation')
    require(out!=Path(ac['out']) and out not in Path(ac['out']).parents and Path(ac['out']) not in out.parents,'New source output overlaps assets')
    for key in ('protocol','calibration_registration'):require(ac[key]==cfg[key],'CPU/GPU source input differs: '+key)
    for values in (ar['source_bindings'],ar['input_bindings'],done['source_bindings'],done['input_bindings'],done['outputs']):verify(values)
    require(all(bound.get(p)==s for p,s in ar['source_bindings'].items()),'Original asset/source execution closure unbound')
    cal=read(cfg['calibration_registration']);records=validate_manifest(manifest,done,cal)
    owner=module(cfg['owner_module'],'full1000_bound_owner');prior=module(cfg['wait_module'],'full1000_bound_batch_verifier')
    old=read(cfg['assets_owner_config']);launch=read(cfg['assets_owner_launch']);ident=launch['identity']
    owner.validate_config(old,ar,sha(cfg['assets_owner_config']))
    require(old['registration']==cfg['assets_registration'] and len(old['stages'])==1
        and old['stages'][0]['id']=='source' and old['stages'][0]['resource']=='cpu' and len(old['stages'][0]['jobs'])==1,
        'CPU assets predecessor scope differs')
    for key,want in (('root',cfg['root']),('out',cfg['H_out']),('budget_path',cfg['ledger']),
                     ('budget_registration',cfg['budget_registration']),('budget_registration_sha256',sha(cfg['budget_registration'])),
                     ('phase_limits',cfg['phase_limits'])):
        require(old[key]==want,'CPU/GPU shared scope differs: '+key)
    require(old['stages'][0]['jobs'][0]['completion']==cfg['assets_completion'] and launch['argv']==ident['argv']
        and launch['argv'][1:]==['-B',cfg['owner_module'],'--config',cfg['assets_owner_config']]
        and launch['registration_sha256']==sha(cfg['assets_registration'])
        and launch['owner_config_sha256']==sha(cfg['assets_owner_config']),'CPU predecessor launch/config differs')
    require(prior.exited(ident,owner.raw_process_state),'CPU assets owner must be fully exited')
    prior_closure=prior.verify_batch(owner,cfg['assets_owner_config'],cfg['assets_registration'],ident['uid'],owner.raw_process_state)
    require(prior_closure['bindings'].get(cfg['assets_completion'])==sha(cfg['assets_completion']),'Exited predecessor did not seal assets')
    budget=read(cfg['budget_registration']);require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
        and budget['phase_limits']==reg['phase_limits']==cfg['phase_limits'],'Original independent H budget differs')
    before=owner.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    verify_frozen_budget(before,reg)
    require(cfg['numerical_reference_field']==['numerical_runtime'],'Original numeric reference field differs')
    ref=read(cfg['numerical_reference']);require(ref['status']=='REAL_NATIVE_QUALIFICATION_PASS','Original native numerical qualification missing')
    for key in cfg['numerical_reference_field']:ref=ref[key]
    require(isinstance(ref,dict) and ref['threads']==6 and ref['deterministic'] is True
        and ref['matmul_tf32'] is False and ref['cudnn_tf32'] is False,'Original numeric reference invalid')
    return dict(cfg=cfg,reg=reg,bound=bound,owner=owner,prior=prior,prior_closure=prior_closure,before=before,
        assets_done=done,records=records,ids=manifest['source_ids'],cal=cal,expected_flags=ref,
        static_bindings=closure_bindings,assets_api=module(cfg['assets_module'],'full1000_bound_assets'))


def verify_live_visual_owner(ctx,config_path):
    cfg=ctx['cfg'];a=ctx['owner'];c=read(cfg['source_owner_config']);a.validate_config(c,ctx['reg'],sha(cfg['source_owner_config']))
    require(c['registration']==cfg['registration'] and c['out']==cfg['H_out'] and len(c['stages'])==1,'Source owner scope differs')
    stage=c['stages'][0];require(stage['id']=='source' and stage['resource']=='gpu' and len(stage['jobs'])==1,'One exclusive GPU source job required')
    job=stage['jobs'][0];argv=job['argv'];entry=a.command_entry(argv);offset=2 if argv[1]=='-B' else 1
    require(entry==Path(__file__).resolve() and argv[offset+1:]==['--config',str(Path(config_path).resolve())]
        and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json'),'Source worker command differs')
    base=Path(c['owner_out']);op=base/'owner_identity.json';oid=read(op)
    require(oid['registration_sha256']==sha(cfg['registration']) and int(oid['pid'])==os.getppid()
        and oid['config_sha256']==sha(cfg['source_owner_config']) and a.same_identity(oid,a.identity(os.getppid()))
        and not (base/'failure.json').exists(),'Bound exclusive owner is not the live parent')
    lp=base/'stages/source/workers'/job['id']/'launch.json';started=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-started<10 and a.same_identity(oid,a.identity(os.getppid())),'Owner did not seal worker launch');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid())
    require(launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='gpu' and launch['argv']==argv
        and a.same_identity(launch['identity'],current) and launch['threads']==c['gpu_threads']==6
        and set(launch['affinity'])==set(c['gpu_affinity'])==set(os.sched_getaffinity(0))
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu_device']),'Source GPU launch/runtime differs')
    gp=base/'stages/source/gpu_admission.json';g=read(gp)
    require(g['status']=='GPU_IDLE_CONFIRMED' and g['registration_sha256']==sha(cfg['registration']),'GPU admission absent')
    return dict(owner_identity=oid,worker_identity=current,bindings={str(p):sha(p) for p in (op,lp,gp)})


def finish_records(records,ids):
    require(len(records)==len(ids)==1000 and len(set(ids))==1000,'Full source codec coverage incomplete')
    require(all(r['source_index']==i and r['source_id']==ids[i] for i,r in enumerate(records)),'Full source codec order differs')
    reused=sum(r['codec_origin']=='EXACT_ID_SOURCE200_REUSE' for r in records)
    fresh=sum(r['codec_origin']=='NEW_FROZEN_SOURCE_ENCODING' for r in records)
    require(reused==200 and fresh==800,'Source200/new800 execution counts differ')
    return dict(reused_codec_sources=reused,newly_encoded_sources=fresh,pending_source_encoding_count=0,source1000_codec_complete=True)


def validate_native(native,ctx):
    require(native.loaded['identity']==ctx['cal']['identity'] and native.flags==ctx['expected_flags'],'Frozen model/numeric flags changed')
    require(native.driver_bindings==ctx['static_bindings'] and all(ctx['bound'].get(p)==s for p,s in native.driver_bindings.items()),
            'Loaded complete visual source closure differs from registration')


def write_source(out,regsha,cp,asset_path,arrays,lengths,origin,assets_api):
    """Write only source bits; expose the same sealed CP to both PHY families."""
    require(origin in ('EXACT_ID_SOURCE200_REUSE','NEW_FROZEN_SOURCE_ENCODING'),'Unknown codec origin')
    reused=origin=='EXACT_ID_SOURCE200_REUSE'
    require(reused==(cp['legacy_source200_index'] is not None),'Codec reuse origin differs from source population')
    arrays,lengths=validate_lengths(arrays,lengths);index=cp['source_index'];out=Path(out)
    archive=out/'sources'/f'{index:04d}.npz';assets_api.save_npz(archive,**arrays)
    with np.load(archive,allow_pickle=False) as z:validate_lengths({k:z[k] for k in z.files},lengths)
    path=out/'source_checkpoints'/f'{index:04d}.json'
    result=dict(status='H_FULL1000_SOURCE_CODEC_SOURCE_COMPLETE',registration_sha256=regsha,source_index=index,
        source_id=cp['source_id'],tokens_sha256=cp['tokens_sha256'],preprocessing_id=cp['preprocessing_id'],
        source_assets_checkpoint=dict(path=str(asset_path),sha256=sha(asset_path)),outputs={str(archive):sha(archive)},
        lengths=lengths,independent_roundtrip=True,codec_origin=origin,legacy_source200_index=cp['legacy_source200_index'],
        independent_roundtrip_reused=reused,new_canonical_roundtrips=0 if reused else 4,
        source_code_protocol='unchanged original H independent TX/RX contexts; m6..m9 public raster',
        new_packet_decodes=0,new_image_renders=0,new_metric_calls=0,online_timing_measured=False)
    if reused:result['reuse_proof']=cp['reuse_proof']
    save(path,result)
    return dict(source_index=index,source_id=cp['source_id'],checkpoint=str(path),sha256=sha(path),codec_origin=origin),merge(result['outputs'],{str(path):sha(path)})


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];out=Path(cfg['out']);regsha=sha(cfg['registration'])
    claim_source_output(out,regsha);started=time.time();began=time.monotonic();native=None
    try:
        supervision=verify_live_visual_owner(ctx,config_path)
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at source boundary')
            elapsed=time.monotonic()-began
            require(elapsed<cfg['max_seconds'] and max(time.time(),started+elapsed)<DEADLINE,'Original source execution deadline')
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),'Source GPU owner changed/disappeared')
            if native is not None:require(not quality_driver.boundary(native),'Original GPU health guard requested stop')
        boundary()
        for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
        import quality_driver
        from var_comm import whole_entropy,entropy
        import h64_source
        source_api=module(cfg['source_driver_module'],'full1000_frozen_source_api')
        native=quality_driver.build_native(Path(cfg['root']),cfg['native_runtime'],stop)
        validate_native(native,ctx)
        primitives=h64_source.reference_primitives(whole_entropy,entropy)
        def factory():return source_api.IndependentProvider(native,primitives)
        outputs={};records=[]
        with native.torch.no_grad():
            for index in range(1000):
                boundary();cp,flat,cached=load_source(ctx,index)
                if cached is None:
                    arrays,lengths=encode_new(flat,source_api,h64_source,factory,primitives);origin='NEW_FROZEN_SOURCE_ENCODING'
                else:arrays,lengths=cached;origin='EXACT_ID_SOURCE200_REUSE'
                record,local=write_source(out,regsha,cp,ctx['records'][index]['checkpoint'],arrays,lengths,origin,ctx['assets_api'])
                outputs.update(local);records.append(record)
                status(out/'status.json',dict(status='RUNNING',completed_sources=index+1,total_sources=1000,
                    registration_sha256=regsha,newly_encoded_sources=sum(r['codec_origin']=='NEW_FROZEN_SOURCE_ENCODING' for r in records),
                    elapsed_seconds=time.monotonic()-began))
        counts=finish_records(records,ctx['ids']);boundary();native.frozen()
        after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        require(after==ctx['before'],'Source stage changed packet ledger')
        verify(ctx['bound']);verify(ctx['assets_done']['outputs']);verify(outputs)
        manifest=out/'manifest.json';save(manifest,dict(status=DONE,source_count=1000,source_ids=ctx['ids'],records=records,**counts))
        outputs[str(manifest)]=sha(manifest)
        done=dict(status=DONE,registration_sha256=regsha,config_sha256=sha(config_path),source_count=1000,source_ids=ctx['ids'],
            records=records,**counts,outputs=outputs,input_bindings=ctx['reg']['input_bindings'],source_bindings=ctx['reg']['source_bindings'],
            predecessor_closure_bindings=ctx['prior_closure']['bindings'],supervision=supervision,numerical_runtime=native.flags,
            frozen_visual_identity=native.loaded['identity'],visual_source_bindings=native.driver_bindings,budget_before=ctx['before'],budget_after=after,
            independent_roundtrip=True,new_canonical_roundtrips=3200,new_image_renders=0,new_metric_calls=0,
            native_loader_unchanged=True,unused_original_metrics_still_loaded=True,online_timing_measured=False,
            new_packet_decodes=0,development_used=False,holdout_used=False,training_updates=0,full1000_calibration_complete=False,
            GPU_used=True,elapsed_seconds=time.monotonic()-began)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,
            traceback=traceback.format_exc(),new_packet_decodes=0,original_outputs_preserved=True));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'status':run(args.config)['status']}))


if __name__=='__main__':main()
