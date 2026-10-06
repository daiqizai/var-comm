"""CPU-only export of the frozen development100 source cache.

Cached tokens are provisional until the independent GPU source job reproduces
them with the registered Encoder/VQ. No reconstruction/metric arrays are read.
"""
import argparse
import hashlib
from pathlib import Path
import signal
import time
import traceback
import zipfile
import numpy as np
import development_source_common as c

STOP=False

def stop(*_):
    global STOP
    STOP=True


def build_plan(pop,calibration,targets,token_ids,roles,classes,pixel_root):
    ids=c.population(pop,calibration)
    c.require(isinstance(targets,list) and len(targets)==100 and len(token_ids)>=100
        and len(roles)==len(token_ids)==len(classes),'Original cached population incomplete')
    c.require(list(token_ids[:100])==ids and list(roles[:100])==['target_development']*100,
        'Development cache IDs/roles/order differ')
    records=[]
    for i,(sid,t,b) in enumerate(zip(ids,targets,pop['data_bindings'])):
        c.require(t['image_id']==sid and type(t['class_index']) is int and int(classes[i])==t['class_index'],
            'Original development target/cache class identity differs')
        records.append(dict(source_index=i,source_id=sid,preprocessing_id=pop['preprocessing_ids'][i],
            evaluation_class_index=t['class_index'],original_development_data_binding=b,
            original_archive=str(Path(pixel_root)/'images'/f'{i:03d}'/'reconstructions.npz'),
            original_archive_sha256=b['source_npz_sha256']))
    return records


def pixels_from_original(source,preprocessing):
    a=np.asarray(source)
    c.require(a.shape==(3,256,256) and np.issubdtype(a.dtype,np.floating) and np.isfinite(a).all()
        and np.all((a>=0)&(a<=1)), 'Original source RGB shape/range differs')
    pixels=np.rint(a*255).astype(np.uint8)
    c.require(hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()==preprocessing,
        'Original source preprocessing bytes changed')
    return pixels


def read_npy_prefix(handle,count=100):
    """Read only the registered leading rows; never materialize another split.

    NPZ/ZIP member metadata and the NPY header describe the container. Data reads
    stop at the byte boundary of row99, even when the member contains1200 rows.
    """
    c.require(count==100,'Only registered leading100 source rows are allowed')
    version=np.lib.format.read_magic(handle)
    c.require(version in ((1,0),(2,0)),'Unsupported cached NPY format')
    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,fortran,dtype=reader(handle)
    c.require(not fortran and not dtype.hasobject and len(shape)>=1 and shape[0]>=count,
        'Cannot safely read original100 prefix from this cache')
    row_items=int(np.prod(shape[1:],dtype=np.int64));size=count*row_items*dtype.itemsize
    c.require(size<=100*680*16 and dtype.itemsize>0,'Unexpected development cache prefix size')
    parts=[];remaining=size
    while remaining:
        block=handle.read(remaining)
        c.require(bool(block),'Truncated original100 NPY prefix');parts.append(block);remaining-=len(block)
    result=np.frombuffer(b''.join(parts),dtype=dtype).reshape((count,*shape[1:])).copy()
    return result,dict(shape=list(shape),dtype=dtype.str,fortran_order=False,
        source_rows_read=100,other_rows_read=0,data_bytes_read=size)


def cache_prefix(archive,key):
    with zipfile.ZipFile(archive) as z:
        name=key+'.npy';c.require(z.namelist().count(name)==1,'Missing/ambiguous original token cache member: '+name)
        with z.open(name) as handle:return read_npy_prefix(handle)


def plan_registered(ctx):
    cfg=ctx['cfg'];targets=c.read(cfg['population_manifest'])
    expected_root=Path(cfg['root'])/'outputs/VAR-PROGRESSIVE-CHANNEL-001'
    c.require(Path(cfg['original_pixel_root'])==expected_root
        and Path(cfg['token_archive'])==Path(cfg['root'])/'outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz'
        and Path(cfg['population_manifest'])==Path(cfg['root'])/'outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001/population.json',
        'Original registered development cache location differs')
    prefixes={};proof={}
    for key in ('image_ids','roles','classes'):
        prefixes[key],proof[key]=cache_prefix(cfg['token_archive'],key)
    plan=build_plan(ctx['pop'],c.read(cfg['calibration_registration']),targets,prefixes['image_ids'].tolist(),
        prefixes['roles'].tolist(),prefixes['classes'].tolist(),expected_root)
    # Token bytes are not read until all100 source identities/roles have passed.
    values,proof['tokens']=cache_prefix(cfg['token_archive'],'tokens')
    c.require(values.shape==(100,680) and np.issubdtype(values.dtype,np.integer)
        and np.all((values>=0)&(values<4096)), 'Original cached token shape/range differs')
    c.require(len({proof[k]['shape'][0] for k in proof})==1,'Original cached metadata/data population lengths differ')
    tokens=np.ascontiguousarray(values,dtype=np.int64);ctx['cache_read_proof']=proof
    for r in plan:c.require(ctx['bound'].get(r['original_archive'])==r['original_archive_sha256']==c.sha(r['original_archive']),
        'Original development source archive not bound/unchanged')
    return plan,tokens


def load_registered(config_path):
    ctx=c.load_registered(config_path,'assets',__file__)
    ctx['plan'],ctx['tokens']=plan_registered(ctx)
    return ctx


def write_asset(out,regsha,row,tokens,pixels,api):
    tokens,pixels=api.validate_arrays(tokens,pixels,row['preprocessing_id'])
    archive=Path(out)/'sources'/f"{row['source_index']:04d}.npz";api.save_npz(archive,tokens=tokens,pixels=pixels)
    cp=Path(out)/'source_checkpoints'/f"{row['source_index']:04d}.json"
    record=dict(row,archive=str(archive),tokens_sha256=api.token_sha(tokens))
    c.save(cp,dict(record,status='H_DEVELOPMENT100_SOURCE_ASSET_READY',registration_sha256=regsha,
        outputs={str(archive):c.sha(archive)},new_source_encoding=False,encoder_tokens_verified=False,
        verification_pending='registered frozen Encoder/VQ recomputation before source coding',
        development_used=True,holdout_used=False,new_packet_decodes=0))
    return dict(record,checkpoint=str(cp),checkpoint_sha256=c.sha(cp)),c.bind((cp,archive))


def finish(records,ctx,outputs):
    c.require(len(records)==100 and all(r['source_index']==i and r['source_id']==ctx['ids'][i] for i,r in enumerate(records)),
        'Original100 asset grid incomplete')
    return dict(status=c.ASSETS_DONE,source_count=100,source_ids=ctx['ids'],preprocessing_ids=ctx['pop']['preprocessing_ids'],
        original_data_bindings=ctx['pop']['data_bindings'],visual_identity=ctx['pop']['identity'],records=records,
        source_codec_complete=False,pending_source_encoding_count=100,cached_tokens_require_frozen_encoder_verification=True,
        population='development',development_used=True,holdout_used=False,new_packet_decodes=0,new_visual_inference=0,
        new_image_renders=0,new_metric_calls=0,training_updates=0)


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];out=Path(cfg['out']);regsha=c.sha(cfg['registration'])
    c.claim(out,regsha,'assets');started=time.time();began=time.monotonic()
    try:
        supervision=c.live_owner(ctx,config_path,__file__,'assets');outputs={};records=[]
        for i,row in enumerate(ctx['plan']):
            c.source_boundary(ctx,supervision,started,began,STOP)
            with np.load(row['original_archive'],allow_pickle=False) as z:
                pixels=pixels_from_original(z['source'],row['preprocessing_id'])
            record,pins=write_asset(out,regsha,row,ctx['tokens'][i],pixels,ctx['assets_api']);records.append(record);outputs.update(pins)
            c.d.atomic(out/'status.json',dict(status='RUNNING',completed_sources=i+1,total_sources=100,registration_sha256=regsha))
        result=finish(records,ctx,outputs);manifest=out/'manifest.json';c.save(manifest,result);outputs[str(manifest)]=c.sha(manifest)
        c.source_boundary(ctx,supervision,started,began,STOP);after=c.final_budget(ctx);c.verify(outputs)
        done=dict(result,registration_sha256=regsha,config_sha256=c.sha(config_path),outputs=outputs,
            original_cache_read_proof=ctx['cache_read_proof'],other_population_token_rows_read=0,
            source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'],
            budget_before=ctx['before'],budget_after=after,supervision=supervision,GPU_used=False,
            elapsed_seconds=time.monotonic()-began)
        c.save(out/'completion.json',done);return done
    except BaseException:
        c.save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,traceback=traceback.format_exc(),new_packet_decodes=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);print(run(a.config)['status'])

if __name__=='__main__':main()
