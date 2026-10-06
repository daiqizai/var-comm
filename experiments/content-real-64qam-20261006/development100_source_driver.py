"""Frozen source coding only for the original development100 after selection.

The encoder is re-evaluated exactly as original development_data; cache tokens
must agree, then every m6..m9 stream undergoes independent canonical decoding.
There is no channel, reconstruction, metric evaluation or policy selection.
"""
import argparse
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import development_source_common as c

STOP=False
def stop(*_):
    global STOP
    STOP=True


def load_registered(config_path):return c.load_registered(config_path,'source',__file__)


def load_asset(ctx,index):
    r=ctx['records'][index];cp=c.read(r['checkpoint']);api=ctx['assets_api'];done=ctx['assets_done']
    c.require(done['outputs'].get(r['checkpoint'])==r['checkpoint_sha256']==c.sha(r['checkpoint']), 'Asset checkpoint changed')
    for k in ('source_index','source_id','preprocessing_id','tokens_sha256','archive'):
        c.require(cp[k]==r[k],'Asset checkpoint mapping differs: '+k)
    c.require(cp['status']=='H_DEVELOPMENT100_SOURCE_ASSET_READY' and cp['registration_sha256']==c.sha(ctx['cfg']['assets_registration'])
        and cp['original_development_data_binding']==ctx['pop']['data_bindings'][index]
        and set(cp['outputs'])=={r['archive']} and done['outputs'].get(r['archive'])==cp['outputs'][r['archive']],
        'Source is not linked to original development binding')
    c.verify(cp['outputs'])
    with np.load(r['archive'],allow_pickle=False) as z:
        c.require(set(z.files)=={'tokens','pixels'},'Unexpected development source cache array')
        tokens,pixels=api.validate_arrays(z['tokens'],z['pixels'],r['preprocessing_id'])
    c.require(api.token_sha(tokens)==r['tokens_sha256'],'Cached source token identity changed')
    return cp,tokens,pixels


def verify_encoder_tokens(native,pixels,expected):
    # Exact original digital.development_data formula and quantization order.
    t=native.torch;loaded=native.loaded
    image=t.as_tensor(pixels[None],dtype=t.float32,device=loaded['device'])/127.5-1
    f=loaded['vae'].quant_conv(loaded['vae'].encoder(image))
    tokens=t.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu().numpy()
    c.require(tokens.shape==(680,) and np.issubdtype(tokens.dtype,np.integer)
        and np.array_equal(tokens,expected),'Frozen Encoder/VQ differs from original cached development tokens; preserve and stop')
    return np.ascontiguousarray(tokens,dtype=np.int64)


def write_source(ctx,index,cp,arrays,lengths):
    cfg=ctx['cfg'];out=Path(cfg['out']);api=ctx['assets_api'];runner=ctx['codec_runner']
    arrays,lengths=runner.validate_lengths(arrays,lengths)
    archive=out/'sources'/f'{index:04d}.npz';api.save_npz(archive,**arrays)
    with np.load(archive,allow_pickle=False) as z:runner.validate_lengths({k:z[k] for k in z.files},lengths)
    path=out/'source_checkpoints'/f'{index:04d}.json';asset=ctx['records'][index]
    result=dict(status='H_DEVELOPMENT100_SOURCE_CODEC_SOURCE_COMPLETE',registration_sha256=c.sha(cfg['registration']),
        source_index=index,source_id=ctx['ids'][index],tokens_sha256=cp['tokens_sha256'],preprocessing_id=cp['preprocessing_id'],
        source_assets_checkpoint=dict(path=asset['checkpoint'],sha256=asset['checkpoint_sha256']),outputs={str(archive):c.sha(archive)},
        lengths=lengths,independent_roundtrip=True,encoder_tokens_verified=True,new_canonical_roundtrips=4,
        codec_origin='NEW_FROZEN_DEVELOPMENT_SOURCE_ENCODING',new_packet_decodes=0,new_image_renders=0,new_metric_calls=0,
        development_used=True,holdout_used=False,online_timing_measured=False)
    c.save(path,result)
    return dict(source_index=index,source_id=ctx['ids'][index],checkpoint=str(path),sha256=c.sha(path)),c.bind((path,archive))


def finish(records,ids):
    c.require(len(records)==len(ids)==len(set(ids))==100 and all(r['source_index']==i and r['source_id']==ids[i]
        for i,r in enumerate(records)),'Development source100 codec coverage incomplete')
    return dict(status=c.SOURCE_DONE,source_count=100,source_ids=ids,records=records,source_codec_complete=True,
        pending_source_encoding_count=0,independent_roundtrip=True,new_canonical_roundtrips=400,encoder_tokens_verified=True,
        population='development',development_used=True,holdout_used=False,new_packet_decodes=0,new_image_renders=0,
        new_metric_calls=0,training_updates=0,policy_selection=False,online_timing_measured=False)


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];out=Path(cfg['out']);regsha=c.sha(cfg['registration'])
    c.claim(out,regsha,'source');started=time.time();began=time.monotonic();native=None
    try:
        supervision=c.live_owner(ctx,config_path,__file__,'source')
        def boundary():
            c.source_boundary(ctx,supervision,started,began,STOP)
            if native is not None:c.require(not quality_driver.boundary(native),'Original GPU health guard requested stop')
        boundary()
        for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
        import quality_driver
        from var_comm import whole_entropy,entropy
        import h64_source
        source_api=c.module(cfg['source_driver_module'],'development_frozen_source_api')
        native=quality_driver.build_native(Path(cfg['root']),cfg['native_runtime'],stop)
        ctx['codec_runner'].validate_native(native,ctx)
        primitives=h64_source.reference_primitives(whole_entropy,entropy)
        def factory():return source_api.IndependentProvider(native,primitives)
        outputs={};records=[]
        with native.torch.no_grad():
            for index in range(100):
                boundary();cp,expected,pixels=load_asset(ctx,index)
                tokens=verify_encoder_tokens(native,pixels,expected)
                arrays,lengths=ctx['codec_runner'].encode_new(tokens,source_api,h64_source,factory,primitives)
                record,local=write_source(ctx,index,cp,arrays,lengths);records.append(record);outputs.update(local)
                c.d.atomic(out/'status.json',dict(status='RUNNING',completed_sources=index+1,total_sources=100,registration_sha256=regsha,
                    new_canonical_roundtrips=(index+1)*4,elapsed_seconds=time.monotonic()-began))
        result=finish(records,ctx['ids']);boundary();native.frozen();after=c.final_budget(ctx)
        c.verify(ctx['assets_done']['outputs']);c.verify(outputs)
        manifest=out/'manifest.json';c.save(manifest,result);outputs[str(manifest)]=c.sha(manifest)
        done=dict(result,registration_sha256=regsha,config_sha256=c.sha(config_path),outputs=outputs,
            source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'],
            budget_before=ctx['before'],budget_after=after,supervision=supervision,numerical_runtime=native.flags,
            frozen_visual_identity=native.loaded['identity'],visual_source_bindings=native.driver_bindings,
            frozen_encoder_calls=100,native_loader_unchanged=True,unused_original_metrics_still_loaded=True,
            GPU_used=True,elapsed_seconds=time.monotonic()-began)
        c.save(out/'completion.json',done);return done
    except BaseException:
        c.save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,traceback=traceback.format_exc(),new_packet_decodes=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);print(run(a.config)['status'])

if __name__=='__main__':main()
