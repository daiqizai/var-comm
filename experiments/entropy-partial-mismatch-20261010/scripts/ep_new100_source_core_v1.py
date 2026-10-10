"""One original Encoder/VQ pass and one actual entropy TX per new source.

No source selection, RX, rendering, PHY or metrics. The original encode_flat AST
and unchanged PartialSourceCodec supply the scientific operations. New arrays
are read back exactly without a second encoder or an invented historical match.
"""
from __future__ import annotations
import ast
import copy
import dis
import hashlib
from pathlib import Path
import numpy as np
import h800_ep_pilot_core_v1 as original

ENCODER_SHA='c8469584a08ca0f1053ff506a74bae54f5427b04fa90025eaa36656d200db458'
CAPS=dict(model_load=1,encoder=100,source_tx=100,source_rx=0,var_render=0,prior_scale=1000,decoder_forward=0)
require=original.require;descriptor=original.descriptor;save=original.save;sha=original.sha


def encoder_projection(path):
    require(sha(path)==ENCODER_SHA,'Original encode_flat source bytes changed')
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='encode_flat']
    require(len(nodes)==1 and not nodes[0].decorator_list,'One undecorated original encode_flat required')
    node=nodes[0];proof=dict(original_file_sha256=ENCODER_SHA,
        original_function_AST_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
        AST_body_changed=False,original_module_top_level_executed=False,encoder_calls_per_source=1)
    namespace=dict(require=require)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    encoder=namespace['encode_flat']
    # Inspect the real frozen function, not a stub: numpy is imported locally,
    # and the only loaded global is its validation function. Native/torch/vae
    # come exclusively from the original function argument.
    loaded_globals=sorted({x.argval for x in dis.get_instructions(encoder) if x.opname=='LOAD_GLOBAL'})
    require(loaded_globals==['require'],'Original encode_flat global dependencies changed')
    proof['loaded_globals']=loaded_globals;proof['resolved_global_dependencies']=True
    return encoder,proof


def token_sha(tokens):return hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()
def pixel_sha(pixels):return hashlib.sha256(pixels.tobytes()).hexdigest()


def load_pixels(record,inside):
    pin=record['pixels_archive'];path=inside(pin['path'])
    require(path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'Closed new pixels archive changed')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'pixels'},'New content archive must contain only pixels');pixels=z['pixels'].copy()
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256) and pixel_sha(pixels)==record['preprocessing_id'] and
        pixel_sha(np.ascontiguousarray(pixels[:,:,::-1]))==record['horizontal_flip_pixel_sha256'],
        'Actual new source pixels must equal content-gate hashes')
    return pixels


def encode_source(record,backend,codec,partial,ledger,encoder,inside,out):
    i=record['source_index'];require(type(i) is int and 0<=i<100,'New100 source index outside fixed scope')
    out=Path(out);out.mkdir();backend.source_index=i;backend.current=('TX',10)
    backend.traces=[];backend.tx_cdfs={};backend.received={};backend.boundary()
    pixels=load_pixels(record,inside)
    def actual_encoder():
        with backend.t.no_grad():return encoder(backend.native,pixels)
    tokens,latent=ledger.call('encoder',actual_encoder,source=i)
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all() and
        latent.dtype==np.float32 and latent.shape==(32,16,16) and np.isfinite(latent).all(),'Actual original encoder output malformed')
    asset=out/'source_assets.npz'
    with asset.open('xb') as stream:np.savez(stream,pixels=pixels,tokens=tokens,F=latent)
    with np.load(asset,allow_pickle=False) as z:
        require(set(z.files)=={'pixels','tokens','F'} and np.array_equal(z['pixels'],pixels) and
            np.array_equal(z['tokens'],tokens) and np.array_equal(z['F'],latent),'Actual single encoder archive readback differs')
    backend.boundary();endpoints=partial.all_endpoints()
    require(len(endpoints)==len(set(endpoints))==24 and sum(K>0 for _,K in endpoints)==18,
        'Frozen six whole plus18 partial source endpoints required')
    encoded=ledger.call('source_tx',lambda:codec.encode_endpoints(tokens,endpoints),source=i)
    trace=copy.deepcopy(backend.traces)
    require(len(trace)==10 and [r['scale'] for r in trace]==list(range(10)) and all(
        r['source']==i and r['role']=='TX' and r['m']==10 for r in trace),'Ten actual fresh source TX priors required')
    require(set(encoded)==set(endpoints),'Missing or extra entropy endpoints')
    arrays={};rows=[]
    for m,K in endpoints:
        item=encoded[m,K];bits=item['bits']
        require(bits.dtype==np.uint8 and bits.ndim==1 and 2<=len(bits)<=1048576 and np.isin(bits,(0,1)).all() and
            item['arithmetic_bits']==len(bits) and item['m']==m and item['K']==K,'Actual entropy stream malformed')
        arrays[f'm{m}_K{K}']=bits.copy()
        rows.append(dict(m=m,K=K,payload_bits=len(bits),bits_sha256=backend.g.image_sha(bits)))
    stream_archive=out/'actual_streams.npz'
    with stream_archive.open('xb') as stream:np.savez(stream,**arrays)
    with np.load(stream_archive,allow_pickle=False) as z:
        require(set(z.files)==set(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items()),'Actual entropy archive readback differs')
    meta=dict(source_index=i,source_id=record['source_id'],canonical_source_id=record['canonical_source_id'],
        archive=descriptor(stream_archive),endpoints=rows,CDF_trace=trace,old_host_bits_used=False,old_host_CDF_used=False,
        original_pixels=record['pixels_archive'],source_assets=descriptor(asset),preprocessing_id=record['preprocessing_id'],
        tokens_sha256=token_sha(tokens),latent_sha256=backend.g.image_sha(latent),encoder_calls=1,source_TX_calls=1,
        actual_prior_calls=10,independent_RX_calls=0,historical_token_agreement_claimed=False)
    meta_path=out/'actual_TX.json';save(meta_path,meta)
    result=dict(source_index=i,source_id=record['source_id'],canonical_source_id=record['canonical_source_id'],
        evaluation_class_index=record['evaluation_class_index'],preprocessing_id=record['preprocessing_id'],
        tokens_sha256=meta['tokens_sha256'],source_assets=meta['source_assets'],metadata=descriptor(meta_path),archive=meta['archive'],
        actual_new_encoder=1,actual_new_TX=1,actual_new_prior=10,independent_RX_calls=0,status='NEW100_ENCODER_AND_TX_COMPLETE_NO_RX_CLAIM')
    save(out/'completion.json',result);backend.traces=[];backend.tx_cdfs={};backend.received={};backend.boundary()
    return result
