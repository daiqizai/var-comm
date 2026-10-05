"""CPU population adapter for the original calibration1000; no codec/model call.

Source200 arithmetic assets are reusable only after source-ID mapping and exact
source token/pixel equality. The other800 remain explicitly unencoded. Running
the exporter requires a new, separately qualified execution registration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import time
import traceback
import numpy as np

STOP=False
DEADLINE=1791564605.9549868
SIZES=(1,2,3,4,5,6,8,10,13,16)
PIXEL_MANIFEST_SHA='49f344e8cf72960b7c164af96e05c203e24f4ae1cbfb8be9f3f02f8842b8e8b8'
DONE='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'


def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def identity(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def bind(paths):return {str(p):sha(p) for p in paths}

def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Immutable source changed: '+p)

def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting SHA: '+p);result[p]=s
    return result

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez(f,**arrays)

def unique_by_basename(values):
    result={}
    for value in values:
        name=Path(value).name;require(name not in result,'Ambiguous duplicate shard basename: '+name);result[name]=value
    return result

def build_population_plan(calibration,pixel_manifest,manifest_path,source200_ids,expected_identity):
    """Pure metadata transformation preserving Native.data('calibration') order."""
    require(calibration['stage']==calibration['calibration_or_development']=='m1_calibration',
            'Only original calibration population is allowed')
    ids=calibration['source_ids'];preprocessing=calibration['preprocessing_ids']
    require(len(ids)==len(set(ids))==len(preprocessing)==1000,'Original1000 identity/order is incomplete or duplicated')
    require(calibration['identity']==expected_identity,'Frozen visual/model identity differs')
    require(len(source200_ids)==len(set(source200_ids))==200 and set(source200_ids)<=set(ids),'Source200 must be an exact unique subset')
    selected={sid:i for i,sid in enumerate(source200_ids)}
    populations=[p for p in pixel_manifest['populations'] if p['name']=='calibration']
    require(len(populations)==1 and populations[0]['count']==1000,'Exactly one original calibration1000 pixel population required')
    pop=populations[0];desc={s['path']:s for s in pop['shards']}
    require(len(desc)==len(pop['shards']) and sum(s['count'] for s in pop['shards'])==1000,'Pixel shard coverage differs')
    latent=calibration['data_bindings']['latents'];by_name=unique_by_basename(latent)
    unique_by_basename(desc)
    locations=calibration['data_bindings']['pixels'];require(len(locations)==1000,'Original pixel index incomplete')
    records=[];seen=set()
    for index,(sid,pre,location) in enumerate(zip(ids,preprocessing,locations)):
        require(location['image_id']==sid,'Pixel index is not in original calibration order')
        rel=location['shard'];require(rel in desc and Path(rel).parts[0]=='calibration','Unregistered/non-calibration pixel shard')
        j=location['index_in_shard'];require(type(j) is int and 0<=j<desc[rel]['count'],'Pixel row out of bounds')
        require((rel,j) not in seen,'Duplicate source shard/index');seen.add((rel,j))
        lp=by_name.get(Path(rel).name);require(lp is not None and 'calibration' in Path(lp).parts,'Matching calibration latent shard missing')
        pp=Path(manifest_path).parent/rel
        require(isinstance(pre,str) and len(pre)==64,'Invalid original preprocessing SHA')
        records.append(dict(source_index=index,source_id=sid,preprocessing_id=pre,
            original_calibration_index=index,index_in_shard=j,latent_shard=str(lp),latent_sha256=latent[lp],
            pixel_shard=str(pp),pixel_sha256=desc[rel]['sha256'],evaluation_class_index=int(location['class_index']),
            legacy_source200_index=selected.get(sid),
            codec_status='REUSE_REQUIRES_EXACT_SOURCE_VALIDATION' if sid in selected else 'NEEDS_FROZEN_SOURCE_ENCODING'))
    require(len(seen)==1000,'Original full population was truncated')
    return dict(schema='H_ORIGINAL1000_POPULATION_PLAN_V1',source_count=1000,source_ids=list(ids),records=records,
        visual_identity=expected_identity,visual_identity_sha256=identity(expected_identity),
        original_data_bindings=calibration['data_bindings'],source200_ids=list(source200_ids),
        reusable_source_count_pending_validation=200,pending_source_encoding_count=800,
        development_used=False,holdout_used=False,new_visual_inference=0,new_packet_decodes=0)

def numpy_cpu(value):
    if hasattr(value,'device'):require(str(value.device)=='cpu','Assets may only be read on CPU')
    if hasattr(value,'detach'):value=value.detach().numpy()
    return np.asarray(value)

def validate_arrays(tokens,pixels,preprocessing):
    tokens=numpy_cpu(tokens);pixels=numpy_cpu(pixels)
    require(tokens.shape==(680,) and np.issubdtype(tokens.dtype,np.integer)
            and np.all((tokens>=0)&(tokens<4096)),'Cached token shape/range differs')
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Original uint8 CHW source pixels required')
    tokens=np.ascontiguousarray(tokens,dtype=np.int64);pixels=np.ascontiguousarray(pixels)
    require(hashlib.sha256(pixels.tobytes()).hexdigest()==preprocessing,'Source preprocessing differs')
    return tokens.copy(),pixels.copy()

def token_sha(tokens):return hashlib.sha256(b'int64:680\0'+np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()

class CalibrationAssetReader:
    """One shard pair in memory; paths come only from the bound calibration plan."""
    def __init__(self,plan,loader=None,expected_bindings=None):
        self.plan=plan;self.loader=loader;self.expected_bindings=expected_bindings
        self.key=None;self.latent=None;self.pixels=None;self.input_bindings={};self.torch=None
    def _load(self,path):
        if self.loader is not None:return self.loader(path)
        if self.torch is None:
            import torch
            torch.set_num_threads(2);torch.set_num_interop_threads(2);self.torch=torch
        return self.torch.load(path,map_location='cpu',weights_only=True)
    def source(self,index):
        require(type(index) is int and 0<=index<1000,'Only full calibration indices0..999 allowed')
        row=self.plan['records'][index];require(row['source_index']==index,'Plan source order differs')
        key=(row['latent_shard'],row['pixel_shard'])
        if key!=self.key:
            lp,pp=map(Path,key)
            require(sha(lp)==row['latent_sha256'] and sha(pp)==row['pixel_sha256'],'Original shard SHA differs')
            side=lp.with_suffix('.json');require(read(side)['sha256']==row['latent_sha256'],'Latent sidecar differs')
            pins=bind((lp,pp,side))
            if self.expected_bindings is not None:
                require(all(self.expected_bindings.get(p)==s for p,s in pins.items()),'Full1000 shard/sidecar not bound before execution')
            self.latent=None;self.pixels=None
            self.latent=self._load(lp);self.pixels=self._load(pp);self.key=key
            self.input_bindings.update(pins)
        j=row['index_in_shard'];sid=row['source_id']
        require(self.latent['image_ids'][j]==self.pixels['image_ids'][j]==sid,'Latent/pixel source ID differs')
        require(int(self.pixels['labels'][j])==row['evaluation_class_index'],'Source evaluation label differs')
        tokens,pixels=validate_arrays(self.latent['full_tokens'][j],self.pixels['targets_u8'][j],row['preprocessing_id'])
        return dict(row,tokens=tokens,pixels=pixels,tokens_sha256=token_sha(tokens))

def validate_reuse(full,s1_cp,s1_tokens,s1_pixels,h_cp,codec_arrays):
    """No inference: certify exact-ID source equivalence before reusing old bits."""
    old_index=full['legacy_source200_index'];require(type(old_index) is int,'This source has no registered old codec')
    require(s1_cp['source_index']==h_cp['source_index']==old_index
            and s1_cp['source_id']==h_cp['source_id']==full['source_id']
            and s1_cp['original_calibration_index']==full['source_index']
            and s1_cp['preprocessing_id']==full['preprocessing_id'],'Source200 index/ID mapping differs')
    tokens,pixels=validate_arrays(s1_tokens,s1_pixels,full['preprocessing_id'])
    require(np.array_equal(tokens,full['tokens']) and np.array_equal(pixels,full['pixels']),
            'Source200 tokens/pixels are not exactly the original full1000 source')
    require(h_cp['independent_roundtrip'] is True,'Original source coder roundtrip not complete')
    lengths={r['m']:r for r in h_cp['lengths']};require(len(lengths)==len(h_cp['lengths'])==4 and set(lengths)=={6,7,8,9},'Original prefix coverage differs')
    result={}
    for m in (6,7,8,9):
        bits=np.asarray(codec_arrays[f'm{m}_bits'])
        require(bits.ndim==1 and np.issubdtype(bits.dtype,np.integer) and np.all((bits==0)|(bits==1))
                and len(bits)==lengths[m]['arithmetic_bits'] and len(bits)>0,'Original arithmetic bit vector/length differs')
        require(lengths[m]['raw_bits']==12*sum(x*x for x in SIZES[:m]),'Original raw prefix length differs')
        result[f'm{m}_bits']=np.array(bits,dtype=np.uint8,copy=True)
    return result

class Source200ReuseReader:
    def __init__(self,s1_assets_completion,h_source_completion,s1_regsha,h_regsha):
        self.s1_path=Path(s1_assets_completion);self.h_path=Path(h_source_completion)
        self.s1=read(self.s1_path);self.h=read(self.h_path);self.s1sha=s1_regsha;self.hsha=h_regsha
        require(self.s1['status']=='S1_EXPORT_ASSETS_COMPLETE' and self.s1['source_count']==200
                and self.s1['registration_sha256']==s1_regsha,'Original S1 assets incomplete')
        require(self.h['status']=='H_SOURCE200_COMPLETE' and self.h['source_count']==200
                and self.h['registration_sha256']==h_regsha and self.h['development_used'] is False
                and self.h['holdout_used'] is False and self.h['training_updates']==0,'Original H codec assets incomplete')
        verify(self.s1['outputs']);verify(self.h['outputs']);self.input_bindings=merge(self.s1['outputs'],self.h['outputs'],bind((self.s1_path,self.h_path)))
    def source(self,full):
        index=full['legacy_source200_index'];require(type(index) is int,'Original H codec unavailable for this ID')
        sp=self.s1_path.parent/'source_checkpoints'/f'{index:04d}.json';hp=self.h_path.parent/'source_checkpoints'/f'{index:04d}.json'
        require(self.s1['outputs'].get(str(sp))==sha(sp) and self.h['outputs'].get(str(hp))==sha(hp),'Original source checkpoints unsealed')
        s,h=read(sp),read(hp)
        require(s['registration_sha256']==self.s1sha and h['registration_sha256']==self.hsha,'Original source registration differs')
        require(s['payload_sha256']==identity({k:v for k,v in s.items() if k!='payload_sha256'}),'S1 source checkpoint identity differs')
        verify(s['outputs']);verify(h['outputs'])
        sa=Path(s['archive']);ha=self.h_path.parent/'sources'/f'{index:04d}.npz'
        require(s['outputs'].get(str(sa))==self.s1['outputs'].get(str(sa))==sha(sa)
                and h['outputs']=={str(ha):sha(ha)} and self.h['outputs'].get(str(ha))==sha(ha),'Original source archive differs')
        with np.load(sa,allow_pickle=False) as z, np.load(ha,allow_pickle=False) as a:
            # Clean image keys in old archive are deliberately never read.
            bits=validate_reuse(full,s,z['tokens'],z['pixels'],h,{f'm{m}_bits':a[f'm{m}_bits'] for m in (6,7,8,9)})
        return bits,dict(status='SOURCE_CODEC_REUSED_BY_EXACT_ID',source_id=full['source_id'],full_source_index=full['source_index'],
            legacy_source200_index=index,tokens_sha256=full['tokens_sha256'],preprocessing_id=full['preprocessing_id'],
            original_input_bindings=bind((sp,hp,sa,ha)),original_lengths=h['lengths'],
            independent_roundtrip_reused=True,new_canonical_decode=False,clean_image_keys_read=False)

def load_registered(config_path):
    path=Path(config_path).resolve();cfg=read(path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_FULL1000_ASSET_ADAPTER_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['allowed_stage_ids']==['source'] and reg['source_stage_scope']=='FULL1000_CPU_ASSETS_ONLY',
        'New independent full1000 asset registration required')
    verify(reg['source_bindings']);verify(reg['input_bindings'])
    require(reg['source_bindings'].get(str(Path(__file__).resolve()))==sha(__file__)
        and reg['input_bindings'].get(str(path))==sha(path),'Adapter/config unbound')
    required=('protocol','calibration_registration','image_manifest','s1_completion','s1_registration','s1_assets_completion',
              's1_source_ids','h_source_registration','h_source_config','h_source_completion')
    for name in required:require(reg['input_bindings'].get(cfg[name])==sha(cfg[name]),'Required preparation dependency unbound: '+name)
    protocol=read(cfg['protocol']);cal=read(cfg['calibration_registration']);hreg=read(cfg['h_source_registration']);hc=read(cfg['h_source_config'])
    h=Path(hc['out']);out=Path(cfg['out'])
    require(cfg['root']==hc['root'] and out.is_absolute() and h in out.parents and cfg['stop_file']==str(h/'STOP'),
            'New asset output/STOP must remain inside the original H scope')
    for original in (Path(cfg['h_source_completion']).parent,Path(cfg['s1_completion']).parent):
        require(out!=original and original not in out.parents and out not in original.parents,'New output overlaps original assets')
    require(protocol['schema']=='H_CODEC_PROTOCOL_V1' and protocol['source']['ordering']=='public raster'
        and protocol['source']['whole_targets']==[6,7,8,9] and protocol['source']['sizes']==list(SIZES)
        and protocol['source']['new_training']==0,'Frozen source codec protocol differs')
    verify(hreg['source_bindings']);verify(hreg['input_bindings'])
    require(all(reg['source_bindings'].get(p)==s for p,s in hreg['source_bindings'].items()),'Original source code closure must remain fully bound')
    require(hreg['input_bindings'].get(cfg['h_source_config'])==sha(cfg['h_source_config'])
        and hreg['input_bindings'].get(cfg['protocol'])==sha(cfg['protocol'])
        and hc['registration']==cfg['h_source_registration'] and hc['calibration_registration']==cfg['calibration_registration']
        and hc['source200']==cfg['s1_source_ids'] and Path(hc['S1'])==Path(cfg['s1_completion']).parent,
        'Original H codec inputs differ')
    s1reg=read(cfg['s1_registration']);s1=read(cfg['s1_completion'])
    verify(s1reg['source_bindings']);verify(s1reg['input_bindings'])
    require(s1['status']=='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION' and s1['registration_sha256']==sha(cfg['s1_registration'])
        and s1['outputs'].get(cfg['s1_assets_completion'])==sha(cfg['s1_assets_completion'])
        and s1reg['input_bindings'].get(cfg['s1_source_ids'])==sha(cfg['s1_source_ids'])
        and s1reg['input_bindings'].get(cfg['calibration_registration'])==sha(cfg['calibration_registration']),
        'S1 population/asset proof is incomplete')
    verify(s1['outputs']);require(sha(cfg['image_manifest'])==PIXEL_MANIFEST_SHA,'Original S1 pixel manifest differs')
    plan=build_population_plan(cal,read(cfg['image_manifest']),cfg['image_manifest'],read(cfg['s1_source_ids'])['source_ids'],cfg['expected_visual_identity'])
    require(cfg['expected_visual_identity']==cal['identity'],'Expected frozen model identity differs')
    return dict(cfg=cfg,reg=reg,plan=plan,reader=CalibrationAssetReader(plan,expected_bindings=reg['input_bindings']),
        reuse=Source200ReuseReader(cfg['s1_assets_completion'],cfg['h_source_completion'],sha(cfg['s1_registration']),sha(cfg['h_source_registration'])))

def claim_empty_output(out,registration_sha256):
    """Accept the empty job directory created by the frozen owner, once only."""
    out=Path(out)
    if out.exists():
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),
                'Existing asset evidence requires independent recovery; no overwrite/retry')
    else:out.mkdir(parents=True)
    # Exclusive creation resolves concurrent claims without overwriting evidence.
    save(out/'attempt.json',dict(status='H_FULL1000_CPU_ASSET_ATTEMPT',
        registration_sha256=registration_sha256,pid=os.getpid(),started_at=time.time()))


def run(config_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Asset preparation must hide CUDA')
    ctx=load_registered(config_path);cfg=ctx['cfg'];out=Path(cfg['out'])
    require(Path(cfg['stop_file']).is_absolute() and 0<cfg['max_seconds']<=3600,'Finite CPU preparation boundary required')
    regsha=sha(cfg['registration']);claim_empty_output(out,regsha);started=time.monotonic()
    try:
        records=[];outputs={}
        for index in range(1000):
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at source boundary')
            require(time.monotonic()-started<=cfg['max_seconds'] and time.time()<DEADLINE,'Original H/CPU preparation deadline')
            full=ctx['reader'].source(index);source=out/'sources'/f'{index:04d}.npz'
            save_npz(source,tokens=full['tokens'],pixels=full['pixels']);local={str(source):sha(source)}
            row={k:v for k,v in full.items() if k not in ('tokens','pixels')};row['archive']=str(source)
            if full['legacy_source200_index'] is not None:
                bits,proof=ctx['reuse'].source(full);codec=out/'codec_reuse'/f'{index:04d}.npz';save_npz(codec,**bits)
                local[str(codec)]=sha(codec);row.update(codec_status='SOURCE_CODEC_REUSED_BY_EXACT_ID',codec_archive=str(codec),reuse_proof=proof)
            else:row.update(codec_status='NEEDS_FROZEN_SOURCE_ENCODING',codec_archive=None)
            cp=out/'source_checkpoints'/f'{index:04d}.json'
            save(cp,dict(status='H_FULL1000_SOURCE_ASSET_READY',registration_sha256=regsha,outputs=local,**row,
                GPU_used=False,new_packet_decodes=0,new_source_encoding=False))
            outputs.update(local);outputs[str(cp)]=sha(cp);records.append(dict(source_index=index,source_id=full['source_id'],
                preprocessing_id=full['preprocessing_id'],tokens_sha256=full['tokens_sha256'],checkpoint=str(cp),checkpoint_sha256=sha(cp),
                archive=str(source),codec_status=row['codec_status'],codec_archive=row['codec_archive'],legacy_source200_index=full['legacy_source200_index']))
        manifest=dict(status=DONE,source_count=1000,source_ids=ctx['plan']['source_ids'],records=records,
            visual_identity=ctx['plan']['visual_identity'],original_data_bindings=ctx['plan']['original_data_bindings'],
            reused_codec_sources=200,pending_source_encoding_count=800,source1000_codec_complete=False,
            requires_independent_GPU_source_execution=True,new_packet_decodes=0,new_visual_inference=0,
            development_used=False,holdout_used=False)
        mp=out/'manifest.json';save(mp,manifest);outputs[str(mp)]=sha(mp)
        inputs=merge(ctx['reg']['input_bindings'],ctx['reader'].input_bindings,ctx['reuse'].input_bindings)
        verify(inputs);verify(ctx['reg']['source_bindings']);verify(outputs)
        done=dict(status=DONE,registration_sha256=regsha,outputs=outputs,input_bindings=inputs,source_bindings=ctx['reg']['source_bindings'],
            source_count=1000,reused_codec_sources=200,pending_source_encoding_count=800,source1000_codec_complete=False,
            new_source_encoding=False,GPU_used=False,new_visual_inference=0,new_packet_decodes=0,
            development_used=False,holdout_used=False,training_updates=0,full1000_calibration_complete=False)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),registration_sha256=regsha,
            new_packet_decodes=0,new_visual_inference=0));raise

def stop(*_):
    global STOP
    STOP=True

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'status':run(args.config)['status']}))

if __name__=='__main__':main()
