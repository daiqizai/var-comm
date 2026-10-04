"""Extract completed images and scalar metrics only; no model or GPU imports.

The fixed16 identities, legacy policy and all scientific values stay unchanged.
Missing images and experiments remain explicitly missing. PNGs are display
copies; original measured float hashes and input receipts are kept separately.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image

FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
CONDITIONS=((1024,7),(1024,13),(2048,7),(2048,13))
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim',
         'ms_ssim','dino_specificity','resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')
METHODS={'SwinJSCC_new_shared':'swin','HiFiDiffCom_SwinJSCC':'hifi'}
DOMAIN=b'float32:3,256,256:RGB\0'

def require(ok,message):
    if not ok: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def identity(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def boolean(value):
    require(value in (True,False,'True','False','true','false','1','0',1,0),'Explicit boolean required')
    return value in (True,'True','true','1',1)
def rgb_sha(value):
    value=np.asarray(value)
    require(value.dtype==np.float32 and value.shape==(3,256,256),'Invalid original float RGB schema')
    require(np.isfinite(value).all() and value.min()>=0 and value.max()<=1,'Invalid original float RGB')
    return hashlib.sha256(DOMAIN+np.ascontiguousarray(value).tobytes()).hexdigest()
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def measured_values(row,historical=False):
    """Copy original numbers; only algebraic diagnostic differences are derived."""
    result={}
    for name in METRICS:
        key='new_'+name if historical and name not in ('psnr_db','lpips_alex','dino_cosine','dino_specificity') else name
        value=row.get(key,'')
        result[name]='' if value in ('',None) else float(value)
        require(result[name]=='' or math.isfinite(result[name]),'Nonfinite original metric: '+name)
    if result['dino_specificity']=='' and row.get('dino_mismatched','') not in ('',None):
        result['dino_specificity']=float(row['dino_cosine'])-float(row['dino_mismatched'])
    if result['semantic_error']=='' and result['resnet50_top1_source_prediction']!='':
        result['semantic_error']=1.-result['resnet50_top1_source_prediction']
    # No confidence exists in the old scorer. A missing score stays missing.
    return result

def select_legacy(row,index):
    if (int(row.get('source_index',-1))!=index or int(float(row.get('N',-1) or -1))!=1024
        or int(float(row.get('snr_db',-1) or -1)) not in (7,13)
        or int(float(row.get('noise_seed',-1) or -1))!=2001):return None
    if row.get('experiment')=='N1024' and row.get('method')=='P1024':method='P'
    elif (row.get('experiment')=='M1' and row.get('method')=='entropy_policy'
          and row.get('phy_family')=='QPSK'):method='M1'
    else:return None
    require(row.get('decoder_id')=='Dc' and not boolean(row['label_conditioned'])
        and boolean(row['replay_parity_passed']),'Legacy selected decoder/access/parity differs')
    return 1024,int(row['snr_db']),index,method

def source_reference_proof(target,record,rows):
    """Compare stored scientific reference hashes; never infer metric equality."""
    common=rgb_sha(target);u8=np.rint(target*255).astype(np.uint8)
    raw=hashlib.sha256(u8.tobytes()).hexdigest()
    require(raw==record['preprocessing_id'],'Original rounded source bytes differ')
    div255=u8.astype(np.float32)/np.float32(255.)
    native=rgb_sha(div255)
    references=sorted({row['reference_sha256'] for row in rows})
    require(set(references)<={common,native},'Unknown historical reference float pixels')
    return dict(source_index=record['source_index'],source_id=record['image_id'],
        common_reference_sha256=common,uint8_div255_reference_sha256=native,
        existing_legacy_metric_reference_sha256=references,
        legacy_scoring_reference_equals_common=references==[common],
        same_original_uint8_pixels=True,raw_uint8_sha256=raw,
        max_abs_common_vs_uint8_div255=float(np.max(np.abs(target.astype(np.float64)-div255.astype(np.float64)))),
        native_scalar_definitions_retained=True,metrics_recomputed=False)

class Extractor:
    def __init__(self,root,out):
        self.root=Path(root).resolve();self.out=Path(out).resolve()
        base=self.root/'outputs/EXTERNAL-COMPARISON-20261004'
        require(base in self.out.parents,'Extraction output must be a new isolated external-comparison directory')
        self.base=base;self.revision=base/'fixed80k_revision';self.hifi=self.revision/'hifi_fixed16_release/evaluation'
        self.hifi_result=self.root/'results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
        self.inputs={};self.assets={};self.rows={};self.provenance=[];self.reference_proofs=[]
        self.out.mkdir(parents=True,exist_ok=True)
        for directory in ('assets','originals','archives'):(self.out/directory).mkdir(exist_ok=True)
        self.commit=subprocess.check_output(['git','-C',str(self.root),'rev-parse','HEAD'],text=True).strip()
        for path in (base/'runtime',self.revision,self.revision/'hifi_fixed16_release'):sys.path.insert(0,str(path))
        import external_eval_common as common
        from hifi16_common import validate_source
        from step0_cache_export import verified_arrays
        from step0_reference_prepare import admitted
        from own_controls_score_common import selected_p2048
        self.common=common;self.validate_source=validate_source;self.verified_arrays=verified_arrays
        self.admitted=admitted;self.selected_p2048=selected_p2048
        for module in ('external_eval_common','hifi16_common','fixed80k_adapter','step0_cache_export','step0_reference_prepare','own_controls_score_common','own_controls_common'):
            self.bind(Path(sys.modules[module].__file__))
        require('torch' not in sys.modules,'CPU extraction unexpectedly imported Torch')

    def bind(self,path,expected=None):
        path=Path(path);digest=sha(path)
        require(expected is None or digest==expected,'Changed admitted input: '+str(path))
        require(str(path) not in self.inputs or self.inputs[str(path)]==digest,'Input changed during extraction')
        self.inputs[str(path)]=digest;return digest

    def read_bound(self,path,bindings=None):
        if bindings is not None:require(str(path) in bindings,'Input missing from completed receipt: '+str(path))
        self.bind(path,None if bindings is None else bindings[str(path)]);return read(path)

    def git_png(self,path):
        path=Path(path);relative=path.relative_to(self.root).as_posix()
        content=subprocess.check_output(['git','-C',str(self.root),'show',self.commit+':'+relative])
        self.bind(path,hashlib.sha256(content).hexdigest())
        return dict(origin='legacy_published_png',git_commit=self.commit,git_path=relative,
            png_sha256=sha(path),float_reconstruction_revalidated=False,
            note='Original published display PNG; original measured float metrics retained separately')

    def save_png(self,key,image=None,path=None,proof=None):
        n,snr,index,method=key
        target=self.out/('originals' if method=='original' else 'assets')/(f'source_{index:03d}.png' if method=='original' else f'source_{index:03d}_N{n}_SNR{snr}_{method}.png')
        if path is not None:
            with Image.open(path) as pic:require(pic.mode=='RGB' and pic.size==(256,256) and pic.format=='PNG','Legacy display PNG schema differs')
            if target.exists():require(sha(target)==sha(path),'Existing extracted image differs')
            else:shutil.copyfile(path,target)
        else:
            rgb_sha(image);display=np.rint(image.transpose(1,2,0)*255).astype(np.uint8)
            if target.exists():
                with Image.open(target) as pic:require(np.array_equal(np.asarray(pic),display),'Existing PNG pixels differ')
            else:Image.fromarray(display).save(target)
        asset=dict(N=n,snr_db=snr,source_index=index,method=method,status='AVAILABLE',path=target.relative_to(self.out).as_posix(),sha256=sha(target),provenance=proof or {})
        require(key not in self.assets,'Duplicate asset');self.assets[key]=asset;return asset

    def add_metric(self,key,row,*,historical=False,origin,checkpoint,record):
        n,snr,index,method=key;require(key not in self.rows,'Duplicate metric frame')
        prefix='new_' if historical else ''
        result=dict(N=n,snr_db=snr,source_index=index,method=method,noise_seed=2001,source_id=record['image_id'],
            true_class_index=int(record['class_index']),preprocessing_id=record['preprocessing_id'],
            metric_origin=origin,metric_checkpoint=str(checkpoint),original_row_sha256=identity(row),
            image_sha256=row.get('history_image_sha256',row.get('image_sha256','')),
            reference_sha256=row.get('history_reference_sha256',row.get('reference_sha256','')),
            metrics_recomputed=False,noise_pairing='same_source_nominal_seed_distinct_noise_namespaces_between_chains',
            original_method=row.get('method',''),original_scientific_row_json=json.dumps(row,sort_keys=True,separators=(',',':')),
            **measured_values(row,historical))
        for field in ('resnet50_prediction','resnet50_source_prediction'):
            result[field]=row.get(prefix+field,'')
        self.rows[key]=result

    def hifi_inputs(self):
        done=self.read_bound(self.hifi/'completion.json')
        require(done['status']=='HIFI_FIXED16_EVALUATION_COMPLETE' and done['rows']==128 and done['sources']==16,'Fixed16 scoring incomplete')
        reconstruction=self.read_bound(self.hifi/'reconstruction_completion.json')
        require(reconstruction['status']=='HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE' and reconstruction['physical_frames']==64
            and done['reconstruction_completion_sha256']==sha(self.hifi/'reconstruction_completion.json'),'Wrong fixed16 completion')
        reg=self.read_bound(self.hifi/'reconstruction_registration.json')
        require(identity(reg)==reconstruction['binding'],'Fixed16 registration mismatch')
        fixed=self.read_bound(self.base/'fixed_examples.json')
        require(fixed['source_indices']==list(FIXED),'Fixed16 source order differs')
        self.records={int(row['source_index']):row for row in fixed['records']}
        require(set(self.records)==set(FIXED),'Fixed source identity missing')
        metric_path=self.hifi_result/'metrics_per_frame.csv';self.bind(metric_path,done['outputs'][str(metric_path)])
        scored=read_csv(metric_path);self.external={}
        for row in scored:
            key=(int(row['N']),int(row['snr_db']),int(row['source_index']),METHODS[row['method']])
            require(key not in self.external and int(row['noise_seed'])==2001 and not boolean(row['label_conditioned']),'Unexpected fixed16 metric identity')
            self.external[key]=row
        require(len(self.external)==128,'Fixed16 measured metric count differs')
        self.targets={}
        for index in FIXED:
            path=self.hifi/'source_checkpoints'/f'{index:04d}.json';cp=self.read_bound(path,reconstruction['outputs'])
            self.validate_source(cp,reconstruction['binding'],index)
            proof=cp['float_reconstructions'];archive=Path(proof['path']);self.bind(archive,reconstruction['outputs'][str(archive)])
            for frame,digest in cp['frame_bindings'].items():self.bind(frame,digest)
            with np.load(archive,allow_pickle=False) as data:
                target=data['source_rgb'].copy();images=data['images'];slots=data['image_slots'].tolist()
                self.targets[index]=target
                require(rgb_sha(target)==reg['source_float_sha256'][list(FIXED).index(index)],'Reference float changed')
                for row,slot in zip(cp['rows'],slots):
                    key=(int(row['N']),int(row['snr_db']),index,METHODS[row['method']]);metric=self.external[key]
                    require(metric['image_sha256']==rgb_sha(images[slot]) and metric['reference_sha256']==rgb_sha(target)
                        and metric['source_id']==self.records[index]['image_id'],'Image/metric/source pairing differs')
                    self.save_png(key,image=images[slot],proof=dict(origin='completed_fixed16_float_cache',archive=str(archive),archive_sha256=sha(archive),image_slot=slot,image_sha256=metric['image_sha256']))
                    self.add_metric(key,metric,origin='completed_hifi_fixed16_metrics',checkpoint=metric_path,record=self.records[index])
            for n,snr in CONDITIONS:self.save_png((n,snr,index,'original'),image=target,proof=dict(origin='completed_fixed16_source_rgb',float_rgb_sha256=rgb_sha(target)))
            source_archive=self.out/'archives'/f'original_{index:03d}.npz'
            if source_archive.exists():
                with np.load(source_archive,allow_pickle=False) as data:require(np.array_equal(data['source_rgb'],target),'Saved original float differs')
            else:np.savez_compressed(source_archive,source_rgb=target)
        return reconstruction

    def p2048_inputs(self):
        study='FINAL_P2048_P3060';_,receipt,bindings=self.admitted(self.root,study)
        for path,digest in bindings.items():self.bind(path,digest)
        folder=self.root/'outputs/HISTORICAL-METRICS-R2-20261003'/study
        regpath=self.root/'results/historical_metrics_r2_20261003'/study/'registration.json'
        reg=self.read_bound(regpath,receipt['bindings'])
        for index in FIXED:
            path=folder/'source_checkpoints'/f'{index:04d}.json';archive=folder/'reconstructions'/f'{index:04d}.npz'
            cp=self.read_bound(path,receipt['bindings']);self.bind(archive,receipt['bindings'][str(archive)])
            target,images,slots=self.verified_arrays(cp,reg,self.records[index],study,archive)
            require(np.array_equal(target,self.targets[index]),'P2048 and external reference floats differ')
            selected=[]
            for row in cp['rows']:
                selection=self.selected_p2048(row)
                if selection is None:continue
                (n,snr,seed,_),meta=selection
                if snr not in (7,13) or seed!=2001:continue
                key=(n,snr,index,'P');slot=slots[row['history_row_id']]
                self.save_png(key,image=images[slot],proof=dict(origin='admitted_historical_float_cache',archive=str(archive),archive_sha256=sha(archive),image_slot=slot,image_sha256=row['history_image_sha256']))
                self.add_metric(key,row,historical=True,origin='completed_historical_P2048',checkpoint=path,record=self.records[index]);selected.append(key)
            require(len(selected)==2,'P2048 selected coverage differs')

    def transmission_png(self,key,row):
        n,snr,index,method=key
        if snr!=7 or index not in (0,25,50,75):return None
        name='TRANSMISSION-COMPARISONS-20261003' if index in (0,25) else 'TRANSMISSION-COMPARISONS-EXTRA-20261003'
        folder=self.root/'outputs'/name;manifestpath=folder/'comparisons_manifest.json'
        if not manifestpath.exists():return None
        manifest=self.read_bound(manifestpath)
        require(manifest['status']=='EXACT_COMPLETED_RGB_EXPORT_PASS','Previous export incomplete')
        matches=[item for item in manifest['frames'] if item['replay_row_id']==row['replay_row_id']]
        require(len(matches)==1,'Legacy displayed frame identity missing/duplicated');entry=matches[0]
        cp=self.root/'outputs/UNIFIED-METRICS-20261002/source_checkpoints'/f'{index:03d}.json'
        self.bind(cp,manifest['checkpoint_bindings'][str(cp)])
        require(entry['exact_completed_rgb_match'] is True and entry['float_rgb_sha256']==row['image_sha256']
            and entry['completed_metrics']==row,'Previous display differs from measured float frame')
        png=folder/entry['file'];self.bind(png,entry['png_sha256'])
        proof=dict(origin='previous_exact_float_export',manifest=str(manifestpath),manifest_sha256=sha(manifestpath),
            image_sha256=row['image_sha256'],original_parity=entry['original_parity'],float_reconstruction_revalidated_by_original_export=True)
        return png,proof

    def legacy_inputs(self):
        folder=self.root/'results/unified_metrics_20261002';done=self.read_bound(self.root/'outputs/UNIFIED-METRICS-20261002/scoring_completion.json')
        require(done['status']=='COMPLETE' and done['parity_passed'] is True,'Original unified scoring incomplete')
        inventory=self.read_bound(folder/'scoring_inventory.json',done['outputs'])
        registration=self.read_bound(folder/'metrics_registration.json',done['outputs'])
        for index in FIXED:
            path=self.root/'outputs/UNIFIED-METRICS-20261002/source_checkpoints'/f'{index:03d}.json'
            cp=self.read_bound(path,inventory['source_checkpoint_sha256'])
            require(cp['binding']==identity(registration) and cp['source_index']==index
                and cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Original unified source seal differs')
            chosen=[]
            for row in cp['rows']:
                key=select_legacy(row,index)
                if key is None:continue
                require(row['source_id']==self.records[index]['image_id'] and row['preprocessing_id']==self.records[index]['preprocessing_id'],'Original source differs')
                chosen.append(row);self.add_metric(key,row,origin='completed_unified_original_metrics',checkpoint=path,record=self.records[index])
                existing=self.transmission_png(key,row)
                if existing is None and index in (0,25,50,75) and key[1]==13:
                    png=(self.root/'results/extreme_bandwidth_20261001_R1_N1024/examples/plugin'/f'source_{index:03d}_snr_13_P1024_seed2001.png' if key[3]=='P' else
                        self.root/'results/scale_causal_partial_residual_20261002/examples/m1'/f'N1024_QPSK_s13_src{index:03d}_entropy_policy.png')
                    if png.exists():existing=png,self.git_png(png)
                if existing is not None:self.save_png(key,path=existing[0],proof=existing[1])
                else:self.assets[key]=dict(N=key[0],snr_db=key[1],source_index=index,method=key[3],status='UNAVAILABLE_CACHE',
                    reason='Metrics exist; reconstruction pixels were not saved. No decoder rerun.')
            require(len(chosen)==4,'Exactly P1024/M1 × two SNR rows required')
            self.reference_proofs.append(source_reference_proof(self.targets[index],self.records[index],chosen))

    def finish(self):
        own=self.base/'own_controls'
        for path in (own/'development_completion.json',self.root/'results/external_comparison_20261004/own_controls/N2048_policy.json'):
            require(not path.exists(),'N2048 M1 state has changed; review completed policy before extraction')
        for n,snr in CONDITIONS:
            if n!=2048:continue
            for index in FIXED:self.assets[n,snr,index,'M1']=dict(N=n,snr_db=snr,source_index=index,method='M1',status='NOT_RUN',
                reason='N2048 M1 has no completed calibration/development result.')
        require(len(self.assets)==320 and len(self.rows)==224,'Extraction scope differs')
        for key,row in self.rows.items():
            asset=self.assets[key];row['asset_status']=asset['status'];row['asset_sha256']=asset.get('sha256','')
            require(asset['status']!='NOT_RUN','Unrun experiment has a measured metric')
        for path,digest in self.inputs.items():require(sha(path)==digest,'Input changed during extraction: '+path)
        require('torch' not in sys.modules,'Extraction loaded an inference framework')
        rows=[self.rows[key] for key in sorted(self.rows)];metricpath=self.out/'per_frame.csv';write_csv(metricpath,rows)
        source_rows=[dict(self.records[index],reference_float_proof=self.reference_proofs[list(FIXED).index(index)]) for index in FIXED]
        manifest=dict(schema='existing_reconstruction_comparison_v1',status='EXISTING_ONLY_CPU_EXTRACTION_COMPLETE',
            source_indices=list(FIXED),noise_seed=2001,conditions=[dict(N=n,snr_db=s) for n,s in CONDITIONS],
            sources=source_rows,assets=[self.assets[key] for key in sorted(self.assets)],metrics=dict(path='per_frame.csv',sha256=sha(metricpath),rows=224),
            training_updates=0,policy_selection_updates=0,reconstruction_inference_calls=0,metric_inference_calls=0,
            source_bindings={str(Path(__file__).resolve()):sha(__file__)},input_bindings=self.inputs,
            archived_originals=[dict(path=f'archives/original_{index:03d}.npz',sha256=sha(self.out/'archives'/f'original_{index:03d}.npz')) for index in FIXED],
            scientific_scope='exploratory original fixed16; not a representative population',
            report_notes=[
                '本轮只提取已完成的重建图和逐帧指标，没有新增推断、评测模型调用、校准或训练。所有16张原图单独提供PNG和原始float32 NPZ。',
                'P1024和M1在全部16张图上的旧指标仍在；仅原先4张固定样例保存了7/13 dB图像，其他12张保留明确缺图格。N2048 M1未完成校准和开发评测，整列标为未运行。',
                '7 dB旧样例由原exact-float导出回执关联指标和PNG；13 dB旧样例验证原Git发布PNG，未重新解码验证float像素。图像仅作展示，不使用PNG重算指标。',
                '每张图保留新旧reference浮点哈希核对；同原始uint8像素已验证。旧原生PSNR/LPIPS/DINO定义和值保留，不声称重新用新参照评测。',
                'Swin/HiFi共享完全相同的接收波形。P/M1只匹配原图、总N、SNR和标称种子2001，其信道噪声命名空间不同，不能称为同波形比较。',
                'P2048为历史最终模型（训练seed2026092304），P1024为已登记模型（seed2026093001）；历史Dc权重版本也需按各自出处区分。Swin为固定80k、预算截断模型。',
                '旧P/M1没有分类top1置信概率，confidently_wrong留空；P2048没有错配DINO，specificity留空。semantic_error由已存分类一致率作1−一致率计算；没有补造未测指标。'])
        write_json(self.out/'manifest.json',manifest)
        return dict(status=manifest['status'],assets=320,available=sum(x['status']=='AVAILABLE' for x in self.assets.values()),
            unavailable_cache=sum(x['status']=='UNAVAILABLE_CACHE' for x in self.assets.values()),not_run=32,metric_rows=224,
            manifest=str(self.out/'manifest.json'),legacy_scoring_reference_all_equal_common=all(x['legacy_scoring_reference_equals_common'] for x in self.reference_proofs))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out')
    args=parser.parse_args();out=Path(args.out) if args.out else Path(args.root)/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/five_column_comparison/extracted'
    job=Extractor(args.root,out);job.hifi_inputs();job.p2048_inputs();job.legacy_inputs();print(json.dumps(job.finish(),ensure_ascii=False))

if __name__=='__main__':main()
