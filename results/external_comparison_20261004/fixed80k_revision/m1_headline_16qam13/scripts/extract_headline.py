"""CPU-only, exact-cache assembly of the N1024 / 16QAM / 13 dB fixed16 panel."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

LEGACY_SHA = '097362e87afc10739b76f39eb8e60bf99fd64767fc871cf5f11fe44c51f107d6'

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def validate_m1(row, require):
    require(row['experiment']=='M1' and row['method']=='entropy_policy'
        and row['phy_family']=='16QAM' and row['action_id']=='N1024/16QAM/m8/K0/whole'
        and int(row['action_m'])==8 and int(row['action_q'])==0 and row['order']=='whole',
        'Headline M1 must be the completed selected m8/K0/whole 16QAM action')

def run(root, legacy_dir, output):
    if hashlib.sha256((legacy_dir/'extract_existing.py').read_bytes()).hexdigest()!=LEGACY_SHA:
        raise RuntimeError('Frozen extraction helper differs')
    old=load(legacy_dir/'extract_existing.py','headline_legacy_extractor')
    # The frozen helper validates every existing HiFi receipt. Only requested
    # image/metric cells are copied into the new, isolated export.
    old.CONDITIONS=((1024,13),)
    class Headline(old.Extractor):
        def save_png(self,key,**kwargs):
            if key[:2]!=(1024,13):return None
            return super().save_png(key,**kwargs)
        def add_metric(self,key,row,**kwargs):
            if key[:2]!=(1024,13):return None
            super().add_metric(key,row,**kwargs)
            self.rows[key].update(E=row.get('E',row.get('energy','')),
                energy_constraint=row.get('energy_constraint','per_frame_2N'),
                action_id=row.get('action_id',''),phy_family=row.get('phy_family','continuous'))
    job=Headline(root,output);job.bind(__file__);job.bind(legacy_dir/'extract_existing.py')
    job.hifi_inputs()
    replay=job.revision/'m1_headline_16qam13/replay_all16'
    manifest=job.read_bound(replay/'manifest.json')
    old.require(manifest['status']=='EXACT_COMPLETED_RGB_EXPORT_PASS' and manifest['images']==32
        and manifest['source_indices']==list(old.FIXED) and manifest['N']==1024 and manifest['snrs']==[13]
        and manifest['noise_seed']==2001 and manifest['exact_completed_rgb_match'] is True
        and manifest['new_metric_evaluation'] is False and manifest['new_training'] is False
        and manifest['new_calibration'] is False,'Replay scope or scientific protocol differs')
    reg=job.read_bound(replay/'registration.json');old.require(old.sha(replay/'registration.json')==manifest['registration_sha256'],'Replay registration changed')
    for path,digest in manifest['input_bindings'].items():job.bind(path,digest)
    binding=old.identity(reg)
    legacy_done=job.read_bound(root/'outputs/UNIFIED-METRICS-20261002/scoring_completion.json')
    old.require(legacy_done['status']=='COMPLETE' and legacy_done['parity_passed'] is True,'Unified scoring incomplete')
    inventory=job.read_bound(root/'results/unified_metrics_20261002/scoring_inventory.json',legacy_done['outputs'])
    for index in old.FIXED:
        path=replay/'source_checkpoints'/f'{index:03d}.json';cp=job.read_bound(path,manifest['outputs'])
        old.require(cp['binding']==binding and cp['source_index']==index and cp['synthetic'] is False
            and cp['payload_sha256']==old.identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Replay source seal changed')
        legacy_path=root/'outputs/UNIFIED-METRICS-20261002/source_checkpoints'/f'{index:03d}.json'
        legacy=job.read_bound(legacy_path,inventory['source_checkpoint_sha256'])
        old.require(cp['original_completed_checkpoint_sha256']==old.sha(legacy_path),'Copied metrics origin differs')
        measured={r['replay_row_id']:r for r in legacy['rows']}
        proof=cp['float_reconstructions'];archive=Path(proof['path']);job.bind(archive,proof['sha256']);job.bind(archive,manifest['outputs'][str(archive)])
        old.require(len(cp['rows'])==2 and {r['method'] for r in cp['rows']}=={'P','M1'},'P/M1 coverage differs')
        with np.load(archive,allow_pickle=False) as data:
            old.require(set(data.files)=={'images','source_rgb','row_ids','image_slots'},'Replay archive schema differs')
            images=data['images'];target=data['source_rgb']
            old.require(images.shape==(2,3,256,256) and images.dtype==np.float32
                and np.array_equal(target,job.targets[index]) and data['image_slots'].tolist()==[0,1]
                and data['row_ids'].tolist()==[r['replay_row_id'] for r in cp['rows']], 'Replay pixels/reference/layout differ')
            for slot,row in enumerate(cp['rows']):
                metric=row['completed_metrics'];key=(1024,13,index,row['method'])
                old.require(row['N']==1024 and row['snr_db']==13 and row['noise_seed']==2001
                    and row['source_index']==index and row['image_slot']==slot
                    and row['source_id']==job.records[index]['image_id'] and row['exact_completed_rgb_match'] is True
                    and row['original_scalar_parity']['status']=='PASS' and metric==measured[row['replay_row_id']]
                    and row['float_rgb_sha256']==metric['image_sha256']==old.rgb_sha(images[slot])
                    and row['reference_sha256']==metric['reference_sha256']==old.rgb_sha(target), 'Replay/image/metric/source pairing changed')
                old.require(not old.boolean(metric['label_conditioned']) and metric['decoder_id']=='Dc','Unexpected label or decoder')
                if row['method']=='M1':validate_m1(metric,old.require)
                else:old.require(metric['experiment']=='N1024' and metric['method']=='P1024','Wrong P1024 point')
                job.save_png(key,image=images[slot],proof=dict(origin='exact_completed_float_reexport',archive=str(archive),archive_sha256=proof['sha256'],image_slot=slot,image_sha256=row['float_rgb_sha256']))
                job.add_metric(key,metric,origin='completed_unified_metrics_exact_reexport',checkpoint=legacy_path,record=job.records[index])
        job.reference_proofs.append(old.source_reference_proof(job.targets[index],job.records[index],[r['completed_metrics'] for r in cp['rows']]))
    old.require(len(job.rows)==64 and len(job.assets)==80 and all(a['status']=='AVAILABLE' for a in job.assets.values()),'Headline must be complete')
    old.require(all(p['legacy_scoring_reference_equals_common'] for p in job.reference_proofs),'Reference floats must agree exactly')
    for key,row in job.rows.items():row.update(asset_status='AVAILABLE',asset_sha256=job.assets[key]['sha256'])
    metricpath=job.out/'per_frame.csv';old.write_csv(metricpath,[job.rows[k] for k in sorted(job.rows)])
    for path,digest in job.inputs.items():old.require(old.sha(path)==digest,'Input changed during extraction: '+path)
    old.require('torch' not in sys.modules,'CPU export imported inference framework')
    energies=[float(r['E']) for r in job.rows.values() if r['method']=='M1']
    out=dict(schema='existing_reconstruction_comparison_v1',status='HEADLINE_EXACT_CACHE_EXTRACTION_COMPLETE',
        source_indices=list(old.FIXED),noise_seed=2001,conditions=[dict(N=1024,snr_db=13)],
        sources=[job.records[i] for i in old.FIXED],assets=[job.assets[k] for k in sorted(job.assets)],
        metrics=dict(path='per_frame.csv',sha256=old.sha(metricpath),rows=64),
        method_labels={'M1':'M1-selected (m8, K=0, 16QAM)'},
        new_inference=True,new_reconstruction_export=True,new_training=False,new_calibration=False,new_metric_evaluation=False,
        extractor_inference_calls=0,input_bindings=job.inputs,reference_proofs=job.reference_proofs,
        m1_actual_energy=dict(n_sources=16,mean=float(np.mean(energies)),min=min(energies),max=max(energies)),
        report_notes=[
            '**本图M1的既有校准选中动作是m8、K=0、whole、16QAM。没有发送部分尺度token，不能把该点的差异归因为熵序或部分尺度传输的收益。**',
            '各方法总信道使用N=1024。Swin、HiFi和P采用连续链逐帧2N能量约束（E=2048）；M1保持原16QAM星座并单列实际能量，不能称为严格同逐帧能量比较。M1的头部68次、正文956次；头部QPSK，正文16QAM。',
            f'M1在这16张图上的实际能量：均值{np.mean(energies):.6f}，最小{min(energies):.6f}，最大{max(energies):.6f}。完整逐帧E保存在指标CSV中；该均值与1000图校准均值不是同一统计量。',
            'Swin/HiFi共享同一实际接收波形。P/M1只匹配源图、N、SNR和名义种子2001，各链噪声命名空间不同。配对区间以源图为单位，不声称跨链同波形。',
            '全部16张源图的新旧评测参考float32哈希完全相同；本次补导出的32张P/M1重建float32哈希也与已有指标登记值完全相同。没有在PNG上重算指标。',
            '分类器置信概率未保存的旧P/M1项保持N/A。本文仅是固定16图的展示与配对描述，不能据此宣称总体或全面优势。'])
    old.write_json(job.out/'manifest.json',out)
    print(json.dumps(dict(status=out['status'],metric_rows=64,image_cells=80,manifest=str(job.out/'manifest.json')),ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--legacy-dir',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();root=a.root.resolve();base=root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision'
    run(root,a.legacy_dir or base/'five_column_comparison',a.out or base/'m1_headline_16qam13/extracted')

if __name__=='__main__':main()
