"""Register existing historical RGB caches on CPU. Never import an old main.

This inspects completed original receipts, tables and float pixels. It writes
only the new admission manifest and coverage audit supplied on the command line.
It neither evaluates a model nor certifies GPU metric parity.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

import historical_cached as cache

ROOTS = {
    'COMMUNICATION_FIXED_MATRIX': 'outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001',
    'COMMUNICATION_POLICIES': 'outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_ANALYSIS_001',
    'LATENT_ORIGINAL': 'outputs/VAR-LATENT-ENHANCEMENT-20260917/development_eval_v1',
    'LEARNED_PREFIX': 'outputs/VAR-PREFIX-JSCC-EVAL-001',
    'PROGRESSIVE': 'outputs/VAR-PROGRESSIVE-CHANNEL-001',
    'WHOLE_FRAME_PRIOR': 'outputs/VAR-WHOLE-FRAME-PRIOR-001',
    'CRC_FAILURE': 'outputs/VAR-CRC-FAILURE-REPLAY-001',
    'WETOK_A0A1': 'outputs/WETOK-COMM-A0A1-20260912-EVALUATION/step_0005000',
    'WETOK_INTERFACE': 'outputs/WETOK-COMM-INTERFACE-20260912-EVALUATION/additional_0005000',
    'WETOK_GEOMETRY': 'outputs/WETOK-GEOMETRY-20260913-EVALUATION/total_0007000',
    'WETOK_INNOVATION': 'outputs/WETOK-INNOVATION-R1-EVALUATION/step_0005000',
    'WETOK_JOINT': 'outputs/WETOK-JOINT-SENDER-R1-EVALUATION/step_0005000',
    'WETOK_GRID': 'outputs/WETOK-JOINT-GRID-CONTROLS-R1-EVALUATION/quality_0005000',
    'WETOK_R2': 'outputs/WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000',
    'WETOK_R3': 'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000',
    'HYBRID_SOURCE': 'outputs/HYBRID-SOURCE-CORRECTION-20260915/development_001',
    'HYBRID_BASE': 'outputs/HYBRID-BASE-CONDITIONING-20260915/development_001',
    'HYBRID_WEIGHTS': 'outputs/HYBRID-WEIGHT-CLOSURE-20260916/development_001',
    'EXTERNAL_AUTHORS': 'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/fast_author_metrics_001',
    'EXTERNAL_DIGITAL_MATCHED': 'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/digital_development_001',
}
LEGACY = {
    'M8_RECIPE': 'outputs/VAR-M8-LOCAL-20260911-EVALUATION',
    'TOKEN_BACKBONE': 'outputs/VAR-TOKEN-BACKBONE-20260911-EVALUATION',
}
COMPLETED_STATUSES = {
    'FIXED_MODE_MATRIX_COMPLETE', 'FROZEN_COMMUNICATION_POLICY_EVALUATION_COMPLETE',
    'DEVELOPMENT_EVALUATION_COMPLETE', 'TRAINED_PREFIX_JSCC_DEVELOPMENT_EVALUATION_COMPLETE',
    'MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN', 'STOP_THIS_FINITE_PREFIX_VAR_RECEIVER',
    'STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9', 'EVALUATION_COMPLETE', 'INTERFACE_EVALUATION_COMPLETE',
    'GEOMETRY_EVALUATION_COMPLETE', 'INNOVATION_EVALUATION_COMPLETE', 'JOINT_EVALUATION_COMPLETE',
    'GRID_QUALITY_COMPLETE', 'R2_QUALITY_COMPLETE', 'R3_QUALITY_COMPLETE', 'DEVELOPMENT_COMPLETE',
    'WEIGHT_CLOSURE_DEVELOPMENT_COMPLETE', 'AUTHOR_METRICS_COMPLETE', 'COMMON_BUDGET_DIGITAL_COMPLETE',
    'PREFIX_REFINEMENT_DEVELOPMENT_COMPLETE', 'TOKEN_BACKBONE_EVALUATION_COMPLETE', 'COMPLETE',
}

def pointer_piece(value):
    return str(value).replace('~', '~0').replace('/', '~1')

def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

class Builder:
    def __init__(self, root, legacy_root):
        self.root = Path(root).resolve()
        self.legacy_root = Path(legacy_root).resolve()
        self.roots = {k: self.root/v for k,v in ROOTS.items()}
        self.roots.update({k:self.legacy_root/v for k,v in LEGACY.items()})
        self.hashes = {}
        self.receipts = {}
        self.proof_index = {}
        self.studies = {}
        self.audit = []
        for base in self.roots.values():
            self.index_receipt(base/'completion.json')
        self.native = self.root/'outputs/ei-liulu-xqvar-eval-20260912-v1/full/wetok'
        self.index_receipt(self.native/'completion.json')
        self.population_path = self.root/'outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001/population.json'
        self.population = cache.read(self.population_path)
        cache.require(len(self.population)==100 and len({r['image_id'] for r in self.population})==100, 'Original development population differs')
        self.id_to_index = {r['image_id']:i for i,r in enumerate(self.population)}
        self.qconfig = self.root/'configs/progressive_channel.yaml'
        import yaml
        q = yaml.safe_load(self.qconfig.read_text())['quality']
        self.quality = dict(module_path=str(self.root/'src/var_comm/quality.py'),
            paths={k:q[k] for k in ('dino_source','dino_checkpoint','alexnet_checkpoint')})
        import lpips
        linear = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
        self.quality['lpips_linear_sha256'] = self.sha(linear)
        self.quality_bindings = {str(self.qconfig):self.sha(self.qconfig),
            self.quality['module_path']:self.sha(self.quality['module_path'])}
        for k in ('dino_checkpoint','alexnet_checkpoint'):
            cache.require(self.sha(q[k])==q[k+'_sha256'], 'Original quality weight changed')
            self.quality_bindings[q[k]]=q[k+'_sha256']
        # The original DINO implementation is executable input too.
        dino_root = Path(q['dino_source'])
        for p in sorted(dino_root.rglob('*.py')):
            if '.git' not in p.parts:
                self.quality_bindings[str(p)] = self.sha(p)

    def sha(self, path):
        key = str(Path(path).resolve())
        if key not in self.hashes:
            self.hashes[key] = cache.sha(key)
        return self.hashes[key]

    def index_receipt(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            return None
        if str(path) in self.receipts:
            return self.receipts[str(path)]
        receipt = cache.read(path)
        self.receipts[str(path)] = receipt
        for field in ('output_hashes',):
            for relative, expected in receipt.get(field, {}).items():
                p = (path.parent/relative).resolve()
                self.proof_index[str(p)] = dict(path=str(p),sha256=expected,receipt=str(path),
                    receipt_sha256=self.sha(path),pointer='/'+field+'/'+pointer_piece(relative))
        for filename, field in [('per_frame.csv','per_frame_sha256'),('per_image.jsonl','per_image_jsonl_sha256')]:
            if field in receipt:
                p = path.parent/filename
                self.proof_index[str(p)] = dict(path=str(p),sha256=receipt[field],receipt=str(path),
                    receipt_sha256=self.sha(path),pointer='/'+field)
        if 'archive_sha256' in receipt:
            p=path.parent/'reconstructions.npz'
            self.proof_index[str(p)] = dict(path=str(p),sha256=receipt['archive_sha256'],receipt=str(path),
                receipt_sha256=self.sha(path),pointer='/archive_sha256')
        return receipt

    def prove(self, spec, path):
        path=Path(path).resolve();key=str(path)
        if key not in self.proof_index:
            self.index_receipt(path.parent/'receipt.json')
            self.index_receipt(path.parent/'frame.json')
        if key not in self.proof_index:
            raise RuntimeError('Missing original file-hash proof: '+key)
        proof=self.proof_index[key]
        cache.require(self.sha(path)==proof['sha256'],'Original file hash mismatch: '+key)
        if key not in {p['path'] for p in spec['proofs']}:
            spec['proofs'].append(proof)
        return path

    def pixel_prove(self, spec, path, table, position, column, key, indices, expected,
                    *, table_format='csv', hash_kind='raw', shape=None, dtype=None):
        path=Path(path).resolve();table=Path(table).resolve()
        self.prove(spec, table)
        item=dict(path=str(path),sha256=self.sha(path),receipt=str(table),receipt_sha256=self.sha(table),
            receipt_format=table_format,row_index=position,column=column,key=key,indices=indices,
            pixel_sha256=expected,hash_kind=hash_kind)
        if shape is not None:item.update(shape=shape,dtype=dtype)
        token=(str(path),key,tuple(indices))
        seen={(p['path'],p['key'],tuple(p['indices'])) for p in spec['pixel_proofs']}
        if token not in seen:spec['pixel_proofs'].append(item)

    def completion(self, base):
        p=base/'completion.json';data=self.index_receipt(p)
        cache.require(data is not None,'Original completion receipt missing: '+str(p))
        status=data.get('status','')
        cache.require(status in COMPLETED_STATUSES, 'Unregistered historical completion status: '+status)
        return dict(path=str(p),sha256=self.sha(p),allowed_statuses=[status])

    def base_spec(self, study, base, tables, method_columns, target_kind, **kwargs):
        spec=dict(synthetic=False,format='indexed_npz',training_updates=0,policy_selection_updates=0,
            proofs=[],pixel_proofs=[],evidence_bindings=dict(self.quality_bindings),completion=self.completion(base),
            quality=dict(self.quality),tables=[dict(path=str(p)) for p in tables],
            source_id_column='image_id',source_index_column='image_index',method_key_columns=method_columns,
            records=[],methods={},rows_per_source=[0]*100,source_count=100,expected_rows=0,
            metric_columns=dict(psnr_db='psnr_db',lpips_alex='lpips_alex',dino_cosine='dino_cosine'),
            metadata_columns={'model_checkpoint':'selected_checkpoint_sha256','training_seed':'training_seed'},
            original_study=study, **kwargs)
        spec['evidence_bindings'][str(self.population_path)]=self.sha(self.population_path)
        for p in tables:self.prove(spec,p)
        for i,r in enumerate(self.population):
            target, digest, preprocessing=self.target(spec,i,target_kind)
            spec['records'].append(dict(source_index=i,image_id=r['image_id'],class_index=int(r['class_index']),
                preprocessing_id=preprocessing,target_rgb_sha256=digest,target=target))
        return spec

    def target(self, spec, i, kind):
        if kind=='progressive':
            p=self.roots['PROGRESSIVE']/f'images/{i:03d}/reconstructions.npz'
            self.prove(spec,p)
            with np.load(p,allow_pickle=False) as a:v=a['source'].copy()
            return dict(path=str(p),key='source',indices=[]),cache.pixel_sha(v),'original_progressive_float_RGB'
        if kind in ('latent','communication'):
            p=self.roots['LATENT_ORIGINAL']/f'images/{i:03d}/reconstructions.npz'
            self.prove(spec,p)
            with np.load(p,allow_pickle=False) as a:v=a['source_rgb'].copy()
            target=dict(path=str(p),key='source_rgb',indices=[])
            preprocessing='original_rounded_uint8_div255_RGB'
            if kind=='communication':
                target['transform']='original_VAR_uint8_normalization_roundtrip'
                v=np.rint(v*np.float32(255)).astype(np.uint8).astype(np.float32)
                v=(((v/np.float32(127.5))-np.float32(1))+np.float32(1))*np.float32(.5)
                preprocessing=target['transform']
            return target,cache.pixel_sha(v),preprocessing
        if kind=='wetok':
            import torch
            table=self.native/'per_image.jsonl'
            rows=[json.loads(line) for line in table.read_text().splitlines()]
            selected=[(j,row) for j,row in enumerate(rows) if int(row['image_index'])==i]
            cache.require(len(selected)==1,'WeTok source row must be unique')
            position,row=selected[0]
            cache.require(row['image_id']==self.population[i]['image_id'] and int(row['class_index'])==int(self.population[i]['class_index']),'Native target population mismatch')
            p=self.native/f'float_images/{i:03d}.pt'
            value=torch.load(p,map_location='cpu',weights_only=True)['source_01']
            typed=hashlib.sha256();typed.update(str(value.dtype).encode());typed.update(json.dumps(list(value.shape)).encode());typed.update(value.numpy().tobytes())
            cache.require(typed.hexdigest()==row['source_tensor_sha256'],'Original typed target hash differs')
            self.pixel_prove(spec,p,table,position,'source_tensor_sha256','source_01',[0],row['source_tensor_sha256'],
                table_format='jsonl',hash_kind='typed_tensor',shape=list(value.shape),dtype=str(value.dtype))
            return dict(path=str(p),key='source_01',indices=[0]),cache.pixel_sha(value[0]),'original_WeTok_source_01_float_RGB'
        raise RuntimeError('Unregistered target kind '+kind)

    def methods(self, spec, rows, *, study, reference=False):
        for row in rows:
            method='|'.join(str(row[k]) for k in spec['method_key_columns'])
            if method in spec['methods']:continue
            if study.startswith('WETOK'):
                label=method in ('digital_adaptive','digital_m8')
                decoder='VAR_D0' if label else 'DeepJSCC_decoder' if 'deepjscc' in method else 'WeTok_decoder'
            elif study=='EXTERNAL_AUTHORS':
                label=False;decoder='SwinJSCC_author_decoder' if method.startswith('swin') else 'ADJSCC_author_decoder'
            else:
                label='deepjscc' not in method
                decoder=('Dc' if method.endswith('_Dc') or 'latent_' in method or method in ('m8_Dc_only','m8_receiver_only_refiner')
                    else 'DeepJSCC_decoder' if not label else 'VAR_D0')
                if study.startswith('HYBRID'):decoder='VAR_D0_plus_learned_residual'
            oracle='oracle' in method.lower() or str(row.get('deployable','1'))=='0'
            reference_only=bool(reference or method.endswith('_D0') and study=='LATENT_ORIGINAL' or oracle)
            spec['methods'][method]=dict(method=method,model_id=study+':'+method,decoder=decoder,
                label_conditioned=bool(label),reference_only=reference_only,oracle=bool(oracle),
                classification_main_eligible=not(label or reference_only),
                paid_class_bits=10 if label and not reference else 0,
                original_population_scope='100 original development sources; original SNR/noise grid retained',
                original_evaluation_precision='original float caches; fresh metric parity required')

    def finish(self, study, spec, rows):
        spec['expected_rows']=len(rows)
        for row in rows:
            i=self.id_to_index[row['image_id']]
            cache.require(int(row[spec['source_index_column']])==i,'Original source index differs')
            spec['rows_per_source'][i]+=1
        cache.require(all(spec['rows_per_source']), 'Completed original table misses development sources')
        spec['proofs']=list({p['path']:p for p in spec['proofs']}.values())
        self.studies[study]=spec
        self.audit.append(dict(study=study,status='CPU_METADATA_REGISTERED_NOT_GPU_QUALIFIED',rows=len(rows),
            sources=100,methods=len(spec['methods']),proof_files=len(spec['proofs']),pixel_proofs=len(spec['pixel_proofs']),
            gpu_parity_passed=False,training_updates=0,policy_selection_updates=0))
        print(f'REGISTERED {study}: {len(rows)} original rows',flush=True)

    def add_standard(self, study):
        if study=='COMMUNICATION_POLICIES':
            return self.add_policies()
        base=self.roots[study];tables=[base/'per_frame.csv']
        kind='wetok' if study.startswith('WETOK') else 'communication' if study=='COMMUNICATION_FIXED_MATRIX' else 'latent' if study in ('LATENT_ORIGINAL','EXTERNAL_AUTHORS','EXTERNAL_DIGITAL_MATCHED') or study.startswith('HYBRID') else 'progressive'
        columns=['method','protocol'] if study.startswith('EXTERNAL_') else ['method'] if study=='LATENT_ORIGINAL' else ['arm']
        if study=='LATENT_ORIGINAL':
            tables=[base/f'images/{i:03d}/per_frame.csv' for i in range(100)]
        spec=self.base_spec(study,base,tables,columns,kind)
        rows=[row for table in tables for row in read_rows(table)]
        if study=='COMMUNICATION_FIXED_MATRIX':
            spec.update(slot_column='image_index_in_archive',archive_template=str(base/'images/{source_index:04d}/reconstructions.npz'))
            spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
            for i in range(100):self.prove(spec,base/f'images/{i:04d}/reconstructions.npz')
        elif study=='LATENT_ORIGINAL':
            metadata=cache.read(base/'metadata.json')
            spec.update(format='latent_matrix_npz',source_index_column='source_index',
                method_order=metadata['method_order'],frame_order=[[s,n] for s in metadata['snrs_db'] for n in metadata['noise_seeds']],
                snr_column='snr_db',seed_column='noise_seed')
            spec['evidence_bindings'][str(base/'metadata.json')]=self.sha(base/'metadata.json')
            for i,record in enumerate(spec['records']):record['matrix_archive']=str(base/f'images/{i:03d}/reconstructions.npz')
        elif study.startswith('HYBRID'):
            spec.update(format='direct_npy',archive_column='image_path',archive_base=str(base),image_hash_column='image_sha256',image_hash_kind='file_sha256')
            spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
            table=tables[0]
            for position,row in enumerate(rows):
                p=Path(row['image_path']);p=p if p.is_absolute() else base/p
                # Original CSV hashes are hashes of the .npy container.
                proof=dict(path=str(p.resolve()),sha256=row['image_sha256'],receipt=str(table),receipt_sha256=self.sha(table),kind='csv',row_index=position,column='image_sha256')
                cache.require(self.sha(p)==proof['sha256'],'Original hybrid reconstruction changed')
                spec['proofs'].append(proof)
        elif study in ('EXTERNAL_AUTHORS','EXTERNAL_DIGITAL_MATCHED'):
            spec.update(archive_column='image_archive',slot_column='image_slot',image_hash_column='image_sha256')
            spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
            for position,row in enumerate(rows):
                try:self.prove(spec,row['image_archive'])
                except RuntimeError as error:
                    if 'Missing original file-hash proof:' not in str(error):raise
                    self.pixel_prove(spec,row['image_archive'],tables[0],position,'image_sha256','images',[int(row['image_slot'])],row['image_sha256'])
            inputs=self.root/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/inputs_001/development_inputs.json'
            values=cache.read(inputs);spec['evidence_bindings'][str(inputs)]=self.sha(inputs)
            by_id={r['image_id']:r for r in values}
            for i,record in enumerate(spec['records']):
                r=by_id[record['image_id']]
                cache.require(self.sha(r['path'])==r['file_sha256'],'External source container differs')
                v=np.load(r['path'],allow_pickle=False).astype(np.float32)/255
                cache.require(cache.pixel_sha(v)==record['target_rgb_sha256'],'External vs registered rounded source differs')
            for row in rows:
                if 'source_pixels_sha256' in row:
                    cache.require(row['source_pixels_sha256']==by_id[row['image_id']]['source_pixels_sha256'],'External source pixel ID changed')
        elif study.startswith('WETOK'):
            spec['quality']['batch_size']=4
            spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
            spec.update(archive_column='image_archive',slot_column='image_ref',image_hash_column='image_sha256')
            if study=='WETOK_A0A1':
                spec.pop('archive_column');spec['archive_template']=str(base/'images/{source_index:03d}/reconstructions.npz')
                for i in range(100):self.prove(spec,base/f'images/{i:03d}/reconstructions.npz')
            else:
                for row in rows:self.prove(spec,row['image_archive'])
        else:
            spec.update(slot_column='reconstruction_index' if study=='PROGRESSIVE' else 'image_ref',
                archive_template=str(base/'images/{source_index:03d}/reconstructions.npz'))
            prog=str(self.roots['PROGRESSIVE']/'images/{source_index:03d}/reconstructions.npz')
            whole=str(self.roots['WHOLE_FRAME_PRIOR']/'images/{source_index:03d}/reconstructions.npz')
            learned=str(self.roots['LEARNED_PREFIX']/'images/{source_index:03d}/reconstructions.npz')
            if study=='WHOLE_FRAME_PRIOR':spec['reference_roots']={'input':prog,'new':spec['archive_template']}
            if study=='CRC_FAILURE':spec['reference_roots']={'input':prog,'new':str(base/'images/{source_index:03d}/retained.npz')}
            if study=='LEARNED_PREFIX':spec['reference_roots']={'old':prog,'transition':whole,'new':spec['archive_template']}
            if study=='M8_RECIPE':
                spec['reference_roots']={'new':spec['archive_template'],'previous:new':learned,'previous:old':prog,'previous:transition':whole}
            if study=='TOKEN_BACKBONE':
                m8=str(self.roots['M8_RECIPE']/'images/{source_index:03d}/reconstructions.npz')
                spec['reference_roots']={'new':spec['archive_template'],'frozen_B_evaluation:new':m8,
                    'frozen_B_evaluation:previous:new':learned,'frozen_B_evaluation:previous:old':prog,
                    'frozen_B_evaluation:previous:transition':whole}
            if 'image_sha256' in rows[0]:spec['image_hash_column']='image_sha256'
            for row in rows:
                i=self.id_to_index[row['image_id']];slot=row[spec['slot_column']]
                template=spec['archive_template']
                if ':' in slot:
                    prefix,_=slot.rsplit(':',1);template=spec['reference_roots'][prefix]
                self.prove(spec,template.format(source_index=i))
        self.methods(spec,rows,study=study)
        self.finish(study,spec,rows)

    def add_policies(self):
        study='COMMUNICATION_POLICIES';base=self.roots[study];table=base/'policy_frames.csv'
        spec=self.base_spec(study,base,[table],['method'],'communication')
        spec.update(locator_by_row_sha256={},row_metadata={})
        spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
        policy_path=self.root/'outputs/COMMUNICATION-CONVERGENCE-20260915/POLICIES_001/policies.json'
        policy=cache.read(policy_path);receipt=cache.read(base/'completion.json')
        cache.require(policy['status']=='CALIBRATION_POLICIES_FROZEN' and policy['development_used_for_fitting'] is False,
                      'Original communication actions are not independently frozen')
        cache.require(self.sha(policy_path)==receipt['policy_sha256'],'Original communication action file changed')
        spec['evidence_bindings'][str(policy_path)]=self.sha(policy_path)
        matrix=self.roots['COMMUNICATION_FIXED_MATRIX']
        cache.require(self.sha(matrix/'completion.json')==receipt['matrix_receipt_sha256'],'Original matrix receipt differs')
        matrix_table=matrix/'per_frame.csv';self.prove(spec,matrix_table)
        key=lambda r:(int(r['image_index']),float(r['snr_db']),int(r['seed']),r['family'],int(r['mode']))
        candidates=read_rows(matrix_table);lookup={key(r):r for r in candidates}
        cache.require(len(lookup)==len(candidates),'Ambiguous original candidate rows')
        rows=read_rows(table)
        for row in rows:
            candidate=lookup[key(row)];family,rule=row['method'].split('_',1)
            actions=policy['actions'][family][rule];snr=float(row['snr_db'])
            support=sorted(float(v) for v in actions);nearest=min(support,key=lambda v:(abs(snr-v),v))
            cache.require(int(row['mode'])==int(actions[str(nearest)]),'Policy row differs from original frozen action')
            for field in ['image_id','family','mode','header_uses','data_uses','complex_uses','accepted_correct','source_correct']:
                cache.require(row[field]==candidate[field],'Original policy candidate identity differs: '+field)
            for field,tolerance in [('psnr_db',1e-4),('lpips',1e-5),('dino',1e-5)]:
                cache.require(abs(float(row[field])-float(candidate[field]))<=tolerance,'Original policy/candidate quality differs')
            i=int(row['image_index']);p=matrix/f'images/{i:04d}/reconstructions.npz';self.prove(spec,p)
            spec['locator_by_row_sha256'][cache.identity(row)]=dict(path=str(p),key='images',indices=[int(candidate['image_index_in_archive'])])
            spec['row_metadata'][cache.identity(row)]=dict(original_candidate_row_sha256=cache.identity(candidate),
                policy_sha256=self.sha(policy_path),selected_on='original independent calibration',policy_selection_updates=0)
        self.methods(spec,rows,study=study)
        self.finish(study,spec,rows)

    def add_wetok_references(self, filename, study, method_column, reference, base_id='WETOK_R3'):
        base=self.roots[base_id];table=base/filename
        spec=self.base_spec(study,base,[table],[method_column],'wetok')
        spec.update(archive_column='image_archive',slot_column='image_ref',image_hash_column='image_sha256')
        spec['quality']['batch_size']=4
        spec['metric_columns'].update(lpips_alex='lpips',dino_cosine='dino')
        rows=read_rows(table)
        if base_id=='WETOK_A0A1':
            spec.pop('archive_column');spec['archive_template']=str(base/'images/{source_index:03d}/reconstructions.npz')
            for i in range(100):self.prove(spec,base/f'images/{i:03d}/reconstructions.npz')
        else:
            for row in rows:self.prove(spec,row['image_archive'])
        self.methods(spec,rows,study=study,reference=reference)
        self.finish(study,spec,rows)

    def add_visual(self, model):
        import torch
        base=self.native.parent/model;table=base/'per_image.jsonl'
        study='VISUAL_SOURCE_'+model.upper().replace('-','_')
        self.index_receipt(base/'completion.json')
        spec=self.base_spec(study,base,[table],['model','name'],'wetok')
        spec['tables'][0]['table_format']='jsonl'
        rows=[json.loads(line) for line in table.read_text().splitlines()]
        spec.update(format='indexed_torch',locator_by_row_sha256={})
        for position,row in enumerate(rows):
            i=int(row['image_index']);p=base/f'float_images/{i:03d}.pt'
            data=torch.load(p,map_location='cpu',weights_only=True)
            image=data['images'][row['name']]
            digest=hashlib.sha256();digest.update(str(image.dtype).encode());digest.update(json.dumps(list(image.shape)).encode());digest.update(image.numpy().tobytes())
            cache.require(digest.hexdigest()==row['image_tensor_sha256'],'Original visual tensor hash differs')
            self.pixel_prove(spec,p,table,position,'image_tensor_sha256','images',[row['name']],row['image_tensor_sha256'],
                table_format='jsonl',hash_kind='typed_tensor',shape=list(image.shape),dtype=str(image.dtype))
            spec['locator_by_row_sha256'][cache.identity(row)]=dict(path=str(p),key='images',indices=[row['name']])
            # These source-only evaluations may have different preprocessing.
            value=data['source_01'];target=cache.rgb(value)
            td=hashlib.sha256();td.update(str(value.dtype).encode());td.update(json.dumps(list(value.shape)).encode());td.update(value.numpy().tobytes())
            cache.require(td.hexdigest()==row['source_tensor_sha256'],'Original visual source tensor differs')
            self.pixel_prove(spec,p,table,position,'source_tensor_sha256','source_01',[],row['source_tensor_sha256'],
                table_format='jsonl',hash_kind='typed_tensor',shape=list(value.shape),dtype=str(value.dtype))
            spec['records'][i].update(target=dict(path=str(p),key='source_01',indices=[]),
                target_rgb_sha256=cache.pixel_sha(target),preprocessing_id='original_'+model+'_source_01_float_RGB')
        for row in rows:
            key='|'.join(str(row[c]) for c in spec['method_key_columns'])
            label=bool(row.get('class_side_bits_if_sent',0))
            spec['methods'][key]=dict(method=key,model_id=key,decoder=model+'_source_decoder',
                label_conditioned=label,reference_only=True,oracle=False,classification_main_eligible=False,
                paid_class_bits=0,free_class_condition=label,source_only=True,physical_channel=False)
        self.finish(study,spec,rows)

    def run_one(self, name, function, *args):
        try:function(*args)
        except Exception as error:
            self.audit.append(dict(study=name,status='NOT_REGISTERED_MISSING_OR_INCONSISTENT_EVIDENCE',
                error=str(error),traceback=traceback.format_exc(),gpu_parity_passed=False))
            print('NEEDS_EVIDENCE '+name+': '+str(error),flush=True)

    def run(self):
        for study in self.roots:self.run_one(study,self.add_standard,study)
        for filename,study,column,reference in [
            ('native_reference.csv','WETOK_NATIVE_REFERENCE','reference',True),
            ('noiseless_mapping.csv','WETOK_NOISELESS_REFERENCE','arm',True),
            ('deep_support_supplement.csv','WETOK_DEEP_SUPPORT','arm',False)]:
            self.run_one(study,self.add_wetok_references,filename,study,column,reference)
        for base_id in ['WETOK_A0A1','WETOK_INTERFACE','WETOK_GEOMETRY','WETOK_INNOVATION','WETOK_JOINT','WETOK_GRID','WETOK_R2']:
            study=base_id+'_NOISELESS_REFERENCE'
            self.run_one(study,self.add_wetok_references,'noiseless_mapping.csv',study,'arm',True,base_id)
        for model in ['wetok','old-official','old-fidelity','xq']:
            self.run_one('VISUAL_SOURCE_'+model.upper().replace('-','_'),self.add_visual,model)
        return dict(status='REGISTERED_ORIGINAL_FLOAT_CACHES',training_updates=0,policy_selection_updates=0,
            gpu_parity_qualification='NOT_RUN',studies=self.studies),dict(
            status='CPU_CACHE_COVERAGE_AUDIT_NOT_GPU_QUALIFICATION',studies=self.audit,
            registered_rows=sum(s['expected_rows'] for s in self.studies.values()),registered_studies=len(self.studies))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--legacy-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--audit',required=True);args=p.parse_args()
    for destination in [args.output,args.audit]:
        cache.require(not Path(destination).exists(),'Refusing to replace registered cache metadata')
    document,audit=Builder(args.root,args.legacy_root).run()
    for path,value in [(args.output,document),(args.audit,audit)]:
        Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
