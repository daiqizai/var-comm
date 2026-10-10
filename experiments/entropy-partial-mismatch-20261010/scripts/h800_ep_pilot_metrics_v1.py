"""Four original metric computations under an explicit new pilot identity.

No model is loaded on import. Original metric definitions/weights are retained;
unused FullSuite networks are not constructed. This is an original-calibration
adapter, not the old holdout evaluator or a claim of old-host numerical identity.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import types
import numpy as np
import h800_ep_pilot_core_v1 as core

SCHEMA='H800_EP_ORIGINAL100_FOUR_METRICS_V1'
POPULATION='original_calibration_pilot'
SOURCE_SHA={
    'quality.py':'589d43a9dede2327c67d363f3b8b879cfb184e40e6a58b1aa2ae9b675d63a1a5',
    'metric_models.py':'dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066',
    'common500_validation_metrics.py':'3f40eda1ba9d79ba4412a57560f9f5420b92e7e426b9152c7907f5f7eba4e7e3',
    'step0_reference_metrics.py':'2c5b712fcf36e859e73ef6a680989ce9a8ef29884ab63b0c3e14e7720d7dae28',
    'score_bpg_cached_holdout_r2.py':'d8e9f85c83f63a288cf0500e53857bc6e81c205cec93897e4d46f9f177c79d8e'}
MANIFEST_SHA='851a92682162e2b2be6e4e53498c50c7c3a9a1d2ea9e559b13cb9215f658ea9f'
REGISTRATION_SHA='61b6b89f3bf0c0886025d57bfb415f7a880bc1ab6e7e0b68d111ce02ea0ab004'
WEIGHT_RECEIPT_SHA='2c6778ede6c15e21947ea0dd1bc9df4c96f97c7d366c3ac6ee0db10704de744a'
DINO_RECEIPT_SHA='1f3a298980eb02983123f963768326a5e0f231ad1bdcf0c4f7dd4efea40eb6a4'
WEIGHT_SHA=dict(alexnet='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    lpips='df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
    convnext='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d',
    dinov2_vitl14='d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428')
FLAGS=dict(matmul_tf32=False,cudnn_tf32=False,precision='highest',cudnn_benchmark=False,
           cudnn_deterministic=True,deterministic=True,threads=6,interop_threads=2)
CAPS=dict(model_constructions=3,reference_preparations=100,image_scores=43200,
    dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=43200,
    convnext_reference=100,convnext_reconstruction=43200,lpips_pair=43200,
    lpips_alexnet_backbone_forward=86400)
require=core.require


def node_hash(node):
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def source_ast(path,name):
    require(name in SOURCE_SHA and core.sha(path)==SOURCE_SHA[name],'Original metric source changed: '+name)
    return ast.parse(Path(path).read_text(encoding='utf-8'),filename=str(path))


def original_definition(tree,name):
    found=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name]
    require(len(found)==1,'Unique original definition required: '+name)
    return found[0]


def compile_nodes(nodes,path,namespace):
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)


def lpips_projection(path,namespace):
    """Keep original loader prefix and its exact perceptual return expression."""
    tree=source_ast(path,'quality.py');old=original_definition(tree,'load_quality_models')
    stops=[i for i,node in enumerate(old.body) if isinstance(node,ast.Assign) and
           any(isinstance(target,ast.Name) and target.id=='dino' for target in node.targets)]
    require(len(stops)==1 and isinstance(old.body[-1],ast.Return) and isinstance(old.body[-1].value,ast.Tuple),
            'Original LPIPS/DINO loader shape changed')
    prefix=copy.deepcopy(old.body[:stops[0]])
    require(len(prefix)==7 and isinstance(prefix[-1],ast.Delete),'Exact original LPIPS loading prefix required')
    fn=copy.deepcopy(old);fn.name='load_only_original_lpips';fn.body=prefix+[ast.Return(value=copy.deepcopy(old.body[-1].value.elts[0]))]
    require([node_hash(n) for n in fn.body[:-1]]==[node_hash(n) for n in old.body[:stops[0]]],
            'LPIPS prefix AST changed')
    compile_nodes([fn],path,namespace)
    metric=original_definition(tree,'quality_metrics')
    assignments=[n for n in ast.walk(metric) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='perceptual_values' for t in n.targets)]
    require(len(assignments)==1,'Original LPIPS pair expression required')
    expression=copy.deepcopy(assignments[0].value)
    exprfn=ast.parse('def original_lpips_pair(perceptual,reconstructed,reference):\n    return None').body[0]
    exprfn.body=[ast.Return(value=expression)];compile_nodes([exprfn],path,namespace)
    return namespace['load_only_original_lpips'],namespace['original_lpips_pair'],dict(
        original_source_sha256=SOURCE_SHA['quality.py'],original_loader_ast=node_hash(old),
        kept_prefix_ast=[node_hash(n) for n in prefix],return_expression_ast=node_hash(old.body[-1].value.elts[0]),
        pair_expression_ast=node_hash(assignments[0].value),deleted_only_after_LPIPS_prefix=True,
        unused_DINO_S_constructed=False,original_module_globals_modified=False)


def psnr_projection(path):
    tree=source_ast(path,'score_bpg_cached_holdout_r2.py');mse=[];metric=[]
    for parent in ast.walk(tree):
        body=getattr(parent,'body',None)
        if not isinstance(body,list):continue
        for i,node in enumerate(body):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mse' for t in node.targets):
                mse.append((node,body[i+1]))
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metric' for t in node.targets) and isinstance(node.value,ast.Call):
                metric.extend(k.value for k in node.value.keywords if k.arg=='psnr_db')
    require(len(mse)==len(metric)==1,'Unique released float64 PSNR expressions required')
    fn=ast.parse('def original_psnr(target,image):\n    return None').body[0]
    fn.body=[copy.deepcopy(mse[0][0]),copy.deepcopy(mse[0][1]),ast.Return(value=copy.deepcopy(metric[0]))]
    namespace=dict(np=np,math=math,require=require);compile_nodes([fn],path,namespace)
    return namespace['original_psnr'],dict(original_source_sha256=SOURCE_SHA['score_bpg_cached_holdout_r2.py'],
        statement_asts=[node_hash(n) for n in fn.body],formula='Original BPG r2 float64 MSE and -10*math.log10; finite values required')


def convnext_projection(path,policy):
    tree=source_ast(path,'common500_validation_metrics.py');old=original_definition(tree,'ConvNeXtValidation')
    namespace=dict(np=np,Path=Path,inspect=inspect,require=require,sha=core.sha,
        VERSION='convnext_tiny_IMAGENET1K_V1_independent_development_v1',WEIGHTS_SHA256=WEIGHT_SHA['convnext'])
    def admission(stage,policy_path,policy_sha256):
        require(stage==POPULATION and policy_path==policy['path'] and policy_sha256==policy['sha256']==core.link.POLICY_SHA and
                core.sha(policy_path)==policy_sha256,'Explicit original-calibration population and whole policy required')
    namespace['admission']=admission
    compile_nodes([copy.deepcopy(old)],path,namespace)
    return namespace['ConvNeXtValidation'],dict(original_source_sha256=SOURCE_SHA['common500_validation_metrics.py'],
        original_class_ast=node_hash(old),constructor_and_predict_AST_unchanged=True,
        private_admission_population=POPULATION,holdout_impersonated=False,original_module_globals_modified=False)


def flags_function(path):
    tree=source_ast(path,'step0_reference_metrics.py');fn=original_definition(tree,'numeric_flags');namespace={}
    compile_nodes([copy.deepcopy(fn)],path,namespace)
    return namespace['numeric_flags']


def bound_assets(config):
    """Byte validation and explicit path relocation; no model/tensor/import."""
    require(set(config['sources'])==set(SOURCE_SHA),'Exact five original computation source pins required')
    for name,pin in config['sources'].items():
        require(pin['sha256']==SOURCE_SHA[name] and core.sha(pin['path'])==pin['sha256'],'Original metric source pin differs')
    for key,digest in dict(manifest=MANIFEST_SHA,registration=REGISTRATION_SHA,
                          weight_receipt=WEIGHT_RECEIPT_SHA,dino_receipt=DINO_RECEIPT_SHA).items():
        require(config[key]['sha256']==digest,'Original metric metadata receipt differs: '+key)
    original=core.checked(config['manifest']);registration=core.checked(config['registration'])
    require(registration['numerical_runtime']==FLAGS,'Final registered numerical flags differ')
    weights=core.checked(config['weight_receipt']);dino=core.checked(config['dino_receipt'])
    require(len(weights['mapping'])==4 and len({x['sha256'] for x in weights['mapping']})==4,'Exactly four frozen weights required')
    result={}
    for name,digest in WEIGHT_SHA.items():
        matches=[r for r in weights['mapping'] if r['sha256']==digest]
        require(len(matches)==1,'Missing or ambiguous original metric weight: '+name)
        row=matches[0];path=Path(row['actual_path'])
        require(path.stat().st_size==row['bytes'] and core.sha(path)==digest,'Metric weight bytes changed: '+name)
        result[name]=dict(path=str(path),sha256=digest,original_path=row['original_path'])
    spec=copy.deepcopy(original['dinov2_vitl14']);oldroot=spec['implementation']['path']
    require(len(spec['implementation']['files'])==157 and dino['file_count']==157,'Original157 DINO implementation required')
    mapping={r['original_path']:r for r in dino['mapping']}
    require(len(mapping)==157,'Exact DINO source mapping required')
    for relative,digest in spec['implementation']['files'].items():
        row=mapping[oldroot+'/'+relative];actual=Path(dino['actual_root'])/relative
        require(str(actual)==row['actual_path'] and row['sha256']==digest and core.sha(actual)==digest,
                'DINO implementation path/digest differs')
    require({p.relative_to(dino['actual_root']).as_posix() for p in Path(dino['actual_root']).rglob('*.py')}==
            set(spec['implementation']['files']),'Unregistered DINO Python source')
    require(spec['weights']['sha256']==WEIGHT_SHA['dinov2_vitl14'],'Standard original DINO-L weights required')
    spec['implementation']['path']=dino['actual_root'];spec['weights']['path']=result['dinov2_vitl14']['path']
    require(config['whole_policy']['sha256']==core.link.POLICY_SHA and core.sha(config['whole_policy']['path'])==core.link.POLICY_SHA,
            'Original full-calibration whole policy required')
    return dict(weights=result,dinov2_manifest={'dinov2_vitl14':spec},numeric_flags=FLAGS,
                original_manifest=config['manifest'],original_registration=config['registration'],
                path_relocation_only=True,extra_FullSuite_models_constructed=0)


def load_module(pin,name):
    require(core.sha(pin['path'])==pin['sha256'],'Bound metric module changed')
    spec=importlib.util.spec_from_file_location(name,pin['path']);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


class FourMetrics:
    def __init__(self,config,ledger,guard):
        self.config=config;self.ledger=ledger;self.guard=guard;assets=bound_assets(config)
        import torch
        import lpips
        self.torch=torch;self.device='cuda:0';self.flags=flags_function(config['sources']['step0_reference_metrics.py']['path'])
        torch.set_num_interop_threads(FLAGS['interop_threads']);torch.set_num_threads(FLAGS['threads'])
        torch.set_float32_matmul_precision(FLAGS['precision'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        require(self.flags(torch)==FLAGS,'Actual metric numerical settings differ')
        self.psnr,psnr_proof=psnr_projection(config['sources']['score_bpg_cached_holdout_r2.py']['path'])
        load_lpips,self.lpips_pair,lpips_proof=lpips_projection(config['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
        linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
        require(core.sha(linear)==WEIGHT_SHA['lpips'],'Loaded LPIPS package linear weights differ')
        metric=load_module(config['sources']['metric_models.py'],'_h800_pilot_original_DINO_L')
        classifier,conv_proof=convnext_projection(config['sources']['common500_validation_metrics.py']['path'],config['whole_policy'])
        guard()
        with metric.no_network():
            self.perceptual=ledger.call('model_constructions',lambda:load_lpips({'alexnet_checkpoint':assets['weights']['alexnet']['path']},self.device),model='LPIPS_ALEX')
            self.evaluator=ledger.call('model_constructions',lambda:metric.MetricEvaluator(assets['dinov2_manifest'],device=self.device,
                requested=('dinov2_vitl14',)),model='DINOv2_ViTL14')
            self.classifier=ledger.call('model_constructions',lambda:classifier(assets['weights']['convnext']['path'],WEIGHT_SHA['convnext'],
                device=self.device,stage=POPULATION,policy_path=config['whole_policy']['path'],policy_sha256=config['whole_policy']['sha256']),model='ConvNeXt_Tiny')
        # Control metadata is explicitly this pilot's scope. Numeric feature code
        # remains the original evaluator and uses its own internally consistent identity.
        self.evaluator.metadata['metric_selection_usage']='DINOv2-L only: registered original-calibration pilot objective; no holdout selection'
        original_forward=self.perceptual.net.forward
        def counted_forward(*args,**kwargs):
            return ledger.call('lpips_alexnet_backbone_forward',lambda:original_forward(*args,**kwargs))
        self.perceptual.net.forward=counted_forward
        self.metadata=dict(schema=SCHEMA,population=POPULATION,metrics=list(core.METRICS),assets=assets,
            original_sources=config['sources'],adapter_source=core.descriptor(__file__),original_math_proof=dict(LPIPS=lpips_proof,ConvNeXt=conv_proof,PSNR=psnr_proof),
            numeric_flags=self.flags(torch),DINO_L_metadata=self.evaluator.metadata,ConvNeXt_identity=self.classifier.identity,
            actual_loaded_lpips_linear=dict(path=str(linear),sha256=WEIGHT_SHA['lpips']),
            runtime_identity=config['runtime_identity'],inference_batch_size=1,inference_precision='float32',AMP=False,
            overall_old_FullSuite_identity_claimed=False,old_host_metric_values_reused=False,
            selection_objective='dinov2_vitl14_cosine',convnext_used_for_selection=False,
            received_truth_supplied_to_reconstruction=False)
        self.identity=core.digest(self.metadata);guard()

    def validate_image(self,rgb):
        require(isinstance(rgb,np.ndarray) and rgb.dtype==np.float32 and rgb.shape==(3,256,256) and
            np.isfinite(rgb).all() and (rgb>=0).all() and (rgb<=1).all(),'Original float32 CHW RGB required')
        self.guard();require(self.flags(self.torch)==FLAGS,'Metric numeric settings changed')
        return self.torch.as_tensor(np.ascontiguousarray(rgb)[None],device=self.device,dtype=self.torch.float32)

    def prepare(self,target):
        def operation():
            x=self.validate_image(target)
            dino=self.ledger.call('dinov2_vitl14_reference',lambda:self.evaluator.prepare_reference(x))
            pred=self.ledger.call('convnext_reference',lambda:self.classifier.predict(target))
            return dict(reference_sha256=array_sha(target),evaluator_identity=self.identity,dino=dino,convnext_prediction=pred)
        return self.ledger.call('reference_preparations',operation)

    def score(self,target,image,class_index,prepared):
        def operation():
            x=self.validate_image(target);y=self.validate_image(image)
            require(prepared['reference_sha256']==array_sha(target) and prepared['evaluator_identity']==self.identity,
                'Reference cache must match source pixels and exact metric identity')
            with self.torch.inference_mode(),self.torch.autocast(device_type='cuda',enabled=False):
                lpips=self.ledger.call('lpips_pair',lambda:self.lpips_pair(self.perceptual,y,x))
                dino=self.ledger.call('dinov2_vitl14_reconstruction',lambda:self.evaluator.score(x,y,[class_index],
                    label_conditioned=False,prepared=prepared['dino']))
                prediction=self.ledger.call('convnext_reconstruction',lambda:self.classifier.predict(image))
            result=dict(psnr_db=self.psnr(target,image),lpips_alex=float(lpips[0].item()),
                dinov2_vitl14_cosine=float(dino[0]['dinov2_vitl14_cosine']),
                convnext_top1_source_prediction=int(prediction==prepared['convnext_prediction']))
            require(all(math.isfinite(float(result[m])) for m in core.METRICS) and
                result['convnext_top1_source_prediction'] in (0,1),'Finite four metrics and binary agreement required')
            return result
        return self.ledger.call('image_scores',operation)


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def pair_cache_key(reference,reconstruction,metric_identity):
    require(isinstance(metric_identity,str) and len(metric_identity)==64,'Exact metric identity required')
    return core.digest(dict(reference_sha256=array_sha(reference),reconstruction_sha256=array_sha(reconstruction),metric_identity=metric_identity))
