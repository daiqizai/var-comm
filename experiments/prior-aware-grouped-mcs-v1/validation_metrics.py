"""Independent ConvNeXt validation, loaded only after calibration policies freeze.

No classifier score from this module is a selection objective. The checkpoint
is local and hash-bound; constructing the model never downloads weights.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import inspect
from pathlib import Path
import numpy as np

VERSION='convnext_tiny_IMAGENET1K_V1_independent_development_v1'
WEIGHTS_SHA256='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
METRICS=('convnext_top1_label','convnext_source_prediction_agreement')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def diagnostics(true_label,source_prediction,reconstruction_prediction):
    values=(true_label,source_prediction,reconstruction_prediction)
    require(all(not isinstance(v,(bool,np.bool_)) and int(v)==v and 0<=int(v)<1000 for v in values),'ImageNet-1k class indices required')
    truth,source,pred=map(int,values);source_ok=source==truth;recon_ok=pred==truth
    return dict(convnext_true_class=truth,convnext_source_prediction=source,convnext_prediction=pred,
        convnext_top1_label=bool(recon_ok),convnext_top1_source_prediction=bool(pred==source),
        convnext_source_prediction_agreement=int(pred==source),
        convnext_source_correct=bool(source_ok),
        convnext_source_correct_to_wrong=bool(source_ok and not recon_ok),
        convnext_source_wrong_to_correct=bool(not source_ok and recon_ok))

def admission(stage,policy_path,policy_sha256):
    require(stage=='development','The independent classifier cannot run on calibration or participate in selection')
    require(len(policy_sha256)==64 and sha(policy_path)==policy_sha256,'Frozen calibration policy receipt required')

class ConvNeXtValidation:
    def __init__(self,weights_path,weights_sha256,*,device,stage,policy_path,policy_sha256):
        admission(stage,policy_path,policy_sha256)
        require(weights_sha256==WEIGHTS_SHA256 and sha(weights_path)==weights_sha256,'Independent classifier weight hash differs')
        import torch
        import torchvision
        from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
        self.torch=torch;self.device=torch.device(device)
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        require(weights.url.endswith('convnext_tiny-983f1562.pth'),'Classifier weight version changed')
        self.transform=weights.transforms()
        # Preserve process RNG around the random construction, then load every
        # frozen parameter. This does not alter subsequent receiver sampling.
        with torch.random.fork_rng(devices=[]):
            self.model=convnext_tiny(weights=None)
        state=torch.load(weights_path,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True);self.model.to(self.device).eval().requires_grad_(False)
        self.names=tuple(weights.meta['categories'])
        self.identity=dict(version=VERSION,architecture='ConvNeXt-Tiny',weights='IMAGENET1K_V1',
            weights_sha256=weights_sha256,torch=torch.__version__,torchvision=torchvision.__version__,
            model_source_sha256=sha(inspect.getfile(type(self.model))),transform=repr(self.transform),
            official_reference='https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html',
            policy_sha256=policy_sha256,population='development',used_for_selection=False,
            pipeline_shared_backbone=False,inference_precision='float32',classification_batch_size=1)

    def predict(self,rgb):
        """Fixed batch one avoids introducing batch-dependent classifier outputs."""
        a=np.asarray(rgb)
        require(a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all()
                and a.min()>=0 and a.max()<=1,'Original float32 RGB required; do not rescore display PNG')
        x=self.torch.from_numpy(np.ascontiguousarray(a)[None]).to(self.device)
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()),'Classifier must remain frozen')
        with self.torch.inference_mode(),self.torch.autocast(device_type=self.device.type,enabled=False):
            logits=self.model(self.transform(x)).float()
        require(bool(self.torch.isfinite(logits).all()) and tuple(logits.shape)==(1,1000),'Invalid classifier output')
        return int(logits.argmax(-1).item())

    def score(self,rgb,*,true_label,source_prediction):
        row=diagnostics(true_label,source_prediction,self.predict(rgb))
        row.update(convnext_true_class_name=self.names[int(true_label)],
                   convnext_source_prediction_name=self.names[int(source_prediction)],
                   convnext_prediction_name=self.names[row['convnext_prediction']],
                   convnext_used_for_selection=False,convnext_weight_sha256=self.identity['weights_sha256'])
        return row

def _groups(rows,source_indices,seeds):
    require(source_indices and seeds and len(set(source_indices))==len(source_indices)
            and len(set(seeds))==len(seeds),'Nonempty unique source and noise populations required')
    groups=defaultdict(dict);original={};expected={(i,s) for i in source_indices for s in seeds}
    for row in rows:
        group=(int(row['N']),float(row['snr_db']),row['method'])
        key=(int(row['source_index']),int(row['noise_seed']))
        require(key in expected and key not in groups[group],'Missing, duplicate or unexpected validation frame')
        check=diagnostics(row['convnext_true_class'],row['convnext_source_prediction'],row['convnext_prediction'])
        require(all(type(row.get(k)) is bool and row[k]==check[k] for k in check if isinstance(check[k],bool)),
                'Transition/accuracy booleans disagree with class predictions')
        require(type(row.get('convnext_source_prediction_agreement')) is int
                and row['convnext_source_prediction_agreement']==check['convnext_source_prediction_agreement'],
                'Canonical independent agreement must be integer zero/one matching the predictions')
        require(row.get('convnext_used_for_selection',False) is False,'Independent metric entered selection')
        reference=(int(row['convnext_true_class']),int(row['convnext_source_prediction']))
        old=original.setdefault(key[0],reference);require(old==reference,'Original-image classifier prediction changed across methods/noises')
        groups[group][key]=row
    require(groups,'No development validation rows')
    for group in groups:require(set(groups[group])==expected,'Incomplete source/noise group: '+str(group))
    return groups

def interval(values,replicates=10000,seed=20261002):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return dict(mean=None,ci_low=None,ci_high=None,n_sources=0)
    require(a.ndim==1 and np.isfinite(a).all(),'Invalid source values')
    draws=np.random.default_rng(seed).integers(0,len(a),(replicates,len(a)))
    lo,hi=np.quantile(a[draws].mean(1),[.025,.975])
    return dict(mean=float(a.mean()),ci_low=float(lo),ci_high=float(hi),n_sources=len(a))

def summarize(rows,*,source_indices=tuple(range(100)),seeds=(2001,2002,2003)):
    groups=_groups(rows,source_indices,seeds);summaries=[];transitions=[]
    for (N,snr,method),by in sorted(groups.items()):
        shared=dict(N=N,snr_db=snr,method=method,n_frames=len(by),used_for_selection=False)
        for metric in METRICS:
            values=[np.mean([by[i,s][metric] for s in seeds]) for i in source_indices]
            summaries.append(dict(shared,metric=metric,**interval(values)))
        for correct,field in ((True,'convnext_source_correct_to_wrong'),(False,'convnext_source_wrong_to_correct')):
            eligible=[i for i in source_indices if by[i,seeds[0]]['convnext_source_correct']==correct]
            numerator=sum(int(by[i,s][field]) for i in eligible for s in seeds)
            values=[np.mean([by[i,s][field] for s in seeds]) for i in eligible]
            transitions.append(dict(shared,transition=field,numerator_frames=numerator,
                denominator_frames=len(eligible)*len(seeds),denominator_sources=len(eligible),
                denominator_definition='original classifier correct' if correct else 'original classifier wrong',
                bootstrap_population='eligible original-correctness source stratum',**interval(values)))
    return summaries,transitions

def paired(rows,method_A,method_B,*,source_indices=tuple(range(100)),seeds=(2001,2002,2003)):
    groups=_groups(rows,source_indices,seeds);result=[]
    cells={(n,s) for n,s,m in groups if m in (method_A,method_B)}
    for N,snr in sorted(cells):
        require((N,snr,method_A) in groups and (N,snr,method_B) in groups,'Paired method missing')
        a,b=groups[N,snr,method_A],groups[N,snr,method_B]
        for metric in METRICS:
            delta=[np.mean([float(a[i,s][metric])-float(b[i,s][metric]) for s in seeds]) for i in source_indices]
            result.append(dict(N=N,snr_db=snr,method_A=method_A,method_B=method_B,metric=metric,
                delta='A_minus_B',n_frames=len(source_indices)*len(seeds),used_for_selection=False,
                bootstrap_unit='source_mean_over_three_noise_repeats',bootstrap_replicates=10000,
                bootstrap_seed=20261002,**interval(delta)))
    return result
