"""Small integration API for frozen P1024 at the new 10 dB development point.

The caller owns registration, persistence, GPU scheduling and final scoring.
Only this new SNR is inferred. Existing 4/7/13 dB float archives are read with
their registered hashes; an absent cache is reported, never regenerated here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from validation_metrics import admission,require,sha

DEV_SEEDS=(2001,2002,2003)
SNR=10
VERSION='P1024_selected40000_new10dB_original_channel_v1'


def rgb(value):
    a=np.asarray(value)
    require(a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all()
            and a.min()>=0 and a.max()<=1,'Unmodified finite float32 CHW256 RGB required')
    return np.ascontiguousarray(a)


def rgb_sha(value):
    return hashlib.sha256(b'float32:3,256,256:RGB\0'+rgb(value).tobytes()).hexdigest()


def selected_identity(adapter):
    selected=adapter.pmeta['selected']
    require(selected['N']==1024 and selected['step']==40000 and adapter.p.uses==1024,
            'Only the frozen final P1024 40000-step checkpoint is admitted')
    require(adapter.pmeta['selected_sha256']==adapter.pcfg['identity']['selected_sha256'],
            'Historical P1024 selection receipt differs')
    return dict(selected_step=40000,selected_checkpoint_sha256=selected['checkpoint_sha256'],
        selected_receipt_sha256=adapter.pmeta['selected_sha256'],
        decoder_sha256=adapter.pmeta['decoder_sha256'],
        model_state_sha256=adapter.pcfg['identity']['models']['P1024'],
        original_selected_record=copy.deepcopy(selected))


class P1024Frozen10dB:
    """Construct after UEP policies freeze, using the admitted original Native.

    ``infer_source`` returns three rows, a float32 [3,3,256,256] image array and
    the original float32 reference. Save these losslessly before new scoring.
    ``finish`` verifies original source/model bindings after the last source.
    No lookup or plugin policy is read at 10 dB.
    """
    def __init__(self,native,*,stage,policy_path,policy_sha256):
        admission(stage,policy_path,policy_sha256)
        self.native=native;self.engine=native.replay_engine()
        self.adapter=self.engine.adapters['N1024'];self.torch=native.torch
        self.device=self.engine.loaded['device']
        require(len(self.engine.records)==100,'Original 100 development sources required')
        self.identity=dict(version=VERSION,**selected_identity(self.adapter),
            N=1024,snr_db=SNR,noise_seeds=list(DEV_SEEDS),policy_sha256=policy_sha256,
            source_role='development',source_count=100,training_updates=0,
            calibration_executed=False,policy_selection_updates=0,new_operating_point_only=True,
            channel_namespace='original token_efficiency.execution.apply_channel / plugin.CELL',
            paired_to_UEP_by='source and nominal noise seed; channel noise namespaces differ',
            receiver_batch_size=1,decoder_batch_size=3,numerical_runtime=native.flags)
        require(not self.adapter.p.training and not any(p.requires_grad for p in self.adapter.p.parameters()),
                'Original P1024 must remain frozen')
        native.frozen()

    def infer_source(self,index,*,guard=lambda:None):
        require(type(index) is int and 0<=index<100,'Invalid development source index')
        guard();e,p,t=self.engine,self.adapter,self.torch
        require(not t.is_autocast_enabled(),'Preserve the original non-autocast P1024 numerical path')
        record=e.records[index];target=rgb(e.target(index))
        with t.no_grad():
            f=e.data['F'][index:index+1].to(self.device)
            wave=p.p.transmit(f)[0].cpu().numpy()
            require(wave.shape==(1024,2) and np.isfinite(wave).all(),'Original continuous waveform changed shape')
            energy=float(np.square(wave.astype(np.float64)).sum())
            require(np.isclose(energy,2048,rtol=1e-5,atol=.02),'P1024 frame energy differs from 2N')
            latent=[];observations=[]
            for seed in DEV_SEEDS:
                guard()
                y=p.plugin.apply_channel(wave,SNR,record['image_id'],seed,p.plugin.CELL)
                require(y.shape==wave.shape and np.isfinite(y).all(),'Invalid original AWGN observation')
                z=p.p.receive(t.tensor(y[None],device=self.device),
                    t.tensor([SNR],dtype=t.float32,device=self.device))[0].cpu()
                latent.append(z);observations.append(p.plugin.waveform_sha(y))
            Z=t.stack(latent);guard()
            require(tuple(Z.shape)==(3,32,16,16) and bool(t.isfinite(Z).all()),'Invalid P1024 receive latent')
            measured,images=p.plugin.images_metrics(Z,t.tensor([index]*3),e.records,p.models,
                e.data['reference'].to(self.device),t.tensor(e.mismatch),save_images=True)
            require(len(measured)==len(images)==3,'Original decoder omitted a noise repeat')
            images=np.stack([rgb(image.numpy()) for image in images])
            # Preserve historical F-error units and double-before-subtraction.
            errors=(Z.double()-e.data['F'][index:index+1].cpu().double()).square().flatten(1).sum(1).tolist()
        rows=[]
        for slot,seed in enumerate(DEV_SEEDS):
            row=dict(source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
                true_class_index=int(record['class_index']),N=1024,snr_db=SNR,noise_seed=seed,
                method='P1024',phy_family='continuous',condition_label=False,
                replay_row_id=f'P1024_new10dB_source{index:03d}_noise{seed}',
                image_slot=slot,image_sha256=rgb_sha(images[slot]),reference_sha256=rgb_sha(target),
                E=energy,waveform_sha256=p.plugin.waveform_sha(wave),observation_sha256=observations[slot],
                latent_sq_err_final=float(errors[slot]),F_error=float(errors[slot]),
                F_error_definition='sum squared latent error over 32x16x16 entries; float64 subtraction',
                selected_step=40000,selected_checkpoint_sha256=self.identity['selected_checkpoint_sha256'],
                decoder_sha256=self.identity['decoder_sha256'],historical_scalar_metrics=measured[slot],
                calibration_executed=False,training_updates=0,policy_selection_updates=0,
                noise_namespace=self.identity['channel_namespace'])
            rows.append(row)
        return rows,images,target

    def finish(self):
        self.native.frozen()
        current=selected_identity(self.adapter)
        require(current=={k:self.identity[k] for k in current},
                'P1024 identity changed during inference')
        require(self.adapter.plugin.old.b.state_sha256(self.adapter.p)==self.identity['model_state_sha256'],
                'P1024 weights changed during inference')
        return self.engine.verify_frozen()


def known_cache_paths(root,index):
    """Candidate locations, not claims that those incomplete exports exist."""
    root=Path(root);base=root/'outputs/EXTERNAL-COMPARISON-20261004'
    return [base/f'own_controls/export-n1024/source_checkpoints/{index:04d}.json',
        base/f'fixed80k_revision/five_column_comparison/replay_all16/source_checkpoints/{index:03d}.json',
        base/f'fixed80k_revision/m1_headline_16qam13/replay_all16/source_checkpoints/{index:03d}.json']


def read_existing_cache(checkpoint,*,checkpoint_sha256,source_index,reference_sha256):
    """Read only admitted P1024 rows at 4/7/13 dB. No neural fallback.

    Returns {status, rows, images, source_rgb, ...}. Images retain original
    slots; caller deduplicates overlapping candidate caches by frame identity.
    ``checkpoint_sha256`` must come from a frozen export completion inventory.
    """
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():
        return dict(status='CACHE_MISSING',checkpoint=str(checkpoint),rows=[])
    require(len(checkpoint_sha256)==64 and sha(checkpoint)==checkpoint_sha256,'Existing float-cache checkpoint changed')
    value=json.loads(checkpoint.read_text(encoding='utf-8'))
    require(value['source_index']==source_index and value.get('synthetic') is False,'Existing cache source/provenance differs')
    proof=value['float_reconstructions'];path=Path(proof['path'])
    require(sha(path)==proof['sha256'] and proof.get('lossless') is True,'Existing float archive changed or is lossy')
    with np.load(path,allow_pickle=False) as archive:
        images=archive['images'].copy()
        require(images.dtype==np.float32 and images.ndim==4 and images.shape[1:]==(3,256,256),'Invalid float image archive')
        fingerprints=[rgb_sha(a) for a in images]
        if set(archive.files)=={'images','native_reference','comparison_reference'}:
            source=rgb(archive['comparison_reference']).copy()
            require(fingerprints==proof['image_sha256'] and rgb_sha(source)==proof['comparison_reference_sha256']
                and rgb_sha(archive['native_reference'])==proof['native_reference_sha256'],'Cached RGB proof differs')
            image_field='image_sha256'
        else:
            require(set(archive.files)=={'images','source_rgb','row_ids','image_slots'},'Unrecognized existing float-cache schema')
            source=rgb(archive['source_rgb']).copy();image_field='float_rgb_sha256'
            require(archive['row_ids'].tolist()==[r['replay_row_id'] for r in value['rows']]
                and archive['image_slots'].tolist()==[r['image_slot'] for r in value['rows']],
                'Old row IDs/image-slot array differs')
        require(rgb_sha(source)==reference_sha256,'Existing cache reference differs from admitted original')
        selected=[];seen=set()
        for row in value['rows']:
            slot=row['image_slot']
            require(type(slot) is int and 0<=slot<len(images) and row[image_field]==fingerprints[slot]
                    and row['reference_sha256']==reference_sha256,'Image/source association differs')
            if row['method'] not in ('P','P1024'):continue
            old=row.get('original_scientific_row',row.get('original_row',{}))
            require(old.get('method')=='P1024' and int(row['N'])==1024,'A non-P1024 row claimed the baseline name')
            snr,seed=int(row['snr_db']),int(row['noise_seed'])
            if snr not in (4,7,13) or seed not in DEV_SEEDS:continue
            require((snr,seed) not in seen,'Duplicate existing P1024 frame');seen.add((snr,seed))
            selected.append(dict(row,method='P1024',image_sha256=fingerprints[slot],
                existing_cache_checkpoint_sha256=checkpoint_sha256,existing_cache_npz_sha256=proof['sha256']))
    return dict(status='CACHE_AVAILABLE',checkpoint=str(checkpoint),rows=selected,
                images=images,source_rgb=source,covered_frames=sorted(seen))
