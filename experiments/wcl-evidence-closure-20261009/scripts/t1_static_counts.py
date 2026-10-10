"""CPU-only exact original-train20k scale counts from existing full_tokens.

Reads immutable cached tensors with weights_only=True. No tokenizer/model call.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS, SIZES, load_integer_codec, read, require, sha, verify, write

def run(args):
    # Hide CUDA before torch import. No torch model is constructed.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    root=Path(args.root); cache=Path(args.cache); out=Path(args.out)
    require(not out.exists(), 'Fresh output required; preserve earlier attempt')
    registration=read(cache/'registration.json'); done=read(cache/'completion.json')
    stats=read(cache/'training_statistics.json'); source=read(args.image_manifest)
    training=read(args.training_registration)
    require(done['status']=='EXACT_FP32_LATENT_CACHE_COMPLETE'
            and done['source_counts']['train']==20000, 'Original complete20k cache required')
    verify(cache/'registration.json',done['registration_sha256'])
    verify(cache/'training_statistics.json',done['statistics_sha256'])
    verify(args.image_manifest,registration['image_manifest_sha256'])
    require(registration['precision']=='float32' and registration['old_fp16_fhat_used'] is False,
            'Exact full-token cache required')
    require(source['vae_checkpoint_sha256']=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c',
            'Tokenizer file identity differs')
    ids=training['train_ids']; require(len(ids)==len(set(ids))==20000, 'Exact original20k IDs required')
    calibration=set(training['calibration_ids']); require(not set(ids)&calibration,'Train/calibration IDs overlap')
    population=[p for p in source['populations'] if p['name']=='train']
    require(len(population)==1 and population[0]['count']==20000,'Original image manifest train20k absent')
    shards=population[0]['shards']; require(len(shards)==200,'Expected original200 cached shards')
    entropy,_,_=load_integer_codec(root)
    counts=np.zeros((10,4096),dtype=np.int64); checked=[]; offset=0
    out.mkdir(parents=True)
    inputs={str(Path(p).resolve()):sha(p) for p in [cache/'registration.json',cache/'completion.json',
        cache/'training_statistics.json',args.image_manifest,args.training_registration]}
    for index,descriptor in enumerate(shards):
        path=cache/'train'/f'shard_{index:04d}.pt'; receipt_path=path.with_suffix('.json'); receipt=read(receipt_path)
        relative=path.relative_to(cache).as_posix()
        require(stats['training_shard_sha256'].get(relative)==receipt['sha256'], 'Training shard not sealed by completed statistics')
        verify(path,receipt['sha256'])
        require(receipt['source_shard_sha256']==descriptor['sha256'],'Original source shard mapping changed')
        record=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        names=[str(x) for x in record['image_ids']]; count=len(names)
        require(count==receipt['count']==descriptor['count'] and names==ids[offset:offset+count],
                'Train IDs/order differ from original registration')
        require(record['source_descriptor']==descriptor,'Cached original source descriptor differs')
        require(np.array_equal(record['source_indices'].cpu().numpy(),np.arange(offset,offset+count)),
                'Cached source indices differ')
        tokens=record['full_tokens'].cpu().numpy()
        require(tokens.shape==(count,680) and np.issubdtype(tokens.dtype,np.integer)
                and np.all((tokens>=0)&(tokens<4096)),'All ten scales of original full_tokens required')
        # Full-token tensors are retokenized with the same frozen tokenizer.
        # The legacy base cache can use a pinned prefix: report its discrepancy,
        # never splice its token prefix into full_tokens.
        base=record['base_tokens'].cpu().numpy()
        actual_difference=int(np.count_nonzero(tokens[:,:255]!=base))
        require(actual_difference==receipt['retokenized_prefix_difference_count'],'Recorded prefix discrepancy differs')
        for s in range(10):
            counts[s]+=np.bincount(tokens[:,OFFSETS[s]:OFFSETS[s+1]].ravel(),minlength=4096)
        checked.append(dict(path=str(path),sha256=receipt['sha256'],receipt_sha256=sha(receipt_path),
            source_count=count,source_start=offset,source_shard_sha256=descriptor['sha256'],
            retokenized_prefix_difference_count=actual_difference))
        offset+=count; del record,tokens,base
        if index%20==19:
            print(f'verified training token shards {index+1}/200; sources {offset}/20000',flush=True)
    require(offset==20000 and np.array_equal(counts.sum(axis=1),20000*np.square(SIZES)), 'Scale count coverage differs')
    # One fixed Jeffreys +0.5 pseudocount, then the already-bound positive24bit CDF.
    cdfs=entropy.probability_cdf(np.log(counts.astype(np.float64)+0.5))
    archive=out/'static_scale_counts_cdf.npz'
    with archive.open('xb') as f: np.savez(f,counts=counts,cdf=cdfs)
    completion=dict(status='T1_STATIC_TRAIN20K_CDF_COMPLETE',source_role='original_train20k',source_count=20000,
        scales=list(range(1,11)),sizes=list(SIZES),token_array='full_tokens',original_images_only=True,
        augmentation_used=False,calibration_used_for_fitting=False,training_updates=0,new_model_calls=0,
        new_packet_decodes=0,statistical_bootstrap_calls=0,smoothing_pseudocount=0.5,cdf_total=1<<24,
        cdf_rule='floor_mass_plus_one_remainder_first_argmax',tokenizer_file_sha256=source['vae_checkpoint_sha256'],
        input_bindings=inputs,shards=checked,source_ids=ids,outputs={str(archive.resolve()):sha(archive)},
        retokenized_prefix_difference_count=sum(r['retokenized_prefix_difference_count'] for r in checked),
        tensor_values_used=['full_tokens'],no_latent_or_generated_output_used=True)
    write(out/'completion.json',completion)
    return completion

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','cache','image-manifest','training-registration','out']: p.add_argument('--'+name,required=True)
    args=p.parse_args(); result=run(args); print(result['status'])

if __name__=='__main__': main()
