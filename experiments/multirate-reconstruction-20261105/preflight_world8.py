"""CPU-only audit of the real saved state and global sample placement."""
import hashlib,json,sys
from pathlib import Path
from runtime import L,OLD,sha,write,setup
setup()
import torch
from train_world8_v1 import Corpus
from world_migration_v1 import positions,new_rank_seed


def main():
    torch.set_num_threads(2)
    out=L/'migration_4to8_v1'
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    request=json.loads((L/'training_segment02/request.json').read_text())
    assert sha(checkpoint['path'])==checkpoint['sha256']
    state=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
    assert state['world_size']==4 and len(state['rng_by_rank'])==4
    assert state['step']==checkpoint['step'] and state['cursor']==checkpoint['cursor']
    assert state['schedule']==request['schedule'] and state['global_batch']==32
    assert sha(state['corpus_path'])==state['corpus_sha256']
    assert len(state['model'])>0 and len(state['optimizer']['state'])>0
    corpus=Corpus(state['corpus_path'],state['seed']);rows={};checks=[]
    for size,microbatch in [(256,4),(512,2)]:
        small=positions(state['cursor'],32,4,microbatch)
        large=positions(state['cursor'],32,8,microbatch)
        assert sorted(sum(small,[]))==sorted(sum(large,[]))==list(range(state['cursor'],state['cursor']+32))
        samples=[];image_hashes=[]
        for position in range(state['cursor'],state['cursor']+32):
            image,record=corpus.image(position,size)
            mode=(7 if state['step']%4==0 else position%8)+(0 if size==256 else 8)
            samples.append(dict(record,mode=mode,position=position))
            image_hashes.append(hashlib.sha256(image.numpy().tobytes()).hexdigest())
        rows[size]=samples
        checks.append(dict(resolution=size,microbatch=microbatch,world4_accumulation=32//(4*microbatch),
            world8_accumulation=32//(8*microbatch),world4_positions=small,world8_positions=large,
            actual_transformed_image_hashes=image_hashes,source_records=samples,global_samples_identical=True))
    seeds={str(rank):new_rank_seed(checkpoint['sha256'],rank) for rank in range(4,8)}
    assert len(set(seeds.values()))==4
    size=256 if state['step']<state['schedule'][0]['updates'] else 512
    write(out/'preflight.json',dict(passed=True,checkpoint=checkpoint,restored_step=state['step'],restored_cursor=state['cursor'],
          global_batch=32,from_world=4,to_world=8,corpus_images=len(corpus.rows),corpus_path=corpus.path,corpus_sha256=corpus.identity,
          schedule=state['schedule'],seed=state['seed'],old_logical_rank_streams_retained=[0,1,2,3],new_rank_seeds=seeds,
          placement_checks=checks,expected_first_batch=rows[size],new_model_updates=0,new_GPU_calls=0,
          old_P2_still_paused=(OLD/'STOP_P2').exists(),bitwise_four_card_trajectory_claim=False))
    print(json.dumps(dict(passed=True,step=state['step'],global_batch=32,global_samples_identical=True,new_GPU_calls=0)))


if __name__=='__main__':main()
