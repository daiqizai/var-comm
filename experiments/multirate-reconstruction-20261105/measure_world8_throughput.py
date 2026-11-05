"""Bounded read-only observation of retained main-training updates."""
import json,statistics,time
from runtime import L,write


def main():
    out=L/'training_world8_v1';transition=L/'migration_4to8_v1'
    request=json.loads((out/'request.json').read_text());resume=request['resume']['step']
    assert json.loads((out/'first_update_audit.json').read_text())['passed']
    anchor=None;deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        assert not (L/'controllers/training_world8_v1/actual_wait.json').exists(),'Training exited during observation'
        p=json.loads((out/'progress.json').read_text())
        if anchor is None and p['step']>=resume+100:
            anchor=p;write(transition/'throughput_anchor.json',anchor)
        if anchor and p['step']>=anchor['step']+300:
            final=p;break
        time.sleep(1)
    else:raise RuntimeError('Read-only throughput window timed out')
    records=[]
    with (out/'updates_rank0.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if anchor['step']<r['step']<=final['step']:records.append(r)
    assert len(records)==final['step']-anchor['step']
    four=json.loads((transition/'four_gpu_timing.json').read_text())
    eight=(final['elapsed_seconds']-anchor['elapsed_seconds'])/(final['step']-anchor['step'])
    result=dict(world=8,global_batch=32,anchor=anchor,final=final,retained_updates=len(records),
          elapsed_seconds=final['elapsed_seconds']-anchor['elapsed_seconds'],mean_seconds_including_checkpoints=eight,
          mean_compute_seconds=statistics.mean(r['seconds'] for r in records),median_compute_seconds=statistics.median(r['seconds'] for r in records),
          four_gpu_mean_seconds_including_checkpoints=four['mean_seconds_including_checkpoints'],
          observed_throughput_ratio=four['mean_seconds_including_checkpoints']/eight,
          comparison='Same global batch/model/loss/precision; four-GPU completed segment versus eight-GPU window with NUMA-local CPU affinity; different source batches',
          first_update_qualification_excluded=True,new_model_calls=0,additional_training_updates_for_benchmark=0)
    write(transition/'throughput_comparison.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
