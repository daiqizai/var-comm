"""One finite segment following normal completion of the first allocation.

Does not reset its predecessor deadline, retry a failure, or migrate world size.
"""
from pathlib import Path
import json
import os
import subprocess
import time

from runtime import CHECKPOINT_ROOT, L, OLD, PY, R, env, sha, write


def live(identity):
    p = Path('/proc') / str(identity['pid']) / 'stat'
    return p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == str(identity['start_ticks'])


def main():
    source = Path(__file__).resolve().parent
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=R, text=True).strip() == 'codex/multirate-reconstruction-20261105'
    assert (OLD / 'STOP_P2').exists()
    assert not (L / 'STOP_MAIN').exists(), 'A user stop must not be cleared by segment admission'
    pause = json.loads((OLD / 'user_pause_verification_v1/complete.json').read_text())
    assert pause['P2_status'] == 'PAUSED_BY_USER_INCOMPLETE'
    prior = L / 'training_v2'
    control = L / 'controllers/training_v2'
    wait = json.loads((control / 'actual_wait.json').read_text())
    assert wait['returncode'] == 0 and wait['child_gone']
    for name in ('owner.json', 'child.json'):
        assert not live(json.loads((control / name).read_text()))
    segment = json.loads((prior / 'segment_complete.json').read_text())
    old = json.loads((prior / 'request.json').read_text())
    assert segment['paused'] and not segment['training_finished']
    checkpoint = json.loads((prior / 'latest_checkpoint.json').read_text())
    assert checkpoint['step'] == segment['step'] == 34027
    for name in ('runtime.py', 'model.py', 'codec.py', 'train_v2.py'):
        assert sha(source / name) == old['bindings'][str(source / name)], name
    for folder in (L / 'controllers').iterdir():
        regpath = folder / 'registration.json'
        if not regpath.exists() or (folder / 'actual_wait.json').exists():
            continue
        reg = json.loads(regpath.read_text())
        assert not set(reg['gpu_devices']).intersection(old['gpus']), folder
    devices = []
    for gpu in old['gpus']:
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[2]) < 100 and int(fields[3]) == 0, fields
        devices.append(fields)
    assert sha(checkpoint['path']) == checkpoint['sha256']
    # Retain this segment boundary despite subsequent replacement of rolling files.
    immutable = CHECKPOINT_ROOT / 'main_v1/segment01_step034027.pt'
    assert not immutable.exists()
    os.link(checkpoint['path'], immutable)
    assert immutable.stat().st_size == checkpoint['bytes']
    frozen = dict(checkpoint, path=str(immutable))
    out = L / 'training_segment02'
    out.mkdir(exist_ok=False)
    request = dict(old, output=str(out), resume=frozen,
                   wall_seconds=39600, owner_max_seconds=43200,
                   training_manifest=checkpoint['corpus'],
                   continuation='separate finite allocation 02 after normal predecessor exit; optimizer/RNG/data/schedule unchanged',
                   predecessor='training_v2',
                   predecessor_original_owner_deadline=old['original_owner_deadline'],
                   predecessor_original_training_deadline=old['original_training_deadline'],
                   allocation_gpu_device_hours_max=48,
                   bindings={str(source / n): sha(source / n) for n in ('runtime.py', 'model.py', 'codec.py', 'train_v2.py', 'resume_segment02.py')})
    request.pop('original_owner_deadline', None)
    request.pop('original_training_deadline', None)
    rp = out / 'request.json'
    write(rp, request)
    write(out / 'admission.json', dict(passed=True, predecessor_actual_wait=wait,
          checkpoint=frozen, predecessor_final_checkpoint_retained=True,
          user_authorization='Continue the new main model through sufficient training; old finite first allocation is not the project ceiling',
          devices_at_admission=devices, world_size=4, global_batch=old['global_batch'],
          unchanged=['model', 'optimizer', 'RNG by rank', 'data cursor', 'global batch', 'loss', 'precision', 'schedule', 'training source'],
          old_P2_still_paused=True, new_max_wall_seconds=43200, new_max_gpu_device_hours=48,
          no_deadline_reset_of_predecessor=True))
    owner = L / 'training_segment02_owner.py'
    assert not owner.exists()
    command = [PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               str(source / 'train_v2.py'), '--request', str(rp)]
    body = 'import sys\nsys.path.insert(0,' + repr(str(source)) + ')\nfrom runtime import *\nsys.path.insert(0,str(R/"experiments/rdvq-prefix-20261104"))\nfrom owned_process import run_role\ne=env();e["CUDA_VISIBLE_DEVICES"]="1,2,4,5"\n'
    body += 'run_role(' + repr(str(L / 'controllers')) + ',"training_segment02",' + repr(command) + ',e,list(range(48,80)),[1,2,4,5],43200,' + repr({**request['bindings'], str(rp): sha(rp)}) + ')\n'
    owner.write_text(body)
    process = subprocess.Popen([PY, str(owner)], stdin=subprocess.DEVNULL,
                stdout=(L / 'training_segment02_owner.stdout').open('xb'),
                stderr=(L / 'training_segment02_owner.stderr').open('xb'),
                start_new_session=True, env=env())
    write(out / 'launch.json', dict(owner_pid=process.pid, time=time.time(),
          resumed_step=checkpoint['step'], request_sha256=sha(rp)))
    print(json.dumps(dict(owner_pid=process.pid, resume_step=checkpoint['step'], new_gpu_device_hours_max=48)))


if __name__ == '__main__':
    main()
