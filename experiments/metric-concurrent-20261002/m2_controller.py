"""Resume the frozen scientific controller with scheduling-only child wrappers.

The operator has already requested a normal checkpoint and retired its previous
owner. This process sends no signals to any scientific worker or controller.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from scheduling_guard import validate_admission, read, write, sha, sources, verify, identity, alive, process, admission_lock


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    a = p.parse_args(); root = Path(a.root).resolve()
    out = root / 'outputs/METRIC-CONCURRENT-R4-20261002'; here = Path(__file__).resolve().parent
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    ownlock = (out/'m2_controller.lock').open('a')
    fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission = validate_admission(root, out / 'admission.json', here)
    handoff = read(out / 'm2_handoff.json')
    for key in ('oldsupervisor','oldworker'):
        expected = identity(handoff[key])
        if alive(expected, process(expected['pid'])):
            raise RuntimeError('Original M2 owner/worker must have exited at its normal checkpoint')
    verify(handoff['checkpoint_bindings'])
    verify(admission['original_execution_bindings'])
    source = root / 'experiments/metric-speed-20261002/controller.py'
    if admission['original_execution_bindings'].get(str(source)) != sha(source):
        raise RuntimeError('Frozen scientific controller is not bound by admission')
    spec = importlib.util.spec_from_file_location('_frozen_m2_scheduling_controller', source)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    me = process(os.getpid()); launch = dict(**identity(me), command=[sys.executable,'-u',str(here/'m2_controller.py'),'--root',str(root)],
        admission_sha256=sha(out/'admission.json'), source_bindings=sources(here), handoff_sha256=sha(out/'m2_handoff.json'),
        scientific_controller=str(source), scientific_controller_sha256=sha(source), time=time.time())
    previous = out / 'm2_controller_launch.json'
    if previous.exists():
        old = read(previous)
        if alive(old, process(old['pid'])): raise RuntimeError('A scheduling owner is still active')
    write(previous, launch)

    def pause(reason):
        write(out/'pause_requested.json',dict(status='PAUSE_REQUESTED',reason=reason,time=time.time(),
            controller_pid=me['pid'],controller_start_ticks=me['start_ticks'],admission_sha256=launch['admission_sha256']))

    def execute(argv, log, started):
        verify(admission['source_bindings']); verify(admission['original_execution_bindings'])
        wrapper = root/'experiments/metric-speed-20261002/wrapper.py'
        if Path(argv[2]).resolve() == wrapper:
            if argv[3:5] != ['--stage',argv[-1]] or argv[-1] not in ('calibration','evaluation','actual','timing'):
                raise RuntimeError('Unexpected original scientific arguments')
            stage = argv[-1]
            if stage == 'timing':
                with admission_lock(out):
                    write(out/'exclusive_timing_requested.json',dict(status='EXCLUSIVE_TIMING_REQUESTED',
                        owner=identity(me),admission_sha256=launch['admission_sha256'],time=time.time()))
                pause('ORIGINAL_TIMING_REQUIRES_EXCLUSIVE_GPU')
                while (out/'active_metrics.json').exists():
                    peer = read(out/'active_metrics.json')
                    if peer.get('status') != 'ACTIVE' or not alive(peer,process(peer['pid'])): break
                    verify(admission['source_bindings']); time.sleep(5)
            argv = [sys.executable,'-u',str(here/'scheduled_process.py'),'--root',str(root),
                    '--admission',str(out/'admission.json'),'--stage',stage]
        with Path(log).open('a') as stream:
            child = subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            started(child.pid); code = child.wait()
        if code == 75:
            pause('ORIGINAL_M2_RESOURCE_BOUNDARY')
            write(out/'m2_resource_wait.json',dict(status='RESOURCE_WAIT',time=time.time(),log=str(log)))
        return code

    class Resumed(original.Controller):
        def takeover(self):
            super().takeover()
            path = self.parent/'supervisor_launch.json'; value = read(path)
            value.update(command=launch['command'],scheduling_extension_launch=str(previous),
                scheduling_extension_launch_sha256=sha(previous),scheduling_only=True)
            write(path,value)
    controller = Resumed(root,source.parent,execute=execute)
    try: controller.run()
    finally:
        write(out/'m2_controller_exit.json',dict(status='EXITED',**identity(me),time=time.time()))


if __name__ == '__main__': main()
