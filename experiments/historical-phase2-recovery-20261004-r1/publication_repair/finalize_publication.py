"""Finish the computed R1 delivery with an explicitly separate, cold audit repair.

No scientific source, queue, checkpoint, image, metric or tolerance is changed.
This process owns only final verification, normal publication and completion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

NAME = 'HISTORICAL-PHASE2-RECOVERY-20261004-R1'
QSHA = 'df619855451424e0329aa592d564a4b18d034bddf52cb48123534c8a13479c80'
FILES = ('finalize_publication.py', 'phase2_final_audit.py', 'test_phase2_final_audit.py', 'README.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def register(root, here, queue, hc):
    out = root / 'outputs' / NAME
    path = here / 'registration.json'
    bindings = {str(here / name): sha(here / name) for name in FILES}
    prior = here / 'prior_publication_failure'
    if not path.exists():
        snapshot = hc.process_snapshot()
        for file in (out / 'controller_launch.json', out / 'controller_dispatch.json',
                     *sorted((out / 'launches').glob('*.json'))):
            hc.require_exited(read(file), snapshot, 'Completed R1 owner ' + file.name)
        require(read(out / 'controller_status.json')['status'] == 'FAILED', 'Review the prior controller state')
        require("KeyError: 'dino_mismatched'" in (out / 'result_publication.log').read_text(),
                'This repair admits only the reviewed absent legacy metric field')
        require(not (out / 'results_publication.json').exists(), 'Existing result publication needs separate review')
        prior.mkdir(exist_ok=False)
        for name in ('controller_launch.json', 'controller_dispatch.json', 'controller_status.json',
                     'results_publication_failure.json', 'result_publication.log'):
            source = out / name
            if source.exists():
                shutil.copy2(source, prior / name)
        bindings.update({str(p): sha(p) for p in prior.iterdir() if p.is_file()})
        checks = read(here / 'cpu_checks.json')
        require(checks['status'] == 'PASS' and checks['source_bindings'] ==
                {str(here / name): sha(here / name) for name in FILES}, 'Repair tests must bind exact sources')
        bindings[str(here / 'cpu_checks.json')] = sha(here / 'cpu_checks.json')
        smoke = read(here / 'actual_audit_smoke.json')
        require(smoke.get('status') == 'PHASE2_RECOVERY_100_SOURCES_VERIFIED'
                and smoke.get('rows') == 4500 and smoke.get('new_pure_strict_grid_proofs') == 285,
                'Complete real-data final audit must pass before publication')
        hc.verify(smoke['bindings'])
        bindings[str(here / 'actual_audit_smoke.json')] = sha(here / 'actual_audit_smoke.json')
        write(path, dict(status='REGISTERED_PUBLICATION_ONLY_REPAIR', queue_sha256=QSHA,
            bindings=bindings, scientific_runtime_source_bindings=queue['source_bindings'],
            change='Final legacy parity audit enumerates METRICS, not unrelated optional TOLERANCES keys',
            original_data_written=False, images_recomputed=False, metrics_recomputed=False,
            tolerances_changed=False, policy_selection_updates=0, training_updates=0))
    registration = read(path)
    require(registration['queue_sha256'] == QSHA and registration['scientific_runtime_source_bindings'] ==
            queue['source_bindings'], 'Repair registration differs')
    hc.verify(registration['bindings'])
    return registration


def run(root):
    root = Path(root).resolve(); here = Path(__file__).resolve().parent
    out = root / 'outputs' / NAME; runtime = out / 'runtime'; qp = out / 'queue_registration.json'
    require(sha(qp) == QSHA, 'Registered scientific queue changed')
    sys.path.insert(0, str(runtime))
    import history_controller as hc
    import history_publish as pub
    import phase2_final_audit as audit
    queue = hc.validate_queue(root, runtime, qp)
    own = hc.lock(out / 'controller.lock')
    try:
        require(not (here / 'failure.json').exists(), 'Repair failure needs review before another invocation')
        registration = register(root, here, queue, hc)
        require(not hc.gpu_pids(), 'Finish publication only after scientific GPU owners exit')
        who = hc.process_snapshot()[os.getpid()]
        launch = dict(pid=os.getpid(), start_ticks=who['start_ticks'], queue_registration_sha256=QSHA,
            source_bindings=queue['source_bindings'], role='publication_only_repair',
            command=[sys.executable, *sys.argv], repair_registration_sha256=sha(here / 'registration.json'))
        write(here / 'launch.json', launch)
        # Publication inputs stay immutable if a reviewed push continuation is
        # later needed; the live controller receipts always name this process.
        if not (here / 'publication_launch_snapshot.json').exists():
            write(here / 'publication_launch_snapshot.json', launch)
        write(out / 'controller_launch.json', launch)
        write(out / 'controller_dispatch.json', dict(status='CONTROLLER_LAUNCHED', **launch))
        write(out / 'controller_status.json', dict(status='RUNNING', job='reviewed_final_publication', **launch))
        # The original published scientific runtime remains byte-identical.
        # Only this finalizer's explicitly bound verification callable differs.
        pub.verify_complete_phase2 = audit.verify_complete_phase2
        original_prepare = pub.prepare_files

        def prepare(*args):
            files, inputs, canonical = original_prepare(*args)
            destination = root / 'experiments/historical-phase2-recovery-20261004-r1/publication_repair'
            for name in FILES:
                source, target = here / name, destination / name
                pub.copy_exact(source, target); files.append(target)
                canonical[str(target)] = sha(target); inputs[str(source)] = sha(source)
            proof = root / 'results/historical_phase2_recovery_20261004_r1/provenance/publication_repair'
            for name in ('registration.json', 'cpu_checks.json', 'actual_audit_smoke.json',
                         'publication_launch_snapshot.json'):
                source, target = here / name, proof / name
                pub.copy_exact(source, target); files.append(target); inputs[str(source)] = sha(source)
            inputs.update(registration['bindings'])
            return files, inputs, canonical

        pub.prepare_files = prepare
        result = pub.publish(root, qp, 'results', runtime=runtime)
        hc.published(result)
        hc.verify(registration['bindings'])
        complete = dict(**launch, status='HISTORICAL_METRICS_COMPLETE', publication=result,
            training_updates=0, policy_selection_updates=0, stop=True,
            final_publication_repair=str(here / 'registration.json'))
        write(out / 'completion.json', complete)
        write(out / 'controller_status.json', dict(status='COMPLETE', stopped_after_registered_scope=True, **launch))
        write(here / 'completion.json', complete)
        print('COMPLETE_PUSHED', result['commit'], flush=True)
    except BaseException as error:
        failure = dict(status='FAILED_REQUIRES_REVIEW', error=repr(error), time=time.time())
        write(here / 'failure.json', failure)
        write(out / 'controller_status.json', failure)
        raise
    finally:
        own.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
