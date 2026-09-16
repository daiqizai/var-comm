"""Read one atomic checkpoint and independently verify retained interface pairing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.common import now, sha256, verify_sources, write_json
from wetok_comm.interface_study import interface_definitions, interface_output, load_interface_study
from wetok_comm.training import paired_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-updates', type=int, default=100)
    parser.add_argument('--output', type=Path, default=EXPERIMENT / 'docs/interface_first_checkpoint_audit.json')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    study, base, parent = load_interface_study()
    if not arguments.execute:
        print('PLAN ONLY: audit atomic paired data, independent model/Adam storage and exact retained counters')
        return
    if arguments.output.exists():
        raise FileExistsError('preserve prior audit evidence and choose another output file')
    torch.set_num_threads(2)
    training = interface_output(study, 'training')
    metadata = json.loads((training / 'metadata.json').read_text())
    verify_sources(metadata['source_hashes'])
    profile_path = interface_output(study, 'profile') / 'profile.json'
    if sha256(profile_path) != metadata['profile_sha256']:
        raise RuntimeError('profile binding changed after training initialization')
    profile = json.loads(profile_path.read_text())
    verify_sources(profile['source_hashes'])
    with (training / 'resume.pt').open('rb') as handle:
        saved = torch.load(handle, map_location='cpu', weights_only=True)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    completed = saved['completed_additional_updates']
    if completed < arguments.minimum_updates or saved['global_data_step'] != study['parent_step'] + completed:
        raise RuntimeError('requested complete update boundary is not yet available')
    if saved['frozen'] != profile['frozen']:
        raise RuntimeError('frozen visual binding changed')
    definitions = interface_definitions(study)
    storages = {}
    for name in definitions:
        if len(saved['histories'][name]) != completed:
            raise RuntimeError('unequal per-arm retained update budgets')
        states = saved['optimizers'][name]['state']
        if not states or any(int(value['step']) != completed for value in states.values()):
            raise RuntimeError('fresh Adam counters do not equal retained additional updates')
        values = list(saved['models'][name].values())
        values.extend(value[key] for value in states.values() for key in ('exp_avg', 'exp_avg_sq'))
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError('nonfinite model or Adam state at the saved boundary')
        storages[name] = {value.untyped_storage().data_ptr() for value in values if value.numel()}
    names = list(definitions)
    for first, name in enumerate(names):
        for other in names[first + 1:]:
            if storages[name] & storages[other]:
                raise RuntimeError('comparison arms share model or optimizer tensor storage')
    power_error = 0.
    for local, batch in enumerate(paired_batches(base, 20000, study['parent_step'], study['parent_step'] + completed)):
        for name in definitions:
            row = saved['histories'][name][local]
            if (row['additional_step'] != local + 1 or row['global_data_step'] != batch['step'] + 1 or
                row['Adam_step'] != local + 1 or row['batch_sha256'] != batch['fingerprint']):
                raise RuntimeError('source order/flip/SNR/noise or counters differ across arms')
            power_error = max(power_error, row['power_max_error'])
    if power_error > 1e-5:
        raise RuntimeError('paid signal energy changed')
    record = {'status': 'INTERFACE_PAIRED_CHECKPOINT_AUDIT_PASS', 'checked_local': now(),
        'retained_additional_updates_per_arm': completed, 'global_data_step': saved['global_data_step'],
        'actual_Adam_steps_per_parameter': completed, 'arms': len(definitions),
        'model_and_Adam_storages_independent': True, 'paired_source_augmentation_SNR_noise_verified': True,
        'maximum_power_error': power_error, 'checkpoint_sha256_at_read': digest.hexdigest(),
        'training_profile_source_bindings_unchanged': True, 'new_development_receiver_outputs_evaluated': 0,
        'last_training_losses_not_quality_results': {name: saved['histories'][name][-1]['loss'] for name in definitions}}
    write_json(arguments.output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
