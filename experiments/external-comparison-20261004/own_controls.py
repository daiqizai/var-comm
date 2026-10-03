"""Isolated N2048 whole/entropy calibration and exact selected N1024 replay.

Native CUDA0 VAR environment, never the Swin Torch1.12 environment:
  python own_controls.py --root ROOT --stage all
Individual stages: qualify, screen, calibrate, development, export-n1024.
No old main, training loop, policy writer or publication command is called.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import time
import numpy as np
import own_controls_common as c


class Runner:
    def __init__(self, root, protocol_path):
        self.paths = c.locations(root); self.root = self.paths['root']; self.out = self.paths['out']
        self.result = self.paths['result']; self.protocol_path = Path(protocol_path).resolve()
        self.protocol = c.validate_protocol(c.read(self.protocol_path)); self.stop = False
        self.native = None; self.targets = None; self.target_bindings = None; self.stage = None
        self.bindings = c.source_gate(root, self.protocol_path)

    def stopped(self, *_):
        self.stop = True

    def guard(self):
        if self.stop:
            raise c.PauseRequested('Requested source-safe pause; completed source checkpoints are retained')

    def status(self, state, **fields):
        c.write(self.out/((self.stage or 'controller')+'_status.json'),
            dict(status=state, stage=self.stage, pid=os.getpid(), updated=time.time(), **fields))

    def load(self):
        self.guard()
        if self.native is None:
            self.status('LOADING_FROZEN_NATIVE_MODELS')
            from own_controls_native import Native
            self.native = Native(self.root, self.stopped)
        self.guard()
        return self.native

    def complete_path(self, stage):
        return self.out/(stage+'_completion.json')

    def completed(self, stage):
        c.require(not (self.out/(stage+'_failure.json')).exists(), 'Previous stage failure requires review: '+stage)
        path = self.complete_path(stage)
        if not path.exists():
            return None
        done = c.read(path)
        c.require(done.get('status') == 'OWN_CONTROL_STAGE_COMPLETE' and done.get('stage') == stage
                  and done.get('source_bindings') == self.bindings and done.get('synthetic') is False
                  and done.get('training_updates') == 0 and done.get('holdout_access') is False,
                  'Prior control completion identity differs: '+stage)
        c.verify(done['inputs']); c.verify(done['outputs'])
        return done

    def finish(self, stage, outputs, inputs=None, **fields):
        c.verify(self.bindings)
        done = dict(status='OWN_CONTROL_STAGE_COMPLETE', stage=stage, source_bindings=self.bindings,
            inputs={**self.bindings, **(inputs or {})}, outputs={str(p):c.sha(p) for p in outputs},
            synthetic=False, training_updates=0, holdout_access=False, **fields)
        c.seal(self.complete_path(stage), done); self.status('COMPLETE', **fields)
        return done

    def require_stage(self, stage):
        value = self.completed(stage)
        c.require(value is not None, 'Required real previous stage is incomplete: '+stage)
        return {str(self.complete_path(stage)):c.sha(self.complete_path(stage)), **value['outputs']}

    def registration(self, stage, data, choices=None, seeds=None, extra=None):
        native = self.load(); role = 'development' if stage in ('development', 'export-n1024') else 'calibration'
        value = dict(version=c.VERSION, stage=stage, protocol=self.protocol, source_bindings=self.bindings,
            original_visual_identity=native.loaded['identity'], numerical_runtime=native.flags,
            original_population=native.proofs[role], training_updates=0, holdout_access=False,
            development_used_for_selection=False,
            choices={str(s):[c.action_dict(a) for a in aa] for s,aa in (choices or {}).items()},
            noise_seeds=list(seeds or []), **(extra or {}))
        path = self.out/(stage+'_registration.json'); c.seal(path, value)
        return c.identity(value), {str(path):c.sha(path)}

    def qualify(self):
        if self.completed('qualify'):
            return
        native = self.load(); value = native.qualify()
        path = self.out/'native_qualification.json'; c.seal(path, dict(value, source_bindings=self.bindings))
        self.finish('qualify', [path], sources=4, frames=value['cases'], development_read=False,
                    policy_selection_updates=0, engineering_only=True)

    def choose(self, summary, screen, mapping, shortlist=None):
        cells = []
        for snr in c.SNRS:
            for kind, method in (('whole', c.METHODS[0]), ('entropy', c.METHODS[1])):
                group = [r for r in summary if r['snr_db'] == snr and
                         (r['q'] == 0 if kind == 'whole' else r['q'] == 0 or r['order'] == 'entropy')]
                if not screen:
                    allowed = next(x['action_ids'] for x in shortlist['cells'] if x['snr_db'] == snr and x['method'] == method)
                    group = [r for r in group if r['action_id'] in allowed]
                ranked, reference = self.native.m1.rank(group)
                if screen:
                    whole, _ = self.native.m1.rank([r for r in group if r['q'] == 0])
                    ids = sorted({r['action_id'] for r in ranked[:2]} | {whole[0]['action_id']})
                    cells.append(dict(snr_db=snr, method=method, action_ids=ids, reliability_reference=reference))
                else:
                    winner = ranked[0]
                    cells.append(dict(snr_db=snr, method=method, action=c.action_dict(mapping[winner['action_id']]),
                                      calibration=winner, reliability_reference=reference))
        return cells

    def calibration(self, screen):
        stage = 'screen' if screen else 'calibrate'
        if self.completed(stage):
            return
        inputs = self.require_stage('qualify' if screen else 'screen')
        native = self.load(); data = native.data('calibration')
        grid = native.actions(); mapping = {c.action_id(a):a for a in grid}
        shortlist = None
        if screen:
            choices = {snr:list(grid) for snr in c.SNRS}; count, seeds = 200, (4101,)
        else:
            path = self.result/'N2048_shortlist.json'; shortlist = c.read(path)
            c.require(shortlist.get('development_read') is False and shortlist.get('source_count') == 200,
                      'Screen selection must use only original calibration')
            choices = {}
            for snr in c.SNRS:
                ids = sorted({aid for cell in shortlist['cells'] if cell['snr_db'] == snr for aid in cell['action_ids']})
                choices[snr] = [mapping[aid] for aid in ids]
                c.require(1 <= len(ids) <= 5, 'Original two-method shortlist exceeds its fixed bound')
            inputs[str(path)] = c.sha(path); count, seeds = 1000, c.CAL_SEEDS
        binding, registration = self.registration(stage, data, choices, seeds,
                                                  extra=dict(prerequisite_bindings=inputs, source_count=count))
        inputs.update(registration); allrows = []; outputs = []; unique = 0; measured_seconds = 0.
        start = time.monotonic()
        for index in range(count):
            self.guard(); path = self.out/stage/'source_checkpoints'/f'{index:04d}.json'
            source = data['records'][index]
            if path.exists():
                saved = c.checkpoint(path, binding, index)
            else:
                began = time.monotonic(); rows = []; memo = native.memo()
                for snr, actions in choices.items():
                    for action in actions:
                        for seed in seeds:
                            self.guard()
                            row, _ = native.score(data, index, action, snr, seed, memo)
                            rows.append(row)
                c.validate_rows(rows, index, source, choices, seeds)
                saved = c.sealed_checkpoint(path, dict(binding=binding, source_index=index, rows=rows,
                    unique_receiver_events=len(memo['render']), unique_quality_images=len(memo['metric']),
                    seconds=time.monotonic()-began, development_read=False, synthetic=False))
            c.validate_rows(saved['rows'], index, source, choices, seeds)
            unique += saved['unique_receiver_events']; measured_seconds += saved['seconds']
            allrows.extend(saved['rows']); outputs.append(path)
            self.status('CALIBRATING', sources_complete=index+1, sources=count, frames=len(allrows),
                distinct_receiver_events=unique, source_seconds_mean=measured_seconds/(index+1),
                estimated_remaining_seconds=(count-index-1)*measured_seconds/(index+1), elapsed_seconds=time.monotonic()-start)
        summary = c.summarize(allrows)
        cells = self.choose(summary, screen, mapping, shortlist)
        summary_path = self.result/(stage+'_summary.csv'); c.write_csv(summary_path, summary)
        destination = self.result/('N2048_shortlist.json' if screen else 'N2048_policy.json')
        c.seal(destination, dict(version=c.VERSION, N=2048, phy='QPSK', source_count=count, noise_seeds=list(seeds),
            snrs=list(c.SNRS), cells=cells, registration_binding=binding,
            development_read=False, training_updates=0, original_N1024_policy_changed=False))
        outputs += [summary_path, destination]; native.frozen()
        self.finish(stage, outputs, inputs, sources=count, frames=len(allrows), policy_sha256=c.sha(destination),
                    policy_selection_updates=len(cells), development_read=False,
                    unique_receiver_events=unique, measured_source_seconds=measured_seconds)

    def target_records(self):
        if self.targets is None:
            from step0_reference_prepare import targets_from_completed
            self.targets, self.target_bindings = targets_from_completed(self.root)
        return self.targets

    def targets_for(self, data, index):
        source = data['records'][index]; target = self.target_records()[index]
        native = np.asarray(source['pixels'], dtype=np.float32)/np.float32(255.)
        c.require(source['image_id'] == target['image_id'] and source['preprocessing_id'] == target['preprocessing_id']
                  and int(source['class_index']) == target['class_index']
                  and np.array_equal(np.rint(target['rgb']*255).astype(np.uint8), source['pixels']),
                  'New external reference and original VAR development uint8 identities differ')
        return native, target['rgb']

    @staticmethod
    def add_image(images, slots, image):
        digest = c.rgb_sha(image)
        if digest not in slots:
            slots[digest] = len(images); images.append(c.pixels(image))
        return slots[digest], digest

    def development(self):
        if self.completed('development'):
            return
        inputs = self.require_stage('calibrate'); policy_path = self.result/'N2048_policy.json'; policy = c.read(policy_path)
        c.require(policy.get('development_read') is False and policy.get('source_count') == 1000
                  and policy.get('noise_seeds') == list(c.CAL_SEEDS), 'Full original calibration is required')
        inputs[str(policy_path)] = c.sha(policy_path)
        native = self.load(); data = native.data('development'); self.target_records()
        inputs.update(self.target_bindings)
        cells = {(r['snr_db'], r['method']):r for r in policy['cells']}
        c.require(set(cells) == {(s,m) for s in c.SNRS for m in c.METHODS}, 'Frozen N2048 selected scope is incomplete')
        choices = {s:[native.action(cells[s,m]['action']) for m in c.METHODS] for s in c.SNRS}
        binding, registration = self.registration('development', data, choices, c.DEV_SEEDS,
                                                  extra=dict(prerequisite_bindings=inputs))
        inputs.update(registration); outputs = []; allrows = []
        for index, source in enumerate(data['records']):
            self.guard(); path = self.out/'development/source_checkpoints'/f'{index:04d}.json'
            original_target, comparison_target = self.targets_for(data, index)
            if path.exists():
                saved = c.checkpoint(path, binding, index)
            else:
                rows = []; images = []; slots = {}; memo = native.memo(); began = time.monotonic()
                for snr in c.SNRS:
                    for seed in c.DEV_SEEDS:
                        for method in c.METHODS:
                            self.guard(); action = native.action(cells[snr,method]['action'])
                            row, image = native.score(data, index, action, snr, seed, memo, images=True)
                            slot, fingerprint = self.add_image(images, slots, image)
                            value = dict(row)
                            for metric in ('psnr_db', 'lpips_alex', 'dino_cosine', 'dino_mismatched'):
                                if metric in value:
                                    value['native_'+metric] = value.pop(metric)
                            value.update(method=method, source_class_index=int(source['class_index']),
                                semantic_side_information=False, receiver_class_embedding=1000, decoder='Dc',
                                image_slot=slot, image_sha256=fingerprint, reference_sha256=c.rgb_sha(comparison_target),
                                native_reference_sha256=c.rgb_sha(original_target),
                                policy_sha256=c.sha(policy_path), policy_selected_on_development=False,
                                source_origin='new_registered_N2048_inference', same_header_whole_control=True)
                            rows.append(value)
                cache = c.save_float_cache(self.out/'development/float_reconstructions'/f'{index:04d}.npz',
                                            images, original_target, comparison_target)
                saved = c.sealed_checkpoint(path, dict(binding=binding, source_index=index, rows=rows,
                    float_reconstructions=cache, synthetic=False, seconds=time.monotonic()-began))
            c.validate_development_rows(saved['rows'], index, source, cells, c.sha(policy_path))
            c.verify_float_cache(saved['float_reconstructions'], saved['rows'], original_target, comparison_target)
            allrows.extend(saved['rows']); outputs += [path, Path(saved['float_reconstructions']['path'])]
            self.status('DEVELOPMENT', sources_complete=index+1, sources=100, rows=len(allrows))
        c.require(len(allrows) == 1800, 'Incomplete N2048 development')
        table = self.result/'N2048_native_per_frame.csv'; c.write_csv(table, allrows); outputs.append(table)
        native.frozen(); self.finish('development', outputs, inputs, sources=100, rows=1800,
            policy_selection_updates=0, policy_sha256=c.sha(policy_path), independent_metrics_pending=True)

    def export_n1024(self):
        if self.completed('export-n1024'):
            return
        native = self.load(); data = native.data('development'); self.target_records()
        engine = native.replay_engine(); inventory = c.old_source_inventory(self.root)
        inputs = {**inventory['bindings'], **self.target_bindings}
        binding, registration = self.registration('export-n1024', data, seeds=c.DEV_SEEDS,
            extra=dict(original_engine_manifest=engine.manifest(), compatibility=native.compatibility,
                       original_scoring_inventory_bindings=inventory['bindings']))
        inputs.update(registration); outputs = []; allrows = []
        for index, source in enumerate(data['records']):
            self.guard(); path = self.out/'export-n1024/source_checkpoints'/f'{index:04d}.json'
            original_target, comparison_target = self.targets_for(data, index)
            completed_path = self.root/'outputs/UNIFIED-METRICS-20261002/source_checkpoints'/f'{index:03d}.json'
            completed = c.original_checkpoint(completed_path, inventory, index, source)
            inputs[str(completed_path)] = c.sha(completed_path)
            measured = {r['replay_row_id']:r for r in completed['rows']}
            selected = {}; originals = {}
            for study in ('N1024', 'M1'):
                wanted = [r for r in engine.by_source[study][index] if int(r['snr_db']) in c.SNRS
                    and int(r['noise_seed']) in c.DEV_SEEDS and
                    ((study == 'N1024' and r['method'] in ('P1024', 'D_U_QPSK')) or
                     (study == 'M1' and int(r['N']) == 1024 and r['phy_family'] == 'QPSK'
                      and r['method'] == 'entropy_policy'))]
                methods = ('P1024', 'D_U_QPSK') if study == 'N1024' else ('entropy_policy',)
                c.require(len(wanted) == 9*len(methods) and
                    {(int(r['snr_db']), int(r['noise_seed']), r['method']) for r in wanted} ==
                    {(s, n, m) for s in c.SNRS for n in c.DEV_SEEDS for m in methods},
                    'Selected N1024 old rows incomplete')
                selected[study] = wanted
                originals.update({native.replay.row_id(study,r):r for r in wanted})
            if path.exists():
                saved = c.checkpoint(path, binding, index)
                c.require(saved['original_completed_checkpoint_sha256'] == c.sha(completed_path), 'Original source metric identity changed')
            else:
                images = []; slots = {}; rows = []; began = time.monotonic()
                for study in ('N1024', 'M1'):
                    wanted = selected[study]
                    expected = {native.replay.row_id(study, r):r for r in wanted}; seen = set()
                    with native.torch.no_grad():
                        for row, image, computed in engine.adapters[study].iterate_source(index, wanted):
                            self.guard(); rid = native.replay.row_id(study, row)
                            c.require(rid in expected and rid not in seen and row == expected[rid], 'Old native row changed')
                            parity = native.replay.parity_check(row, computed, required_quality=True)
                            old = measured[rid]; fingerprint = c.rgb_sha(image)
                            c.require(old['image_sha256'] == fingerprint and old['reference_sha256'] == c.rgb_sha(original_target),
                                      'N1024 replay differs from the already completed exact float RGB')
                            slot, _ = self.add_image(images, slots, image)
                            method = row['method'] if study == 'N1024' else 'M1_entropy_N1024_frozen'
                            rows.append(dict(source_index=index, source_id=source['image_id'], N=1024,
                                snr_db=int(row['snr_db']), noise_seed=int(row['noise_seed']), method=method,
                                source_class_index=int(source['class_index']), image_slot=slot, image_sha256=fingerprint,
                                reference_sha256=c.rgb_sha(comparison_target), native_reference_sha256=c.rgb_sha(original_target),
                                original_replay_row_id=rid, original_scientific_row=row, original_completed_metrics=old,
                                original_scalar_parity=parity, original_scientific_row_sha256=native.replay.source_row_hash(row),
                                source_origin='exact_old_selected_replay', policy_selection_updates=0))
                            seen.add(rid)
                    c.require(seen == set(expected), 'Old selected replay omitted an output')
                cache = c.save_float_cache(self.out/'export-n1024/float_reconstructions'/f'{index:04d}.npz',
                                            images, original_target, comparison_target)
                saved = c.sealed_checkpoint(path, dict(binding=binding, source_index=index, rows=rows,
                    float_reconstructions=cache, original_completed_checkpoint_sha256=c.sha(completed_path),
                    synthetic=False, seconds=time.monotonic()-began))
            c.validate_export_rows(saved['rows'], index, source, originals, measured)
            c.verify_float_cache(saved['float_reconstructions'], saved['rows'], original_target, comparison_target)
            outputs += [path, Path(saved['float_reconstructions']['path'])]; allrows.extend(saved['rows'])
            self.status('EXPORTING_EXACT_N1024', sources_complete=index+1, sources=100, rows=len(allrows))
        manifest = engine.verify_frozen(); inputs.update(manifest['input_sha256'])
        self.finish('export-n1024', outputs, inputs, sources=100, rows=len(allrows),
                    policy_selection_updates=0, exact_previous_rgb=True, independent_metrics_pending=True)

    def run(self, stage):
        import fcntl
        self.out.mkdir(parents=True, exist_ok=True)
        lock = (self.out/'run.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, self.stopped); signal.signal(signal.SIGINT, self.stopped)
        try:
            stages = c.STAGES if stage == 'all' else (stage,)
            for name in stages:
                self.stage = name; self.guard(); self.status('STARTING')
                if name == 'qualify': self.qualify()
                elif name == 'screen': self.calibration(True)
                elif name == 'calibrate': self.calibration(False)
                elif name == 'development': self.development()
                elif name == 'export-n1024': self.export_n1024()
                else: raise ValueError('Unknown registered control stage')
            return 0
        except BaseException as error:
            # The original frozen GPU/thermal boundary uses ResourceBusy.
            module = __import__('sys').modules.get('latent_enhancement.runtime')
            busy = getattr(module, 'ResourceBusy', ()) if module else ()
            if isinstance(error, c.PauseRequested) or (busy and isinstance(error, busy)):
                self.status('PAUSED', reason=str(error), resume_same_configuration=True); return 75
            c.seal(self.out/((self.stage or stage)+'_failure.json'), dict(status='FAILED_REQUIRES_REVIEW',
                error=repr(error), stage=self.stage, automatic_retry=False, original_sources_modified=False))
            self.status('FAILED_REQUIRES_REVIEW', error=repr(error)); raise
        finally:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--stage', required=True, choices=(*c.STAGES, 'all'))
    parser.add_argument('--protocol', default=str(c.HERE/'own_controls_protocol.json'))
    args = parser.parse_args()
    return Runner(args.root, args.protocol).run(args.stage)


if __name__ == '__main__':
    raise SystemExit(main())
