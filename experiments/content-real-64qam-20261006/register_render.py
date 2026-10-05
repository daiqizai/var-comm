"""Seal one render-only H revision after its CPU owner and workers close.

This helper never starts a process, decoder, model, or GPU. All new files are
created only after the real predecessor closure and immutable bindings pass.
The separate human-controlled launcher must require registration_completion.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEADLINE = 1791564605.9549868
PHASES = {'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
          'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def merge(*maps):
    result = {}
    for mapping in maps:
        for path, value in mapping.items():
            require(path not in result or result[path] == value, 'Conflicting immutable binding: '+path)
            result[path] = value
    return result


def bind(paths): return {str(p): sha(p) for p in paths}
def pinned(record):
    p = Path(record['path'])
    require(p.is_absolute() and sha(p) == record['sha256'], 'Pinned input changed: '+str(p))
    return p


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_cpu_closed(a, prior, config_path, registration_path, launch_path,
                      cpu_config_path, state_reader):
    """No zombie, orphan, incomplete batch or substituted merge can release GPU."""
    cfg, reg, launch = read(config_path), read(registration_path), read(launch_path)
    rsha = sha(registration_path)
    a.validate_config(cfg, reg, sha(config_path))
    require(cfg['registration'] == str(registration_path)
            and reg['allowed_stage_ids'] == ['initial_true200','report']
            and all(s['resource'] == 'cpu' for s in cfg['stages']), 'Wrong prerequisite CPU scope')
    identity = launch['identity']
    expected = [launch['argv'][0], '-B', str(Path(a.__file__).resolve()), '--config', str(config_path)]
    require(launch['argv'] == identity['argv'] == expected
            and launch['registration_sha256'] == rsha
            and launch['owner_config_sha256'] == sha(config_path), 'CPU owner launch changed')
    require(prior.exited(identity, state_reader), 'CPU owner is live or unreaped')
    require(not (Path(cfg['owner_out'])/'STOP').exists() and not (Path(cfg['out'])/'STOP').exists(),
            'CPU/H STOP blocks render registration')
    closure = prior.verify_batch(a, str(config_path), str(registration_path), identity['uid'], state_reader)
    jobs = [job for stage in cfg['stages'] for job in stage['jobs']]
    require(len(jobs) == 3 and len(cfg['stages'][0]['jobs']) == 2
            and len(cfg['stages'][1]['jobs']) == 1, 'Expected two CPU workers plus one merge')
    for job in jobs:
        argv = job['argv']; require(argv.count('--config') == 1
            and argv[argv.index('--config')+1] == str(cpu_config_path), 'CPU job configuration differs')
        for name in ('failure.json','merge_failure.json','STOP'):
            require(not (Path(job['out'])/name).exists(), 'Preserved CPU failure/STOP: '+job['out'])
    merge_job = cfg['stages'][1]['jobs'][0]; cpu_cfg = read(cpu_config_path)
    cp = Path(cpu_cfg['out'])/'completion.json'; done = read(cp)
    require(merge_job['completion'] == str(cp) and closure['bindings'].get(str(cp)) == sha(cp)
            and cpu_cfg['registration'] == str(registration_path), 'Actual CPU merge receipt is unbound')
    require(done['status'] == 'H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE'
            and done['registration_sha256'] == rsha and done['driver_config_sha256'] == sha(cpu_config_path)
            and done['source_count'] == 200 and done['images_scored'] is False
            and done['source_decode_complete'] is False and done['GPU_used'] is False
            and done['development_used'] is False, 'CPU reception is not complete in the expected scope')
    require(len(done['worker_identities']) == 2, 'CPU worker identities incomplete')
    for child in done['worker_identities']:
        require(any(a.same_identity(child, c) for c in closure['child_identities'])
                and prior.exited(child, state_reader), 'CPU reception worker not closed')
    for field in ('outputs','input_bindings','source_bindings'): a.verify(done[field])
    return dict(status='H_INITIAL_CPU_OWNER_AND_ALL_CHILDREN_CLOSED', owner_identity=identity,
        child_identities=closure['child_identities'], cpu_completion=str(cp),
        bindings=merge(closure['bindings'], bind((config_path,registration_path,launch_path,cpu_config_path))))


def make_configs(request, previous, cpu, visual, execution):
    """Pure configuration construction; no future receipt or self-hash cycle."""
    regpath = execution/'execution_registration.json'
    renderpath = execution/'render_config.json'; ownerpath = execution/'owner_config.json'
    runtime = Path(cpu['runtime_dir']); output = request['render_out']
    cfg = {k:cpu[k] for k in ('root','protocol','catalogue','shortlist','source200','S1_completion',
                             'S1_assets_completion','budget_registration','ledger','phase_limits','stop_file')}
    cfg.update(H_out=previous['out'], out=output, registration=str(regpath),
        render_owner_config=str(ownerpath), owner_module=request['owner_module'],
        wait_module=request['wait_module'], cpu_driver_module=str(runtime/'h_payload_driver.py'),
        rx_module=str(runtime/'h_payload_rx.py'), source_driver_module=str(runtime/'h_source_driver.py'),
        runtime_dir=str(runtime), uep_runtime=visual['uep_runtime'], native_runtime=visual['native_runtime'],
        calibration_registration=visual['calibration_registration'],
        cpu_config=request['cpu_config']['path'], cpu_registration=request['cpu_registration']['path'],
        cpu_completion=str(Path(cpu['out'])/'completion.json'),
        cpu_owner_config=request['cpu_owner_config']['path'],
        cpu_owner_registration=request['cpu_registration']['path'], cpu_owner_launch=request['cpu_owner_launch']['path'],
        max_seconds=request['max_seconds'], overall_deadline_unix=DEADLINE)
    owner = copy.deepcopy(previous)
    owner.update(owner_out=str(execution), registration=str(regpath))
    owner['stages'] = [dict(id='render',resource='gpu',requires=[],max_seconds=request['max_seconds'],jobs=[
        dict(id='render',argv=[request['python'],'-B',str(runtime/'h_payload_render_driver.py'),'--config',str(renderpath)],
             cwd=cpu['root'],out=output,completion=str(Path(output)/'completion.json'),
             accepted_statuses=['H_INITIAL_TRUE200_RX_COMPLETE'],
             receipt_expect=dict(source_count=200,images_scored=True,source_decode_complete=True,
                                 new_packet_decodes=0,policy_selection=False,development_used=False,holdout_used=False))])]
    # Prior qualification and S1 gate are inherited verbatim, including their
    # original registration paths. CPU/GPU affinity, threads and locks stay fixed.
    return cfg, owner


def prepare(request_path):
    request_path = Path(request_path).resolve(); r = read(request_path)
    require(r['schema'] == 'H_RENDER_REGISTRATION_REQUEST_V1', 'Wrong registration request')
    require(sys.platform.startswith('linux'), 'Actual registration requires Linux process evidence')
    require(time.time() < DEADLINE, 'Original H deadline has expired')
    paths = {k:pinned(r[k]) for k in ('cpu_owner_config','cpu_registration','cpu_owner_launch','cpu_config',
                                    'visual_source_config','prepared_qualification')}
    old, reg, cpu = read(paths['cpu_owner_config']), read(paths['cpu_registration']), read(paths['cpu_config'])
    oldbound = merge(reg['input_bindings'],reg['source_bindings'])
    for key in ('owner_module','wait_module'):
        require(oldbound.get(r[key]) == sha(r[key]), 'Frozen predecessor verifier unbound: '+key)
    for key in ('cpu_owner_config','cpu_config','visual_source_config'):
        require(oldbound.get(str(paths[key])) == sha(paths[key]), 'CPU revision lacks required input: '+key)
    a = module(r['owner_module'],'render_reg_bound_owner')
    prior = module(r['wait_module'],'render_reg_bound_prior')
    a.verify(reg['source_bindings']); a.verify(reg['input_bindings'])
    closure = verify_cpu_closed(a,prior,paths['cpu_owner_config'],paths['cpu_registration'],
                               paths['cpu_owner_launch'],paths['cpu_config'],a.raw_process_state)
    require(old['phase_limits'] == cpu['phase_limits'] == PHASES
            and old['budget_path'] == cpu['ledger'] and old['budget_registration'] == cpu['budget_registration'],
            'Original CPU/budget scope changed')
    before = a.budget_snapshot(cpu['ledger'],sha(cpu['budget_registration']),PHASES,quiescent=True)
    done = read(closure['cpu_completion'])
    require(before['created'] and before['phase_charged']['initial_true200']
            == done['ledger']['phase_charged']['initial_true200'] <= 19200, 'CPU charged count changed')
    for phase in ('whole_calibration','partial_calibration','development','engineering_reserve'):
        require(before['phase_charged'][phase] == 0, 'Later/shared-ledger stage already started: '+phase)
    visual = read(paths['visual_source_config'])
    require(visual['root'] == cpu['root'] and str(Path(visual['S1'])/'completion.json') == cpu['S1_completion']
            and visual['source200'] == cpu['source200'], 'Original visual data scope differs')
    prepared = read(paths['prepared_qualification'])
    require(prepared['status'] == 'PREPARED_SELECTION_RENDER_CPU_QUALIFICATION_PASS'
            and prepared['actual_H_decoder_calls'] == 0 and prepared['GPU_used'] is False
            and prepared['execution_registered'] is False and prepared['results']
            and all(row['exit_code'] == 0 for row in prepared['results']), 'Prepared renderer tests not PASS')
    a.verify(prepared['source_bindings'])
    runtime = Path(cpu['runtime_dir']); entry = runtime/'h_payload_render_driver.py'
    require(prepared['source_bindings'].get(str(entry)) == sha(entry)
            and prepared['source_bindings'].get(str(runtime/'h_payload_rx.py')) == sha(runtime/'h_payload_rx.py'),
            'Current receiver/driver lacks matching CPU qualification')
    sources = merge(reg['source_bindings'],prepared['source_bindings'],r.get('additional_source_bindings',{}),bind((Path(__file__).resolve(),)))
    inputs = merge(reg['input_bindings'],closure['bindings'],bind((request_path,*paths.values())))
    a.verify(sources); a.verify(inputs)
    render = module(entry,'render_reg_bound_driver')
    render.require_native_bindings(visual['native_runtime'],merge(inputs,sources))
    for p in (Path(visual['calibration_registration']),runtime/'h_source_driver.py'):
        require(merge(inputs,sources).get(str(p)) == sha(p), 'Original visual input/source unbound: '+str(p))
    execution, output, h = Path(r['execution_dir']), Path(r['render_out']), Path(old['out'])
    for p in (execution,output):
        require(p.is_absolute() and h in p.parents and not p.exists(), 'New independent H output required: '+str(p))
    require(execution != output and execution not in output.parents and output not in execution.parents,
            'Render evidence and science paths overlap')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file(), 'Visual interpreter missing')
    require(0 < r['max_seconds'] <= 86400, 'Render runtime must remain finite')
    require(old['overall_deadline_unix'] == DEADLINE, 'Inherited owner deadline differs')
    require(set(old['gpu_affinity']) <= set(os.sched_getaffinity(0)), 'Original GPU CPU affinity unavailable')
    cfg, owner = make_configs(r,old,cpu,visual,execution)
    for k in ('protocol','catalogue','shortlist','source200','calibration_registration','S1_completion','S1_assets_completion',
              'cpu_config','cpu_registration','cpu_completion','cpu_owner_config','cpu_owner_registration','cpu_owner_launch',
              'budget_registration','owner_module','wait_module','cpu_driver_module','rx_module','source_driver_module'):
        p = cfg[k]; require(merge(inputs,sources).get(p) == sha(p), 'Render dependency unbound: '+k)
    require(a.budget_snapshot(cpu['ledger'],sha(cpu['budget_registration']),PHASES,quiescent=True) == before,
            'Ledger changed during read-only registration gates')
    execution.mkdir()
    try:
        gatepath = execution/'predecessor_closure.json'; save(gatepath,dict(closure,budget=before,new_packet_decodes=0,GPU_used=False))
        cp, op, rp = execution/'render_config.json', execution/'owner_config.json', execution/'execution_registration.json'
        save(cp,cfg); save(op,owner); inputs = merge(inputs,bind((gatepath,cp,op)))
        new = dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            owner_config_sha256=sha(op),allowed_stage_ids=['render'],phase_limits=PHASES,
            source_bindings=sources,input_bindings=inputs,scientific_protocol_sha256=sha(cfg['protocol']),
            budget_registration_sha256=sha(cfg['budget_registration']),cpu_completion_sha256=sha(cfg['cpu_completion']),
            shortlist_sha256=sha(cfg['shortlist']),predecessor_closure_sha256=sha(gatepath),
            source_count=200,frame_count=done['frame_count'],calibration_noise_seeds=[6101,6102,6103],
            qualification_start_unix=owner['qualification_started_unix'],qualification_deadline_unix=DEADLINE,
            GPU_jobs=1,GPU_threads=owner['gpu_threads'],GPU_affinity=owner['gpu_affinity'],
            new_packet_decodes=0,policy_selection=False,development_used=False,holdout_started=False,
            C_started=False,H_full_delivery_claimed=False,scientific_protocol_modified=False,
            cache='Per-source actual RX profile/payload + frozen receiver identity only; no source-truth shortcut',
            implemented_scope='Receive canonicalization, actual RGB reconstruction and PSNR/MSE of closed initial_true200 traces only',
            shared_budget_exclusion='No later H/C or shared-ledger job may start until this owner and GPU worker close',
            future_stage_automatic=False,created_unix=time.time())
        a.validate_config(owner,new,sha(op)); a.verify(sources); a.verify(inputs); save(rp,new)
        # This loader is purely read-only; live-owner/GPU admission is enforced
        # separately in run(), immediately before constructing a visual model.
        ctx = render.load_registered(str(cp))
        require(ctx['completion']['frame_count'] == done['frame_count'], 'Read-only render preflight changed scope')
        require(a.budget_snapshot(cpu['ledger'],sha(cpu['budget_registration']),PHASES,quiescent=True) == before,
                'Read-only preflight changed original packet ledger')
        result = dict(status='H_RENDER_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((rp,cp,op,gatepath)),new_packet_decodes=0,workers_started=False,GPU_used=False,
            read_only_driver_preflight='PASS',budget=before,
            required_launch_condition='Require exact registration_completion outputs and no registration_failure; reverify CPU closure and quiescent unchanged budget before separate launch. Owner admits one exclusive GPU job; no later shared-ledger work until its owner and child exit.')
        save(execution/'registration_completion.json',result)
        return result
    except BaseException as error:
        failure = execution/'registration_failure.json'
        if not failure.exists(): save(failure,dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),
            traceback=traceback.format_exc(),workers_started=False,new_packet_decodes=0,GPU_used=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    result=prepare(parser.parse_args().config)
    print(json.dumps({k:result[k] for k in ('status','registration_sha256','owner_config_sha256')}))


if __name__ == '__main__': main()
