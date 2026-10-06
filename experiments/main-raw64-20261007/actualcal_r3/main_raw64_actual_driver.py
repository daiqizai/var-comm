"""Finite raw64 calibration: explicit registration, two CPU workers, normal close.

No model, image pixels, ranking or successor is executed. Historical raw traces
are admitted point by point; new calls use the reserved phase of the SAME ledger.
The original q/proxy runner supplies only small process/file utilities.
"""
import argparse
from contextlib import contextmanager, closing
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import numpy as np

import main_raw64_cpu_runner as u
import main_raw64_calibration_checkpoints as c
import main_raw64_calibration_budget as budget

read, sha, save, require, combine = u.read, u.sha, u.save, u.require, u.combine
SCHEMA = 'MAIN_RAW64_ACTUAL_CALIBRATION_REQUEST_V1'
REGISTERED = 'MAIN_RAW64_ACTUAL_CALIBRATION_REGISTERED_V1'
WORKER_DONE = 'MAIN_RAW64_ACTUAL_WORKER_COMPLETE_V1'
DONE = 'MAIN_RAW64_ACTUAL_CPU_COMPLETE_V1'
NORMAL = 'MAIN_RAW64_ACTUAL_CALIBRATION_NORMALLY_COMPLETE_V1'
QSTATUS = 'MAIN_RAW64_ACTUAL_CPU_INTERFACE_QUALIFICATION_PASS_V1'
TEST_MODULES = ['test_main_raw64_calibration_checkpoints', 'test_main_raw64_calibration_budget', 'test_main_raw64_actual_driver']
TEST_COUNT = 35


def pin(path, bound):
    require(Path(path).is_absolute() and bound.get(path) == sha(path), 'Missing/changed direct pin: '+path)
    return read(path)


def module(path, name, bound):
    require(bound.get(path) == sha(path), 'Unbound imported source: '+path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def gone(identity):
    p = Path('/proc') / str(identity['pid'])
    require(not p.exists() or u.process(identity['pid'])['start_ticks'] != identity['start_ticks'], 'Original process is still present')


def waited(path, bound, expected=None):
    end = pin(path, bound)
    require(end['process_waited'] is True and end['exit_code'] == 0 and bound.get(end['log']) == end['log_sha256'] == sha(end['log']), 'Normal wait0 and closed log required')
    if expected is not None:
        require(u.same(end['identity'], expected), 'Waited identity differs')
    gone(end['identity'])
    return end


def prescreen_closed(spec, bound, catalogue_path, science_path):
    done = pin(spec['completion'], bound); science = pin(spec['science'], bound)
    reg = pin(spec['registration'], bound); request = pin(spec['request'], bound)
    require(done['status'] == 'RAW64_PRESCREEN_NORMAL_COMPLETE' and done['original_worker_success'] is True and done['automatic_calibration'] is False, 'Normal frozen prescreen required')
    require(done['science_completion_sha256'] == sha(spec['science']) and done['exit_receipt_sha256'] == sha(spec['worker_exit']), 'Prescreen normal receipt differs')
    require(reg['status'] == 'FROZEN_READ_ONLY_PRESCREEN_EXECUTION' and reg['request_sha256'] == science['request_sha256'] == sha(spec['request']), 'Prescreen registration differs')
    require(science['status'] == 'RAW64_SIX_SNR_FINITE_SHORTLIST_COMPLETE' and science['source_count'] == 200 and science['wire_count'] == 433 and science['SNRs'] == list(c.SNRS) and science['new_PHY_calls'] == 0 and science['GPU_used'] is False and science['automatic_successor'] is False, 'Exact prospective shortlist scope required')
    end = waited(spec['worker_exit'], bound, science['worker_identity'])
    owner = pin(spec['owner_identity'], bound); gone(owner)
    outer=pin(spec['sequence_completion'],bound)
    require(outer['status']=='RAW64_PRESCREEN_SEQUENCE_NORMAL_COMPLETE' and outer['original_owner_success'] is True and outer['owner_completion_sha256']==sha(spec['completion']) and outer['owner_exit_sha256']==sha(spec['owner_exit']) and outer['automatic_calibration'] is False and outer['new_PHY']==0 and outer['GPU'] is False,'Prescreen durable outer sequence not normally complete')
    oe=waited(spec['owner_exit'],bound,owner)
    require(oe['capture_error'] is None and oe['identity']['argv'][-2:]==['--request',spec['request']],'Wrong prescreen owner exit')
    gone(pin(spec['sequence_identity'],bound))
    require(end['identity']['argv'][-2:] == ['--worker', spec['request']], 'Wrong prescreen worker command')
    require(not (Path(spec['completion']).parent/'failure.json').exists(), 'Failed prescreen cannot be used')
    for p, h in science['outputs'].items():
        require(bound.get(p) == h == sha(p), 'Prescreen output not bound')
    short = pin(spec['shortlist'], bound)
    require(science['outputs'].get(spec['shortlist']) == sha(spec['shortlist']) and short['catalogue_sha256'] == sha(catalogue_path) and short['science_registration_sha256'] == sha(science_path), 'Shortlist belongs to another catalogue/science contract')
    require(len(set(short['original_construction_source_ids'])) == 200 and short['final_objective'] == c.PRIMARY and short['development_used'] is False and short['holdout_used'] is False, 'Construction population/objective changed')
    require(request['deadline_unix'] > 0, 'Original finite study deadline required')
    return short


def historical(spec, bound):
    """Authenticate small original normal controls; verify source/trace bytes on use."""
    observation = pin(spec['observation'], bound); reg = pin(spec['registration'], bound)
    normal = pin(spec['completion'], bound); science = pin(spec['science'], bound)
    cfg = pin(spec['config'], bound); owner = pin(spec['owner_config'], bound)
    require(observation['status'] == 'MAIN_ACTUAL1000_CPU_NORMAL_CLOSED_OUTPUTS_VERIFIED' and all(observation[k] is True for k in ('original_owner_success', 'processes_exited', 'original_receipts_validated', 'paid_inventory_validated')), 'Original normal paid-inventory observation required')
    require(observation['registration_sha256'] == sha(spec['registration']) and observation['science_completion_sha256'] == sha(spec['science']) and observation['owner_completion_sha256'] == sha(spec['completion']), 'Original observation does not name these receipts')
    require(normal['status'] == 'MAIN_ACTUAL_CALIBRATION_NORMALLY_COMPLETE_R1' and science['status'] == 'MAIN_ACTUAL_CALIBRATION_CPU_COMPLETE' and normal['registration_sha256'] == science['registration_sha256'] == sha(spec['registration']), 'Original normal actual calibration required')
    require(reg['status'] == 'MAIN_ACTUAL_CALIBRATION_EXECUTION_REGISTERED_R1' and normal['source_bindings'] == science['source_bindings'] == reg['source_bindings'] and normal['input_bindings'] == science['input_bindings'] == reg['input_bindings'], 'Original receipts must retain registered source/input maps')
    require(normal['owner_config_sha256'] == sha(spec['owner_config']) and science['config_sha256'] == sha(spec['config']) and cfg['registration'] == owner['registration'] == spec['registration'], 'Original configuration identity differs')
    require(normal['frame_count'] == science['frame_count'] == 30000 and science['source_count'] == 1000 and normal['packet_calls'] == science['packet_calls'] == 60000, 'Original calibration grid differs')
    require(normal['budget_after'] == science['budget'] and not normal['budget_after']['unresolved'], 'Original ledger did not close')
    inherited = combine(reg['source_bindings'], reg['input_bindings'], normal['outputs'], science['outputs'])
    for path in (spec['config'], spec['owner_config']):
        require(inherited.get(path) == sha(path), 'Original config not sealed')
    gone(normal['owner_identity'])
    for path in normal['outputs']:
        if Path(path).name == 'exit.json':
            end = pin(path, inherited); folder = Path(path).parent
            launch_path, logpath = str(folder/'launch.json'), str(folder/'worker.log')
            launch = pin(launch_path, inherited)
            require(end['wait_completed'] is True and end['exit_code'] == 0 and end['registration_sha256'] == sha(spec['registration']) and end['launch_sha256'] == sha(launch_path) and end['log_sha256'] == inherited.get(logpath) == sha(logpath) and u.same(end['identity'], launch['identity']) and u.same(launch['owner_identity'],normal['owner_identity']), 'Original worker/merge actual exit contract differs')
            require(end['completion_sha256'] in normal['outputs'].values(), 'Original exit does not name sealed scientific completion')
            gone(end['identity'])
    require(sum(Path(p).name == 'exit.json' for p in normal['outputs']) == 3, 'Original two workers and merge must have waited')
    require(normal['outputs'].get(spec['science']) == sha(spec['science']), 'Science not sealed by original owner')
    return dict(cfg=cfg, reg=reg, normal=normal, science=science, bound=combine(bound, inherited), receipts=[spec[k] for k in ('observation','registration','completion','science','config','owner_config')])


def assets(cfg, historical_ctx, bound):
    old = historical_ctx['cfg']; rows = pin(cfg['asset_manifest'], bound); done = pin(cfg['asset_completion'], bound); cal = pin(cfg['calibration_registration'], bound)
    require(all(cfg[k] == old[k] for k in ('asset_manifest','asset_completion','calibration_registration')), 'Reuse must retain original source assets/Encoder provenance')
    require(done['status'] == 'H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE' and done['source_count'] == 1000 and done['new_packet_decodes'] == 0 and done['development_used'] is False, 'Original calibration source export required')
    ids, records = rows['source_ids'], rows['records']
    require(ids == cal['source_ids'] and len(ids) == len(set(ids)) == len(records) == 1000 and cal['stage'] == cal['calibration_or_development'] == 'm1_calibration', 'Exact original calibration1000 source order required')
    for i, r in enumerate(records):
        require(r['source_index'] == i and r['source_id'] == ids[i] and r['preprocessing_id'] == cal['preprocessing_ids'][i] and done['outputs'].get(r['checkpoint']) == r['checkpoint_sha256'], 'Original source identity/Encoder asset checkpoint differs')
        require(bound.get(r['checkpoint']) == r['checkpoint_sha256'] and r['archive'] in bound, 'Source asset missing immutable pin')
    return records


def inspect_inputs(r):
    require(r['schema'] == SCHEMA, 'Explicit finite actual calibration request required')
    sources, inputs = r['source_bindings'], r['input_bindings']; bound = combine(sources, inputs)
    u.verify(sources)
    for mod in (u, c, budget):
        require(sources.get(str(Path(mod.__file__).absolute())) == sha(mod.__file__), 'Imported runtime support is not bound')
    require(sources.get(str(Path(__file__).absolute())) == sha(__file__), 'Unbound actual driver')
    rootcfg = pin(r['root_config'], bound); rootreg = pin(r['root_registration'], bound)
    require(rootreg['config_sha256'] == sha(r['root_config']) and rootcfg['registration'] == r['root_registration'], 'Original new-root registration required')
    require(r['root'] == rootcfg['root'] and r['python'] == rootcfg['python'] and r['deadline_unix'] == rootcfg['deadline_unix'] and r['ledger'] == rootcfg['ledger'] and r['stop_file'] == rootcfg['stop_file'], 'Study root, interpreter, deadline or sole ledger changed')
    require(r['ledger_lock'] == str(Path(r['root'])/'cpu_runner.lock') and r['adapter_module'] == rootcfg['adapter_module'] and r['adapter_config'] == rootcfg['adapter_config'], 'Original packet implementation and same owner lock required')
    require(r['original_cpu_plan'] == rootcfg['plan'] and r['science_registration'] == rootcfg['science_registration'], 'Original science/PHY plan changed')
    require(all(bound.get(p) == h for p,h in rootreg['source_bindings'].items()), 'Original physical source closure missing')
    cat = pin(r['adapter_config']['catalogue'], bound); science = pin(r['science_registration'], bound)
    require(science['scope']['N'] == 1024 and science['scope']['SNRs'] == list(c.SNRS) and science['scope']['order'] == 'raster' and science['scope']['failure_rule'] == 'KEEP', 'Frozen scientific scope differs')
    short = prescreen_closed(r['prescreen'], bound, r['adapter_config']['catalogue'], r['science_registration'])
    old = historical(r['legacy_batch'], bound); inherited = combine(bound, old['bound'])
    require(old['reg']['source_bindings'].get(r['original_receiver_module']) == sources.get(r['original_receiver_module']) == sha(r['original_receiver_module']), 'Exact original pure KEEP receiver must be inherited')
    require(sources.get(r['receiver_module']) == sha(r['receiver_module']), 'New mixed public receiver must be bound')
    records = assets(r, old, inherited)
    plan = c.make_plan(cat['profiles'], short, [x['source_id'] for x in records], science_sha256=sha(r['science_registration']), shortlist_sha256=sha(r['prescreen']['shortlist']), source_manifest_sha256=sha(r['asset_manifest']))
    require(plan['catalogue_digest'] == cat['catalogue_digest'] and plan['maximum_new_packet_calls'] <= rootreg['phase_caps']['actual_calibration'], 'Finite calibration exceeds reserved allowance')
    aff = r['resources']['affinities']
    require(r['resources']['workers'] == 2 and r['resources']['threads'] == 2 and r['resources']['nice'] == 15 and len(aff) == 2 and all(len(x) == len(set(x)) == 2 for x in aff) and len(set(sum(aff,[]))) == 4, 'Two disjoint CPU pairs required')
    require(0 < r['max_seconds'] <= science['resources']['wall_clock_ceiling_days']*86400 and Path(r['python']).is_absolute(), 'Finite run parameters required')
    return dict(cfg=r, bound=inherited, plan=plan, records=records, profiles=cat['profiles'], catalogue_data=cat, legacy=old, rootcfg=rootcfg, rootreg=rootreg)


def qualification(r, bound):
    q = pin(r['prepared_qualification'], bound)
    require(TEST_COUNT > 0 and q['status'] == QSTATUS and q['tests_run'] == TEST_COUNT and q['test_modules'] == TEST_MODULES and q['python'] == r['python'] and q['process_waited'] is True and q['exit_code'] == 0 and q['GPU_used'] is False and q['new_packet_decodes'] == 0, 'Exact actual CPU interface qualification required')
    require(all(q['source_bindings'].get(p) == h for p,h in r['source_bindings'].items()), 'Qualification omitted runtime/test source')
    for name in TEST_MODULES:
        p = str(Path(__file__).absolute().with_name(name+'.py')); require(r['source_bindings'].get(p) == sha(p), 'Formal test source missing')
    log = q['log']; require(q['outputs'].get(log) == sha(log), 'Qualification log not sealed')
    txt = Path(log).read_text(); require(('Ran '+str(TEST_COUNT)+' tests') in txt and txt.rstrip().endswith('OK') and 'skipped=' not in txt, 'Complete tests without skips required')
    return q


def build_request(materials_path,target):
    """Metadata-only assembler after real normal predecessors exist.

    materials supplies exact paths, never downloaded basenames. Large original
    token archives are authenticated by their sealed manifests and hashed only
    when a CPU worker reads that source. No quality values are loaded here.
    """
    m=read(materials_path); rootcfg=read(m['root_config']); rootreg=read(rootcfg['registration'])
    oldcfg=read(m['legacy_batch']['config']); oldreg=read(m['legacy_batch']['registration'])
    oldnormal=read(m['legacy_batch']['completion']); oldscience=read(m['legacy_batch']['science'])
    sources=combine(rootreg['source_bindings'],oldreg['source_bindings'],{p:sha(p) for p in m['source_paths']})
    for mod in (u,c,budget):sources[str(Path(mod.__file__).absolute())]=sha(mod.__file__)
    for name in ['main_raw64_actual_driver']+TEST_MODULES:
        path=str(Path(__file__).absolute().with_name(name+'.py'));sources[path]=sha(path)
    paths=[m['root_config'],rootcfg['registration'],m['root_completion'],m['root_owner_exit'],m['prepared_qualification'],m['receiver_module'],m['original_receiver_module'],*m['legacy_batch'].values(),*m['prescreen'].values()]
    inputs=combine(rootreg['input_bindings'],oldreg['input_bindings'],oldnormal['outputs'],oldscience['outputs'],{p:sha(p) for p in paths if p not in sources},{str(Path(materials_path).absolute()):sha(materials_path)})
    ex=read(m['root_owner_exit']);inputs[ex['log']]=ex['log_sha256']
    pre=read(m['prescreen']['science']);px=read(m['prescreen']['worker_exit']);po=read(m['prescreen']['owner_exit']);inputs=combine(inputs,pre['outputs'],{px['log']:px['log_sha256'],po['log']:po['log_sha256']})
    r={k:rootcfg[k] for k in ('root','project_root','python','deadline_unix','ledger','stop_file','adapter_module','adapter_config','science_registration')}
    r.update(schema=SCHEMA,root_config=m['root_config'],root_registration=rootcfg['registration'],root_completion=m['root_completion'],root_owner_exit=m['root_owner_exit'],original_cpu_plan=rootcfg['plan'],original_ledger_module=str(Path(u.__file__).absolute().with_name('main_raw64_ledger.py')),ledger_lock=str(Path(rootcfg['root'])/'cpu_runner.lock'),legacy_batch=m['legacy_batch'],prescreen=m['prescreen'],source_bindings=sources,input_bindings=inputs)
    r.update({k:m[k] for k in ('execution_dir','out','pythonpath','max_seconds','resources','prepared_qualification','receiver_module','original_receiver_module')})
    r.update({k:oldcfg[k] for k in ('asset_manifest','asset_completion','calibration_registration')})
    require(r['ledger'] not in inputs and oldcfg['ledger'] not in inputs,'Mutable ledgers must never be SHA-bound as files')
    ctx=inspect_inputs(r);qualification(r,ctx['bound']);save(target,r)
    return dict(path=str(Path(target).absolute()),sha256=sha(target),source_count=1000,frame_count=ctx['plan']['frame_count'],maximum_new_packet_calls=ctx['plan']['maximum_new_packet_calls'],registration_created=False,model_calls=0,new_packet_decodes=0)


def budget_registration(r, plan):
    keys = ('root_registration','root_config','root_completion','root_owner_exit','original_ledger_module')
    inputs = {r[k]: sha(r[k]) for k in keys if k != 'original_ledger_module'}
    normal = read(r['root_completion']); inputs = combine(inputs, normal['outputs'])
    ex = read(r['root_owner_exit']); inputs[ex['log']] = ex['log_sha256']
    sources = {str(Path(m.__file__).absolute()): sha(m.__file__) for m in (c,budget)}
    sources[r['original_ledger_module']] = sha(r['original_ledger_module'])
    return dict(status=budget.STATUS, phase=budget.PHASE, **{k:r[k] for k in keys}, ledger=r['ledger'], phase_caps=read(r['root_registration'])['phase_caps'], budget_before=normal['budget'], stage_call_cap=plan['maximum_new_packet_calls'], event_namespace='MAIN_RAW64_ACTUAL_V1', plan_sha256=c.digest(plan), source_bindings=sources, input_bindings=inputs)


def register(request_path):
    r = read(request_path); ctx = inspect_inputs(r); qualification(r, ctx['bound']); u.check_stop(r)
    root, e, out = (Path(r[k]).absolute() for k in ('root','execution_dir','out'))
    require(root in e.parents and root in out.parents and e != out and e not in out.parents and out not in e.parents and not e.exists() and not out.exists(), 'Fresh separate execution/output below new root required')
    with u.owner_lock(r):
        e.mkdir(parents=True)
        try:
            save(e/'plan.json', ctx['plan'])
            br = budget_registration(r, ctx['plan']); save(e/'budget_registration.json', br)
            budget.inspect_registration(e/'budget_registration.json')
            cfg = dict(r, _path=str(e/'config.json'), registration=str(e/'registration.json'), plan=str(e/'plan.json'), budget_registration=str(e/'budget_registration.json'))
            save(cfg['_path'], cfg)
            inputs = combine(ctx['bound'], {str(e/'plan.json'):sha(e/'plan.json'), str(e/'budget_registration.json'):sha(e/'budget_registration.json'), str(Path(request_path).absolute()):sha(request_path)})
            reg = dict(status=REGISTERED, scope='actual_calibration_only', source_bindings=r['source_bindings'], input_bindings={p:h for p,h in inputs.items() if p not in r['source_bindings']}, config_sha256=sha(cfg['_path']), plan_sha256=c.digest(ctx['plan']), budget_before=br['budget_before'], expected_frames=ctx['plan']['frame_count'], automatic_successor=False, GPU_used=False)
            save(cfg['registration'], reg)
            budget.CalibrationBudget.install(cfg['budget_registration'])
            save(e/'registration_completion.json', dict(status='MAIN_RAW64_ACTUAL_REGISTRATION_SEALED_V1', outputs={p:sha(p) for p in (cfg['_path'],cfg['registration'],cfg['budget_registration'],cfg['plan'])}, new_packet_decodes=0))
            return cfg['_path']
        except BaseException:
            save(e/'registration_failure.json', dict(traceback=traceback.format_exc(), retry_allowed=False)); raise


def load_registered(config_path):
    cfg = read(config_path); require(cfg['_path'] == str(Path(config_path).absolute()), 'Explicit registered config path required')
    reg = read(cfg['registration']); require(reg['status'] == REGISTERED and reg['config_sha256'] == sha(config_path), 'Registration/config mismatch')
    seal = read(Path(cfg['execution_dir'])/'registration_completion.json'); u.verify(seal['outputs'])
    require(seal['outputs'].get(cfg['registration']) == sha(cfg['registration']) and seal['outputs'].get(config_path) == sha(config_path), 'Missing final registration seal')
    ctx = inspect_inputs(dict(cfg, input_bindings=reg['input_bindings']))
    plan = pin(cfg['plan'], ctx['bound']); require(plan == ctx['plan'] and c.digest(plan) == reg['plan_sha256'], 'Frozen plan changed')
    ctx.update(cfg=cfg, reg=reg, regsha=sha(cfg['registration']), gate=budget.CalibrationBudget(cfg['budget_registration']))
    qualification(cfg, ctx['bound']); return ctx


def budget_snapshot(ctx):
    return ctx['gate'].snapshot()


def command(cfg, index=None):
    return [cfg['python'],'-B',str(Path(__file__).absolute()),'--config',cfg['_path'],'--stage','merge' if index is None else 'worker'] + ([] if index is None else ['--index',str(index)])


def proof_path(cfg,index):
    return Path(cfg['execution_dir'])/('merge' if index is None else 'worker_'+str(index))/'admission.json'


def check_live(ctx, proof):
    cfg = ctx['cfg']; u.check_stop(cfg); u.require_resources(cfg, 0 if proof['worker_index'] is None else proof['worker_index'])
    require(u.same(u.process(os.getpid()),proof['worker_identity']) and u.same(u.process(os.getppid()),proof['owner_identity']), 'Original live owner/child identity required')
    import fcntl
    with Path(cfg['ledger_lock']).open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(f,fcntl.LOCK_UN); raise ValueError('Registered owner lock is not held')


def admit_worker(ctx,index):
    cfg = ctx['cfg']; path = proof_path(cfg,index); until=time.monotonic()+15
    while not path.exists():
        u.check_stop(cfg); require(os.getppid()>1 and time.monotonic()<until,'No live owner admission'); time.sleep(.02)
    proof=read(path)
    require(proof['registration_sha256']==ctx['regsha'] and proof['config_sha256']==sha(cfg['_path']) and proof['worker_index']==index and proof['worker_identity']['argv']==command(cfg,index),'Foreign worker/merge admission')
    check_live(ctx,proof); return proof


def token_source(ctx,index):
    record=ctx['records'][index]; cp=pin(record['checkpoint'],ctx['bound']); archive=record['archive']
    require(cp['source_index']==index and cp['source_id']==record['source_id'] and cp['tokens_sha256']==record['tokens_sha256'] and cp['archive']==archive and cp['outputs'].get(archive)==ctx['bound'].get(archive)==sha(archive),'Original raw token source changed')
    with np.load(archive,allow_pickle=False) as z:
        require(set(z.files)=={'tokens','pixels'},'Original source NPZ schema differs'); tokens=z['tokens']
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),'Original680 tokens required')
    require(hashlib.sha256(b'int64:680\0'+tokens.astype('<i8',copy=False).tobytes()).hexdigest()==record['tokens_sha256'],'Original token hash differs')
    prefix=(0,1,5,14,30,55,91,155,255,424,680)
    return record,[tokens[prefix[i]:prefix[i+1]] for i in range(10)]


def scientific_runtime(ctx,proof):
    cfg=ctx['cfg']; acfg=dict(cfg,plan=cfg['original_cpu_plan'],source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'])
    adapter=module(cfg['adapter_module'],'main_raw64_packet_adapter',ctx['bound']).create(acfg)
    rx=module(cfg['receiver_module'],'main_raw64_keep_receiver',ctx['bound'])
    old=module(cfg['original_receiver_module'],'main_raw64_original_receiver_for_cal',ctx['bound'])
    legacy_profiles=pin(ctx['legacy']['cfg']['profiles'],ctx['bound']); aliases=pin(ctx['legacy']['cfg']['aliases'],ctx['bound'])
    cat=rx.MixedPublicCatalogue(ctx['catalogue_data'],legacy_profiles,aliases,legacy_backend_identity=adapter.backend.identity,original_receiver=old)
    legacy=old.PublicCatalogue(legacy_profiles,aliases,catalogue_digest=c.digest(legacy_profiles),aliases_digest=c.digest(aliases),backend_identity=adapter.backend.identity)
    def charge(event,request,callback):
        check_live(ctx,proof); return ctx['gate'].decode_once(event,request,callback,proof['worker_identity'])
    receiver=rx.Receiver(cat,backend=adapter.backend,legacy_phy=adapter.legacy,header=adapter.header,charge=charge)
    return dict(adapter=adapter,rx=rx,cat=cat,legacy=legacy,old=old,receiver=receiver)


def paid_rows(path,event_ids):
    with closing(sqlite3.connect('file:'+Path(path).as_posix()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        result={}
        for eid in event_ids:
            row=db.execute('SELECT * FROM events WHERE event_id=?',(eid,)).fetchone(); require(row is not None,'Missing actual paid event'); result[eid]=dict(row)
        return result


def legacy_source(ctx,index):
    old=ctx['legacy']; cp_path=str(Path(old['cfg']['out'])/'source_checkpoints'/f'{index:04d}.json')
    cp=pin(cp_path,old['bound']); require(cp['status']=='MAIN_ACTUAL_CALIBRATION_SOURCE_COMPLETE' and cp['source_index']==index and cp['source_id']==ctx['records'][index]['source_id'] and cp['source_tokens_sha256']==ctx['records'][index]['tokens_sha256'] and cp['registration_sha256']==sha(ctx['cfg']['legacy_batch']['registration']),'Wrong original source trace checkpoint')
    tp=str(Path(old['cfg']['out'])/'traces'/f'{index:04d}.json'); require(cp['outputs'].get(tp)==old['bound'].get(tp)==sha(tp),'Original trace not sealed')
    rows=read(tp); require(len(rows)==cp['frame_count']==30,'Original30-frame source grid required')
    mapping={(r['candidate_id'],r['snr_db'],r['noise_seed']):r for r in rows}; require(len(mapping)==30,'Duplicate original trace frame')
    pins={p:sha(p) for p in old['receipts']+[cp_path,tp]}
    return mapping,tp,pins


def execute_point(ctx,proof,runtime,record,scales,point,oldrows,oldtrace,oldpins):
    cfg=ctx['cfg']; cat=runtime['cat']; output=[]
    payload=runtime['old'].serialize_raw(scales,cat.entry(point['profile_id']))
    for seed in c.NOISE:
        check_live(ctx,proof); counter=c.frame_counter(record['source_index'],point['snr_db'],seed)
        wave,tx=runtime['rx'].transmit_frame(cat,runtime['adapter'].backend,runtime['adapter'].legacy,runtime['adapter'].header,point['profile_id'],payload,counter)
        expected,received=c.prepare_frame(record,point,seed,payload,wave,tx)
        prior=oldrows.get((point['candidate_id'],point['snr_db'],seed))
        # Accepted unknown old IDs cannot be reinterpreted under a larger domain.
        reuse=prior is not None and not (prior['header']['header_crc_ok'] and not prior['header']['header_ok'])
        if reuse:
            row=prior; events=paid_rows(ctx['legacy']['cfg']['ledger'],row['packet_event_ids'])
            evidence=dict(paid_results_verified=True,normal_exit_verified=True,normal_receipts=ctx['legacy']['receipts'],trace_path=oldtrace,bindings=combine(oldpins,{record['checkpoint']:ctx['bound'][record['checkpoint']]}))
        else:
            prefix=f'MAIN_RAW64_ACTUAL_V1/{point["wire_key"]}/snr{point["snr_db"]}/source{record["source_index"]:04d}/noise{seed}'
            row=runtime['receiver'].receive(received,float(point['snr_db']),counter,phase='actual_calibration',event_prefix=prefix)
            row.update(expected,slot=point['slot'],family_memberships=point['family_memberships'],new_metric_calls=0,quality_scored=False)
            row['body_received_float32_sha256']=None if not row['header_ok'] else c.array_sha(received[68:68+cat.entry(row['rx_profile_id'])['groups'][0]['symbols']].astype(np.float32))
            tp=Path(cfg['out'])/'raw_traces'/f'{record["source_index"]:04d}'/f'snr{point["snr_db"]}'/point['candidate_id']/(str(seed)+'.json'); save(tp,[row])
            ap=str(proof_path(cfg,proof['worker_index'])); evidence=dict(paid_results_verified=True,live_owner_admission_verified=True,admission_receipts=[ap],trace_path=str(tp),bindings={str(tp):sha(tp),ap:sha(ap),record['checkpoint']:ctx['bound'][record['checkpoint']]})
            events=paid_rows(cfg['ledger'],row['packet_event_ids'])
        output.append(c.admit_trace(row,expected,point,received,catalogue=cat,present_actual=runtime['old'].present_actual,paid_events=events,evidence=evidence,legacy_catalogue=runtime['legacy'] if reuse else None))
    return c.write_point(cfg['out'],ctx['plan'],record['source_index'],point,output,ctx['regsha'])


def worker(config_path,index):
    ctx=load_registered(config_path); cfg=ctx['cfg']; require(index in (0,1),'Only two registered workers')
    proof=admit_worker(ctx,index); root=Path(cfg['out'])/('worker_'+str(index)); root.mkdir(exist_ok=False)
    try:
        runtime=scientific_runtime(ctx,proof); outputs={}; calls=reused=0
        for i in range(index,1000,2):
            record,scales=token_source(ctx,i); rows,tp,pins=legacy_source(ctx,i)
            for point in ctx['plan']['schedule']:
                cp=execute_point(ctx,proof,runtime,record,scales,point,rows,tp,pins); outputs[cp['path']]=cp['sha256']; value=read(cp['path'])
                calls+=value['newly_charged_packet_calls']; reused+=value['reused_frames']
        check_live(ctx,proof)
        done=dict(status=WORKER_DONE,registration_sha256=ctx['regsha'],config_sha256=sha(cfg['_path']),worker_index=index,worker_identity=proof['worker_identity'],source_indices=list(range(index,1000,2)),frame_count=500*3*len(ctx['plan']['schedule']),new_packet_calls=calls,reused_frames=reused,outputs=outputs,GPU_used=False,quality_ranked=False)
        save(root/'completion.json',done)
    except BaseException:
        save(root/'failure.json',dict(traceback=traceback.format_exc(),worker_identity=proof['worker_identity'],retry_allowed=False)); raise


def closed_workers(ctx):
    result=[]; cfg=ctx['cfg']; attempt=read(Path(cfg['execution_dir'])/'attempt.json')
    require(attempt['registration_sha256']==ctx['regsha'],'Wrong original owner attempt')
    for i in (0,1):
        stage=Path(cfg['execution_dir'])/('worker_'+str(i)); end=read(stage/'exit.json'); proof=read(stage/'admission.json'); done=read(Path(cfg['out'])/('worker_'+str(i))/'completion.json')
        require(end['process_waited'] is True and end['exit_code']==0 and sha(end['log'])==end['log_sha256'] and u.same(end['identity'],proof['worker_identity']) and u.same(end['identity'],done['worker_identity']),'Worker was not normally reaped')
        require(proof['registration_sha256']==done['registration_sha256']==ctx['regsha'] and done['config_sha256']==sha(cfg['_path']) and end['identity']['argv']==command(cfg,i) and done['status']==WORKER_DONE and done['worker_index']==i and done['source_indices']==list(range(i,1000,2)),'Wrong worker/source assignment')
        require(proof['config_sha256']==sha(cfg['_path']) and u.same(proof['owner_identity'],attempt['identity']),'Worker belongs to another owner attempt')
        gone(end['identity']); result.append((done,stage))
    return result


def audit_inventory(ctx):
    workers=closed_workers(ctx); cfg=ctx['cfg']; before=budget_snapshot(ctx); require(before['unresolved']==0,'Actual ledger must be quiescent before merge')
    actual=ctx['gate'].ledger.events('actual_calibration'); seen=set(); outputs={}; points=[]; calls=reused=0
    original=module(cfg['original_receiver_module'],'main_raw64_audit_pure_receiver',ctx['bound'])
    catalogue=type('BoundCatalogue',(),{'entry':lambda self,pid:deepcopy(ctx['profiles'][pid])})()
    for i,(done,stage) in enumerate(workers):
        expected={str(Path(cfg['out'])/c.point_relative_path(j,p)) for j in range(i,1000,2) for p in ctx['plan']['schedule']}
        require(set(done['outputs'])==expected,'Worker point inventory differs')
        subtotal=subreuse=0
        for j in range(i,1000,2):
            for point in ctx['plan']['schedule']:
                path=str(Path(cfg['out'])/c.point_relative_path(j,point)); pinvalue=dict(path=path,sha256=done['outputs'][path]); cp=c.read_point(pinvalue,ctx['plan'],j,point,ctx['regsha'])
                c.verify_files(cp['input_bindings'])
                for row in cp['frames']:
                    require(row['source_index']==j and row['source_id']==ctx['records'][j]['source_id'] and row['source_tokens_sha256']==ctx['records'][j]['tokens_sha256'] and row['population_role']=='calibration' and all(row[k]==point[k] for k in ('candidate_id','profile_id','wire_key','snr_db')) and row['public_frame_counter']==c.frame_counter(j,point['snr_db'],row['noise_seed']),'Point frame source/public identity differs')
                    prov=row['execution_provenance']; originals=[r for r in read(prov['original_trace']['path']) if c.digest(r)==prov['original_row_sha256']]
                    require(len(originals)==1,'Original frame bytes missing/duplicated')
                    base=originals[0]
                    require(all(row[k]==v for k,v in base.items() if k not in ('slot','family_memberships')) and row['effective_catalogue_digest']==ctx['plan']['catalogue_digest'] and row['slot']==point['slot'] and row['family_memberships']==point['family_memberships'],'Projected CPU frame changed original facts')
                    require(all(row[k]==v for k,v in original.present_actual(row['header'],row['body'],catalogue).items()),'RX hard state differs from actual header/body')
                    events=actual if prov['origin']=='NEW_PAID_RECEIVE' else paid_rows(ctx['legacy']['cfg']['ledger'],row['packet_event_ids'])
                    # Pure event validation; no backend/no waveform regeneration in merge.
                    c._paid(row,events,catalogue)
                    if prov['origin']=='NEW_PAID_RECEIVE':
                        if row['header_ok']:
                            body_request=c.event_request(row,actual[row['packet_events'][1]['event_id']],'body',catalogue)
                            group=catalogue.entry(row['rx_profile_id'])['groups'][0]
                            require(body_request['physical_definition_sha256']==c.digest(original.physical_definition(group,group['layout']['implementation'])),'Paid actual received-ID physical definition differs')
                        for eid in row['packet_event_ids']:
                            require(eid not in seen and u.same(json.loads(actual[eid]['worker']),done['worker_identity']),'Duplicate/foreign current paid event'); seen.add(eid)
                    else:
                        require(prov['origin']=='HISTORICAL_EXACT_REUSE' and prov['receiver_interpretation_unchanged'] is True and set(prov['normal_receipts'])==set(ctx['legacy']['receipts']),'Unadmitted historical reuse')
                subtotal+=cp['newly_charged_packet_calls']; subreuse+=cp['reused_frames']
                points.append(dict(source_index=j,source_id=ctx['records'][j]['source_id'],snr_db=point['snr_db'],candidate_id=point['candidate_id'],checkpoint=pinvalue)); outputs[path]=pinvalue['sha256']
        require(subtotal==done['new_packet_calls'] and subreuse==done['reused_frames'] and done['frame_count']==500*3*len(ctx['plan']['schedule']),'Worker totals differ from checkpoints')
        calls+=subtotal; reused+=subreuse
        for p in (stage/'exit.json',stage/'admission.json',Path(cfg['out'])/('worker_'+str(i))/'completion.json'):
            outputs[str(p)]=sha(p)
        end=read(stage/'exit.json'); outputs[end['log']]=end['log_sha256']
    require(set(actual)==seen and len(seen)==calls==before['phase_charged']['actual_calibration'] and budget_snapshot(ctx)==before,'Current point inventory and root budget disagree')
    return dict(points=sorted(points,key=lambda p:(p['source_index'],p['snr_db'],p['candidate_id'])),outputs=outputs,frame_count=ctx['plan']['frame_count'],new_packet_calls=calls,reused_frames=reused,budget=before)


def merge(config_path):
    ctx=load_registered(config_path); proof=admit_worker(ctx,None); audit=audit_inventory(ctx); check_live(ctx,proof)
    done=dict(status=DONE,registration_sha256=ctx['regsha'],config_sha256=sha(config_path),worker_identity=proof['worker_identity'],source_count=1000,frame_count=audit['frame_count'],new_packet_calls=audit['new_packet_calls'],reused_frames=audit['reused_frames'],budget_before=ctx['reg']['budget_before'],budget_after=audit['budget'],points=audit['points'],outputs=audit['outputs'],source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'],GPU_used=False,quality_ranked=False,development_used=False,holdout_used=False,automatic_successor_started=False)
    save(Path(ctx['cfg']['out'])/'completion.json',done)


def spawn(ctx,index,children):
    cfg=ctx['cfg']; stage=proof_path(cfg,index).parent; stage.mkdir(); argv=command(cfg,index); logpath=stage/'worker.log'; log=logpath.open('xb')
    try: child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=u.environment(cfg),preexec_fn=lambda:u.set_resources(cfg['resources']['affinities'][0 if index is None else index]))
    except BaseException: log.close(); raise
    row=[child,log,logpath,dict(pid=child.pid,argv=argv,identity_verified=False),stage/'exit.json']; children.append(row); save(stage/'spawn.json',row[3])
    ident=u.capture(child,argv); row[3]=ident
    save(stage/'admission.json',dict(registration_sha256=ctx['regsha'],config_sha256=sha(cfg['_path']),owner_identity=u.process(os.getpid()),worker_identity=ident,worker_index=index))
    return row


def monitor(ctx,children,start):
    while any(x[0].poll() is None for x in children):
        u.check_stop(ctx['cfg']); require(time.monotonic()-start<ctx['cfg']['max_seconds'],'Finite calibration wall-time exceeded'); require(all(x[0].poll() in (None,0) for x in children),'A registered child failed'); time.sleep(.2)
    for row in children: require(u.wait_closed(*row)==0,'Normal child exit0 required')


def run(config_path):
    ctx=load_registered(config_path); cfg=ctx['cfg']; require(sys.platform.startswith('linux') and str(Path(sys.executable).absolute())==cfg['python'],'Original LDPC interpreter required')
    u.check_stop(cfg); u.set_resources(sum(cfg['resources']['affinities'],[])); children=[]; start=time.monotonic()
    with u.owner_lock(cfg):
        save(Path(cfg['execution_dir'])/'attempt.json',dict(registration_sha256=ctx['regsha'],identity=u.process(os.getpid())))
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        def interrupted(signum,frame):
            for s in previous: signal.signal(s,signal.SIG_IGN)
            raise InterruptedError('Actual owner interrupted')
        for s in previous: signal.signal(s,interrupted)
        try:
            require(budget_snapshot(ctx)==ctx['reg']['budget_before'],'No actual calls before one registered owner attempt'); Path(cfg['out']).mkdir(parents=True,exist_ok=False)
            spawn(ctx,0,children); spawn(ctx,1,children); monitor(ctx,children,start)
            spawn(ctx,None,children); merge_children=children[-1:]; monitor(ctx,merge_children,start)
            science_path=str(Path(cfg['out'])/'completion.json'); science=read(science_path)
            require(science['status']==DONE and science['registration_sha256']==ctx['regsha'] and u.same(science['worker_identity'],merge_children[0][3]) and science['budget_after']==budget_snapshot(ctx),'Wrong scientific merge completion')
            attempt=str(Path(cfg['execution_dir'])/'attempt.json'); outputs={science_path:sha(science_path),attempt:sha(attempt)}
            for row in children:
                for p in (row[4],row[4].parent/'admission.json',row[2]): outputs[str(p)]=sha(p)
            done=dict(status=NORMAL,registration_sha256=ctx['regsha'],config_sha256=sha(config_path),owner_identity=u.process(os.getpid()),outputs=outputs,source_count=1000,frame_count=science['frame_count'],new_packet_calls=science['new_packet_calls'],reused_frames=science['reused_frames'],budget_before=ctx['reg']['budget_before'],budget_after=science['budget_after'],all_waited=True,worker_exit_codes=[0,0],merge_exit_code=0,automatic_successor_started=False,GPU_used=False)
            save(Path(cfg['execution_dir'])/'completion.json',done); return done
        except BaseException:
            detail=traceback.format_exc()
            try: u.request_stop(cfg,'Actual calibration failed')
            except BaseException: detail+='\nSTOP_WRITE_FAILED\n'+traceback.format_exc()
            errors=u.drain_children(children)
            save(Path(cfg['execution_dir'])/'failure.json',dict(traceback=detail,all_children_waited=True,drain_evidence_errors=errors,retry_allowed=False)); raise
        finally:
            for s,handler in previous.items(): signal.signal(s,handler)


def closed_calibration(spec):
    """Downstream admission after the durable caller has reaped this CPU owner."""
    u.verify(spec['bindings'])
    for k in ('config','registration','completion','owner_exit'): require(spec['bindings'].get(spec[k])==sha(spec[k]),'Direct normal calibration control pin required')
    cfg=pin(spec['config'],spec['bindings']); require(cfg['registration']==spec['registration'],'Wrong self registration pin')
    ctx=load_registered(spec['config']); done=pin(spec['completion'],spec['bindings']); end=waited(spec['owner_exit'],spec['bindings'],done['owner_identity'])
    require(end['identity']['argv']==[cfg['python'],'-B',str(Path(__file__).absolute()),'--config',cfg['_path'],'--stage','run'],'Wrong actual owner argv')
    require(done['status']==NORMAL and done['registration_sha256']==ctx['regsha'] and done['config_sha256']==sha(cfg['_path']) and done['all_waited'] is True and done['worker_exit_codes']==[0,0] and done['merge_exit_code']==0 and done['automatic_successor_started'] is False,'Incomplete normal actual calibration')
    attempt=str(Path(cfg['execution_dir'])/'attempt.json'); require(done['outputs'].get(attempt)==sha(attempt) and u.same(read(attempt)['identity'],done['owner_identity']),'Normal owner identity differs from single attempt')
    require(not (Path(cfg['execution_dir'])/'failure.json').exists(),'Failed owner is not normal'); u.verify(done['outputs'])
    science_path=str(Path(cfg['out'])/'completion.json'); science=read(science_path); audit=audit_inventory(ctx)
    require(science['status']==DONE and science['registration_sha256']==ctx['regsha'] and science['config_sha256']==sha(cfg['_path']) and science['points']==audit['points'] and science['outputs']==audit['outputs'],'Actual merge checkpoint inventory differs')
    require(science['source_bindings']==ctx['reg']['source_bindings'] and science['input_bindings']==ctx['reg']['input_bindings'] and all(science[k] is False for k in ('GPU_used','quality_ranked','development_used','holdout_used','automatic_successor_started')),'Scientific scope/source identity differs')
    for k in ('frame_count','new_packet_calls','reused_frames'): require(done[k]==science[k]==audit[k],'Normal scientific summary differs')
    require(done['budget_before']==science['budget_before']==ctx['reg']['budget_before'] and done['budget_after']==science['budget_after']==audit['budget'],'Actual normal budget differs')
    merge_stage=Path(cfg['execution_dir'])/'merge'; merge_end=read(merge_stage/'exit.json'); merge_proof=read(merge_stage/'admission.json')
    require(merge_end['process_waited'] is True and merge_end['exit_code']==0 and merge_end['log_sha256']==sha(merge_end['log']) and u.same(merge_end['identity'],science['worker_identity']) and u.same(merge_end['identity'],merge_proof['worker_identity']) and merge_end['identity']['argv']==command(cfg,None),'Merge was not normally reaped'); gone(merge_end['identity'])
    bindings=combine(ctx['bound'],spec['bindings'],done['outputs'],audit['outputs'],{science_path:sha(science_path),cfg['registration']:ctx['regsha']})
    return dict(ctx=ctx,done=done,bindings=bindings,plan=ctx['plan'],points=audit['points'],records=ctx['records'],budget_before=audit['budget'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--config'); p.add_argument('--materials');p.add_argument('--request-out');p.add_argument('--stage',choices=('prepare','register','run','worker','merge'),required=True); p.add_argument('--index',type=int); a=p.parse_args()
    if a.stage=='prepare':print(c.canonical(build_request(a.materials,a.request_out)),flush=True)
    elif a.stage=='register': print(register(a.request),flush=True)
    elif a.stage=='run': print(run(a.config)['status'],flush=True)
    elif a.stage=='worker': worker(a.config,a.index)
    else: merge(a.config)


if __name__=='__main__': main()
