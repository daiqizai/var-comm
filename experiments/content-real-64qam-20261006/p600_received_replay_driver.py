"""Separate registered P600 GPU replay of sealed received latents, with no channel.

The H18 metric predecessor must have closed normally. This worker writes its
scientific receipt while its own exclusive owner is alive; the original owner
certifies this worker's successful exit afterwards.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np

DONE = 'P600_RECEIVED_LATENT_CONVNEXT_COMPLETE'
SCOPE = 'P600_EXISTING_RECEIVED_LATENT_RGB_AND_INDEPENDENT_CONVNEXT_ONLY'
DEADLINE = 1791564605.9549868
STOP = False
RGB_BYTES = 600 * 3 * 256 * 256 * 4
MAX_ARCHIVE_BYTES = 1024**3
STORAGE_RESERVE_BYTES = 8 * 1024**3
REQUIRED = ('owner_module','wait_module','cpu_driver_module','h_metric_driver_module',
    'core_module','replay_driver_module','visual_owner_config','protocol','budget_registration',
    'latent_completion','latent_inventory','scalar_inventory','metrics_registration',
    'numerical_reference','selected_P_policy','quality_driver_module','static_closure_module',
    'visual_source_closure','validation_module','convnext_weights','independent_binding')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def bind(paths): return {str(p): sha(p) for p in paths}


def merge(*maps):
    result = {}
    for values in maps:
        for p, s in values.items():
            require(p not in result or result[p] == s, 'Conflicting SHA: ' + p)
            result[p] = s
    return result


def verify(values):
    for p, s in values.items():
        require(Path(p).is_absolute() and sha(p) == s, 'Changed bound file: ' + p)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded
    spec.loader.exec_module(loaded); return loaded


def stop(*_):
    global STOP
    STOP = True


def original_metrics_closed(spec, metric_module):
    """One complete predecessor context reused throughout this worker."""
    require(set(spec) == {'config','owner_config','registration','launch','completion'},
            'Exact normal H18 metric predecessor required')
    cfg = read(spec['config']); reg = read(spec['registration'])
    require(cfg['metric_driver_module'] == metric_module
            and cfg['registration'] == spec['registration']
            and cfg['visual_owner_config'] == spec['owner_config'], 'Metric predecessor lineage differs')
    # These modules have ordinary sibling imports; all imported paths must first
    # be in the original frozen registration, which is verified before loading.
    oldbound = merge(reg['source_bindings'], reg['input_bindings']); verify(oldbound)
    require(oldbound.get(metric_module) == sha(metric_module), 'Original metric entry unbound')
    sys.path.insert(0, str(Path(metric_module).parent))
    md = module(metric_module, '_p600_closed_h_metric_driver')
    ctx = md.load_registered(spec['config'])
    expected = dict(status=md.DONE, source_count=100, frame_count=5400, H_policy_snr_points=18,
        MAIN_frames=0, new_packet_decodes=0, policy_selection=False, development_used=True,
        holdout_used=False, unified_neural_metrics_run=True, overall_development_complete=False,
        H_full_delivery_claimed=False)
    closed = ctx['cpu'].closed_batch(ctx['owner'], ctx['wait'],
        dict(config=spec['owner_config'], registration=spec['registration'],
             launch=spec['launch'], completion=spec['completion']), cfg['owner_module'], expected,
        ctx['owner'].raw_process_state)
    old = closed['owner']; done = closed['done']
    require(len(old['stages']) == 1 and old['stages'][0]['id'] == 'development'
            and old['stages'][0]['resource'] == 'gpu' and len(old['stages'][0]['jobs']) == 1,
            'One normally completed metric GPU job required')
    job = old['stages'][0]['jobs'][0]
    require(job['id'] == 'development_metrics' and job['argv'][1:] ==
            ['-B', metric_module, '--config', spec['config']], 'Wrong predecessor GPU command')
    require(done['source_ids'] == ctx['source_ids'] and done['budget_before'] == done['budget_after']
            == closed['owner_done']['budget'] == ctx['before'], 'Metric closure budget/population differs')
    return dict(metric=md, context=ctx, config=cfg, closed=closed, before=ctx['before'],
        spec=dict(spec),metric_module=metric_module,
        bindings=merge(oldbound, ctx['bindings'], closed['bindings'], done['outputs'], bind(spec.values())))


def original_assets(ctx, corectx):
    pop, manifest, completion = ctx['population'], ctx['manifest'], ctx['assets']
    require(pop['stage'] == pop['calibration_or_development'] == 'm1_development'
            and pop['source_ids'] == manifest['source_ids'] == corectx['source_ids']
            and pop['preprocessing_ids'] == corectx['preprocessing_ids']
            and len(manifest['records']) == 100, 'Original development source100 differs')
    require(manifest['status'] == completion['status'] ==
            'H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
            and manifest['source_count'] == completion['source_count'] == 100,
            'Sealed original source-only assets required')
    for index, item in enumerate(manifest['records']):
        path = item['checkpoint']
        require(completion['outputs'].get(path) == item['checkpoint_sha256'] == sha(path),
                'Source checkpoint missing/unsealed before model loading')
        cp = read(path)
        for row in (item, cp):
            require(row['source_index'] == index and row['source_id'] == pop['source_ids'][index]
                    and row['preprocessing_id'] == pop['preprocessing_ids'][index]
                    and row['original_development_data_binding'] == pop['data_bindings'][index],
                    'Original pixel/source identity differs')
        label = cp['evaluation_class_index']
        require(type(label) is int and 0 <= label < 1000 and item['evaluation_class_index'] == label
                and item['archive'] == cp['archive'] and len(cp['outputs']) == 1
                and cp['outputs'].get(cp['archive']) == completion['outputs'].get(cp['archive']),
                'Source archive/evaluation label differs')
        require(all(int(corectx['scalar'][index, s, n]['true_class_index']) == label
                    for s in (13,19) for n in (2001,2002,2003)), 'Original P/source label differs')
    return pop, manifest, completion


def audited_input_paths(cfg, audit):
    for key in ('scalar_inventory','metrics_registration'):
        require(audit['input_bindings'].get(cfg[key]) == sha(cfg[key]),
                'Configured original P input is not the exact latent-audited file: ' + key)
    if cfg['numerical_reference'] in audit['input_bindings']:
        require(audit['input_bindings'][cfg['numerical_reference']] == sha(cfg['numerical_reference']),
                'Latent-audited native qualification changed')


def inspect_inputs(cfg, bound, *, prior=None):
    """Metadata-only registrar preflight; does not need new config/reg/owner files.

    ``bound`` is the proposed exact source/input union. No models, image arrays,
    registration writes or launches occur. Returned context is reused by a
    single load/run; no historical graph is rechecked inside source iteration.
    """
    verify(bound)
    require(cfg['schema'] == 'P600_RECEIVED_REPLAY_CONFIG_V1', 'P600 replay config required')
    pending = {'visual_owner_config'}
    for p in (str(Path(__file__).absolute()), *[cfg[k] for k in REQUIRED if k not in pending], *cfg['metric_batch'].values()):
        require(bound.get(p) == sha(p), 'Replay dependency/input not preregistered: ' + p)
    require(cfg['replay_driver_module'] == str(Path(__file__).absolute()), 'Wrong registered replay entry')
    if prior is None:
        prior = original_metrics_closed(cfg['metric_batch'], cfg['h_metric_driver_module'])
    else:
        # Registrar may reuse its in-memory completed preflight, never a JSON
        # substitute or an assumed dead PID. Production load does not use this.
        require(prior['spec'] == cfg['metric_batch'] and prior['metric_module'] == cfg['h_metric_driver_module']
                and prior['closed']['normal_owner_success'] is True,
                'Reused predecessor context is not this normal metric closure')
        for p in cfg['metric_batch'].values():
            require(prior['bindings'].get(p) == sha(p), 'Predecessor changed during registrar preflight')
    pc, mc = prior['context'], prior['config']; visual = pc['render_cfg']
    for k in ('root','H_out','owner_module','wait_module','cpu_driver_module','ledger',
              'budget_registration','phase_limits','stop_file','protocol'):
        require(cfg[k] == mc[k], 'Original H execution input changed: ' + k)
    for k in ('runtime_dir','native_runtime','uep_runtime','var_source','dino_source',
              'static_closure_module','numerical_reference'):
        require(cfg[k] == visual[k], 'Original frozen native prerequisite changed: ' + k)
    for k in ('validation_module','convnext_weights','independent_binding','metrics_registration'):
        require(cfg[k] == mc[k], 'Original metric/classifier identity changed: ' + k)
    prior['metric'].u.assert_budget(prior['before'], prior['before'], pc['group'])
    require(all(bound.get(p) == s for p,s in prior['bindings'].items()), 'Original metric closure not inherited')
    audit = read(cfg['latent_completion']); inv = read(cfg['latent_inventory'])
    require(audit['outputs'] == {cfg['latent_inventory']: sha(cfg['latent_inventory'])}, 'Actual latent output differs')
    audited_input_paths(cfg,audit)
    for collection in (audit['input_bindings'], audit['source_bindings']):
        require(all(bound.get(p) == s for p,s in collection.items()), 'Original latent audit dependencies not inherited')
    core = module(cfg['core_module'], '_p600_received_replay_core')
    cctx = core.context(audit, inv, read(cfg['scalar_inventory']), read(cfg['metrics_registration']),
        read(cfg['numerical_reference']), inventory_sha256=sha(cfg['latent_inventory']))
    require(inv['input_bindings'].get(cfg['selected_P_policy']) == sha(cfg['selected_P_policy'])
            == cctx['selected_P_sha256'], 'Original paid P checkpoint selection differs')
    independent = read(cfg['independent_binding'])
    require(independent['reference_implementation_sha256'] == sha(cfg['validation_module'])
            and independent['weights_sha256'] == sha(cfg['convnext_weights']) == core.CONVNEXT
            and independent['used_for_selection'] is False, 'Independent classifier admission differs')
    graph = read(cfg['visual_source_closure'])
    closure = module(cfg['static_closure_module'], '_p600_native_source_closure')
    actual = closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
    require(graph['status'] == 'EXACT_SOURCE_CLOSURE_MATCH' and graph['source_bindings'] == actual
            and closure.compare_bindings(actual,bound)['status'] == 'EXACT_SOURCE_CLOSURE_MATCH',
            'Complete current native source closure differs')
    for p,s in read(visual['visual_source_closure'])['source_bindings'].items():
        require(actual.get(p) == s, 'Earlier executed native source changed: ' + p)
    require(cfg['quality_driver_module'] == str(Path(cfg['uep_runtime'])/'quality_driver.py')
            and actual.get(cfg['quality_driver_module']) == sha(cfg['quality_driver_module']),
            'Original qualified native loader differs')
    population, manifest, assets = original_assets(pc, cctx)
    out = Path(cfg['out']); hout = Path(cfg['H_out'])
    require(out.is_absolute() and hout in out.parents and cfg['stop_file'] == str(hout/'STOP')
            and cfg['overall_deadline_unix'] == DEADLINE and type(cfg['max_seconds']) is int
            and 0 < cfg['max_seconds'] <= 86400, 'Finite original deadline/output required')
    for old in (mc['out'], visual['out'], str(Path(cfg['latent_completion']).parent),
                pc['group']['cfg']['out'], str(Path(pc['group']['cfg']['asset_completion']).parent)):
        old = Path(old)
        require(out != old and out not in old.parents and old not in out.parents, 'Replay overlaps immutable input')
    require(type(cfg['max_archive_bytes']) is int and RGB_BYTES < cfg['max_archive_bytes'] <= MAX_ARCHIVE_BYTES,
            'Finite sufficient archive cap required')
    require(not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(), 'STOP blocks replay')
    return dict(cfg=cfg,bound=bound,prior=prior,owner=pc['owner'],core=core,corectx=cctx,
        population=population,manifest=manifest,assets=assets,static_bindings=actual,before=prior['before'])


def load_registered(config_path):
    path = str(Path(config_path).absolute()); cfg = read(path); reg = read(cfg['registration'])
    require(reg['status'] == 'H_EXECUTION_REVISION_REGISTERED' and reg['branch'] == 'H'
            and reg['source_stage_scope'] == SCOPE and reg['allowed_stage_ids'] == ['development'],
            'Independent P600 replay-only registration required')
    bound = merge(reg['source_bindings'], reg['input_bindings'])
    for p in (path,cfg['visual_owner_config']):
        require(bound.get(p) == sha(p), 'Own new config/owner unbound: ' + p)
    ctx = inspect_inputs(cfg,bound)
    require(ctx['before'] == reg['budget_before'] and cfg['phase_limits'] == reg['phase_limits'],
            'Current registration budget differs')
    ctx['reg'] = reg
    return ctx


def load_pixels(ctx, index):
    pop, manifest, completion = ctx['population'],ctx['manifest'],ctx['assets']
    item = manifest['records'][index]; cp_path = item['checkpoint']; cp = read(cp_path)
    require(completion['outputs'][cp_path] == item['checkpoint_sha256'] == sha(cp_path), 'Source checkpoint changed')
    require(cp['source_index'] == index and cp['source_id'] == pop['source_ids'][index]
            and cp['preprocessing_id'] == pop['preprocessing_ids'][index], 'Pixel source identity changed')
    verify(cp['outputs'])
    require(all(completion['outputs'].get(p) == s for p,s in cp['outputs'].items()), 'Source archive is not sealed')
    with np.load(cp['archive'],allow_pickle=False) as z: pixels = z['pixels'].copy()
    require(pixels.dtype == np.uint8 and pixels.shape == (3,256,256)
            and hashlib.sha256(pixels.tobytes()).hexdigest() == pop['preprocessing_ids'][index], 'Original pixel bytes differ')
    return pixels, merge(cp['outputs'],bind((cp_path,)))


def gpu_admission(ctx, path):
    c = ctx['cfg']; a = ctx['owner']; oc = read(c['visual_owner_config'])
    a.validate_config(oc,ctx['reg'],sha(c['visual_owner_config']))
    require(oc['registration'] == c['registration'] and oc['out'] == c['H_out']
            and len(oc['stages']) == 1 and oc['stages'][0]['id'] == 'development'
            and oc['stages'][0]['resource'] == 'gpu' and len(oc['stages'][0]['jobs']) == 1,
            'Exactly one separate P GPU job required')
    job = oc['stages'][0]['jobs'][0]; argv = job['argv']
    oldargv = ctx['prior']['closed']['owner']['stages'][0]['jobs'][0]['argv']
    require(argv[0] == oldargv[0] and job['id'] == 'p600_received_replay'
            and argv[1:] == ['-B',str(Path(__file__).absolute()),'--config',str(Path(path).absolute())]
            and job['out'] == c['out'] and job['completion'] == str(Path(c['out'])/'completion.json'),
            'Current command/interpreter is not the registered original UM GPU runtime')
    base = Path(oc['owner_out']); ip = base/'owner_identity.json'; identity = read(ip)
    require(int(identity['pid']) == os.getppid() and a.same_identity(identity,a.identity(os.getppid()))
            and identity['config_sha256'] == sha(c['visual_owner_config'])
            and identity['registration_sha256'] == sha(c['registration']) and not (base/'failure.json').exists(),
            'Live exclusive replay parent absent')
    lp = base/'stages/development/workers/p600_received_replay/launch.json'; began = time.monotonic()
    while not lp.exists():
        require(time.monotonic()-began < 10 and a.same_identity(identity,a.identity(os.getppid())),
                'Replay worker launch not sealed'); time.sleep(.1)
    launch = read(lp); current = a.identity(os.getpid()); gp = base/'stages/development/gpu_admission.json'; gate = read(gp)
    require(a.same_identity(current,launch['identity']) and launch['argv'] == argv and launch['resource'] == 'gpu'
            and launch['registration_sha256'] == sha(c['registration']) and launch['threads'] == oc['gpu_threads'] == 6
            and set(launch['affinity']) == set(oc['gpu_affinity']) == set(os.sched_getaffinity(0))
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(oc['gpu_device']), 'Replay resource/identity differs')
    require(gate['status'] == 'GPU_IDLE_CONFIRMED' and gate['registration_sha256'] == sha(c['registration']),
            'Original exclusive GPU admission absent')
    return dict(owner_identity=identity,worker_identity=current,bindings=bind((ip,lp,gp)))


def build_backends(ctx):
    cfg = ctx['cfg']; core = ctx['core']
    for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')): sys.path.insert(0,p)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    quality = module(cfg['quality_driver_module'], '_p600_original_quality_driver')
    native = quality.build_native(Path(cfg['root']),cfg['native_runtime'],stop)
    require(native.driver_bindings == ctx['static_bindings']
            and all(ctx['bound'].get(p) == s for p,s in native.driver_bindings.items()), 'Loaded native closure differs')
    require(native.flags == ctx['corectx']['native_runtime'], 'Original complete native numerical qualification differs')
    decoder = core.FrozenDc(native,ctx['corectx'])
    validation = module(cfg['validation_module'],'_p600_original_convnext')
    classifier = validation.ConvNeXtValidation(cfg['convnext_weights'],core.CONVNEXT,device='cuda:0',
        stage='development',policy_path=cfg['selected_P_policy'],policy_sha256=ctx['corectx']['selected_P_sha256'])
    decoder.check()
    return native,quality,decoder,classifier,core.SealedLatents(ctx['corectx'],native.torch)


def write_source(out, result, input_bindings, regsha, core):
    out = Path(out); index = result['source_index']; sid = result['source_id']
    archive = out/'images'/f'{index:04d}.npz'; rows_path = out/'sources'/f'{index:04d}.json'
    cp_path = out/'source_checkpoints'/f'{index:04d}.json'
    require(len(result['rows']) == 6 and result['RGB_parity_frames'] == result['ConvNeXt_frames'] == 6,
            'Complete source replay required')
    images, rows = {}, []
    for row in result['rows']:
        i,s,n = core.row_key(row); key = f'P1024_snr{s}_noise{n}'
        require(i == index and row['source_id'] == sid and key not in images, 'Wrong/duplicate source frame')
        pixels = core.rgb(result['images'][s,n])
        require(core.rgb_sha(pixels) == row['image_sha256'], 'Replay RGB changed before persistence')
        images[key] = pixels
        rows.append(dict(row,p_replay_image_archive=str(archive),p_replay_image_key=key))
    archive.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as f:
        np.savez(f,**images); f.flush(); os.fsync(f.fileno())
    save(rows_path,rows); outputs = bind((archive,rows_path))
    save(cp_path,dict(status='P600_RECEIVED_REPLAY_SOURCE_COMPLETE',source_index=index,source_id=sid,
        frame_count=6,registration_sha256=regsha,outputs=outputs,input_bindings=input_bindings,
        RGB_parity_frames=6,ConvNeXt_frames=6,source_predictions=1,new_noise_draws=0,new_packet_decodes=0))
    return rows,merge(outputs,bind((cp_path,))),archive.stat().st_size


def read_source_output(out, index, sid, outputs, regsha, core):
    out = Path(out); cp = out/'source_checkpoints'/f'{index:04d}.json'
    rp = out/'sources'/f'{index:04d}.json'; ap = out/'images'/f'{index:04d}.npz'
    for p in (cp,rp,ap): require(outputs.get(str(p)) == sha(p),'Missing/changed replay output')
    check = read(cp)
    require(check['status'] == 'P600_RECEIVED_REPLAY_SOURCE_COMPLETE' and check['source_index'] == index
            and check['source_id'] == sid and check['frame_count'] == 6 and check['registration_sha256'] == regsha
            and check['outputs'] == bind((ap,rp)) and check['RGB_parity_frames'] == check['ConvNeXt_frames'] == 6
            and check['new_noise_draws'] == check['new_packet_decodes'] == 0, 'Replay checkpoint differs')
    verify(check['input_bindings']); rows = read(rp)
    require(len(rows) == 6 and {core.row_key(r) for r in rows} ==
            {(index,s,n) for s in (13,19) for n in (2001,2002,2003)}, 'Persisted source grid differs')
    with np.load(ap,allow_pickle=False) as z:
        require(set(z.files) == {r['p_replay_image_key'] for r in rows},'Unreferenced/missing archived image')
        for row in rows:
            require(row['source_id'] == sid and row['p_replay_image_archive'] == str(ap)
                    and core.rgb_sha(z[row['p_replay_image_key']]) == row['image_sha256'], 'Archived RGB parity differs')
    return rows


def claim_output(out, regsha):
    out = Path(out)
    if out.exists(): require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), 'Prior attempt preserved; no retry')
    else: out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='P600_RECEIVED_REPLAY_ATTEMPT',registration_sha256=regsha,
        pid=os.getpid(),started_unix=time.time(),new_packet_decodes=0,new_noise_draws=0))


def run(config_path):
    ctx = load_registered(config_path); cfg = ctx['cfg']; out = Path(cfg['out']); regsha = sha(cfg['registration'])
    claim_output(out,regsha); began = time.monotonic(); wall = time.time(); native = None
    try:
        supervision = gpu_admission(ctx,config_path)
        require(shutil.disk_usage(out).free >= cfg['max_archive_bytes']+STORAGE_RESERVE_BYTES,
                'Insufficient registered RGB storage headroom')
        # Reuse original hardware monitor. Manual STOP is observed only at a
        # complete-source boundary; hardware/process safety can interrupt calls.
        hardware = ctx['prior']['metric'].GPUHealth()
        def model_guard():
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),
                    'Exclusive replay owner disappeared')
            hardware.check()
            if native is not None: quality.boundary(native)
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),
                    'STOP at replay source boundary')
            elapsed = time.monotonic()-began
            require(elapsed < cfg['max_seconds'] and max(time.time(),wall+elapsed) < DEADLINE,'Original finite deadline')
            hardware.check(force=True); model_guard()
        boundary(); native,quality,decoder,classifier,latents = build_backends(ctx)
        metadata = out/'replay_identity.json'
        save(metadata,dict(frozen_visual_identity=native.loaded['identity'],native_runtime=ctx['corectx']['native_runtime'],
            classifier_identity=classifier.identity,latent_inventory_sha256=sha(cfg['latent_inventory']),
            old_scalar_inventory_sha256=sha(cfg['scalar_inventory']),Dc_batch_size=3,classifier_batch_size=1,
            unused_original_native_models_loaded=True,new_noise_draws=0,new_packet_decodes=0))
        outputs = bind((metadata,)); written = 0
        for index,sid in enumerate(ctx['corectx']['source_ids']):
            boundary(); pixels,inputs = load_pixels(ctx,index)
            result = ctx['core'].replay_source(ctx['corectx'],index,pixels,latents=latents,decoder=decoder,
                classifier=classifier,guard=model_guard)
            rows,new,size = write_source(out,result,inputs,regsha,ctx['core'])
            outputs.update(new); written += size
            require(written <= cfg['max_archive_bytes'],'Registered RGB archive limit exceeded')
            progress = dict(status='RUNNING',completed_sources=index+1,total_sources=100,
                completed_frames=(index+1)*6,registration_sha256=regsha,elapsed_seconds=time.monotonic()-began)
            tmp = out/'status.tmp'; tmp.write_text(json.dumps(progress)+'\n',encoding='utf-8'); os.replace(tmp,out/'status.json')
            del pixels,result,rows
        boundary(); decoder.finish(); allrows = []
        for index,sid in enumerate(ctx['corectx']['source_ids']):
            boundary(); allrows.extend(read_source_output(out,index,sid,outputs,regsha,ctx['core']))
        proof = ctx['core'].validate_complete(ctx['corectx'],allrows)
        after = ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        ctx['prior']['metric'].u.assert_budget(after,ctx['before'],ctx['prior']['context']['group'])
        fp = out/'frame_metrics.json'; save(fp,allrows); outputs.update(bind((fp,)))
        csvpath = out/'frame_metrics.csv'
        fields = sorted({k for r in allrows for k,v in r.items() if v is None or isinstance(v,(str,int,float,bool))})
        with csvpath.open('x',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); writer.writeheader()
            writer.writerows({k:r.get(k) for k in fields} for r in allrows); f.flush(); os.fsync(f.fileno())
        outputs.update(bind((csvpath,))); verify(ctx['bound']); verify(outputs); boundary(); decoder.finish()
        done = dict(status=DONE,registration_sha256=regsha,config_sha256=sha(config_path),
            source_count=100,frame_count=600,P_policy_snr_points=2,N=1024,snrs_db=[13,19],noise_seeds=[2001,2002,2003],
            source_ids=ctx['corectx']['source_ids'],outputs=outputs,input_bindings=ctx['reg']['input_bindings'],
            source_bindings=ctx['reg']['source_bindings'],visual_source_bindings=native.driver_bindings,
            metric_predecessor_completion_sha256=sha(cfg['metric_batch']['completion']),
            latent_inventory_sha256=sha(cfg['latent_inventory']),scalar_inventory_sha256=sha(cfg['scalar_inventory']),
            RGB_exact_parity=True,legacy_scalars_preserved=True,ConvNeXt_scored=True,source_predictions=100,
            reconstruction_predictions=600,Dc_batch_size=3,classifier_batch_size=1,
            frozen_visual_identity=native.loaded['identity'],numerical_runtime=ctx['corectx']['flags'],
            classifier_identity=classifier.identity,core_validation=proof,archive_bytes=written,
            budget_before=ctx['before'],budget_after=after,budget_writes=0,new_packet_decodes=0,new_noise_draws=0,
            GPU_used=True,development_used=True,holdout_used=False,policy_selection=False,training_updates=0,
            unused_original_native_models_loaded=True,statistical_aggregation_run=False,
            MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
            online_latency_measured=False,supervision=supervision,
            own_owner_success_not_yet_certified=True,elapsed_seconds=time.monotonic()-began)
        save(out/'completion.json',done); return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,
            traceback=traceback.format_exc(),new_noise_draws=0,new_packet_decodes=0)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    args = parser.parse_args()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    result = run(args.config)
    print(json.dumps({k:result[k] for k in ('status','source_count','frame_count','new_packet_decodes','new_noise_draws')}))


if __name__ == '__main__': main()
