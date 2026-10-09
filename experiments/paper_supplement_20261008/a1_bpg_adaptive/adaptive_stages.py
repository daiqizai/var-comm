"""Finite adaptive-BPG calibration and received-byte evaluation, without models.

The old codec/PHY primitives are pinned and imported read-only. The original
native256 results and root budget are never modified. Every actual decoder call
is charged to a separate finite ledger before execution.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def load(request):
    cfg = json.loads(Path(request).read_text())
    sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for path, expected in cfg['source_bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Changed pinned runtime: ' + path)
    # Explicit original modules, without importing old runners or starting jobs.
    codec_path = next(p for p in cfg['source_bindings'] if p.endswith('/bpg_codec.py'))
    phy_path = next(p for p in cfg['source_bindings'] if p.endswith('/bpg_phy.py'))
    sys.path[0:0] = [str(Path(__file__).resolve().parent), str(Path(codec_path).parent), str(Path(phy_path).parent)]
    from bpg_codec import require, sha, read, digest
    require(cfg['schema'] == 'A1B_ADAPTIVE_BPG_REQUEST_V1', 'Unexpected adaptive request schema')
    require(cfg['phase_caps'] == {'qualification': 44, 'prescreen': 5760, 'calibration': 10800, 'holdout': 18000}
            and cfg['total_new_PHY_cap'] == 34604, 'Finite supplemental scope changed')
    require(cfg['SNRs'] == [1, 4, 7, 10, 13, 19] and cfg['noise_seeds'] == [2001, 2002, 2003], 'Changed SNR/noise grid')
    require(cfg['prior_BLER_minimum_actual_frames'] == cfg['missing_BLER_fixed_frames'] == 32
            and cfg['missing_BLER_noise_seeds'] == list(range(9001, 9033)), 'Changed pre-screen size')
    require(cfg['resources'] == {'workers': 8, 'threads': 2, 'nice': 15,
                                'affinities': [[16+2*i, 17+2*i] for i in range(8)]}, 'Changed resources')
    require(len(cfg['records']['calibration']) == 100 and len(cfg['records']['holdout']) == 500, 'Changed population')
    for key in ['original_request', 'probe_request', 'probe_completion', 'source_rule',
                'inherited_qualification', 'inherited_calibration', 'python', 'bpgenc', 'bpgdec']:
        require(sha(cfg[key]['path']) == cfg[key]['sha256'], 'Changed pinned input: ' + key)
    for descriptor in cfg['original_native_codec_completion'].values():
        require(sha(descriptor['path']) == descriptor['sha256'], 'Changed native codec completion')
    old = read(cfg['original_request']['path'])
    require(cfg['records'] == old['records'] and cfg['catalogue'][:-1] == old['catalogue'], 'Original population/catalogue changed')
    require(cfg['protocol'] == old['protocol'], 'Original PHY numeric protocol must be reused')
    require(cfg['catalogue'][-1] == dict(profile_id=3014, q=6, rate='1/3', k=1912, n=5736,
                                       capacity_bytes=235, N=1024, header_symbols=68, body_symbols=956), 'New 64QAM1/3 layout changed')
    rule = read(cfg['source_rule']['path']); probe = read(cfg['probe_completion']['path'])
    require(rule['status'] == 'SOURCE_RULE_FROZEN_MCS_NOT_SELECTED' and not rule['MCS_frozen'], 'Source rule missing')
    require(rule['probe_completion_sha256'] == cfg['probe_completion']['sha256']
            and probe['request_sha256'] == cfg['probe_request']['sha256'], 'Actual probe/source rule identity differs')
    require(rule['resolutions'] == [256,128,64,32] and rule['coarse_qps'] == [0,8,16,24,32,40,48,51], 'Source rule grid changed')
    require(rule['capacity_bytes'] == sorted({p['capacity_bytes'] for p in cfg['catalogue']}), 'Source capacity grid changed')
    require(probe['status'] == 'A1A_CALIBRATION32_BPG_CODEC_PROBE_COMPLETE_V1', 'Actual source probe incomplete')
    require(not cfg['holdout_used_for_selection'] and not cfg['root_ledger_mutation'], 'Scientific scope changed')
    cfg['_path'], cfg['_sha256'] = str(Path(request).resolve()), sha(request)
    return cfg


def guard(cfg):
    from bpg_codec import require
    from probe_source_codec import STOP
    require(not STOP, 'Worker stop signal observed: ' + str(STOP))
    require(time.time() < cfg['deadline_unix'], 'Adaptive supplement deadline reached')
    require(not any(Path(p).exists() for p in cfg['stop_files']), 'STOP observed')


def environment(cfg, worker):
    os.sched_setaffinity(0, cfg['resources']['affinities'][worker])
    os.setpriority(os.PRIO_PROCESS, 0, cfg['resources']['nice'])
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cfg['_worker'] = worker


def completed(cfg, stage):
    from bpg_codec import read, require, verify_outputs
    p = Path(cfg['out']) / stage / 'completion.json'
    d = read(p); verify_outputs(d)
    require(d['request_sha256'] == cfg['_sha256'] and d['status'] == 'ADAPTIVE_BPG_STAGE_COMPLETE_V1', 'Incomplete prerequisite: ' + stage)
    return d


def source_pixels(record):
    from codec_stage import source_pixels as original
    return original(record)


def psnr(image, reference):
    import numpy as np
    mse = float(np.mean(np.square(image.astype(np.float64) - reference.astype(np.float64))))
    return 'Infinity' if mse == 0 else -10.0 * math.log10(mse)


def number(value):
    return math.inf if value == 'Infinity' else float(value)


def link_and_ledger(cfg):
    from bpg_phy import Link
    from supplement_ledger import Ledger
    return Link(cfg), Ledger(cfg['independent_ledger'], cfg['phase_caps'], cfg['_sha256'])


def verify_old_qualification(cfg, link):
    from bpg_codec import read, require, digest
    old = read(cfg['inherited_qualification']['path'])
    require(digest(old['backend_identity']) == digest(link.identity), 'Original numerical PHY identity changed')
    require(all(digest(link.plans[k]) == digest(v) for k, v in old['layouts'].items()), 'Original14 LDPC layouts changed')
    return old


def qualification(cfg):
    from bpg_codec import read, require, sha, write, verify_outputs
    from adaptive_codec import choice, ReceiverCodec
    link, ledger = link_and_ledger(cfg)
    verify_old_qualification(cfg, link)
    receiver = ReceiverCodec(cfg, 'calibration')
    checks, outputs = [], {}
    profile = cfg['catalogue'][-1]
    for i in range(2):
        guard(cfg)
        payload = bytes([0x55 if i == 0 else 0xAA]) * profile['capacity_bytes']
        result = link.receive(link.transmit(payload, profile, i), 80, i, ledger, 'qualification', 'new64QAM1/3:pattern%d' % i)
        require(result['body'] and result['body']['parser_accepted']
                and base64.b64decode(result['body']['payload_b64']) == payload, 'New64QAM1/3 roundtrip failed')
        checks.append({'profile_id': profile['profile_id'], 'pattern': i, 'roundtrip_exact': True})
    strongest = max(cfg['catalogue'], key=lambda p: p['capacity_bytes'])
    for i, record in enumerate(cfg['records']['calibration'][:20]):
        guard(cfg)
        path = Path(cfg['probe_root']) / 'sources' / ('%04d' % i) / 'completion.json'
        actual = read(path); verify_outputs(actual)
        require(actual['source'] == record, 'Wrong probe source')
        selected = choice(actual['rows'], strongest['capacity_bytes'])
        if selected is None:
            checks.append({'source_index': i, 'status': 'SOURCE_UNFIT_AT_LARGEST_CAPACITY', 'roundtrip_run': False})
            continue
        payload = Path(selected['stream']).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == selected['stream_sha256'], 'Probe stream changed')
        counter = 100 + i
        result = link.receive(link.transmit(payload, strongest, counter), 80, counter, ledger, 'qualification', 'adaptive20:%d' % i)
        require(result['body'] and result['body']['parser_accepted'], 'Adaptive container roundtrip rejected')
        received = base64.b64decode(result['body']['payload_b64'])
        image, status, receipt = receiver.decode(received)
        require(received == payload and status == 'BPG_DECODED', 'Actual complete-container roundtrip failed')
        import numpy as np
        from PIL import Image
        expected = np.asarray(Image.open(selected['restored_png']).convert('RGB')).transpose(2,0,1).astype(np.float32) / np.float32(255)
        require(np.array_equal(image, expected), 'Receiver bicubic/source-probe restoration differ')
        outputs[str(path)] = sha(path); outputs[receipt] = sha(receipt)
        checks.append({'source_index': i, 'source_id': record['source_id'], 'resolution': selected['resolution'],
                       'qp': selected['qp'], 'roundtrip_exact': True, 'received_bytes_only': True})
    write(Path(cfg['out']) / 'qualification' / 'worker_0.json',
          {'request_sha256': cfg['_sha256'], 'checks': checks, 'layouts': link.plans,
           'backend_identity': link.identity, 'inherited_14_layouts_reused': True, 'outputs': outputs})


def screen(cfg):
    """Length-matched actual packet outcomes are used only to shortlist three MCS."""
    from bpg_codec import read, require, sha, write, verify_outputs, digest
    from adaptive_codec import choice
    import numpy as np
    completed(cfg, 'qualification')
    link, ledger = link_and_ledger(cfg)
    verify_old_qualification(cfg, link)
    inherited = read(cfg['inherited_calibration']['path'])
    counts = {(snr, p['profile_id']): [0,0] for snr in cfg['SNRs'] for p in cfg['catalogue']}
    outputs = {}
    for i, record in enumerate(cfg['records']['calibration']):
        path = Path(cfg['inherited_calibration']['path']).parent / 'source_checkpoints' / ('%04d.json' % i)
        require(inherited['outputs'].get(str(path)) == sha(path), 'Unbound original calibration checkpoint')
        old = read(path); verify_outputs(old)
        require(old['source_id'] == record['source_id'] and digest(old['runtime_identity']) == digest(link.identity), 'Original calibration identity differs')
        for row in old['rows']:
            if row['source_encoding_fit']:
                profile = next(p for p in cfg['catalogue'] if p['profile_id'] == row['profile_id'])
                require([row['q'],row['k'],row['n']] == [profile['q'],profile['k'],profile['n']], 'Cannot reuse mismatched BLER')
                pair = counts[row['snr_db'], row['profile_id']]
                pair[0] += 1
                pair[1] += row['status'] == 'BPG_DECODED' and row['undetected_payload_difference'] is False
        outputs[str(path)] = sha(path)
    delivery, diagnostics = {}, []
    for snr in cfg['SNRs']:
        for profile in cfg['catalogue']:
            guard(cfg)
            trials, successes = counts[snr, profile['profile_id']]
            origin = 'same_k_n_q_decoder_original_actual_calibration_packets'
            if trials < cfg['prior_BLER_minimum_actual_frames']:
                trials, successes = 32, 0
                origin = 'fixed32_new_actual_random_container_packets'
                for i, seed in enumerate(cfg['missing_BLER_noise_seeds']):
                    guard(cfg)
                    entropy = int.from_bytes(hashlib.sha256(('adaptive-bpg-screen:%d:%d' % (profile['profile_id'], i)).encode()).digest()[:16], 'little')
                    payload = np.random.Generator(np.random.PCG64(entropy)).integers(0,256,profile['capacity_bytes'],dtype=np.uint8).tobytes()
                    counter = (profile['profile_id'] - 3000) * 32 + i
                    tx = link.transmit(payload, profile, counter)
                    rx = link.noisy(tx, 'adaptive_prescreen', snr, profile['profile_id'], seed)
                    result = link.receive(rx, snr, counter, ledger, 'prescreen', 'screen:SNR%d:profile%d:noise%d' % (snr,profile['profile_id'],seed))
                    exact = bool(result['body'] and result['body']['parser_accepted']
                                 and base64.b64decode(result['body']['payload_b64']) == payload)
                    successes += exact
            delivery[snr, profile['profile_id']] = successes / trials
            diagnostics.append({'snr_db':snr, 'profile_id':profile['profile_id'], 'k':profile['k'], 'n':profile['n'],
                                'q':profile['q'], 'trials':trials, 'exact_delivery':successes,
                                'estimated_packet_delivery': successes / trials, 'origin':origin})
    source_scores = {p['profile_id']: [] for p in cfg['catalogue']}
    gray_scores = []
    for i, record in enumerate(cfg['records']['calibration'][:32]):
        path = Path(cfg['probe_root']) / 'sources' / ('%04d' % i) / 'completion.json'
        actual = read(path); verify_outputs(actual)
        source = source_pixels(record).astype(np.float32) / np.float32(255)
        gray_scores.append(number(psnr(np.full_like(source, .5), source)))
        for p in cfg['catalogue']:
            selected = choice(actual['rows'], p['capacity_bytes'])
            source_scores[p['profile_id']].append(None if selected is None else
                                                  (math.inf if selected['psnr_is_infinite'] else selected['psnr_db']))
        outputs[str(path)] = sha(path)
    shortlists, scores = {}, {}
    for snr in cfg['SNRs']:
        predicted = {}
        for p in cfg['catalogue']:
            frequency = delivery[snr,p['profile_id']]
            values = [g if x is None or frequency == 0 else frequency * x + (1-frequency) * g
                      for x,g in zip(source_scores[p['profile_id']], gray_scores)]
            predicted[p['profile_id']] = sum(values) / 32
        order = sorted(predicted, key=lambda p: (-predicted[p],p))[:3]
        shortlists[str(snr)] = order
        scores[str(snr)] = {str(k):('Infinity' if math.isinf(v) else v) for k,v in predicted.items()}
    write(Path(cfg['out']) / 'screen' / 'worker_0.json',
          {'request_sha256': cfg['_sha256'], 'shortlists': shortlists, 'predicted_PSNR_proxy': scores,
           'matched_packet_statistics': diagnostics, 'source_count':32, 'holdout_used':False,
           'scope':'selection approximation only; packet delivery is payload-dependent; original14 statistics precede codebook extension; this is not final noisy image quality',
           'outputs':outputs})


def profile_map(cfg, population):
    from bpg_codec import read
    if population == 'calibration':
        completed(cfg, 'screen')
        selected = read(Path(cfg['out']) / 'screen' / 'worker_0.json')['shortlists']
    else:
        completed(cfg, 'freeze')
        selected = {s:[p] for s,p in read(Path(cfg['out']) / 'freeze' / 'worker_0.json')['selected_profile_ids'].items()}
    return {int(s):[next(p for p in cfg['catalogue'] if p['profile_id'] == profile) for profile in ids]
            for s,ids in selected.items()}


def source_worker(cfg, stage, index):
    from bpg_codec import read, require, sha, write, verify_outputs
    from adaptive_codec import source_rows, ReceiverCodec
    import numpy as np
    codec_only = stage.startswith('codec_')
    population = stage[6:] if codec_only else stage
    profiles = profile_map(cfg, population)
    if not codec_only:
        completed(cfg, 'codec_' + population)
        link, ledger = link_and_ledger(cfg)
        actual = read(Path(cfg['out']) / 'qualification' / 'worker_0.json')
        from bpg_codec import digest
        require(digest(actual['backend_identity']) == digest(link.identity)
                and digest(actual['layouts']) == digest(link.plans), 'Qualified PHY identity changed')
        from bpg_phy import rgb_sha
        receiver = ReceiverCodec(cfg, population)
    outputs = {}
    for i in range(index, len(cfg['records'][population]), 8):
        guard(cfg)
        record = cfg['records'][population][i]
        cp = Path(cfg['out']) / stage / 'sources' / ('%04d.json' % i)
        if cp.exists():
            done = read(cp); verify_outputs(done)
            require(done['request_sha256'] == cfg['_sha256'] and done['source_id'] == record['source_id'], 'Changed completed source')
            outputs[str(cp)] = sha(cp); outputs.update(done['outputs']); continue
        pixels = source_pixels(record)
        if codec_only:
            coded = source_rows(cfg,population,i,pixels)
            path = Path(cfg['out']) / 'codec' / population / ('%04d' % i) / 'completion.json'
            write(cp, {'request_sha256':cfg['_sha256'],'source_id':record['source_id'],
                       'source_index':i,'outputs':{str(path):sha(path)}})
        else:
            encoded_path = Path(cfg['out']) / 'codec' / population / ('%04d' % i) / 'completion.json'
            coded = read(encoded_path); verify_outputs(coded)
            target = pixels.astype(np.float32) / np.float32(255)
            rows, images, slots, lookup, local = [], [], [], {}, {str(encoded_path):sha(encoded_path)}
            for si,snr in enumerate(cfg['SNRs']):
                for profile in profiles[snr]:
                    selected = coded['fits'][str(profile['profile_id'])]['selected']
                    for ni,seed in enumerate(cfg['noise_seeds']):
                        guard(cfg)
                        counter = (si * len(cfg['records'][population]) + i) * 3 + ni
                        frame = '%s:%04d:SNR%d:noise%d:MCS%d' % (population,i,snr,seed,profile['profile_id'])
                        status, outcome, calls, different, tx_energy = 'SOURCE_UNFIT',None,0,None,None
                        image = np.full((3,256,256),.5,dtype=np.float32)
                        if selected:
                            payload = Path(selected['stream']).read_bytes()
                            require(hashlib.sha256(payload).hexdigest() == selected['stream_sha256'], 'Source BPG changed')
                            tx = link.transmit(payload,profile,counter)
                            tx_energy = float(np.square(tx.astype(np.float64)).sum())
                            # Preserve original standard-noise namespace, but a new waveform is a new observation.
                            rx = link.noisy(tx,population,snr,i,seed)
                            outcome = link.receive(rx,snr,counter,ledger,population,frame)
                            calls,status = outcome['packet_calls'],outcome['status']
                            if outcome['body'] and outcome['body']['parser_accepted']:
                                received = base64.b64decode(outcome['body']['payload_b64'])
                                different = received != payload  # offline diagnostic, never rescues the decoder
                                decoded,status,receipt = receiver.decode(received)
                                local[receipt] = sha(receipt)
                                if decoded is not None:
                                    image = decoded
                        body = dict(outcome['body']) if outcome and outcome['body'] else None
                        if body and body.get('payload_b64') is not None:
                            body['received_BPG_sha256'] = hashlib.sha256(base64.b64decode(body.pop('payload_b64'))).hexdigest()
                        row = dict(frame_id=frame,source_index=i,source_id=record['source_id'],snr_db=snr,noise_seed=seed,
                                   profile_id=profile['profile_id'],q=profile['q'],k=profile['k'],n=profile['n'],
                                   allocated_symbols=1024,header_symbols=68,body_symbols=956,padding_symbols=0,
                                   capacity_bytes=profile['capacity_bytes'],source_encoding_fit=selected is not None,
                                   selected_resolution=selected['resolution'] if selected else None,
                                   selected_qp=selected['qp'] if selected else None,
                                   source_stream_bytes=selected['complete_BPG_bytes'] if selected else None,
                                   status=status,gray_substitution=status!='BPG_DECODED',packet_decoder_calls=calls,
                                   header=outcome['header'] if outcome else None,body=body,actual_frame_energy=tx_energy,
                                   psnr_db=psnr(image,target),undetected_payload_difference=different,
                                   image_sha256=rgb_sha(image),reference_sha256=rgb_sha(target),
                                   preprocessing_id=record['preprocessing_id'],receiver_batch_size=1,quality_selection=False)
                        if population == 'holdout':
                            key = row['image_sha256']
                            if key not in lookup:
                                lookup[key] = len(images); images.append(image)
                            row['image_slot'] = lookup[key]; slots.append(lookup[key])
                        rows.append(row)
            extra = {}
            if population == 'holdout':
                archive = Path(cfg['out']) / stage / 'reconstructions' / ('%04d.npz' % i)
                archive.parent.mkdir(parents=True,exist_ok=True)
                require(not archive.exists(),'Unreceipted reconstruction archive retained')
                with archive.open('xb') as f:
                    np.savez_compressed(f,images=np.stack(images),source_rgb=target,row_ids=np.asarray([r['frame_id'] for r in rows]),image_slots=np.asarray(slots,dtype=np.int64))
                    f.flush();os.fsync(f.fileno())
                local[str(archive)] = sha(archive)
                extra = dict(evaluation_class_index=record['evaluation_class_index'],
                             float_reconstructions=dict(path=str(archive),sha256=sha(archive),dtype='float32',layout='CHW',lossless=True,
                                                        image_sha256=[rgb_sha(x) for x in images],reference_sha256=rgb_sha(target),rows=len(rows),
                                                        unique_images=len(images),row_ids=[r['frame_id'] for r in rows],image_slots=slots))
            write(cp,dict(status='ADAPTIVE_BPG_SOURCE_RX_COMPLETE_V1',request_sha256=cfg['_sha256'],source_index=i,
                          source_id=record['source_id'],preprocessing_id=record['preprocessing_id'],rows=rows,
                          reference_sha256=rgb_sha(target),method='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024',
                          runtime_identity=link.identity,outputs=local,training_updates=0,**extra))
        outputs[str(cp)] = sha(cp); outputs.update(read(cp)['outputs'])
        print(json.dumps({'stage':stage,'completed_source_index':i,'worker':index}),flush=True)
    write(Path(cfg['out'])/stage/('worker_%d.json'%index),
          {'request_sha256':cfg['_sha256'],'source_indices':list(range(index,len(cfg['records'][population]),8)), 'outputs':outputs})


def freeze(cfg):
    from bpg_codec import read, require, sha, write
    completed(cfg,'calibration')
    profiles = profile_map(cfg,'calibration')
    scores = {snr:{p['profile_id']:[] for p in profiles[snr]} for snr in cfg['SNRs']}
    outputs = {}
    for i,record in enumerate(cfg['records']['calibration']):
        path = Path(cfg['out'])/'calibration'/'sources'/('%04d.json'%i)
        source = read(path); require(source['source_id']==record['source_id'],'Wrong calibration source')
        for snr in cfg['SNRs']:
            for profile in profiles[snr]:
                rows = [r for r in source['rows'] if r['snr_db']==snr and r['profile_id']==profile['profile_id']]
                require([r['noise_seed'] for r in rows]==cfg['noise_seeds'],'Incomplete actual cal noise grid')
                scores[snr][profile['profile_id']].append(sum(number(r['psnr_db']) for r in rows)/3)
        outputs[str(path)]=sha(path)
    means={snr:{pid:sum(v)/100 for pid,v in values.items()} for snr,values in scores.items()}
    winners={str(snr):min(v,key=lambda pid:(-v[pid],pid)) for snr,v in means.items()}
    write(Path(cfg['out'])/'freeze'/'worker_0.json',
          {'request_sha256':cfg['_sha256'],'selected_profile_ids':winners,
           'calibration_mean_PSNR':{str(s):{str(p):('Infinity' if math.isinf(v) else v) for p,v in d.items()} for s,d in means.items()},
           'calibration_source_count':100,'noise_count':3,'holdout_used_for_selection':False,
           'objective':cfg['calibration_objective'],'outputs':outputs})


def work(cfg,stage,worker):
    environment(cfg,worker); guard(cfg)
    if stage=='qualification': qualification(cfg)
    elif stage=='screen': screen(cfg)
    elif stage=='freeze': freeze(cfg)
    else: source_worker(cfg,stage,worker)


def run(cfg,stage):
    import fcntl
    from bpg_codec import read,require,sha,write,verify_outputs
    from supplement_ledger import Ledger,root_snapshot
    base=Path(cfg['out']);directory=base/stage;directory.mkdir(parents=True,exist_ok=True)
    with (base/'owner.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'completion.json').exists():
            completed(cfg,stage);print(json.dumps({'status':'VERIFIED_COMPLETED_STAGE_REUSED','stage':stage}));return
        ledger=Ledger(cfg['independent_ledger'],cfg['phase_caps'],cfg['_sha256']);ledger.assert_resume()
        before=root_snapshot(cfg['original_root_ledger_readonly']);guard(cfg)
        count=1 if stage in ('qualification','screen','freeze') else 8
        attempt=directory/('attempt_'+str(time.time_ns()));attempt.mkdir()
        processes=[];logs=[]
        try:
            for worker in range(count):
                log=attempt/('worker_%d.log'%worker);f=log.open('xb');logs.append((f,log))
                argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--request',cfg['_path'],'--stage',stage,'--worker',str(worker)]
                pending=[];handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
                for s in handlers:signal.signal(s,lambda signum,frame:pending.append(signum))
                try:
                    processes.append(subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
                finally:
                    for s,handler in handlers.items():signal.signal(s,handler)
                if pending:raise KeyboardInterrupt('Deferred owner stop: '+str(pending))
            while any(p.poll() is None for p in processes):
                guard(cfg);require(not any(p.poll() not in (None,0) for p in processes),'Adaptive worker failed');time.sleep(.2)
            codes=[p.wait() for p in processes]
            for f,_ in logs:f.flush();os.fsync(f.fileno());f.close()
            require(codes==[0]*count,'Actual worker nonzero exit')
            after=root_snapshot(cfg['original_root_ledger_readonly']);require(before==after,'Original root budget changed')
            snap=ledger.assert_resume()
            outputs={str(p):sha(p) for _,p in logs}
            rows=0
            for i in range(count):
                path=directory/('worker_%d.json'%i);d=read(path);verify_outputs(d);outputs[str(path)]=sha(path);outputs.update(d['outputs'])
                if stage in ('calibration','holdout'):
                    for source_index in d['source_indices']:
                        rows+=len(read(directory/'sources'/('%04d.json'%source_index))['rows'])
            expected={'calibration':5400,'holdout':9000}.get(stage)
            if expected is not None:require(rows==expected,'Incomplete actual frame grid')
            write(directory/'completion.json',dict(status='ADAPTIVE_BPG_STAGE_COMPLETE_V1',stage=stage,
                request_sha256=cfg['_sha256'],actual_children_waited=True,worker_exit_codes=codes,
                root_budget_before=before,root_budget_after=after,ledger=snap,frame_count=rows,outputs=outputs))
            print(json.dumps({'status':'ADAPTIVE_BPG_STAGE_COMPLETE_V1','stage':stage,'frames':rows,'ledger':snap}),flush=True)
        except BaseException as error:
            for s in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(s,signal.SIG_IGN)
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:p.wait()
            for f,_ in logs:
                if not f.closed:f.flush();os.fsync(f.fileno());f.close()
            write(attempt/'failure.json',{'error':repr(error),'actual_children_waited':True,'exit_codes':[p.returncode for p in processes],
                                         'logs':{str(p):sha(p) for _,p in logs},'ledger':ledger.snapshot()})
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True)
    p.add_argument('--stage',choices=['qualification','screen','codec_calibration','calibration','freeze','codec_holdout','holdout'],required=True)
    p.add_argument('--worker',type=int);a=p.parse_args()
    def stopped(signum,frame):
        if a.worker is None:
            raise KeyboardInterrupt('Adaptive supplement stopped: '+str(signum))
        from probe_source_codec import STOP
        STOP.append(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,stopped)
    cfg=load(a.request)
    if a.worker is None:run(cfg,a.stage)
    else:work(cfg,a.stage,a.worker)


if __name__=='__main__':main()
