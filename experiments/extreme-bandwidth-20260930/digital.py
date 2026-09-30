"""N512 digital references, calibration and frozen-policy development.

No training occurs here. All digital variants of one action reuse its actual
received bits. Class1000 is the verified unconditional VAR embedding.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import assets
import digital_protocol as phy
import numpy as np
import torch
from torch.nn import functional as fn
from PIL import Image
from latent_enhancement.latent import complete_latent, original_rgb
from token_efficiency.digital_grid import population
from var_comm.quality import dino_features

ROOT, HERE = assets.ROOT, assets.HERE
OUT = assets.OUT / 'digital'
RESULT = assets.RESULT
SNRS, CAL_SEEDS, DEV_SEEDS = assets.SNRS, assets.CAL_SEEDS, assets.DEV_SEEDS
TIMING = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
EXAMPLES = [0, 25, 50, 75]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def seal(path, obj):
    path = Path(path)
    if path.exists():
        if read(path) != obj:
            raise RuntimeError('Frozen digital identity differs: ' + str(path))
    else:
        write(path, obj)


def csv_rows(path, rows):
    if not rows:
        raise ValueError('Empty digital table')
    rows=[{('lambda' if k=='lambda_' else k):v for k,v in row.items()} for row in rows]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def status(stage, **kw):
    write(OUT / 'status.json', dict(stage=stage, time=time.time(), pid=os.getpid(), **kw))


def source_bindings():
    files = [*HERE.glob('*.py'), HERE/'EXECUTION_PLAN.md', HERE/'protocol.json',
        HERE/'training_protocol.json', ROOT/'src/var_comm/scale_channel.py',
        ROOT/'src/var_comm/token_trellis.cpp', ROOT/'src/var_comm/study.py',
        ROOT/'src/var_comm/quality.py', ROOT/'src/var_comm/next_scale_prior.py',
        ROOT/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',
        ROOT/'experiments/token_channel_efficiency_20260923/src/token_efficiency/phy.py']
    return {str(p): assets.sha(p) for p in files}


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if assets.sha(path) != expected:
            raise RuntimeError('Bound digital source changed: ' + path)


def context(loaded, calibration):
    return dict(scope='AUTHORIZED_N512_ONLY', protocol=phy.registration(),
        identity=loaded['identity'], sources=[r['image_id'] for r in calibration['records']],
        preprocessing_ids=[r['preprocessing_id'] for r in calibration['records']],
        calibration_bindings=calibration['bindings'], snrs=SNRS, calibration_seeds=CAL_SEEDS,
        selection=dict(reliability='min mean(header failure OR body CRC failure); tie larger m, then action ID',
            objective='min Dc LPIPS under PSNR >= condition-specific reliability-reference PSNR-.25',
            ties='lower failure fraction, then fixed action ID', source_first_mean=True),
        reference_generation='original complete_latent, greedy argmax, no CFG/sampling',
        no_development_tuning=True, mismatch_permutation=assets.mismatch_permutation(),
        latent_failure='no Fhat on header failure; latent error blank, zero-erasure proxy diagnostic only',
        source_bindings=source_bindings())


def split_tokens(tokens):
    values = np.asarray(tokens, dtype=np.int64)
    if values.shape != (680,):
        raise ValueError('All official source tokens required at TX')
    return list(np.split(values, np.cumsum([p*p for p in phy.SIZES])[:-1]))


@torch.no_grad()
def prefix_latent(vae, prefix):
    """Official ten-scale cumulative updates, including short m4/m5."""
    if tuple(vae.quantize.v_patch_nums) != phy.SIZES:
        raise ValueError('Official ten-scale quantizer required')
    f = vae.quantize.embedding.weight.new_zeros(1, 32, 16, 16)
    for i, tokens in enumerate(prefix):
        values = np.asarray(tokens)
        if values.shape != (phy.SIZES[i]**2,):
            raise ValueError('Received prefix scale shape')
        e = vae.quantize.embedding(torch.as_tensor(values[None], device=f.device, dtype=torch.long))
        e = e.transpose(1, 2).reshape(1, 32, phy.SIZES[i], phy.SIZES[i])
        f, _ = vae.quantize.get_next_autoregressive_input(i, 10, f, e)
    return f


class ReceivedCache:
    """Quality-only cache, keyed exclusively by actual received prefix/class."""
    def __init__(self, capacity=128):
        self.capacity = capacity; self.values = OrderedDict()
        self.hits = 0; self.misses = 0

    @torch.no_grad()
    def completed(self, loaded, prefix, label):
        h = hashlib.sha256(f'{label}|{len(prefix)}|'.encode())
        for tokens in prefix:
            h.update(np.asarray(tokens, dtype='<u2').tobytes())
        key = h.hexdigest()
        if key in self.values:
            self.hits += 1; self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1; assets.check()
        latent = complete_latent(loaded['vae'], loaded['var'], prefix, label, loaded['device']).cpu()
        self.values[key] = latent
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return latent


@torch.no_grad()
def render_received(event, loaded, cache=None, include_prefix=True):
    """No clean feature, true class or TX action enters this function."""
    if not event['header_ok']:
        return dict(C=None, U=None, prefix=None)
    prefix = event['prefix']
    if cache is None:
        c = complete_latent(loaded['vae'], loaded['var'], prefix, event['decoded_label'], loaded['device']).cpu()
        u = complete_latent(loaded['vae'], loaded['var'], prefix, 1000, loaded['device']).cpu()
    else:
        c = cache.completed(loaded, prefix, event['decoded_label'])
        u = cache.completed(loaded, prefix, 1000)
    return dict(C=c, U=u, prefix=prefix_latent(loaded['vae'], prefix).cpu() if include_prefix else None)


@torch.no_grad()
def quality_outputs(loaded, record, outputs, dino_reference=None, mismatch_reference=None, images=False):
    """Render same received latents with specified frozen decoder; score separately."""
    target = torch.as_tensor(record['pixels'][None], dtype=torch.float32, device=loaded['device']) / 255.
    keys, unique, mapping = [], [], {}
    for name, (latent, decoder) in outputs.items():
        h = 'header_erasure' if latent is None else decoder + ':' + hashlib.sha256(latent.numpy().tobytes()).hexdigest()
        if h not in mapping:
            mapping[h] = len(unique); unique.append((latent, decoder)); keys.append(h)
    results, rendered = [], []
    for i in range(0, len(unique), 4):
        assets.check(); chunk = unique[i:i+4]; pics = []
        for latent, decoder in chunk:
            if latent is None:
                pic = target.new_full((1, 3, 256, 256), .5)
            elif decoder == 'Dc':
                pic = loaded['decoder'](latent.to(loaded['device']))
            elif decoder == 'D0':
                pic = original_rgb(loaded['vae'], latent.to(loaded['device']))
            else:
                raise ValueError('Frozen decoder ID')
            pics.append(pic)
        pics = torch.cat(pics); targets = target.expand(len(pics), -1, -1, -1)
        psnr = -10 * torch.log10((targets - pics).square().flatten(1).mean(1))
        lp = loaded['lpips'](targets*2-1, pics*2-1).flatten()
        embedding = dino_features(loaded['dino'], pics)
        ref = dino_reference
        if ref is None:
            ref = dino_features(loaded['dino'], target)[0]
        cos = fn.cosine_similarity(embedding, ref[None].to(embedding.device), dim=1)
        mis = None if mismatch_reference is None else fn.cosine_similarity(embedding, mismatch_reference[None].to(embedding.device), dim=1)
        for j in range(len(pics)):
            row = dict(psnr_db=float(psnr[j]), lpips_alex=float(lp[j]), dino_cosine=float(cos[j]))
            if mis is not None:
                row['dino_mismatched'] = float(mis[j])
            results.append(row)
            if images:
                rendered.append(pics[j].cpu().numpy())
    answer = {}
    for name, (latent, decoder) in outputs.items():
        h = 'header_erasure' if latent is None else decoder + ':' + hashlib.sha256(latent.numpy().tobytes()).hexdigest()
        index = mapping[h]
        answer[name] = (results[index], rendered[index] if images else None)
    return answer


def event_fields(event):
    names = ('header_ok','header_crc_ok','header_fields_legal','body_crc_ok','decoded_label',
        'decoded_mode','source_complete','trusted_prefix_scales','hard_candidate_prefix_scales',
        'raw_candidate_used_after_crc_failure','source_error')
    return {k: event[k] for k in names}


def latent_fields(clean, latent):
    if latent is None:
        return dict(latent_valid=False, decoder_applied=False, latent_sq_err_final='',
            zero_erasure_proxy_sq_error=float(clean.double().square().sum()))
    return dict(latent_valid=True, decoder_applied=True,
        latent_sq_err_final=float((clean.double()-latent[0].double()).square().sum()),
        zero_erasure_proxy_sq_error='')


def validate_grid(rows, records, seeds, scopes, scope_fields):
    ids={r['image_id']:(i,r['preprocessing_id']) for i,r in enumerate(records)}
    key=lambda row:(row['source_id'],row['snr_db'],row['noise_seed'],*(row[k] for k in scope_fields))
    keys={key(row) for row in rows}
    expected={(sid,snr,seed,*scope) for sid in ids for snr in SNRS for seed in seeds for scope in scopes}
    if len(keys)!=len(rows) or keys!=expected:
        raise RuntimeError('Digital source/noise/action grid duplicated or incomplete')
    for row in rows:
        if (row['source_index'],row['preprocessing_id'])!=ids[row['source_id']]:
            raise RuntimeError('Digital source order or preprocessing changed')


def reliability_then_quality(candidates, phy_family, snr, condition):
    group = [r for r in candidates if r['phy_family']==phy_family and r['snr_db']==snr and r['class_condition']==condition]
    if len(group) != (2 if phy_family=='QPSK' else 3):
        raise ValueError('Complete legal candidate set required')
    reliable = min(group, key=lambda r:(r['failure_fraction'], -r['action_m'], r['action_id']))
    feasible = [r for r in group if r['psnr_db'] >= reliable['psnr_db']-.25]
    selected = min(feasible, key=lambda r:(r['lpips_alex'], r['failure_fraction'], r['action_id']))
    return dict(action_id=selected['action_id'], action_m=selected['action_m'],
        reliability_reference_action_id=reliable['action_id'], reliability_reference_m=reliable['action_m'],
        reliability_reference_psnr=reliable['psnr_db'], calibration=selected,
        feasible_action_ids=[r['action_id'] for r in feasible])


def legacy_references(loaded, records, role):
    """Reuse compatible existing Source A conditional m6/m7 scalar references."""
    folder = ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/source'/role
    if not (folder/'completion.json').exists():
        return {}, dict(compatible=False, reason='Source A reference absent')
    reg, done = read(folder/'registration.json'), read(folder/'completion.json')
    if reg['decoder_sha256'] != loaded['identity']['models']['decoder']:
        return {}, dict(compatible=False, reason='Dc identity differs')
    verify_bindings(reg['bindings'])
    path = folder/'source_codec_per_image.csv'
    if assets.sha(path) != done['files'][path.name]:
        raise RuntimeError('Source A scalar reference hash differs')
    lookup = {}
    wanted = {f'{d}_{kind}_m{m}' for d in ('D0','Dc') for kind in ('VAR','prefix') for m in (6,7)}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['method'] in wanted:
                lookup[(row['source_id'],row['method'])] = row
    for r in records:
        for name in wanted:
            row = lookup[(r['image_id'],name)]
            if row['preprocessing_id'] != r['preprocessing_id']:
                raise RuntimeError('Source A preprocessing differs')
    return lookup, dict(compatible=True, path=str(path), sha256=assets.sha(path),
        registration_sha256=assets.sha(folder/'registration.json'), reuse='C m6/m7 VAR and prefix, D0/Dc; no finite channel claim')


@torch.no_grad()
def references(loaded, role='calibration', data=None):
    data = assets.load_calibration() if data is None else data
    records, F, T = data['records'], data['F'], data['T']
    old_refs, compatibility = legacy_references(loaded, records, role)
    reference_identity = dict(assets=loaded['identity'], source_bindings=source_bindings(),
        role=role, sources=[r['image_id'] for r in records], preprocessing_ids=[r['preprocessing_id'] for r in records],
        compatibility=compatibility)
    seal(OUT/'references'/role/'identity.json', reference_identity)
    cache = ReceivedCache(); allrows = []
    for i, record in enumerate(records):
        assets.check(); cell = OUT/'references'/role/f'{i:04d}.json'
        if cell.exists():
            saved = read(cell)
            if saved['identity_sha256'] != assets.sha(cell.parent/'identity.json'):
                raise RuntimeError('Reference resume identity changed')
            allrows.extend(saved['rows']); continue
        scales = split_tokens(T[i].numpy()); rows=[]; outputs={}; metadata={}
        for m in range(4,8):
            for decoder in ('Dc','D0'):
                for kind in ('C','U','prefix'):
                    name=f'{decoder}_{kind}_m{m}'
                    old_name=f'{decoder}_{"prefix" if kind=="prefix" else "VAR"}_m{m}'
                    legacy = old_refs.get((record['image_id'],old_name)) if kind!='U' else None
                    base=dict(population=role,source_id=record['image_id'],source_index=i,
                        preprocessing_id=record['preprocessing_id'],method=name,action_m=m,
                        source_bits=phy.raw_bits(m),N='',E='',decoder_id=decoder,class_condition=kind,
                        reference_only=True,finite_channel_result=False)
                    if legacy:
                        rows.append(dict(**base,psnr_db=float(legacy['psnr_db']),lpips_alex=float(legacy['lpips_alex']),
                            dino_cosine=float(legacy['dino_cosine']),reference_origin='compatible_Source_A',
                            latent_valid='',latent_sq_err_final=''))
                    else:
                        prefix=scales[:m]
                        latent=prefix_latent(loaded['vae'],prefix).cpu() if kind=='prefix' else cache.completed(loaded,prefix,1000 if kind=='U' else int(record['class_index']))
                        outputs[name]=(latent,decoder); metadata[name]=base
        if outputs:
            scored=quality_outputs(loaded,record,outputs)
            for name,(metrics,_) in scored.items():
                rows.append(dict(**metadata[name],**metrics,reference_origin='new_noiseless_reference',
                    latent_valid=True,latent_sq_err_final=float((F[i].double()-outputs[name][0][0].double()).square().sum())))
        seal(cell,dict(identity_sha256=assets.sha(cell.parent/'identity.json'),rows=rows));allrows.extend(rows)
        status('source_references',population=role,sources=i+1,total=len(records))
    path=OUT/f'source_references_{role}.csv';csv_rows(path,allrows)
    merged=[]
    for pop in ('calibration','development'):
        p=OUT/f'source_references_{pop}.csv'
        if p.exists():
            with p.open() as handle:merged.extend(csv.DictReader(handle))
    csv_rows(RESULT/'source_references.csv',merged)
    verify_bindings(reference_identity['source_bindings']); assets.assert_frozen(loaded)
    write(OUT/'references'/role/'completion.json',dict(status='SOURCE_REFERENCE_COMPLETE',sources=len(records),rows=len(allrows),
        reference_only=True,sha256=assets.sha(path),legacy_compatibility=compatibility))
    return allrows


@torch.no_grad()
def calibrate(loaded):
    data=assets.load_calibration();cfg=context(loaded,data);OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    seal(OUT/'calibration_config.json',cfg)
    seal(RESULT/'digital_config.json',cfg)
    selected_path=RESULT/'digital_selected_policy.json'
    if selected_path.exists():
        policy=read(selected_path)
        if policy['config_sha256']!=assets.sha(OUT/'calibration_config.json'):
            raise RuntimeError('Digital frozen selection identity differs')
        return policy
    cache=ReceivedCache();allrows=[]
    for i,record in enumerate(data['records']):
        assets.check();cell=OUT/'calibration_cells'/f'{i:04d}.json';cell.parent.mkdir(parents=True,exist_ok=True)
        if cell.exists():
            saved=read(cell)
            if saved['config_sha256']!=assets.sha(OUT/'calibration_config.json'):
                raise RuntimeError('Digital calibration resume identity differs')
            allrows.extend(saved['rows']);continue
        scales=split_tokens(data['T'][i].numpy());rows=[]
        ref=dino_features(loaded['dino'],torch.as_tensor(record['pixels'][None],dtype=torch.float32,device=loaded['device'])/255.)[0]
        for action in phy.legal_actions():
            wave,ledger=phy.transmit(scales,int(record['class_index']),action)
            for snr in SNRS:
                for seed in CAL_SEEDS:
                    assets.check();y=phy.apply_channel(wave,snr,record['image_id'],seed,action.phy)
                    event=phy.receive(y,snr,action.phy);latents=render_received(event,loaded,cache,include_prefix=False)
                    scored=quality_outputs(loaded,record,{c:(latents[c],'Dc') for c in ('C','U')},ref)
                    for condition,(metrics,_) in scored.items():
                        rows.append(dict(source_id=record['image_id'],source_index=i,preprocessing_id=record['preprocessing_id'],
                            snr_db=snr,noise_seed=seed,class_condition=condition,decoder_id='Dc',
                            waveform_sha256=phy.waveform_sha(wave),observation_sha256=phy.waveform_sha(y),
                            **ledger,**event_fields(event),**metrics,**latent_fields(data['F'][i],latents[condition])))
        seal(cell,dict(config_sha256=assets.sha(OUT/'calibration_config.json'),rows=rows));allrows.extend(rows)
        status('digital_calibration',sources=i+1,total=1000,cache_hits=cache.hits,cache_misses=cache.misses)
    if len(allrows)!=150000:
        raise RuntimeError('Full1000 x5 SNR x3noise x5legalactions x2conditions incomplete')
    validate_grid(allrows,data['records'],CAL_SEEDS,
        [(a.phy,a.m,c) for a in phy.legal_actions() for c in ('C','U')],
        ('phy_family','action_m','class_condition'))
    csv_rows(OUT/'calibration_per_frame.csv',allrows)
    grouped=defaultdict(list)
    for row in allrows:grouped[(row['phy_family'],row['snr_db'],row['action_m'],row['class_condition'])].append(row)
    candidates=[]
    for (family,snr,m,condition),rows in sorted(grouped.items()):
        if len(rows)!=3000 or len({r['source_id'] for r in rows})!=1000:
            raise RuntimeError('Candidate calibration source/noise grid incomplete')
        failed=[not r['header_ok'] or not r['body_crc_ok'] for r in rows]
        energy=np.array([r['E'] for r in rows]);valid=[r for r in rows if r['latent_valid']]
        candidates.append(dict(**phy.action_record(m,family),snr_db=snr,class_condition=condition,decoder_id='Dc',
            sources=1000,noise_repeats=3,frames=len(rows),failure_fraction=float(np.mean(failed)),
            header_failure_fraction=float(np.mean([not r['header_ok'] for r in rows])),
            body_crc_failure_fraction=float(np.mean([r['header_ok'] and not r['body_crc_ok'] for r in rows])),
            psnr_db=float(np.mean([r['psnr_db'] for r in rows])),lpips_alex=float(np.mean([r['lpips_alex'] for r in rows])),
            dino_cosine=float(np.mean([r['dino_cosine'] for r in rows])),latent_valid_frames=len(valid),
            latent_sq_err_valid_only=float(np.mean([r['latent_sq_err_final'] for r in valid])) if valid else '',
            E_mean=float(energy.mean()),E_min=float(energy.min()),E_p05=float(np.percentile(energy,5)),
            E_p95=float(np.percentile(energy,95)),E_max=float(energy.max())))
    csv_rows(RESULT/'digital_candidates.csv',candidates)
    csv_rows(RESULT/'digital_action_ledger.csv',[phy.action_record(m,p) for p in phy.PHY_FAMILIES for m in range(4,8)])
    levels={str(snr):{p:{c:reliability_then_quality(candidates,p,snr,c) for c in ('C','U')} for p in phy.PHY_FAMILIES} for snr in SNRS}
    verify_bindings(cfg['source_bindings']);assets.assert_frozen(loaded)
    policy=dict(config_sha256=assets.sha(OUT/'calibration_config.json'),protocol=phy.registration(),
        development_read=False,selection=cfg['selection'],levels=levels)
    seal(selected_path,policy)
    seal(OUT/'calibration_complete.json',dict(status='DIGITAL_FULL1000_CALIBRATION_COMPLETE',sources=1000,frames=len(allrows),
        candidates=len(candidates),policy_sha256=assets.sha(selected_path),config_sha256=policy['config_sha256'],
        per_frame=str(OUT/'calibration_per_frame.csv'),per_frame_sha256=assets.sha(OUT/'calibration_per_frame.csv'),
        development_read=False,cache_hits=cache.hits,cache_misses=cache.misses))
    return policy


def selected_outputs(family, choices, received):
    outputs={};info={}
    for condition in ('C','U'):
        m=choices[condition]['action_m'];event,latents,meta=received[m]
        for decoder in ('Dc','D0'):
            name=f'D_{condition}_{family}'+('' if decoder=='Dc' else '_D0')
            outputs[name]=(latents[condition],decoder);info[name]=(condition,event,meta,m)
        name=f'D_prefix_{condition}_{family}'
        outputs[name]=(latents['prefix'],'Dc');info[name]=('prefix',event,meta,m)
    for condition,selector in [('C','U'),('U','C')]:
        m=choices[selector]['action_m'];event,latents,meta=received[m]
        name=f'D_{condition}_{family}_at_{selector}_action'
        outputs[name]=(latents[condition],'Dc');info[name]=(condition,event,meta,m)
    return outputs,info


@torch.no_grad()
def development_data(loaded):
    records,bindings=population('development')
    if len(records)!=100:
        raise ValueError('Original100 development population required')
    F=[];T=[];reference=[]
    for record in records:
        assets.check();image=torch.as_tensor(record['pixels'][None],dtype=torch.float32,device=loaded['device'])/127.5-1
        f=loaded['vae'].quant_conv(loaded['vae'].encoder(image));F.append(f[0].cpu())
        T.append(torch.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu())
        reference.append(dino_features(loaded['dino'],(image+1)/2)[0].cpu())
    return dict(records=records,bindings=bindings,F=torch.stack(F),T=torch.stack(T),reference=torch.stack(reference))


@torch.no_grad()
def evaluate(loaded, policy=None):
    policy_path=RESULT/'digital_selected_policy.json';policy=read(policy_path) if policy is None else policy
    complete=read(OUT/'calibration_complete.json')
    if complete['policy_sha256']!=assets.sha(policy_path) or policy['development_read']:
        raise RuntimeError('Digital calibration must finish before development')
    cfg=read(OUT/'calibration_config.json');verify_bindings(cfg['source_bindings'])
    if cfg['identity']!=loaded['identity']:
        raise RuntimeError('Selected digital policy and evaluation visual assets differ')
    data=development_data(loaded)
    identity=dict(config_sha256=assets.sha(OUT/'calibration_config.json'),policy_sha256=assets.sha(policy_path),
        assets=loaded['identity'],bindings=data['bindings'],sources=[r['image_id'] for r in data['records']],
        preprocessing_ids=[r['preprocessing_id'] for r in data['records']],mismatch_permutation=cfg['mismatch_permutation'])
    seal(OUT/'development_identity.json',identity)
    cache=ReceivedCache();allrows=[]
    for i,record in enumerate(data['records']):
        assets.check();cell=OUT/'development_cells'/f'{i:03d}.json';cell.parent.mkdir(exist_ok=True)
        if cell.exists():
            saved=read(cell)
            if saved['identity_sha256']!=assets.sha(OUT/'development_identity.json'):
                raise RuntimeError('Digital development resume identity differs')
            allrows.extend(saved['rows']);continue
        scales=split_tokens(data['T'][i].numpy());rows=[]
        for snr in SNRS:
            for family in phy.PHY_FAMILIES:
                choices=policy['levels'][str(snr)][family]
                waves={m:phy.transmit(scales,int(record['class_index']),phy.Action(m,family)) for m in {c['action_m'] for c in choices.values()}}
                for seed in DEV_SEEDS:
                    assets.check();received={}
                    for m,(wave,ledger) in waves.items():
                        y=phy.apply_channel(wave,snr,record['image_id'],seed,family);event=phy.receive(y,snr,family)
                        latents=render_received(event,loaded,cache)
                        received[m]=(event,latents,dict(**ledger,waveform_sha256=phy.waveform_sha(wave),observation_sha256=phy.waveform_sha(y)))
                    outputs,information=selected_outputs(family,choices,received)
                    save=i in EXAMPLES and snr in (4,13) and seed==2001
                    scored=quality_outputs(loaded,record,outputs,data['reference'][i],data['reference'][cfg['mismatch_permutation'][i]],save)
                    for name,(metrics,image) in scored.items():
                        condition,event,ledger,m=information[name];latent,decoder=outputs[name]
                        rows.append(dict(source_id=record['image_id'],source_index=i,preprocessing_id=record['preprocessing_id'],
                            snr_db=snr,noise_seed=seed,training_seed='',model_id=phy.PROTOCOL,system='digital',method=name,
                            decoder_id=decoder,class_condition=condition,action_m=m,policy_action='',lambda_='',rx_ms='',
                            **{k:v for k,v in ledger.items() if k!='action_m'},
                            **event_fields(event),**metrics,**latent_fields(data['F'][i],latent)))
                        if save:
                            path=RESULT/'examples/digital'/f'source_{i:03d}_snr_{snr}_{name}_seed{seed}.png';path.parent.mkdir(parents=True,exist_ok=True)
                            Image.fromarray(np.clip(image.transpose(1,2,0)*255,0,255).astype(np.uint8)).save(path)
        seal(cell,dict(identity_sha256=assets.sha(OUT/'development_identity.json'),rows=rows));allrows.extend(rows)
        status('digital_development',sources=i+1,total=100,cache_hits=cache.hits,cache_misses=cache.misses)
    if len(allrows)!=24000:
        raise RuntimeError('Digital development100x5x3x16methods incomplete')
    methods=[f'D_{c}_{p}{suffix}' for p in phy.PHY_FAMILIES for c in ('C','U') for suffix in ('','_D0')]
    methods += [f'D_prefix_{c}_{p}' for p in phy.PHY_FAMILIES for c in ('C','U')]
    methods += [f'D_{c}_{p}_at_{selector}_action' for p in phy.PHY_FAMILIES for c,selector in [('C','U'),('U','C')]]
    validate_grid(allrows,data['records'],DEV_SEEDS,[(m,) for m in methods],('method',))
    csv_rows(RESULT/'digital_per_frame.csv',allrows)
    references(loaded,'development',data)
    timing(loaded,policy,data)
    verify_bindings(cfg['source_bindings']);assets.assert_frozen(loaded)
    write(OUT/'evaluation_complete.json',dict(status='DIGITAL_DEVELOPMENT_COMPLETE',sources=100,rows=len(allrows),methods=16,
        policy_sha256=assets.sha(policy_path),config_sha256=assets.sha(OUT/'calibration_config.json'),
        frozen_weights_unchanged=True,training_updates=0,cache_hits=cache.hits,cache_misses=cache.misses))
    return allrows


def sync(loaded):
    if loaded['device'].type=='cuda':torch.cuda.synchronize(loaded['device'])


@torch.no_grad()
def timing(loaded,policy,data):
    path=RESULT/'digital_timing.csv'
    expected={(i,s,c,p,r) for i in TIMING for s in SNRS for c in ('C','U') for p in phy.PHY_FAMILIES for r in range(2)}
    if path.exists():
        with path.open() as handle:rows=list(csv.DictReader(handle))
        grid={(int(r['source_index']),int(r['snr_db']),r['class_condition'],r['phy_family'],int(r['repeat'])) for r in rows}
        if len(rows)==400 and grid==expected:return
    rows=[]
    def transmit_online(record,action):
        sync(loaded);start=time.perf_counter()
        image=torch.as_tensor(record['pixels'][None],dtype=torch.float32,device=loaded['device'])/127.5-1
        f=loaded['vae'].quant_conv(loaded['vae'].encoder(image))
        source=[v[0].cpu().numpy() for v in loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False)]
        sync(loaded);source_done=time.perf_counter();wave,ledger=phy.transmit(source,int(record['class_index']),action)
        sync(loaded);done=time.perf_counter()
        return wave,ledger,dict(tx_ms=1000*(done-start),source_encoding_ms=1000*(source_done-start),phy_tx_ms=1000*(done-source_done))
    def receive_online(y,snr,condition,family):
        sync(loaded);start=time.perf_counter();event=phy.receive(y,snr,family);phy_done=time.perf_counter()
        if not event['header_ok']:
            image=np.full((3,256,256),.5,dtype=np.float32);complete_done=phy_done;decode_done=time.perf_counter()
        else:
            latent=complete_latent(loaded['vae'],loaded['var'],event['prefix'],1000 if condition=='U' else event['decoded_label'],loaded['device'])
            sync(loaded);complete_done=time.perf_counter();image=loaded['decoder'](latent)[0].cpu().numpy()
            sync(loaded);decode_done=time.perf_counter()
        return dict(rx_ms=1000*(decode_done-start),phy_rx_ms=1000*(phy_done-start),
            var_completion_ms=1000*(complete_done-phy_done),decode_ms=1000*(decode_done-complete_done),
            ran_VAR=bool(event['header_ok']),**event_fields(event))
    for i in TIMING:
        record=data['records'][i]
        for family in phy.PHY_FAMILIES:
            for condition in ('C','U'):
                for _ in range(3):
                    assets.check();action=phy.Action(policy['levels']['13'][family][condition]['action_m'],family)
                    wave,ledger,tx=transmit_online(record,action)
                    receive_online(phy.apply_channel(wave,13,record['image_id'],2001,family),13,condition,family)
                for snr in SNRS:
                    action=phy.Action(policy['levels'][str(snr)][family][condition]['action_m'],family)
                    for repeat in range(2):
                        assets.check();wave,ledger,tx=transmit_online(record,action)
                        y=phy.apply_channel(wave,snr,record['image_id'],2001,family);rx=receive_online(y,snr,condition,family)
                        rows.append(dict(source_id=record['image_id'],source_index=i,preprocessing_id=record['preprocessing_id'],
                            snr_db=snr,noise_seed=2001,method=f'D_{condition}_{family}',decoder_id='Dc',class_condition=condition,
                            repeat=repeat,**ledger,**tx,**rx,waveform_sha256=phy.waveform_sha(wave),observation_sha256=phy.waveform_sha(y),
                            endpoints='CPU uint8 RGB -> CPU waveform; CPU observation -> CPU float RGB; noise outside RX',cache_used=False))
        status('digital_timing',sources=TIMING.index(i)+1,total=10)
    csv_rows(path,rows)


def archive_calibration(max_part_bytes=8_000_000):
    """Publish all150000 actual calibration rows as exact reconstructable shards."""
    completion=read(OUT/'calibration_complete.json')
    path=OUT/'calibration_per_frame.csv';data=path.read_bytes()
    if assets.sha(path)!=completion['per_frame_sha256'] or completion['frames']!=150000:
        raise RuntimeError('Completed calibration table identity differs')
    with path.open(newline='') as handle:
        count=sum(1 for _ in csv.DictReader(handle))
    lines=data.splitlines(keepends=True);header=lines[0]
    if count!=150000 or len(lines)-1!=count:
        raise RuntimeError('Calibration export requires150000 single-line CSV records')
    folder=RESULT/'digital_calibration_shards';folder.mkdir(parents=True,exist_ok=True)
    parts=[];block=[];size=len(header);offset=0
    def flush():
        nonlocal block,size,offset
        content=header+b''.join(block);name=f'part_{len(parts):03d}.csv';target=folder/name
        if target.exists() and target.read_bytes()!=content:
            raise RuntimeError('Existing calibration shard differs')
        if not target.exists():target.write_bytes(content)
        parts.append(dict(path='digital_calibration_shards/'+name,sha256=assets.sha(target),bytes=len(content),
            rows=len(block),first_data_row=offset+1,last_data_row=offset+len(block)))
        offset+=len(block);block=[];size=len(header)
    for line in lines[1:]:
        if len(header)+len(line)>max_part_bytes:raise RuntimeError('Single calibration row exceeds shard size')
        if block and size+len(line)>max_part_bytes:flush()
        block.append(line);size+=len(line)
    if block:flush()
    manifest=dict(schema=1,table='digital_full1000_calibration_per_frame.csv',rows=count,bytes=len(data),
        sha256=assets.sha(path),header_sha256=hashlib.sha256(header).hexdigest(),max_part_bytes=max_part_bytes,
        reconstruction='first header plus ordered shard data rows; preserve exact original bytes',
        config_sha256=completion['config_sha256'],policy_sha256=completion['policy_sha256'],parts=parts)
    seal(folder/'manifest.json',manifest)
    rebuilt=header+b''.join((RESULT/p['path']).read_bytes().split(b'\n',1)[1] for p in parts)
    if hashlib.sha256(rebuilt).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Calibration shard exact reconstruction failed')
    return manifest


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    seal(OUT/'protocol_registration.json',phy.registration())
    loaded=assets.setup()
    if stage=='references':references(loaded)
    elif stage=='calibration':calibrate(loaded)
    elif stage=='evaluation':evaluate(loaded)
    else:raise ValueError('Registered digital stage')
    status('COMPLETE',completed_stage=stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('references','calibration','evaluation'),required=True)
    args=parser.parse_args()
    try:main(args.stage)
    except assets.old.b.ResourceBusy as error:
        status('PAUSED_SAFE_BOUNDARY',reason=str(error));raise SystemExit(75)
    except Exception:
        import traceback
        write(OUT/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        status('FAILED',traceback=traceback.format_exc());raise
