"""One finite role. RX is a separate process, supplied only actual RX inputs."""
import argparse
import base64
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from contract import *
import native


def b64(v):
    return base64.b64encode(v).decode()


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


class Clock:
    def __init__(self, torch=None):
        self.t = torch
        self.parts = {}

    def call(self, key, fn):
        if self.t is not None:
            self.t.cuda.synchronize()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            if self.t is not None:
                self.t.cuda.synchronize()
            self.parts[key] = self.parts.get(key, 0.) + time.perf_counter()-start


def resource(r, initial=False):
    a = subprocess.check_output([
        'nvidia-smi', '-i', str(r['resource']['gpu_index']),
        '--query-gpu=uuid,memory.free,utilization.gpu,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert a[0] == r['resource']['gpu_uuid']
    assert int(a[1]) >= (81920 if initial else 8192), ('GPU free memory', a)
    if initial:
        assert int(a[2]) <= 10, ('GPU busy before stage', a)
    return dict(unix=time.time(), uuid=a[0], free_MiB=int(a[1]),
                utilization=int(a[2]), temperature=a[3], power_W=a[4],
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                load_average=list(os.getloadavg()))


def firewall(r, method, out):
    """Deny source archives, TX witnesses and all nonpublic result/cache trees."""
    allowed = [E, R/'experiments', R/'src', T/'assets', out,
               T/method/'rx_inputs', T/method/'failed_codec_evidence', RT/'envs']
    allowed += [Path(x['path']) for x in read(r['model_routes']['path'])['files']]
    allowed += [Path(x['path']) for x in read(r['model_routes']['path'])['author_files']]
    routes = read(r['model_routes']['path'])
    allowed += [Path(routes['author_root']), Path(routes['dist']['path']), Path(r['P_checkpoint']['path']),
                Path(r['Swin_checkpoint']['path']), Path(r['model_routes']['path'])]
    allowed += [Path(r['static'][k]['path']) for k in ('completion', 'table')]
    allowed += [Path(r['arpc_public']['weights_root']),
                Path(r['arpc_public']['environment']),
                B/'baseline_strength_20261011/vendor/SwinJSCC',
                B/'baseline_strength_20261011/native_build']
    # The frozen environments use byte-identical relocated site-packages.
    # Only library roots are admitted, never the enclosing source/data asset tree.
    allowed += [Path(p) for p in sys.path if p and (
        p.endswith('/site-packages') or '/envs/' in p)]
    allowed = [p.resolve() for p in allowed]
    forbidden = [Path(s['archive']['path']).resolve() for s in r['sources']]
    blocked = []
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p in forbidden or (p.is_relative_to(Path('/mnt/pfs')) and
                              not any(p == a or p.is_relative_to(a) for a in allowed)):
            blocked.append(str(p))
            raise PermissionError('RX boundary: '+str(p))
    sys.addaudithook(audit)
    return blocked


def nest_tx(pixels, r, snr, ctx, meter):
    import numpy as np
    from timing_endpoints import fresh_tokens
    from ep_source_codec import fallback_endpoints
    import tb_plan
    n, codec, _, calls = ctx
    tokens = meter.call('tokenizer', lambda: fresh_tokens(n, pixels))
    policy = r['nest_policies'][str(snr)]
    candidates = tb_plan.fallback('PARTIAL') if policy['target_m'] == 10 else fallback_endpoints(policy['target_m'], policy['target_K'])
    attempts = []
    def source():
        # Use the current frozen multi-endpoint implementation: one probability
        # traversal, separately finished endpoint streams, then actual-length
        # fallback. Never subtract/slice a larger terminated stream.
        if policy['target_m'] == 10:
            from m10_codec import AuditCodec
            encoded, _ = AuditCodec.encode_all(codec, tokens)
        else:
            encoded = codec.encode_endpoints(tokens, candidates)
        for m, K in candidates:
            v = encoded[m,K]
            fits = len(v['bits']) <= policy['source_capacity_bits']
            attempts.append(dict(m=m, K=K, bits=len(v['bits']), fits=fits))
            if fits:
                return dict(payload=v['bits'].tolist(), m=m, K=K,
                            q=policy['q'], nominal_rate=policy['nominal_rate'])
        raise RuntimeError('No terminated source stream fits; frozen protocol has no invented null')
    result = meter.call('source_and_actual_fallback', source)
    return result, dict(attempts=attempts,
        tokens_sha256=hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest())


def arpc_tx(pixels, r, snr, c, meter):
    from PIL import Image
    t = c.torch
    def caption():
        values = c.processor(images=Image.fromarray(pixels.transpose(1,2,0)), return_tensors='pt').to('cuda')
        with t.inference_mode():
            generated = c.blip.generate(**values, do_sample=False, num_beams=3, max_length=20, min_length=5)
        return c.processor.decode(generated[0], skip_special_tokens=True)
    text = meter.call('caption', caption)
    def tokenize():
        x = t.from_numpy(pixels.copy()).unsqueeze(0).cuda().float().div(255).mul(2).sub(1)
        with t.inference_mode():
            raw, _, _ = c.vae.encode_for_raw_features(x, scale_schedule=c.schedule)
            _, tokens = c.util.mask_quant(c.vae, c.q_schedule, raw, 'cuda')
        return tokens
    tokens = meter.call('tokenizer', tokenize)
    def encode():
        assert len(c.tokenizer(text, truncation=False).input_ids) <= 512
        with t.inference_mode():
            trans, modes, _ = c.sender.encoding(c.args, c.prior, c.vae, c.schedule,
                                              text, c.tokenizer, c.textencoder, tokens)
        streams = {e: c.pack(text, trans[:e], modes[:e], e)[0] for e in range(1, 8)}
        policy = r['arpc_policies'][str(snr)]
        e = max([e for e in streams if len(streams[e])*8 <= policy['source_capacity_bits']], default=0)
        return dict(payload_b64=b64(streams[e] if e else b''), endpoint=e,
                    q=policy['q'], nominal_rate=policy['nominal_rate']), {str(e): len(v) for e, v in streams.items()}
    result, lengths = meter.call('source_probability_container_fallback', encode)
    return result, dict(lengths_bytes=lengths, caption_calls=1, tokenizer_calls=1, probability_passes=1)


def native_rx(actual, r, method, ctx, meter):
    import numpy as np
    image = np.full((3,256,256), .5, dtype=np.float32)
    if not actual['accepted']:
        return image, actual['status']
    if method == 'NEST':
        from ep_source_codec import InvalidSourceStream
        n, codec, render, _ = ctx
        p = actual['profile']
        try:
            v = meter.call('independent_source_decode', lambda: codec.decode(
                p['family'], np.asarray(actual['payload'], dtype=np.uint8), p['m'], p['K']))
        except InvalidSourceStream:
            return image, 'SOURCE_PARSE_REJECT'
        with n.torch.no_grad():
            image = meter.call('VAR_and_Dc', lambda: render(n, v['received_tokens'], p['m'], p['K']))
        return image, 'ARITHMETIC_SOURCE_DECODED'
    if method == 'ARPC':
        c = ctx
        e = actual['profile']['endpoint']
        if e == 0:
            return image, 'NULL_PROFILE_GRAY'
        def decode():
            caption, trans, modes = c.unpack(base64.b64decode(actual['payload_b64']), e)
            with c.torch.inference_mode():
                tokens = c.decode(c.args, c.prior, c.vae, c.schedule, caption,
                                 c.tokenizer, c.textencoder, None, trans, modes)
            return caption, tokens
        # Source parser rejection rules match the original source receiver.
        from wire import SourceParseError
        try:
            caption, tokens = meter.call('independent_source_decode', decode)
        except SourceParseError:
            return image, 'SOURCE_PARSE_REJECT'
        except (ValueError, AssertionError) as exc:
            # Do not turn generic model exceptions into scientific packet failures.
            raise RuntimeError('ARPC source decode needs diagnosis; preserve actual bytes') from exc
        def render():
            c.prior.rng.manual_seed(0)
            with c.torch.inference_mode():
                out = c.receiver.decompress_cfg(c.prior, c.vae, c.schedule, caption,
                    c.tokenizer, c.textencoder, e-1, tokens, cfg_list=3, tau_list=.5)
            return out[0].float().cpu().numpy()
        return meter.call('ARPC_completion_decoder', render), 'ARPC_DECODED'
    import bpg
    decoded, status = meter.call('BPG_decode_and_resize', lambda: bpg.decode(
        base64.b64decode(actual['payload_b64']), r['bpg_config'], r['_tmpfs']))
    return (image if decoded is None else decoded), status


def phy_runtime(r, method, out):
    native.paths()
    import torch
    native.flags(torch)
    if method == 'NEST':
        import ep_phy, tb_plan
        rt = ep_phy.create_runtime(R, checked(r['phy_qualification']), configure_threads=False)
        cat = read(checked(r['catalogue']))
        assert tb_plan.extend_catalogue(rt.catalogue) == cat
        rt.catalogue, rt.profiles = cat, {str(p['profile_id']): p for p in cat['profiles']}
    elif method == 'ARPC':
        m = native.load(R/'experiments/arpc-external-baseline-20261019/phy_v1/link.py', '_timing_arpc_link')
        Runtime = native.replace_function(m, 'Runtime', [
            ('t.set_num_threads(2);t.set_num_interop_threads(1)', 'assert t.get_num_threads()==6 and t.get_num_interop_threads()==2')])
        rt = Runtime()
    else:
        sys.path.insert(0, str(T/'assets/bpg_support'))
        m = native.load(T/'assets/bpg_support/bpg_phy.py', '_timing_bpg_link')
        Link = native.replace_function(m, 'Link', [
            ("root='/home/liulu/projects/VAR_COMM'", 'root='+repr(str(R)))])
        rt = Link(r['bpg_config'])
    rt.header.s.ROOT = B/'baseline_strength_20261011/native_build'
    rt.header.s.load_native()
    assert torch.get_num_threads() == 6 and torch.get_num_interop_threads() == 2
    assert not torch.cuda.is_initialized()
    return rt


def phy_call(data, r, method, row, rt, calls, meter):
    import numpy as np
    snr, counter = row['snr_db'], row['counter']
    if method == 'NEST':
        p = next(p for p in rt.profiles.values() if
                 (p['m'],p['K'],p['q'],p['nominal_rate'],p['family']) ==
                 (data['m'],data['K'],data['q'],data['nominal_rate'],
                  'EC_VAR_PARTIAL' if data['K'] else 'EC_VAR_WHOLE'))
        wave, _ = meter.call('phy_tx', lambda: rt.transmit(p['profile_id'], np.asarray(data['payload'],dtype=np.uint8), counter))
        import ep_phy
        observation = wave+ep_phy.standard_noise(row['source_id'],6201)*10**(-snr/20)
        rx = meter.call('phy_rx', lambda: rt.receive(observation,snr,counter,rt.profiles,calls,row['key'],phase='fixed16_timing'))
        b, p = rx['body'], rx['rx_profile']
        accepted = bool(rx['header']['header_ok'] and b and b['crc_accepted'] and b['parser_accepted'])
        actual = dict(accepted=accepted,status=rx['status'],profile=p,payload=b['payload'] if accepted else None)
    elif method == 'ARPC':
        p = rt.profile(data['endpoint'],data['q'],data['nominal_rate'])
        wave, _ = meter.call('phy_tx',lambda:rt.transmit(base64.b64decode(data['payload_b64']),p['profile_id'],counter))
        import ep_phy
        observation = wave+ep_phy.standard_noise(row['source_id'],6201)*10**(-snr/20)
        rx = meter.call('phy_rx',lambda:rt.receive(observation,snr,counter,calls,row['key']))
        b = rx['body']
        accepted = bool(rx['header']['header_ok'] and b and b['crc_accepted'] and b['parser_accepted'])
        actual = dict(accepted=accepted,status=rx['status'],profile=rx['profile'],
                      payload_b64=b64(b['payload']) if accepted else None)
    else:
        if data['payload_b64'] is None:
            return dict(accepted=False,status='SOURCE_UNFIT',payload_b64=None),dict(no_frame=True)
        p = next(p for p in r['bpg_config']['catalogue'] if p['profile_id']==data['profile_id'])
        wave = meter.call('phy_tx',lambda:rt.transmit(base64.b64decode(data['payload_b64']),p,counter))
        observation = rt.noisy(wave,'A3_DEVELOPMENT_TIMING',snr,row['source_index'],2001)
        rx = meter.call('phy_rx',lambda:rt.receive(observation,snr,counter,calls,'timing',row['key']))
        b = rx['body']
        accepted = bool(rx['header']['header_ok'] and b and b['crc_accepted'] and b['parser_accepted'])
        actual = dict(accepted=accepted,status=rx['status'],payload_b64=b['payload_b64'] if accepted else None)
    assert wave.shape == observation.shape == (1024,2)
    return actual,dict(wave_sha256=array_sha(wave),observation_sha256=array_sha(observation),
                       energy=float(np.square(wave.astype(np.float64)).sum()),packet_calls=calls.counts.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=METHODS)
    ap.add_argument('role', choices=['tx','phy','rx'])
    ap.add_argument('--registration',required=True)
    a = ap.parse_args()
    r = binding(a.registration)
    out = T/a.method/a.role
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, set(r['resource']['cpu_affinity']))
    cache = out/'runtime_cache'
    (cache/'tmp').mkdir(parents=True)
    os.environ.update(TMPDIR=str(cache/'tmp'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        TRITON_CACHE_DIR=str(cache/'triton'), HF_HOME=str(cache/'hf'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    tempfile.tempdir=str(cache/'tmp')
    r['_tmpfs']='/dev/shm/five_method_timing_20261020_v1/'+a.method+'_'+a.role
    if a.method=='BPG':
        Path(r['_tmpfs']).mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv,registration_sha256=a.registration))
    completed=[]
    try:
        save(out/'resource_before.json',resource(r, initial=a.role!='phy'))
        blocked=firewall(r,a.method,out) if a.role=='rx' else []
        import numpy as np
        t=None;ctx=None;calls=native.Calls()
        setup=time.perf_counter()
        if a.role!='phy' and a.method!='BPG':
            import torch,ctypes
            torch.cuda.set_device(0)
            lib=ctypes.CDLL('libcuda.so.1')
            device=ctypes.c_int()
            assert lib.cuDeviceGet(ctypes.byref(device),0)==0
            buf=ctypes.create_string_buffer(32)
            assert lib.cuDeviceGetPCIBusId(buf,32,device)==0
            pci=buf.value.decode()
            assert pci.lower().lstrip('0')==r['resource']['gpu_pci'].lower().lstrip('0'),pci
            save(out/'actual_device.json',dict(PCI=pci,CUDA_UUID=str(torch.cuda.get_device_properties(0).uuid),
                NVML_UUID=r['resource']['gpu_uuid'],CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        if a.role=='phy':
            assert a.method in ('NEST','ARPC','BPG')
            ctx=phy_runtime(r,a.method,out)
        elif a.method in ('P100000','Swin200000'):
            ctx,calls=native.continuous(r,out,a.method);t=ctx.t
        elif a.method=='NEST':
            ctx=native.nest_codec(r,out);t=ctx[0].torch;calls=ctx[3]
        elif a.method=='ARPC':
            ctx=native.arpc(r,out,a.role);t=ctx.torch
        elif a.method=='BPG':
            import bpg
            bpg.legacy()
        if t is not None:t.cuda.synchronize()
        save(out/'setup.json',dict(seconds=time.perf_counter()-setup,python=sys.version,
            platform=platform.platform(),torch=None if t is None else t.__version__,
            cuda=None if t is None else t.version.cuda,threads=6,interop_threads=2,
            CPU_affinity=sorted(os.sched_getaffinity(0)),separate_process_role=a.role,
            cached_scientific_calls=0))
        rxinputs=T/a.method/'rx_inputs'
        if a.role=='phy' or (a.role=='tx' and a.method in ('P100000','Swin200000')):
            rxinputs.mkdir()
        (out/'cases').mkdir()
        for row0 in r['grid']:
            assert time.monotonic()-start<r['stage_wall_seconds'][a.role], 'Stage wall limit'
            assert not (T/'STOP').exists()
            row=dict(row0,source_id=r['sources'][row0['source_slot']]['source_id'])
            key=row['key'];case=out/'cases'/key;case.mkdir()
            save(case/'reservation.json',dict(row,process=process(os.getpid()),resource=resource(r)))
            meter=Clock(t)
            evidence={}
            if a.role=='tx':
                s=r['sources'][row['source_slot']]
                with np.load(checked(s['archive']),allow_pickle=False) as z:
                    pixels=z['pixels'].copy()
                assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
                if a.method=='NEST':
                    result,evidence=meter.call('source_tx',lambda:nest_tx(pixels,r,row['snr_db'],ctx,meter))
                    assert evidence['tokens_sha256']==s['tokens_sha256'], 'Fresh tokenizer mismatch'
                elif a.method=='ARPC':
                    result,evidence=meter.call('source_tx',lambda:arpc_tx(pixels,r,row['snr_db'],ctx,meter))
                elif a.method=='BPG':
                    pid=r['bpg_policies'][str(row['snr_db'])]
                    profile=next(p for p in r['bpg_config']['catalogue'] if p['profile_id']==pid)
                    payload,evidence=meter.call('source_tx',lambda:bpg.encode(pixels,r['bpg_config'],profile['capacity_bytes'],r['_tmpfs']))
                    result=dict(payload_b64=None if payload is None else b64(payload),profile_id=pid)
                else:
                    wave,witness=meter.call('source_tx',lambda:ctx.encode(pixels,row['snr_db'],row['counter'],meter))
                    tick=time.perf_counter()
                    observed=ctx.noise(wave,row['source_id'],row['snr_db'])
                    evidence=dict(channel_seconds=time.perf_counter()-tick,wave_sha256=array_sha(wave),observation_sha256=array_sha(observed))
                    with (rxinputs/(key+'.npz')).open('xb') as f:np.savez(f,observed=observed)
                    result=dict(observation=pin(rxinputs/(key+'.npz')))
                save(case/'output.json',result)
            elif a.role=='phy':
                p=T/a.method/'tx/cases'/key/'output.json'
                data=read(p)
                result,evidence=phy_call(data,r,a.method,row,ctx,calls,meter)
                save(rxinputs/(key+'.json'),result)
            else:
                if a.method in ('P100000','Swin200000'):
                    with np.load(rxinputs/(key+'.npz'),allow_pickle=False) as z:observed=z['observed'].copy()
                    image,actual=meter.call('native_rx',lambda:ctx.receive(observed,row['snr_db'],row['counter'],meter,key))
                    status=actual['status']
                else:
                    actual=read(rxinputs/(key+'.json'))
                    image,status=meter.call('native_rx',lambda:native_rx(actual,r,a.method,ctx,meter))
                assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all()
                assert image.min()>=0 and image.max()<=1
                evidence=dict(image_sha256=array_sha(image),status=status,source_truth_used=False)
                # Save one fixed first-source warmup image per SNR for output inspection.
                if row['source_slot']==0 and row['phase']=='warmup' and row['repeat']==0:
                    with (case/'image.npz').open('xb') as f:np.savez(f,image=image)
            report=dict(row,seconds=meter.parts,evidence=evidence,model_calls=calls.counts.copy())
            save(case/'complete.json',report)
            completed.append(report)
            print(json.dumps(dict(method=a.method,role=a.role,completed=len(completed),total=288,key=key)),flush=True)
        assert len(completed)==288 and not blocked
        save(out/'completion.json',dict(status='COMPLETE',registration_sha256=a.registration,
            calls=len(completed),formal=sum(v['phase']=='measure' for v in completed),
            seconds=time.monotonic()-start,blocked_reads=blocked,
            resource_after=resource(r),results=completed))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed=len(completed),automatic_retry=False))
        raise


if __name__=='__main__':
    main()
