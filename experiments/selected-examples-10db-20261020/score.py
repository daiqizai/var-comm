"""Only PSNR and LPIPS of the twenty fixed, exact-hash receiver outputs."""
import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
import execute as x


def main(expected, metric_expected):
    reg = x.bind(expected)
    assert x.sha(x.O/'metric_registration.json') == metric_expected
    plan = x.read(x.O/'metric_registration.json')
    for d in plan['sources']:
        x.checked(d)
    assert x.read(x.T/'dispatch/actual_wait.json')['returncode'] == 0
    assert x.read(x.T/'control/completion.json')['rx_reconstructions'] == 20
    out = x.T/'metrics'
    out.mkdir()
    x.setup(out)
    x.save(out/'launch.json', dict(process=x.process(os.getpid()),
        parent_process=x.process(os.getppid()), argv=sys.argv))
    results = []
    try:
        import numpy as np
        import torch
        import lpips
        import worker
        x.save(out/'resource_before.json', worker.resource(x.binding(x.OLD), initial=True))
        x.native.flags(torch)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction((6 << 30)/torch.cuda.get_device_properties(0).total_memory)
        sys.path.insert(0, str(Path(plan['metric_math']['path']).parent))
        metric = x.native.load(x.checked(plan['metric_math']), '_selected_exact_two_metrics')
        psnr, psnr_proof = metric.psnr_projection(x.checked(plan['psnr_source']))
        loader, pair, lpips_proof = metric.lpips_projection(
            x.checked(plan['lpips_source']), dict(Path=Path, lpips=lpips, torch=torch))
        linear = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
        assert x.sha(linear) == metric.WEIGHT_SHA['lpips']
        weight = x.checked(plan['alexnet'])
        x.save(out/'model_reserved.json', dict(model='LPIPS-Alex', weights=plan['alexnet']))
        model = loader(dict(alexnet_checkpoint=str(weight)), 'cuda:0')
        x.save(out/'model_complete.json', dict(model='LPIPS-Alex'))
        calls = dict(started=0, completed=0)
        def before(*_):
            calls['started'] += 1
            assert calls['started'] <= 40
        def after(*_):
            calls['completed'] += 1
        model.net.register_forward_pre_hook(before)
        model.net.register_forward_hook(after)
        x.save(out/'metric_identity.json', dict(original_math=dict(psnr=psnr_proof,lpips=lpips_proof),
            alexnet=plan['alexnet'], lpips_linear=x.pin(linear), torch=torch.__version__,
            lpips_package=x.pin(Path(lpips.__file__)),
            input='Original RGB uint8 / 255 versus exact native float32 CHW reconstruction; no resize/crop',
            LPIPS='Original Alex v0.1 expression, inputs mapped from [0,1] to [-1,1]',
            PSNR='Original released float64 MSE; data range 1',
            precision='FP32, TF32 off, deterministic; six threads / two interop',
            population='Four previously selected development examples, single fixed receive result'))
        (out/'pairs').mkdir()
        for c in reg['cases']:
            name = c['method']+'_'+str(c['source_index'])
            d = out/'pairs'/name
            d.mkdir()
            result = x.read(x.T/c['method']/'cases'/c['key']/'complete.json')
            assert result['image_sha256'] == c['expected_image_sha256']
            source = next(s for s in reg['sources'] if s['source_index'] == c['source_index'])
            with np.load(x.checked(source['archive']), allow_pickle=False) as z:
                target = z['pixels'].astype(np.float32)/np.float32(255)
            with np.load(x.checked(result['image']), allow_pickle=False) as z:
                image = z['image'].copy()
            assert worker.array_sha(image) == result['image_sha256']
            assert target.shape == image.shape == (3,256,256)
            x.save(d/'reserved.json', dict(source_id=c['source_id'],image=result['image'],
                expected_image_sha256=c['expected_image_sha256']))
            with torch.inference_mode():
                reference = torch.from_numpy(target[None]).cuda()
                reconstructed = torch.from_numpy(image[None]).cuda()
                value = float(pair(model, reconstructed, reference)[0].item())
            values = dict(psnr_db=psnr(target,image),lpips_alex=value)
            assert all(math.isfinite(v) for v in values.values())
            row = dict(method=c['method'],source_index=c['source_index'],source_id=c['source_id'],
                key=c['key'],snr_db=10,status=result['status'],
                noise_seed=c['noise_seed'],noise_namespace=c['noise_namespace'],
                image=result['image'],image_sha256=result['image_sha256'],**values)
            x.save(d/'complete.json', row)
            results.append(row)
        assert len(results) == 20 and calls == dict(started=40,completed=40)
        x.save(out/'completion.json', dict(status='COMPLETE',results=results,
            new_pairs=20,psnr=20,lpips=20,lpips_backbone_calls=calls,model_constructions=1,
            DINO=0,DreamSim=0,bootstrap=0,noise_averaging=False,
            metrics_on_native_float=True,PNG_is_uniform_rounded_display_copy=True))
    except BaseException:
        x.save(out/'failure.json', dict(traceback=traceback.format_exc(),
            completed=len(results),automatic_retry=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registration', required=True)
    p.add_argument('--metric-registration', required=True)
    a = p.parse_args()
    main(a.registration, a.metric_registration)
