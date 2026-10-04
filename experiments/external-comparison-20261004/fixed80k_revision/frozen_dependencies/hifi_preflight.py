"""Early engineering-only Swin/ADM integration probe before formal training.

This uses the registered random initialization, not a selected trained codec.
Its receipt cannot satisfy the post-training HiFi qualification gate. It has no
SIGTERM handler: the outer controller waits for this short probe to finish.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from swin_data import RGBPopulation,read,sha
from swin_model import build_official
from swin_protocol import RATES,configure_phy,decode_header,layout,standard_noise
from swin_replay import transmit,receive
from swin_state import identity,register,write
from swin_train import configure_runtime,available_gpu,PauseRequested,validate_config,model_hash
from hifi_swin_operator import SwinHiFiOperator,strict_decoded_context,qualify_native_parity
from hifi_swin_sampler import FrozenHiFiSwinReceiver,ADM_SHA256
from hifi_qualification import (source_bindings,parameter_guard,assert_unchanged,
    native_channel_parity,finite_difference,EPSILONS,FD_RELATIVE_TOLERANCE,FD_ABSOLUTE_TOLERANCE)

HERE=Path(__file__).resolve().parent


def run(args):
    import fcntl
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request={key:str(Path(value).resolve()) for key,value in vars(args).items()}
    path=output/'preflight.json'
    if path.exists():
        result=read(path)
        if result.get('status')!='ENGINEERING_PREFLIGHT' or result.get('passed') is not True or result.get('request')!=request or result.get('torch')!=torch.__version__:
            raise RuntimeError('Existing engineering preflight identity differs')
        for file,expected in {**result['source_bindings'],**result['input_bindings']}.items():
            if sha(file)!=expected: raise RuntimeError('Engineering preflight binding changed: '+file)
        print('ENGINEERING_PREFLIGHT already passed; this is not a trained-model qualification',flush=True)
        return
    if (output/'failure.json').exists(): raise RuntimeError('Earlier preflight failure requires diagnosis')
    available_gpu(); configure_runtime(); phy=configure_phy(args.root)
    config=read(args.config); validate_config(config)
    swin_qualification=Path(args.swin_output).resolve()/'qualification.json'
    admitted=read(swin_qualification)
    if admitted.get('status')!='PASS' or admitted.get('formal_training_updates')!=0 or admitted.get('config_sha256')!=identity(config):
        raise RuntimeError('Matching real Swin qualification is required before the HiFi preflight')
    for file,expected in admitted['bindings'].items():
        if sha(file)!=expected: raise RuntimeError('Prior Swin qualification input changed: '+file)
    if (Path(args.swin_output)/'registration.json').exists() or (Path(args.swin_output)/'latest.json').exists():
        raise RuntimeError('Early preflight must precede formal Swin training registration')
    bindings=source_bindings(args.swin_vendor,args.diffcom_vendor)
    bindings.update({str(p):sha(p) for p in (phy,phy.with_name('token_trellis.cpp'))})
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    bindings[str(Path(args.config).resolve())]=sha(args.config)
    population=RGBPopulation(args.image_cache,args.reference_registration,'calibration')
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    codec=build_official(args.swin_vendor)
    initial_codec_sha=model_hash(codec)
    if initial_codec_sha!=admitted['initial_model_sha256']:
        raise RuntimeError('Preflight random Swin does not match the qualified training initialization')
    codec.eval().requires_grad_(False)
    initial_swin=parameter_guard(codec.native)
    files={**population.bindings,str(swin_qualification):sha(swin_qualification),
        str(Path(args.adm_checkpoint).resolve()):ADM_SHA256}
    for file,expected in files.items():
        if sha(file)!=expected: raise RuntimeError('Engineering preflight input differs: '+file)
    frames=[]; checks=[]; started=time.perf_counter()
    try:
        for index,N in enumerate(RATES):
            available_gpu()
            image=population.batch([index],'cuda:0')
            for snr in (1,13):
                checks.append(dict(N=N,snr=snr,source_index=index,source_id=population.ids[index],
                    kind='native_encoder_decoder',result=qualify_native_parity(codec.native,image,snr,RATES[N])))
            snr=7
            noise=standard_noise(population.ids[index],2026100451,N,snr)
            checks.append(dict(N=N,snr=snr,source_index=index,source_id=population.ids[index],
                kind='official_full_forward',result=native_channel_parity(codec,image,N,snr,noise[:layout(N)['data_N']])))
            signal=transmit(codec,image,N,snr)
            observed=signal+noise/10**(snr/20)
            rx=decode_header(observed[layout(N)['data_N']:],snr,N)
            if not rx.accepted: raise RuntimeError('Fixed engineering header failed; no favorable-frame resampling')
            context=strict_decoded_context(N,snr,{'power':rx.power,'indices':rx.indices},crc_accepted=rx.accepted)
            operator=SwinHiFiOperator(codec.native,context)
            measurement=operator.measurement(observed[:layout(N)['data_N']])
            baseline,_=receive(codec,observed,N,snr)
            np.testing.assert_allclose(measurement['x_mse'][0].cpu().numpy(),baseline,rtol=2e-5,atol=3e-6)
            for objective in ('waveform','confirming_rgb'):
                checks.append(dict(N=N,snr=snr,source_index=index,source_id=population.ids[index],
                    kind='full_input_gradient',result=finite_difference(operator,image,measurement,objective)))
            frames.append(dict(N=N,snr=snr,observed=observed,source_id=population.ids[index],source_index=index,
                signal_energy=float(np.square(signal).sum()),rx=asdict(rx)))
            write(output/'status.json',dict(status='ENGINEERING_OPERATOR_PROBE',N=N,completed_checks=len(checks),scientific_result=False))
        assert_unchanged(codec.native,initial_swin)
        available_gpu()
        receiver=FrozenHiFiSwinReceiver(codec.native,args.diffcom_vendor,args.adm_checkpoint)
        initial_adm=parameter_guard(receiver.unet)
        probes=[]
        for frame in frames:
            available_gpu()
            image,probe=receiver.receive_frame(frame['observed'],frame['N'],frame['snr'],sampling_seed=23,step_limit=2)
            if probe.get('NFE')!=2 or probe.get('diagnostic_probe') is not True or probe.get('header_accepted') is not True:
                raise RuntimeError('Engineering probe did not execute two actual ADM reverse steps')
            if tuple(image.shape)!=(3,256,256) or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
                raise RuntimeError('Engineering ADM probe produced invalid RGB')
            probes.append(dict(N=frame['N'],snr=frame['snr'],source_id=frame['source_id'],source_index=frame['source_index'],
                actual_frame_energy=frame['signal_energy'],actual_rx_context=frame['rx'],
                observed_waveform_sha256=hashlib.sha256(frame['observed'].tobytes()).hexdigest(),sampler_receipt=probe))
            write(output/'status.json',dict(status='ENGINEERING_ADM_PROBE',completed_budgets=len(probes),scientific_result=False))
        swin_after=assert_unchanged(codec.native,initial_swin)
        adm_after=assert_unchanged(receiver.unet,initial_adm)
        for file,expected in {**bindings,**files}.items():
            if sha(file)!=expected: raise RuntimeError('Engineering probe input changed: '+file)
        result=dict(status='ENGINEERING_PREFLIGHT',passed=True,request=request,
            full_input_gradient_pass=True,native_forward_parity_pass=True,model_parameters_unchanged=True,
            parameter_gradients_accumulated=False,initialization='random',initial_model_sha256=initial_codec_sha,
            formal_training_updates=0,swin_qualification_sha256=sha(swin_qualification),
            adm_checkpoint_sha256=ADM_SHA256,source_bindings=bindings,input_bindings=files,
            torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
            numeric=dict(dtype='float32',TF32=False,deterministic=True),source_role='original_calibration',
            source_ids=population.ids[:2],source_indices=[0,1],development_read=False,holdout_read=False,
            checks=checks,real_ADM_probes=probes,model_state_before_after=dict(
                swin_before=initial_swin['state_sha256'],swin_after=swin_after,
                adm_before=initial_adm['state_sha256'],adm_after=adm_after),
            finite_difference_epsilons=list(EPSILONS),finite_difference_relative_tolerance=FD_RELATIVE_TOLERANCE,
            finite_difference_absolute_tolerance=FD_ABSOLUTE_TOLERANCE,seconds=time.perf_counter()-started,
            qualification_only=True,engineering_only=True,scientific_result=False,synthetic=False,
            selected_checkpoint_qualification=False,training_qualification_replacement=False,
            formal_evaluation_step_limit_permitted=False,SIGTERM_handler=False)
        register(path,result); write(output/'status.json',dict(status=result['status'],passed=True,scientific_result=False))
        print('ENGINEERING_PREFLIGHT passed; trained selected checkpoint still requires its own qualification',flush=True)
    except PauseRequested:
        raise
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),
            scientific_result=False,no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','swin-output','swin-vendor','diffcom-vendor','adm-checkpoint','image-cache','reference-registration','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default=str(HERE/'swin_training_protocol.json'))
    arguments=parser.parse_args()
    try:
        run(arguments)
    except PauseRequested as error:
        write(Path(arguments.output)/'status.json',dict(status='PAUSED',reason=str(error),scientific_result=False))
        raise SystemExit(75)
