"""Engineering gate for frozen Swin + HiFi; never a reconstruction result.

Reads two original calibration images, tests full input derivatives and native
forward parity, then runs two real ADM reverse steps at each total budget.
The diagnostic step limit is confined to this qualification command.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from swin_data import RGBPopulation,read,sha
from swin_protocol import RATES,configure_phy,decode_header,layout,standard_noise
from swin_replay import load_selected,transmit,receive
from swin_state import register,write
from swin_train import configure_runtime,available_gpu,PauseRequested
from hifi_swin_operator import SwinHiFiOperator,strict_decoded_context,qualify_native_parity
from hifi_swin_sampler import FrozenHiFiSwinReceiver,ADM_SHA256

HERE=Path(__file__).resolve().parent
HIFI_FILES=('hifi_swin_operator.py','hifi_swin_sampler.py','hifi_swin_schedule.py',
    'hifi_swin_schedule_tests.py','hifi_swin_tests.py')
EPSILONS=(1e-3,3e-4)
FD_RELATIVE_TOLERANCE=.03
FD_ABSOLUTE_TOLERANCE=2e-7


def state_fingerprint(model):
    value=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        data=tensor.detach().cpu().contiguous().numpy()
        value.update(key.encode()+b'\0'+str(data.dtype).encode()+b'\0'+str(data.shape).encode()+b'\0')
        value.update(data.tobytes())
    return value.hexdigest()


def parameter_guard(model):
    parameters=tuple(model.parameters())
    if any(p.grad is not None for p in parameters):
        raise RuntimeError('Qualification requires clean model parameter gradients')
    return dict(state_sha256=state_fingerprint(model),versions=tuple(p._version for p in parameters),
        requires_grad=tuple(p.requires_grad for p in parameters))


def assert_unchanged(model,before):
    after=parameter_guard(model)
    if after != before: raise RuntimeError('Qualification changed model tensors or autograd flags')
    return after['state_sha256']


def source_bindings(swin_vendor,diffcom_vendor):
    files={HERE/name for name in HIFI_FILES}
    files.update(HERE.glob('swin_*.py'))
    files.add(Path(__file__).resolve())
    for root in (Path(swin_vendor).resolve(),Path(diffcom_vendor).resolve()):
        if not root.is_dir(): raise RuntimeError('Frozen upstream vendor directory is absent')
        files.update(p for p in root.rglob('*') if p.is_file() and
            p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts and '.git' not in p.parts)
    return {str(path.resolve()):sha(path) for path in sorted(files)}


def native_channel_parity(codec,image,N,snr,noise):
    """Replay the same noise through the official full Swin.forward method."""
    native=codec.native
    with torch.no_grad():
        _,mask=codec.encode_features(image,snr,RATES[N])
        half=mask.numel()//2
        if not torch.equal(mask.flatten()[:half],mask.flatten()[half:]):
            raise RuntimeError('Native complex channel mask pairing changed')
        active=torch.where(mask.flatten()[:half]>0)[0]
        standard=torch.as_tensor(noise,device=image.device,dtype=torch.float32)
        if tuple(standard.shape) != (layout(N)['data_N'],2): raise RuntimeError('Native parity noise shape differs')
        own=codec.training_forward(image,snr,RATES[N],standard[None])
        original=native.channel.gaussian_noise_layer
        existed='gaussian_noise_layer' in native.channel.__dict__
        def supplied(channel_input,std,name=None):
            values=torch.zeros_like(channel_input)
            values[active]=torch.complex(standard[:,0],standard[:,1])*float(std)
            return channel_input+values
        native.channel.gaussian_noise_layer=supplied
        try:
            reference=native(image,given_SNR=snr,given_rate=RATES[N])[0].clamp(0,1)
        finally:
            if existed: native.channel.gaussian_noise_layer=original
            else: delattr(native.channel,'gaussian_noise_layer')
        torch.testing.assert_close(own,reference,rtol=2e-5,atol=3e-6)
        return dict(max_abs=float((own-reference).abs().max()),
            mean_abs=float((own-reference).abs().mean()),real_official_forward=True,
            same_standard_noise=True,batch_size=1,passed=True)


def finite_difference(operator,image,measurement,objective_name):
    """Two predetermined central differences along the full gradient sign.

    Sign direction avoids cancellation of a near-zero random directional
    derivative. No mask re-selection, TX gain, clipping of perturbed input,
    learned-parameter gradient, or tolerance search is used.
    """
    if objective_name not in ('waveform','confirming_rgb'): raise ValueError(objective_name)
    def objective(candidate):
        predicted=operator.forward(operator.encode(candidate))
        if objective_name=='waveform':
            return .5*(predicted-measurement['ofdm_sig']).square().mean()
        reconstructed=operator.decode(operator.transpose(predicted))
        return .5*(reconstructed-measurement['x_mse']).square().mean()
    candidate=image.detach().clone().requires_grad_(True)
    value=objective(candidate)
    gradient,=torch.autograd.grad(value,candidate)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('Nonfinite full input derivative')
    direction=gradient.sign().detach()
    analytic=float((gradient*direction).double().sum())
    if analytic <= 1e-10: raise RuntimeError('Degenerate full input derivative cannot be qualified')
    comparisons=[]
    with torch.no_grad():
        for epsilon in EPSILONS:
            upper=float(objective(image+epsilon*direction))
            lower=float(objective(image-epsilon*direction))
            numeric=(upper-lower)/(2*epsilon)
            error=abs(numeric-analytic)
            tolerance=FD_ABSOLUTE_TOLERANCE+FD_RELATIVE_TOLERANCE*max(abs(numeric),abs(analytic))
            comparisons.append(dict(epsilon=epsilon,analytic=analytic,central_difference=numeric,
                absolute_error=error,tolerance=tolerance,passed=error<=tolerance))
    if not all(r['passed'] and math.isfinite(r['central_difference']) for r in comparisons):
        raise RuntimeError('Full input derivative finite differences failed: '+repr(comparisons))
    return dict(objective=objective_name,loss=float(value),gradient_l2=float(gradient.double().norm()),
        gradient_nonzero_elements=int(torch.count_nonzero(gradient)),direction='full_gradient_sign',
        input_gradient_includes_encoder_and_decoder=objective_name=='confirming_rgb',
        fixed_received_mask_and_gain=True,comparisons=comparisons,passed=True)


def run(args):
    import fcntl
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=output/'qualification.json'
    request={key:str(Path(value).resolve()) for key,value in vars(args).items()}
    if path.exists():
        receipt=read(path)
        if receipt.get('status')!='HIFI_SWIN_QUALIFICATION_PASS' or receipt.get('request')!=request or receipt.get('torch')!=torch.__version__:
            raise RuntimeError('Existing qualification identity is not admitted')
        for file,expected in {**receipt['source_bindings'],**receipt['input_bindings']}.items():
            if sha(file)!=expected: raise RuntimeError('Frozen qualification binding changed: '+file)
        print('HiFi Swin qualification already passed and all bindings verified',flush=True)
        return
    if (output/'failure.json').exists(): raise RuntimeError('Earlier qualification failure requires diagnosis')
    available_gpu(); configure_runtime(); phy=configure_phy(args.root)
    bindings=source_bindings(args.swin_vendor,args.diffcom_vendor)
    bindings.update({str(p):sha(p) for p in (phy,phy.with_name('token_trellis.cpp'))})
    population=RGBPopulation(args.image_cache,args.reference_registration,'calibration')
    codec,selected=load_selected(args.swin_output,args.swin_vendor)
    registered=read(Path(args.swin_output)/'registration.json')['bindings']
    for file,expected in registered.items():
        if file in bindings and bindings[file]!=expected:
            raise RuntimeError('A source shared with the trained Swin registration changed: '+file)
    initial_swin=parameter_guard(codec.native)
    files={**population.bindings,selected['checkpoint']:selected['checkpoint_sha256'],
        str(Path(args.adm_checkpoint).resolve()):ADM_SHA256}
    for name in ('selected_swin.json','registration.json','completion.json'):
        file=Path(args.swin_output).resolve()/name; files[str(file)]=sha(file)
    for file,expected in files.items():
        if sha(file)!=expected: raise RuntimeError('Qualification input binding changed: '+file)
    source_ids=population.ids[:2]
    frames=[]; checks=[]; started=time.perf_counter()
    try:
        for index,N in enumerate(RATES):
            available_gpu()
            image=population.batch([index],'cuda:0')
            for snr in (1,13):
                parity=qualify_native_parity(codec.native,image,snr,RATES[N])
                checks.append(dict(N=N,snr=snr,source_index=index,source_id=source_ids[index],
                    kind='native_encoder_decoder',result=parity))
            snr=7
            noise=standard_noise(source_ids[index],2026100451,N,snr)
            checks.append(dict(N=N,snr=snr,source_index=index,source_id=source_ids[index],
                kind='official_full_forward',result=native_channel_parity(codec,image,N,snr,noise[:layout(N)['data_N']])))
            signal=transmit(codec,image,N,snr)
            observed=signal+noise/10**(snr/20)
            rx=decode_header(observed[layout(N)['data_N']:],snr,N)
            if not rx.accepted: raise RuntimeError('Fixed qualification header failed; do not resample a favorable frame')
            context=strict_decoded_context(N,snr,{'power':rx.power,'indices':rx.indices},crc_accepted=rx.accepted)
            operator=SwinHiFiOperator(codec.native,context)
            measurement=operator.measurement(observed[:layout(N)['data_N']])
            baseline,_=receive(codec,observed,N,snr)
            np.testing.assert_allclose(measurement['x_mse'][0].cpu().numpy(),baseline,rtol=2e-5,atol=3e-6)
            for objective in ('waveform','confirming_rgb'):
                checks.append(dict(N=N,snr=snr,source_index=index,source_id=source_ids[index],
                    kind='full_input_gradient',result=finite_difference(operator,image,measurement,objective)))
            frames.append(dict(N=N,snr=snr,observed=observed,source_id=source_ids[index],source_index=index,
                signal_energy=float(np.square(signal).sum()),rx=asdict(rx)))
            write(output/'status.json',dict(status='OPERATOR_QUALIFYING',N=N,completed_checks=len(checks),scientific_result=False))
        assert_unchanged(codec.native,initial_swin)
        available_gpu()
        receiver=FrozenHiFiSwinReceiver(codec.native,args.diffcom_vendor,args.adm_checkpoint)
        initial_adm=parameter_guard(receiver.unet)
        probes=[]
        for frame in frames:
            available_gpu()
            image,probe=receiver.receive_frame(frame['observed'],frame['N'],frame['snr'],sampling_seed=23,step_limit=2)
            if probe.get('NFE')!=2 or probe.get('diagnostic_probe') is not True or probe.get('header_accepted') is not True:
                raise RuntimeError('Expected exactly two actual ADM reverse steps at each budget')
            if tuple(image.shape)!=(3,256,256) or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
                raise RuntimeError('Two-step ADM qualification output is invalid')
            probes.append(dict(N=frame['N'],snr=frame['snr'],source_id=frame['source_id'],
                source_index=frame['source_index'],actual_frame_energy=frame['signal_energy'],
                observed_waveform_sha256=hashlib.sha256(frame['observed'].tobytes()).hexdigest(),
                actual_rx_context=frame['rx'],sampler_receipt=probe))
            write(output/'status.json',dict(status='ADM_PROBING',completed_budgets=len(probes),scientific_result=False))
        swin_after=assert_unchanged(codec.native,initial_swin)
        adm_after=assert_unchanged(receiver.unet,initial_adm)
        for file,expected in {**bindings,**files}.items():
            if sha(file)!=expected: raise RuntimeError('Qualification input mutated during probe: '+file)
        receipt=dict(status='HIFI_SWIN_QUALIFICATION_PASS',request=request,full_input_gradient_pass=True,
            native_forward_parity_pass=True,model_parameters_unchanged=True,
            parameter_gradients_accumulated=False,selected_checkpoint_sha256=selected['checkpoint_sha256'],
            adm_checkpoint_sha256=ADM_SHA256,source_bindings=bindings,input_bindings=files,
            torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
            numeric=dict(dtype='float32',TF32=False,deterministic=True),
            source_role='original_calibration',source_ids=source_ids,source_indices=[0,1],
            development_read=False,holdout_read=False,checks=checks,real_ADM_probes=probes,
            model_state_before_after=dict(swin_before=initial_swin['state_sha256'],swin_after=swin_after,
                adm_before=initial_adm['state_sha256'],adm_after=adm_after),
            finite_difference_epsilons=list(EPSILONS),finite_difference_relative_tolerance=FD_RELATIVE_TOLERANCE,
            finite_difference_absolute_tolerance=FD_ABSOLUTE_TOLERANCE,
            seconds=time.perf_counter()-started,qualification_only=True,engineering_only=True,
            scientific_result=False,synthetic=False,formal_evaluation_step_limit_permitted=False)
        register(path,receipt); write(output/'status.json',dict(status=receipt['status'],scientific_result=False))
        print('HIFI_SWIN_QUALIFICATION_PASS',flush=True)
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
    arguments=parser.parse_args()
    try:
        run(arguments)
    except PauseRequested as error:
        write(Path(arguments.output)/'status.json',dict(status='PAUSED',reason=str(error),scientific_result=False))
        raise SystemExit(75)
