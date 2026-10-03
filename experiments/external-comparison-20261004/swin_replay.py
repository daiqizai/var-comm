"""Physical Swin frames; receiver functions never accept transmitter context."""
import time
import numpy as np
import torch
from swin_protocol import layout, transmit_frame, observe_frame, decode_header


def load_selected(training_output, vendor, device='cuda:0'):
    """Admit only the completed calibration-selected shared checkpoint."""
    from pathlib import Path
    from swin_data import read,sha
    from swin_model import build_official
    output=Path(training_output).resolve()
    completion=read(output/'completion.json')
    if completion.get('status') != 'SWIN_TRAINING_COMPLETE' or completion.get('synthetic') is not False:
        raise RuntimeError('Swin training has not produced an admitted completed selection')
    for path,expected in completion['outputs'].items():
        if sha(path) != expected: raise RuntimeError('Selected Swin output changed: '+path)
    selected=read(output/'selected_swin.json')
    regsha=sha(output/'registration.json')
    checkpoint=Path(selected['checkpoint']).resolve()
    if not checkpoint.is_relative_to(output/'checkpoints') or sha(checkpoint) != selected['checkpoint_sha256']:
        raise RuntimeError('Selected Swin checkpoint identity differs')
    if selected['registration_sha256'] != regsha or completion['registration_sha256'] != regsha:
        raise RuntimeError('Selected Swin registration differs')
    registration=read(output/'registration.json')
    vendor=Path(vendor).resolve()
    for path in vendor.rglob('*.py'):
        if '__pycache__' in path.parts: continue
        if registration['bindings'].get(str(path.resolve())) != sha(path):
            raise RuntimeError('Selected Swin vendor source differs')
    model=build_official(vendor,device)
    payload=torch.load(checkpoint,map_location='cpu')
    if payload['step'] != selected['step'] or payload['registration_sha256'] != regsha:
        raise RuntimeError('Selected Swin checkpoint metadata differs')
    model.load_state_dict(payload['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,selected


@torch.no_grad()
def transmit(model, image, N, snr):
    if len(image) != 1:
        raise ValueError('Physical single-frame TX requires batch1')
    data, power, indices = model.encode_data(image, snr, layout(N)['channels'])
    signal = transmit_frame(data[0].cpu().numpy(), float(power[0]), indices[0].cpu().tolist(), N)
    return signal


@torch.no_grad()
def receive(model, observed, N, snr):
    spec = layout(N)
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != (N, 2) or not np.isfinite(observed).all():
        raise ValueError('Physical single-frame observation differs')
    context = decode_header(observed[spec['data_N']:], snr, N)
    if not context.accepted:
        return np.full((3, 256, 256), .5, dtype=np.float32), context
    device = next(model.parameters()).device
    data = torch.as_tensor(observed[:spec['data_N']], device=device, dtype=torch.float32)[None]
    power = data.new_tensor([context.power])
    indices = torch.tensor([context.indices], device=device, dtype=torch.long)
    return model.receive_data(data, snr, power, indices)[0].cpu().numpy(), context


@torch.no_grad()
def physical_batch(model, images, N, snr, noises):
    """Batch neural work, but encode/decode every actual paid metadata packet."""
    spec = layout(N)
    data, powers, indices = model.encode_data(images, snr, spec['channels'])
    data_cpu, power_cpu, indices_cpu = data.cpu().numpy(), powers.cpu().numpy(), indices.cpu().numpy()
    if np.asarray(noises).shape != (len(images), N, 2):
        raise ValueError('Actual calibration noise population differs')
    observations, contexts, energies = [], [], []
    for i in range(len(images)):
        signal = transmit_frame(data_cpu[i], power_cpu[i], indices_cpu[i].tolist(), N)
        observation, context = observe_frame(signal, noises[i], snr, N)
        observations.append(observation)
        contexts.append(context)
        energies.append(float(np.square(signal).sum()))
    result = images.new_full(images.shape, .5)
    accepted = [i for i, context in enumerate(contexts) if context.accepted]
    if accepted:
        y = torch.as_tensor(np.stack([observations[i] for i in accepted]), device=images.device, dtype=torch.float32)
        power = y.new_tensor([contexts[i].power for i in accepted])
        mask = torch.tensor([contexts[i].indices for i in accepted], device=images.device, dtype=torch.long)
        result[accepted] = model.receive_data(y, snr, power, mask)
    return result, [dict(header_accepted=c.accepted, header_crc_accepted=c.crc_accepted,
        header_fields_legal=c.fields_legal, actual_energy=energy,
        fallback='none' if c.accepted else 'fixed_gray_0.5') for c, energy in zip(contexts, energies)]

