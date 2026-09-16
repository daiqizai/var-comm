"""Prepared geometry alternative; it never changes the running frozen interface study."""

import copy

import torch
from torch import nn

from .interface_study import InterfaceJSCC
from .model import CommunicationEncoder, CommunicationReceiver


class RowCompatibleEncoder(CommunicationEncoder):
    def __init__(self, config):
        if config['channel_positions'] * config['channel_features'] != 6120:
            raise ValueError('geometry candidate must retain exactly 3060 complex uses')
        if not 0 < config['channel_positions'] <= 256 or config['channel_features'] <= 0:
            raise ValueError('unsupported source-to-channel geometry')
        if config['channel_features'] >= 32:
            super().__init__(config)
            return
        scaffold = {**config, 'channel_features': 32}
        super().__init__(scaffold)
        orthogonal = self.channel_lift.weight.detach().clone()
        self.features = config['channel_features']
        self.channel_lift = nn.Linear(32, self.features, bias=False)
        self.channel_lift.weight.data.copy_(orthogonal[:self.features])
        self.output = nn.Linear(config['width'], self.features)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class GeometryJSCC(InterfaceJSCC):
    def __init__(self, variant, config):
        nn.Module.__init__(self)
        if variant not in ('single_pass', 'multiscale_no_history', 'multiscale_conditioned'):
            raise ValueError('unknown matched receiver structure')
        self.variant, self.interface = variant, 'continuous_mean'
        self.encoder = RowCompatibleEncoder(config)
        self.receiver = CommunicationReceiver(self.encoder, config)


def matched_geometry_initializations(base, variant):
    control_config = copy.deepcopy(base['model'])
    if (control_config['channel_positions'], control_config['channel_features']) != (153, 40):
        raise ValueError('the frozen source geometry is no longer 153 by 40')
    candidate_config = {**control_config, 'channel_positions': 204, 'channel_features': 30}
    torch.manual_seed(base['training']['initialization_seed'])
    control = GeometryJSCC(variant, control_config)
    torch.manual_seed(base['training']['initialization_seed'])
    candidate = GeometryJSCC(variant, candidate_config)
    shared, geometry_specific = [], []
    control_state, candidate_state = control.state_dict(), candidate.state_dict()
    with torch.no_grad():
        for name, value in candidate_state.items():
            previous = control_state[name]
            if value.shape == previous.shape:
                value.copy_(previous)
                shared.append(name)
            else:
                geometry_specific.append(name)
    return control, candidate, {'shared_initial_tensors': shared, 'geometry_specific_tensors': geometry_specific,
        'control_parameters': sum(value.numel() for value in control.parameters()),
        'candidate_parameters': sum(value.numel() for value in candidate.parameters()),
        'control_geometry': [153, 40], 'candidate_geometry': [204, 30]}
