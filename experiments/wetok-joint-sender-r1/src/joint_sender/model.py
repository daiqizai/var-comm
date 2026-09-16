"""Existing receiver mathematics with only the communication sender update policy changed."""

from innovation_comm.model import InnovationSystem


VARIANTS = ('single_pass', 'multiscale_no_history', 'multiscale_state_history')


class JointSenderSystem(InnovationSystem):
    def __init__(self, parent, variant, fusion_config, update_sender=True):
        if variant not in VARIANTS:
            raise ValueError('this comparison only uses the three registered basic receivers')
        super().__init__(parent, variant, fusion_config)
        self.update_sender = bool(update_sender)
        self.encoder.requires_grad_(self.update_sender)
