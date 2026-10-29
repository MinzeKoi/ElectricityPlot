"""Task26 candidate representation; not a training or inference entry point."""
import numpy as np
import torch
from torch import nn
from virtual_submeter.models.center_mask_tcn import CenterMaskTCN, tensor_input


class CenteredTCN(CenterMaskTCN):
    def __init__(self, original):
        with torch.random.fork_rng(devices=[]):
            super().__init__(16)
            self.load_state_dict(original.state_dict())
            old = self.project
            self.project = nn.Conv1d(3, 16, 1)
            with torch.no_grad():
                self.project.weight[:, :2].copy_(old.weight)
                self.project.weight[:, 2:].zero_()
                self.project.bias.copy_(old.bias)


def centered_input(meta, x, mask=None):
    base = tensor_input(meta, x, mask)
    if x.shape[1] != 127:
        raise ValueError('Task26 requires C127')
    observed = np.isfinite(x)
    core = observed[:, 48:79]
    if np.any(core.sum(axis=1) < 24):
        raise ValueError('Task26 requires at least 24 observed core points')
    mean = np.where(core, x[:, 48:79], 0).sum(axis=1) / core.sum(axis=1)
    shape = np.where(observed, (x - mean[:, None]) / 1000.0, 0).astype(np.float32)
    return torch.cat([base, torch.from_numpy(shape[:, None, :])], dim=1)
