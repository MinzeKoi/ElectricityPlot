"""Task26 training core; no command-line entry or authorization implied."""
import numpy as np
import torch
from virtual_submeter.models.center_mask_tcn import CenterMaskTCN
from task26_centered.model import CenteredTCN, centered_input


def fit_seed(seed, initial, train, blocks, indices, meta, plan, on_checkpoint):
    """Fit supplied data; caller must enforce authorization, identity and isolation."""
    torch.set_num_threads(plan['cpu_threads'])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    original = CenterMaskTCN()
    for name, value in original.state_dict().items():
        if not torch.equal(value, initial[name]):
            raise ValueError('control initialization differs')
    model = CenteredTCN(original)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
    x = centered_input(meta, train['x'], train['mask'])
    y = torch.from_numpy((train['y'] >= 200).astype(np.float32))
    if any(not block['mask'][:, 48:79].all() for block in blocks.values()):
        raise ValueError('validation requires complete core31')
    validation = {key: centered_input(meta, block['x'], block['mask']) for key, block in blocks.items()}
    checkpoints = []
    for epoch in range(plan['checkpoints']):
        model.train()
        for step in range(plan['steps_per_checkpoint']):
            idx = indices[epoch * plan['steps_per_checkpoint'] + step]
            if len(idx) != 256 or int((y[idx] == 1).sum()) != 16:
                raise ValueError('batch quota changed')
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('nonfinite training state')
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = {key: torch.cat([torch.sigmoid(model(batch)) for batch in value.split(plan['batch_size'])]).numpy()
                      for key, value in validation.items()}
        checkpoints.append(scores)
        on_checkpoint(epoch + 1, model, scores)
    return model, checkpoints
