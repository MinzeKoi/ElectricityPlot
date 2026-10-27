"""Opt-in offline center TCN; future observations are part of its context."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..data.common import DataError, read_json, sha256
from .arrays import aggregate_arrays, windows

KIND = "microwave_center_mask_tcn_v1"


class CenterMaskTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(2, 16, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(16, 16, 3, padding=d, dilation=d)
                                     for d in [1, 2, 4, 8, 16, 32]])
        self.head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.constant_(self.head[-1].bias, -3.)

    def forward(self, channels):
        mask = channels[:, 1:2]
        h = self.project(channels) * mask
        for conv in self.blocks:
            h = (h + torch.relu(conv(h))) * (2 ** -.5) * mask
        return self.head(h[:, :, h.shape[2]//2])[:, 0]


def tensor_input(meta, x, mask=None):
    observed = np.isfinite(x)
    if (x.ndim != 2 or x.shape[1] not in (31, 127) or np.isinf(x).any()
            or np.any(x[observed] < 0)):
        raise DataError("Expected nonnegative C31/C127 with NaN missing values")
    if mask is not None and not np.array_equal(mask, observed):
        raise DataError("NaN/observation mask mismatch")
    z = np.zeros(x.shape, np.float32)
    z[observed] = (np.log1p(x[observed])-meta["input_log_mean"])/meta["input_log_std"]
    return torch.from_numpy(np.stack([z, observed.astype(np.float32)], axis=1))


def expand_context(t, a, centers, low, high, length=127):
    """Walk actual rows, stop at a time break or second consecutive missing grid."""
    centers = np.asarray(centers, np.int64)
    half = length//2
    x = np.full((len(centers), length), np.nan, np.float32)
    x[:, half] = a[centers]
    reasons = []
    for direction in (-1, 1):
        alive = np.ones(len(centers), bool)
        missing = np.zeros(len(centers), np.int8)
        reason = np.full(len(centers), "radius", dtype="U16")
        for distance in range(1, half+1):
            ix = centers + direction*distance
            safe = np.clip(ix, 0, len(t)-1)
            prev = np.clip(ix-direction, 0, len(t)-1)
            boundary = (ix < 0) | (ix >= len(t)) | (t[safe] < low) | (t[safe] >= high)
            broken = (t[safe]-t[prev] != direction*8) & ~boundary
            stop = alive & (boundary | broken)
            reason[stop & boundary] = "boundary"
            reason[stop & broken] = "time_break"
            alive &= ~stop
            observed = np.isfinite(a[safe])
            missing = np.where(observed, 0, missing+1)
            stop = alive & (missing >= 2)
            reason[stop] = "missing_run"
            alive &= ~stop
            take = alive & observed
            x[take, half+direction*distance] = a[safe[take]]
        reasons.append(reason)
    mask = np.isfinite(x)
    outer = mask.sum(axis=1)-mask[:, half-15:half+16].sum(axis=1)
    return {"x": x, "mask": mask, "outer_observed": outer,
            "outer_fraction": outer.astype(np.float32)/max(length-31, 1),
            "left_stop": reasons[0], "right_stop": reasons[1],
            "mask_identity": np.packbits(mask, axis=1)}


def predict_windows(metadata, model, x):
    if x.shape[1] != metadata["window_samples"]:
        raise DataError("Context length does not match model")
    half = x.shape[1]//2
    if not np.isfinite(x[:, half-15:half+16]).all():
        raise DataError("Ordinary inference requires complete core31")
    torch.set_num_threads(metadata["cpu_threads"])
    z = tensor_input(metadata, x)
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(b)).numpy() for b in z.split(metadata["batch_size"])]) if len(z) else np.empty(0, np.float32)


def load_state_model(path):
    path = Path(path)
    meta = read_json(path/"model.json")
    if (meta.get("kind") != KIND or meta.get("truth_threshold_w") != 200
            or meta.get("window_samples") not in (31,127)
            or meta.get("inference_support") != "strict_complete_31"
            or not np.isfinite([meta["input_log_mean"], meta["input_log_std"]]).all()
            or meta["input_log_std"] <= 0 or not 0 <= meta["decision_threshold_probability"] <= 1
            or sha256(path/"weights.pt") != meta["weights_sha256"]):
        raise DataError("Invalid center TCN artifact")
    model = CenterMaskTCN()
    model.load_state_dict(torch.load(path/"weights.pt", map_location="cpu", weights_only=True))
    return meta, model.eval()


def context_at_complete_centers(metadata, t, a):
    core, centers = windows(t, a, 31, 8)
    if metadata["window_samples"] == 31:
        return core, centers
    d = expand_context(t, a, centers, int(t[0]), int(t[-1])+8)
    np.testing.assert_array_equal(d["x"][:,48:79], core)
    return d["x"], centers


def predict_state_rows(metadata, model, rows):
    t, a = aggregate_arrays(rows, 8)
    x, c = context_at_complete_centers(metadata, t, a)
    scores = np.full(len(t), np.nan, np.float32)
    scores[c] = predict_windows(metadata, model, x)
    return [{"timestamp": r["timestamp"], "probability_on": float(s) if np.isfinite(s) else None,
             "state": bool(s >= metadata["decision_threshold_probability"]) if np.isfinite(s) else None}
            for r, s in zip(rows, scores)]
