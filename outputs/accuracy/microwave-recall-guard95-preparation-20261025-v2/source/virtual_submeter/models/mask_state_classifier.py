"""Opt-in C31 mask model. Ordinary inference retains strict complete support."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.common import DataError, read_json, sha256
from .arrays import aggregate_arrays, windows

KIND = "microwave_mask_state_classifier_v1"


class MaskStateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(2, 8, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8*31, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.constant_(self.head[-1].bias, -3.)

    def forward(self, channels):
        return self.head(self.features(channels))[:, 0]


def tensor_input(meta, x, mask=None):
    observed = np.isfinite(x)
    if x.ndim != 2 or x.shape[1] != 31 or np.isinf(x).any() or np.any(x[observed] < 0):
        raise DataError("Expected nonnegative C31 measurements with NaN missing values")
    if mask is not None and not np.array_equal(mask, observed):
        raise DataError("Observation mask does not match measured values")
    z = np.zeros(x.shape, np.float32)
    # Zero is only an internal standardized-space missing code, never zero Watts.
    z[observed] = (np.log1p(x[observed])-meta["input_log_mean"])/meta["input_log_std"]
    return torch.from_numpy(np.stack([z, observed.astype(np.float32)], axis=1))


def predict_windows(metadata, model, x):
    if not np.isfinite(x).all():
        raise DataError("Ordinary mask inference requires strict complete C31 support")
    torch.set_num_threads(metadata["cpu_threads"])
    z = tensor_input(metadata, x)
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(b)).numpy() for b in z.split(metadata["batch_size"])]) if len(z) else np.empty(0, np.float32)


def load_state_model(path):
    path = Path(path)
    meta = read_json(path/"model.json")
    if (meta.get("kind") != KIND or meta.get("truth_threshold_w") != 200 or meta.get("window_samples") != 31
            or meta.get("inference_support") != "strict_complete_31"
            or not np.isfinite([meta["input_log_mean"], meta["input_log_std"]]).all()
            or meta["input_log_std"] <= 0 or not 0 <= meta["decision_threshold_probability"] <= 1
            or sha256(path/"weights.pt") != meta["weights_sha256"]):
        raise DataError("Invalid mask state model artifact")
    model = MaskStateClassifier()
    model.load_state_dict(torch.load(path/"weights.pt", map_location="cpu", weights_only=True))
    return meta, model.eval()


def predict_state_rows(metadata, model, rows):
    t, a = aggregate_arrays(rows, 8)
    x, c = windows(t, a, 31, 8)
    scores = np.full(len(t), np.nan, np.float32)
    scores[c] = predict_windows(metadata, model, x)
    return [{"timestamp": r["timestamp"], "probability_on": float(s) if np.isfinite(s) else None,
             "state": bool(s >= metadata["decision_threshold_probability"]) if np.isfinite(s) else None}
            for r, s in zip(rows, scores)]
