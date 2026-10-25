"""Independent, end-to-end state network. This interface never returns Watts."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.common import DataError, read_json, sha256
from .arrays import aggregate_arrays, windows


class StateClassifier(nn.Module):
    def __init__(self, width=31, features=False):
        super().__init__()
        self.enriched = features
        self.features = nn.Sequential(nn.Conv1d(3 if features else 1, 8, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * width, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.constant_(self.head[-1].bias, -3.)

    def forward(self, x):
        channels = x[:, None, :]
        if self.enriched:
            delta = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
            channels = torch.stack([x, delta, x - x.mean(dim=1, keepdim=True)], dim=1)
        return self.head(self.features(channels))[:, 0]


def predict_windows(metadata, model, x):
    torch.set_num_threads(metadata["cpu_threads"])
    z = torch.from_numpy(((np.log1p(x) - metadata["input_log_mean"]) / metadata["input_log_std"]).astype(np.float32))
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(b)).numpy() for b in z.split(metadata["batch_size"])]) if len(z) else np.empty(0, np.float32)


def load_state_model(path):
    path = Path(path)
    meta = read_json(path / "model.json")
    if (meta.get("kind") != "microwave_state_classifier_v1" or meta.get("truth_threshold_w") != 200
            or type(meta.get("window_samples")) is not int or meta["window_samples"] < 7 or meta["window_samples"] % 2 != 1
            or type(meta.get("features")) is not bool
            or not np.isfinite([meta["input_log_mean"], meta["input_log_std"]]).all() or meta["input_log_std"] <= 0
            or not 0 <= meta["decision_threshold_probability"] <= 1
            or sha256(path / "weights.pt") != meta["weights_sha256"]):
        raise DataError("Invalid independent state artifact")
    model = StateClassifier(meta["window_samples"], meta["features"])
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    return meta, model.eval()


def predict_state_rows(metadata, model, rows):
    stamps, values = aggregate_arrays(rows, 8)
    x, centers = windows(stamps, values, metadata["window_samples"], 8)
    probabilities = np.full(len(stamps), np.nan, np.float32)
    probabilities[centers] = predict_windows(metadata, model, x)
    return [{"timestamp": row["timestamp"], "probability_on": float(p) if np.isfinite(p) else None,
             "state": bool(p >= metadata["decision_threshold_probability"]) if np.isfinite(p) else None}
            for row, p in zip(rows, probabilities)]
