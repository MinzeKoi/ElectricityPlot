"""Versioned model manifests; checkpoints are hash-checked and loaded weights-only."""

from pathlib import Path
import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256


def load_model(directory: Path):
    metadata = read_json(directory / "model.json")
    if metadata.get("schema_version") != 1 or metadata.get("status") != "trained":
        raise DataError("Model is missing, incompatible or not trained")
    appliances = metadata.get("appliances", [])
    if not appliances or len(set(appliances)) != len(appliances):
        raise DataError("Invalid model appliance list")
    if metadata.get("state_threshold_view", "power_w") not in ("power_w", "raw_power_w"):
        raise DataError("Unknown state threshold view")
    if metadata.get("state_adapter"):
        from ..adaptation.state import validate_head
        validate_head(metadata["state_adapter"], appliances)
    if metadata.get("accuracy_state_head"):
        from ..accuracy.learners import validate_adapter
        validate_adapter(metadata["accuracy_state_head"], metadata)
        if metadata.get("state_adapter"):
            raise DataError("A2 and historical state adapters cannot be combined")
        if sha256(directory / "queries.json") != metadata["queries_sha256"]:
            raise DataError("A2 query history hash mismatch")
    if metadata.get("adaptation") == "weak_window_state_only":
        if sha256(directory / "queries.json") != metadata["queries_sha256"]:
            raise DataError("Adaptation query history hash mismatch")
    if metadata["kind"] in ("constant_mean", "zero"):
        values = np.asarray(metadata["constant_power_w"], dtype=np.float32)
        if values.shape != (len(appliances),) or not np.isfinite(values).all() or (values < 0).any():
            raise DataError("Invalid constant model")
        return metadata, values
    if metadata["kind"] != "small_seq2point":
        raise DataError("Unknown model kind")
    import torch
    from .network import SmallSeq2Point
    checkpoint = safe_child(directory, metadata["weights"]["path"])
    if sha256(checkpoint) != metadata["weights"]["sha256"]:
        raise DataError("Model checkpoint hash mismatch")
    settings = metadata["settings"]
    model = SmallSeq2Point(settings["window_samples"], len(appliances), settings["channels"], settings["hidden_units"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return metadata, model
