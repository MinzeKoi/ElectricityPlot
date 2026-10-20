"""Source-only fitting and validation-only checkpoint selection for M2."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from ..data.common import DataError, read_json, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition
from ..environment import environment_report
from .arrays import aggregate_arrays, labels_at_centers, windows
from .settings import load_settings


def dataset(manifest_path, partitions, appliances, settings, interval):
    xs, ys = [], []
    counts = {}
    for partition in partitions:
        # This consumer refuses target adaptation and test labels, even if a
        # caller supplies an inappropriate partition list.
        data = read_partition(manifest_path, partition["partition_id"], "training", include_labels=True)
        stamps, aggregate = aggregate_arrays(data["aggregate"], interval)
        x, centers = windows(stamps, aggregate, settings["window_samples"], interval)
        y = labels_at_centers(data, appliances, stamps, centers)
        usable = np.isfinite(y).any(axis=1)
        xs.append(x[usable])
        ys.append(y[usable])
        counts[partition["partition_id"]] = int(usable.sum())
    if not xs or sum(len(x) for x in xs) == 0:
        raise DataError("No complete labeled source windows; do not fill gaps to force training")
    x, y = np.concatenate(xs), np.concatenate(ys)
    if np.any(np.isfinite(y).sum(axis=0) == 0):
        raise DataError("At least one source appliance has no valid labels")
    return x, y, counts


def train(manifest_path: Path, settings_path: Path, output: Path, kind="small_seq2point", appliances=None) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise DataError("Training destination exists; choose a new run directory")
    if kind not in ("small_seq2point", "constant_mean", "zero"):
        raise DataError("Unsupported baseline kind")
    settings = load_settings(settings_path)
    protocol = read_json(manifest_path)
    appliances = list(protocol["target_appliances"] if appliances is None else appliances)
    if not appliances or len(set(appliances)) != len(appliances) or any(a not in protocol["target_appliances"] or a not in settings["on_thresholds_w"] for a in appliances):
        raise DataError("Appliances must be unique frozen targets with state thresholds")
    interval = protocol["sample_interval_seconds"]
    if settings["window_samples"] * interval > min(protocol["max_model_context_seconds"], protocol["split_gap_seconds"]):
        raise DataError("Model context exceeds frozen protocol isolation; create a new protocol")
    training = [p for p in protocol["partitions"] if p["role"] == "source_train"]
    validation = [p for p in protocol["partitions"] if p["role"] == "source_validation"]
    target_ids = {p["household_id"] for p in protocol["partitions"] if p["role"] in ("target_adaptation", "frozen_test")}
    train_ids = {p["household_id"] for p in training}
    valid_ids = {p["household_id"] for p in validation}
    if not train_ids or not valid_ids or train_ids & valid_ids or (train_ids | valid_ids) & target_ids:
        raise DataError("Training/validation/target household isolation is invalid")
    x, y, train_counts = dataset(manifest_path, training, appliances, settings, interval)
    vx, vy, valid_counts = dataset(manifest_path, validation, appliances, settings, interval)
    logged = np.log1p(x)
    mean, std = float(logged.mean()), max(float(logged.std()), 1e-6)
    thresholds = np.array([settings["on_thresholds_w"][a] for a in appliances], np.float32)
    scales = np.array([max(float(np.nanmax(y[:, i])), thresholds[i], 1) for i in range(len(appliances))], np.float32)
    metadata = {
        "schema_version": 1, "status": "trained", "project": "ElectricityPlot", "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(), "appliances": appliances,
        "settings": settings, "sample_interval_seconds": interval,
        "protocol_sha256": sha256(manifest_path), "protocol_id": protocol["protocol_id"], "purpose": protocol["purpose"],
        "code_sha256": source_fingerprint(), "settings_sha256": sha256(settings_path),
        "input_log_mean": mean, "input_log_std": std, "power_scales_w": scales.tolist(),
        "source_train_windows": train_counts, "source_validation_windows": valid_counts,
        "training_households": sorted(train_ids), "validation_households": sorted(valid_ids),
        "target_labels_read": False, "adaptation": "none", "confidence": None,
        "upstream_limitations": protocol["upstream_limitations"],
        "state_definition": "predicted_power_w >= fixed on_thresholds_w; thresholds chosen before test evaluation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".train-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        history = []
        fit_started = time.perf_counter()
        if kind in ("constant_mean", "zero"):
            constants = np.nanmean(y, axis=0) if kind == "constant_mean" else np.zeros(len(appliances), np.float32)
            metadata["constant_power_w"] = constants.tolist()
            metadata["selected_epoch"] = None
            metadata["validation_normalized_mae"] = float(np.nanmean(np.abs(vy - constants) / scales))
        else:
            import torch
            from .network import SmallSeq2Point
            torch.set_num_threads(settings["cpu_threads"])
            torch.manual_seed(settings["seed"])
            torch.use_deterministic_algorithms(True)
            model = SmallSeq2Point(settings["window_samples"], len(appliances), settings["channels"], settings["hidden_units"])
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
            tx = torch.from_numpy((logged - mean) / std)
            ty = torch.from_numpy(np.nan_to_num(y / scales, nan=0.0))
            mask = torch.from_numpy(np.isfinite(y).astype(np.float32))
            weights = torch.from_numpy(np.where(y >= thresholds, settings["active_loss_weight"], 1).astype(np.float32))
            val_x = torch.from_numpy((np.log1p(vx) - mean) / std)
            generator = torch.Generator().manual_seed(settings["seed"])
            best = float("inf")
            best_state = None
            for epoch in range(1, settings["epochs"] + 1):
                model.train()
                permutation = torch.randperm(len(tx), generator=generator)
                running_loss = 0.0
                for indices in permutation.split(settings["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(tx[indices])
                    weighted_mask = mask[indices] * weights[indices]
                    loss = (((prediction - ty[indices]) ** 2) * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.detach()) * len(indices)
                model.eval()
                with torch.inference_mode():
                    val_prediction = np.concatenate([model(batch).numpy() for batch in val_x.split(settings["batch_size"])]) * scales
                score = float(np.nanmean(np.abs(val_prediction - vy) / scales))
                if not np.isfinite(score):
                    raise DataError("Training produced a non-finite validation metric")
                history.append({"epoch": epoch, "training_weighted_mse": running_loss / len(tx), "validation_normalized_mae": score})
                print(f"epoch {epoch}/{settings['epochs']}: source validation normalized MAE={score:.6f}", file=sys.stderr, flush=True)
                if score < best:
                    best = score
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    metadata["selected_epoch"] = epoch
            torch.save(best_state, staging / "weights.pt")
            metadata["weights"] = {"path": "weights.pt", "sha256": sha256(staging / "weights.pt")}
            metadata["validation_normalized_mae"] = best
            metadata["torch_version"] = torch.__version__
            metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        metadata["timing_seconds"] = {"fit": time.perf_counter() - fit_started, "total": time.perf_counter() - started}
        identity = {key: metadata[key] for key in ("kind", "settings", "protocol_sha256", "code_sha256", "appliances")}
        identity["learned"] = metadata.get("weights", metadata.get("constant_power_w"))
        metadata["model_id"] = kind + "-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        write_json(staging / "model.json", metadata)
        write_json(staging / "settings.json", settings)
        write_json(staging / "history.json", {"epochs": history})
        environment = environment_report()
        environment.update({"numpy_version": np.__version__, "training_environment": "M2 CPU local environment", "second_developer_machine": "not_measured"})
        write_json(staging / "environment.json", environment)
        (staging / "protocol.json").write_bytes(manifest_path.read_bytes())
        write_json(staging / "queries.json", {"records": [], "budget_used": 0})
        staging.rename(output)
    return {"status": "baseline_trained", "model_id": metadata["model_id"], "model_dir": str(output),
            "purpose": protocol["purpose"], "validation_normalized_mae": metadata["validation_normalized_mae"]}
