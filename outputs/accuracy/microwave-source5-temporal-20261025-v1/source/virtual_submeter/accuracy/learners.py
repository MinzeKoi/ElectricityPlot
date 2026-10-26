"""A2 frozen temporal features and window-supervised, source-anchored heads.

Fitting receives visible feature arrays and budgeted weak records, never paths
to target labels. Positive bags mean at least one activation, not all points.
"""

import copy
import math

import numpy as np

from ..adaptation.state import features as legacy_features, fit_head
from ..data.common import DataError, utc
from ..models.arrays import aggregate_arrays, windows

VARIANTS = ("L0", "L1", "L2_max_unprotected", "L2_max_bound", "L2_max_replay", "L2_max", "L2_smooth", "L1_stable")


def temporal_features(metadata, model, rows):
    """Return hidden representations at complete centers, retaining null gaps."""
    import torch
    if metadata["kind"] != "small_seq2point":
        raise DataError("Temporal state features require the frozen A1 CNN")
    settings = metadata["settings"]
    torch.set_num_threads(settings["cpu_threads"])
    stamps, aggregate = aggregate_arrays(rows, metadata["sample_interval_seconds"])
    x, centers = windows(stamps, aggregate, settings["window_samples"], metadata["sample_interval_seconds"])
    hidden = np.full((len(rows), settings["hidden_units"]), np.nan, dtype=np.float64)
    if len(centers):
        x = (np.log1p(x) - metadata["input_log_mean"]) / metadata["input_log_std"]
        with torch.inference_mode():
            values = [model.head[:-1](model.features(batch[:, None, :])).numpy()
                      for batch in torch.from_numpy(x).split(settings["batch_size"])]
        hidden[centers] = np.concatenate(values)
    return hidden


def normalization(hidden):
    complete = hidden[np.isfinite(hidden).all(axis=1)]
    if not len(complete):
        raise DataError("No complete source temporal features")
    return {"mean": complete.mean(axis=0).tolist(), "std": np.maximum(complete.std(axis=0), 0.1).tolist()}


def design(rows, appliance, threshold, hidden=None, normalizer=None):
    raw = legacy_features(rows, appliance, threshold)
    # L1 is an offset to the original source logit: the slope stays exactly 1.
    if hidden is None:
        return raw[:, :2]
    standardized = np.clip((hidden - normalizer["mean"]) / normalizer["std"], -8, 8)
    return np.column_stack((raw[:, :2], standardized))


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def pool(logits, method, temperature):
    """Stable bag logit and derivative; smooth uses normalized log-mean-exp."""
    if method == "max":
        index = int(np.argmax(logits))
        weights = np.zeros(len(logits))
        weights[index] = 1
        return float(logits[index]), weights
    if method != "smooth" or not math.isfinite(temperature) or temperature <= 0:
        raise DataError("Invalid MIL pooling")
    maximum = float(np.max(logits))
    exponent = np.exp((logits - maximum) / temperature)
    return maximum + temperature * float(np.log(exponent.mean())), exponent / exponent.sum()


def source_fit(train_x, train_y, validation_x, validation_y, settings, seed):
    """Train a tiny pointwise head; epoch selection sees source validation only."""
    import torch
    rng = np.random.default_rng(seed)
    valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices) or len(np.unique(train_y[indices])) != 2:
        raise DataError("Source state training needs both observed classes")
    if len(indices) > settings["source_max_samples"]:
        indices = rng.choice(indices, settings["source_max_samples"], replace=False)
    x = torch.tensor(train_x[indices], dtype=torch.float64)
    y = torch.tensor(train_y[indices], dtype=torch.float64)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        weight[1] = 1
    optimizer = torch.optim.Adam([weight], lr=settings["source_learning_rate"])
    positive_weight = min(settings["source_positive_weight_cap"], float((1 - y).sum() / y.sum().clamp_min(1)))
    vmask = np.isfinite(validation_y) & np.isfinite(validation_x).all(axis=1)
    if not vmask.any() or len(np.unique(validation_y[vmask])) != 2:
        raise DataError("Source validation needs observed active and inactive states")
    vx, vy = validation_x[vmask], validation_y[vmask].astype(bool)
    history, best, selected, chosen = [], -1, None, None
    for epoch in range(1, settings["source_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(positive_weight))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == settings["source_epochs"]:
            values = weight.detach().numpy().copy()
            state = vx @ values >= 0
            tp = int((state & vy).sum())
            denominator = int(state.sum() + vy.sum())
            f1 = 2 * tp / denominator if denominator else 0
            history.append({"epoch": epoch, "loss": float(loss.detach()), "source_validation_f1": f1})
            if f1 > best:
                best, selected, chosen = f1, values, epoch
    if selected is None or not np.isfinite(selected).all():
        raise DataError("Source head did not produce finite parameters")
    # Equal-class replay is a fixed source-only regularizer, not user labels.
    replay = []
    for label in (0, 1):
        choices = np.flatnonzero(valid & (train_y == label))
        replay.extend(rng.choice(choices, min(len(choices), settings["replay_samples"] // 2), replace=False).tolist())
    replay = np.asarray(replay, dtype=int)
    return selected, (train_x[replay], train_y[replay]), {
        "selected_epoch": chosen, "source_validation_f1": best, "sample_count": len(indices),
        "positive_weight": positive_weight, "replay_samples": len(replay), "history": history,
        "target_labels_read": False, "replay_semantics": "source-only equal-class sample; identical across variants and budgets"}


def bags_for(rows, records, appliance, x):
    stamps = np.asarray([utc(r["timestamp"]) for r in rows])
    bags = []
    for record in records:
        if record["appliance_type"] != appliance or record["response"] not in ("used", "not_used"):
            continue
        indices = np.flatnonzero((stamps >= utc(record["window_start"])) & (stamps < utc(record["window_end"])))
        if not len(indices) or not np.isfinite(x[indices]).all():
            raise DataError("Confirmed window lacks complete model features")
        bags.append((x[indices], float(record["response"] == "used")))
    return bags


def fit_parameters(prior, bags, replay, settings, variant):
    """Fresh source initialization for every prefix; no future-based fallback."""
    weight = np.array(prior, dtype=float, copy=True)
    if not bags:
        return weight, {"usable_windows": 0, "update_rule": "exact_source_identity", "loss": []}
    protected = variant != "L2_max_unprotected"
    single_class = len({label for _, label in bags}) == 1
    bias_only = variant in ("L1", "L1_stable") or (protected and single_class)
    pooling = "smooth" if variant == "L2_smooth" else "max"
    use_replay = variant in ("L2_max_replay", "L2_max", "L2_smooth")
    use_prior = variant in ("L2_max", "L2_smooth")
    losses = []
    for _ in range(settings["target_epochs"]):
        gradient = np.zeros_like(weight)
        loss = 0.0
        for bag, label in bags:
            # A negative answer constrains the highest activation even for the
            # smooth positive-bag candidate: a rare peak cannot be averaged away.
            logit, derivative = pool(bag @ weight, pooling if label else "max", settings["smooth_temperature"])
            probability = float(sigmoid(logit))
            gradient += (probability - label) * (derivative @ bag) / len(bags)
            loss += float(np.logaddexp(0, logit) - label * logit) / len(bags)
        if use_replay:
            rx, ry = replay
            rlogits = rx @ weight
            gradient += settings["replay_weight"] * (rx.T @ (sigmoid(rlogits) - ry)) / len(ry)
            loss += settings["replay_weight"] * float(np.mean(np.logaddexp(0, rlogits) - ry * rlogits))
        if use_prior:
            gradient += settings["prior_l2"] * (weight - prior)
            loss += settings["prior_l2"] * float(np.sum((weight - prior) ** 2)) / 2
        if bias_only:
            gradient[1:] = 0
        weight -= settings["target_learning_rate"] * gradient
        if protected:
            delta = weight - prior
            if bias_only:
                # Reuse the stricter existing bound for every answer mixture
                # in the R2 repair; preserve historical L1 behavior.
                bound = settings["single_class_bias_bound"] if single_class or variant == "L1_stable" else settings["bias_bound"]
                delta[0] = np.clip(delta[0], -bound, bound)
            else:
                delta *= min(1.0, settings["parameter_radius"] / max(float(np.linalg.norm(delta)), 1e-12))
            weight = prior + delta
        if not np.isfinite(weight).all():
            raise DataError("Non-finite A2 state parameters")
        losses.append(loss)
    return weight, {"usable_windows": len(bags), "class_counts": {"used": sum(y == 1 for _, y in bags), "not_used": sum(y == 0 for _, y in bags)},
                    "update_rule": "bounded_bias_only" if bias_only else "bounded_head" if protected else "unprotected_head",
                    "pooling": pooling, "source_replay": use_replay, "source_prior_l2": use_prior,
                    "parameter_delta_l2": float(np.linalg.norm(weight - prior)), "loss": losses}


def fit_adapter(rows, records, metadata, settings, variant, source_head=None, hidden=None, replay=None, legacy_settings=None):
    if variant not in VARIANTS:
        raise DataError("Unknown A2 learner")
    appliances = metadata["appliances"]
    thresholds = metadata["settings"]["on_thresholds_w"]
    if variant == "L0":
        head, history, counts = fit_head(rows, records, appliances, thresholds, legacy_settings)
        return {"schema_version": 1, "variant": variant, "coefficients": head["coefficients"],
                "active_appliances": head["active_appliances"], "normalizer": None}, {"loss": history, "usable_windows": counts}
    normalizer = source_head["normalizer"] if variant.startswith("L2") else None
    coefficients, active, history = {}, [], {}
    for appliance in appliances:
        x = design(rows, appliance, thresholds[appliance], hidden if normalizer else None, normalizer)
        prior = np.asarray(source_head["coefficients"][appliance] if normalizer else [0.0, 1.0])
        bags = bags_for(rows, records, appliance, x)
        weight, history[appliance] = fit_parameters(prior, bags, replay[appliance] if replay else None, settings, variant)
        coefficients[appliance] = weight.tolist()
        # L2 always uses its source state output, including zero-query devices.
        if bags or normalizer:
            active.append(appliance)
    return {"schema_version": 1, "variant": variant, "coefficients": coefficients,
            "active_appliances": active, "normalizer": normalizer}, history


def validate_adapter(head, metadata):
    if head.get("schema_version") != 1 or head.get("variant") not in VARIANTS:
        raise DataError("Invalid A2 head schema")
    appliances = metadata["appliances"]
    temporal = head["variant"].startswith("L2")
    dimension = metadata["settings"]["hidden_units"] + 2 if temporal else 3 if head["variant"] == "L0" else 2
    if set(head.get("coefficients", {})) != set(appliances):
        raise DataError("A2 head appliance mismatch")
    active = head.get("active_appliances")
    if not isinstance(active, list) or len(active) != len(set(active)) or not set(active) <= set(appliances):
        raise DataError("Invalid active A2 appliances")
    for values in head["coefficients"].values():
        array = np.asarray(values, dtype=float)
        if array.shape != (dimension,) or not np.isfinite(array).all():
            raise DataError("Invalid A2 head coefficients")
    if temporal:
        norm = head.get("normalizer") or {}
        for key in ("mean", "std"):
            array = np.asarray(norm.get(key), dtype=float)
            if array.shape != (dimension - 2,) or not np.isfinite(array).all() or (key == "std" and (array <= 0).any()):
                raise DataError("Invalid A2 source feature normalization")


def score_adapter(rows, metadata, head, hidden=None):
    validate_adapter(head, metadata)
    result = np.full((len(rows), len(metadata["appliances"])), np.nan)
    for column, appliance in enumerate(metadata["appliances"]):
        if appliance not in head["active_appliances"]:
            result[:, column] = [float(r["appliances"][appliance]["raw_state"]) if r["appliances"][appliance]["raw_state"] is not None else np.nan for r in rows]
            continue
        threshold = metadata["settings"]["on_thresholds_w"][appliance]
        x = legacy_features(rows, appliance, threshold) if head["variant"] == "L0" else design(
            rows, appliance, threshold, hidden if head["normalizer"] else None, head["normalizer"])
        if x.shape[1] != len(head["coefficients"][appliance]):
            raise DataError("Temporal A2 inference requires frozen hidden features")
        result[:, column] = sigmoid(x @ np.asarray(head["coefficients"][appliance]))
    return result


def apply_scores(rows, appliances, scores, active=None):
    if scores.shape != (len(rows), len(appliances)):
        raise DataError("A2 scores do not match prediction support")
    for index, row in enumerate(rows):
        for column, appliance in enumerate(appliances):
            if active is not None and appliance not in active:
                continue
            item, value = row["appliances"][appliance], scores[index, column]
            if np.isfinite(value) != (item["raw_power_w"] is not None):
                raise DataError("A2 cannot change the valid prediction support")
            item["state_score"] = float(value) if np.isfinite(value) else None
            item["state"] = item["raw_state"] = bool(value >= 0.5) if np.isfinite(value) else None


def apply_adapter(rows, metadata, hidden=None):
    head = metadata["accuracy_state_head"]
    values = score_adapter(rows, metadata, head, hidden)
    apply_scores(rows, metadata["appliances"], values, head["active_appliances"])


def composed_rows(base_rows, appliances, values):
    rows = copy.deepcopy(base_rows)
    apply_scores(rows, appliances, values)
    return rows
