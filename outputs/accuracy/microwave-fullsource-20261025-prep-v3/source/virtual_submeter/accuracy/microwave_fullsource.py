"""Full-source preparation and pure acceptance rules. Never opens target truth.

Execution adapters are deliberately not enabled until the coordinator fixes the
upstream recipe. Preparation, source selection and guards are independently usable.
"""
import csv
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from ..data.common import DataError, inventory, iso, read_json, sha256, utc, write_json
from ..models.arrays import aggregate_arrays, labels_at_centers, numeric, windows
from .microwave_diagnosis import metrics, normalization, save_csv, segments, split_mask
from .microwave_sampling import BatchStream, check_artifacts, selection, support, supported_block, validate_manifest
from .microwave_background import BackgroundStream, make_pools

ROOT = Path(__file__).resolve().parents[3]
ORDER = ["natural", "event", "event_background"]


def permit(house, role, channel):
    allowed = ((house in ("refit_2", "refit_19") and role == "train")
               or (house == "refit_5" and role == "validation"))
    if channel not in ("aggregate", "microwave") or not allowed:
        raise DataError("Preparation allows only House2/19 training and House5 validation")


def date_blocks(start, end, days=2):
    """Fixed UTC calendar blocks; omit partial edge dates without viewing scores."""
    first = ((start + 86399)//86400)*86400
    last = (end//86400)*86400
    return [(t, t+days*86400) for t in range(first, last-days*86400+1, days*86400)]


def source_data(part, base, role):
    house, raw, hashes = part["household_id"], {}, {}
    for channel in ("aggregate", "microwave"):
        permit(house, role, channel)
        item = part["artifacts"][channel]
        path = (base/item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise DataError("Source data hash mismatch")
        hashes[channel] = {"path": str(path), "sha256": item["sha256"]}
        with path.open(encoding="utf-8", newline="") as stream:
            raw[channel] = list(csv.DictReader(stream))
        if {r["household_id"] for r in raw[channel]} != {house}:
            raise DataError("Source household mismatch")
    t, a = aggregate_arrays(raw["aggregate"], 8)
    low, high = utc(part["start"]), utc(part["end"])
    mapping = {}
    for row in raw["microwave"]:
        stamp = utc(row["timestamp"])
        if stamp in mapping or stamp % 8 or not low <= stamp < high:
            raise DataError("Label timestamps duplicate, unaligned or outside partition")
        mapping[stamp] = numeric(row["power_w"])
    if np.any(np.diff(t) <= 0) or np.any(t % 8) or t[0] < low or t[-1] >= high:
        raise DataError("Aggregate timestamps invalid")
    full = np.array([mapping.get(int(s), np.nan) for s in t], np.float32)
    ids, events = segments(t, full)
    x, centers = windows(t, a, 31, 8)
    y = labels_at_centers(raw, ["microwave"], t, centers)[:, 0]
    np.testing.assert_array_equal(y, full[centers])
    valid = np.isfinite(y) & split_mask(t, centers, ids, events, low, high)
    data = {"x": x[valid], "y": y[valid], "t": t[centers[valid]],
            "raw_index": centers[valid], "event": ids[centers[valid]],
            "house": np.full(int(valid.sum()), 0 if house == "refit_2" else 1, np.int64)}
    return data, mapping, (low, high), events, hashes


def prepare(run, plan_path):
    if run.exists():
        raise DataError("Run exists; preparation never overwrites")
    p = read_json(plan_path)
    if p != read_json(ROOT/"configs/microwave_fullsource_20261025.json"):
        raise DataError("Plan differs from registered full-source protocol")
    for recipe_name in ("sampling", "background"):
        recipe = read_json(ROOT/f"configs/microwave_{recipe_name}_20261025.json")
        for key in ("seeds", "window_samples", "sample_seconds", "batch_size", "positive_per_batch",
                    "uniform_negative_per_batch", "checkpoints", "steps_per_checkpoint", "learning_rate", "pos_weight"):
            if p[key] != recipe[key]:
                raise DataError(f"Upstream {recipe_name} recipe differs: {key}")
    manifest_path = ROOT/p["audit_manifest"]
    manifest = read_json(manifest_path)
    parts, audit = validate_manifest(manifest, manifest_path.parent)
    protocol_path = ROOT/p["protocol"]
    protocol = read_json(protocol_path)
    validation = [v for v in protocol["partitions"]
                  if v["role"] == "source_validation" and v["household_id"] == "refit_5"]
    if len(validation) != 1:
        raise DataError("Exactly one original House5 validation partition required")
    run.mkdir(parents=True)
    shutil.copytree(ROOT/"src/virtual_submeter", run/"source/virtual_submeter",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path, name in ((plan_path, "plan.json"), (audit, "audit_receipt.json"),
                       (manifest_path, "audit_manifest.json"),
                       (protocol_path, "upstream_protocol.json"),
                       (ROOT/"docs/MICROWAVE_FULLSOURCE_PLAN_20261025.md", "protocol.md"),
                       (ROOT/"scripts/run_microwave_fullsource.py", "run_microwave_fullsource.py"),
                       (ROOT/"tests/test_microwave_fullsource.py", "test_microwave_fullsource.py"),
                       (ROOT/"configs/microwave_sampling_20261025.json", "sampling_recipe.json"),
                       (ROOT/"configs/microwave_background_20261025.json", "background_recipe.json")):
        shutil.copyfile(path, run/name)
    arrays, mappings, bounds, evidence, identities = [], {}, {}, {}, []
    for part in parts:
        house = part["household_id"]
        d, mappings[house], bounds[house], events, hashes = source_data(part, manifest_path.parent, "train")
        # The strict replay must match the original A/W50 source identity.
        original = [v for v in protocol["partitions"] if v["household_id"] == house and v["role"] == "source_train"]
        if len(original) != 1 or any(original[0]["artifacts"][c]["sha256"] != hashes[c]["sha256"] for c in hashes):
            raise DataError("Audited sources differ from original A/W50 training")
        arrays.append(d)
        evidence[house] = {**support(d, *bounds[house]), "inputs": hashes,
                           "label_complete_segments": sum(e["complete"] for e in events),
                           "complete_observed_segments": sum(e["complete"] and
                               int((d["event"] == e["event_id"]).sum()) == e["samples"] for e in events)}
        identities.extend({"house": house, "role": "train", "timestamp": iso(int(t)),
                           "window_id": f"{house}:{int(t)}:31"} for t in d["t"])
    train = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    np.savez_compressed(run/"train.npz", **train)
    write_json(run/"runtime.json", {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__})
    write_json(run/"normalization.json", normalization(train["x"]))
    pools, pool_stats = make_pools(train, mappings, bounds)
    np.savez_compressed(run/"pools.npz", **pools)
    write_json(run/"pool_support.json", pool_stats)
    val, _, vb, ve, vh = source_data(validation[0], protocol_path.parent, "validation")
    evidence["refit_5"] = {"inputs": vh, "blocks": {}}
    (run/"validation").mkdir()
    supported = []
    for i, (low, high) in enumerate(date_blocks(*vb, p["validation_block_complete_utc_days"])):
        mask = (val["t"]-120 >= low) & (val["t"]+128 <= high)
        for event in ve:
            if event["start"] < low or event["end"] > high:
                mask &= val["event"] != event["event_id"]
        d = {k: v[mask] for k, v in val.items()}
        key = f"block_{i}"
        np.savez_compressed(run/f"validation/{key}.npz", **d)
        ok = supported_block(d, p)
        evidence["refit_5"]["blocks"][key] = {**support(d, low, high), "start": iso(low), "end": iso(high), "supported": ok}
        complete = sum(e["complete"] and low <= e["start"] and e["end"] <= high
                       and int((d["event"] == e["event_id"]).sum()) == e["samples"] for e in ve)
        evidence["refit_5"]["blocks"][key].update(
            complete_observed_segments=complete,
            partially_supported_segments=evidence["refit_5"]["blocks"][key]["events"]-complete)
        if ok:
            supported.append(key)
        identities.extend({"house": "refit_5", "role": key, "timestamp": iso(int(t)),
                           "window_id": f"refit_5:{int(t)}:31"} for t in d["t"])
    write_json(run/"support.json", evidence)
    save_csv(run/"window_manifest.csv", identities)
    # Only existing predictions/models are hashed; no House17 label is opened,
    # even for hashing. Their original protocol hashes remain metadata only.
    reference = ROOT/p["reference_run"]
    baseline_hashes = {}
    for relative in ("reference/models", "reference/predictions", "models/W50", "predictions/W50"):
        baseline_hashes[relative] = inventory(reference/relative)
        if not baseline_hashes[relative]:
            raise DataError(f"Missing original A/W50 baseline evidence: {relative}")
    write_json(run/"baseline_identity.json", {"reference_run": str(reference), "artifacts": baseline_hashes,
               "development_partition": next(v for v in protocol["partitions"] if v["household_id"] == "refit_17" and v["role"] == "frozen_test"),
               "truth_access": "metadata_only_no_label_hashing_or_parsing"})
    # Full sampler and augmentation replay, with no network construction or fit.
    replay = {}
    for seed in p["seeds"]:
        left, right = BatchStream(train, "event", seed, p), BatchStream(train, "event", seed, p)
        aug = BackgroundStream(train, pools, seed, p)
        digest = hashlib.sha256()
        for step in range(1, p["checkpoints"]*p["steps_per_checkpoint"]+1):
            indices = left.next()
            np.testing.assert_array_equal(indices, right.next())
            output, _ = aug.apply(indices, step)
            digest.update(indices.tobytes())
            digest.update(output.tobytes())
        replay[str(seed)] = {**aug.summary(), "replay_sha256": digest.hexdigest()}
    write_json(run/"dry_replay.json", {"fits": 0, "seeds": replay})
    viable = bool(pool_stats["template_centers"] and pool_stats["background_centers"]
                  and all(r["synthesized"] > 0 for r in replay.values()))
    arms = ORDER if viable else ORDER[:2]
    matrix = {"arms": arms, "seeds": p["seeds"], "fits": len(arms)*3,
              "background_removed_before_training": not viable,
              "reason": None if viable else "no_usable_fullsource_background_composition",
              "supported_validation_blocks": supported,
              "preparation_passed": len(supported) >= p["minimum_supported_blocks"],
              "limited_source_evidence": len(supported) != len(evidence["refit_5"]["blocks"])}
    write_json(run/"matrix.json", matrix)
    write_json(run/"registration.json", {"artifacts": inventory(run), "actual_fits": 0,
               "house17_labels_opened": False, "final_house_data_accessed": False,
               "execution_adapter_status": "pending_upstream_recipe_lock_and_coordinator_execution",
               "matrix_sha256": sha256(run/"matrix.json")})
    return {"status": "prepared_not_trained", **matrix, "actual_fits": 0}


def select_checkpoint(scores, blocks, p):
    point = selection(scores, blocks, p)
    if point is None:
        return None
    point["limited_evidence"] = len(point["supported_blocks"]) != len(blocks)
    rows = [metrics(scores[k], blocks[k]["y"], point["threshold"]) for k in point["supported_blocks"]]
    point["mean_recall"] = float(np.mean([r["recall"] for r in rows]))
    point["block_metrics"] = dict(zip(point["supported_blocks"], rows))
    return point


def select_source(rows, p):
    """Input contains House5 selected checkpoint metrics only, one row per seed."""
    base = rows["natural"]
    if any([r["seed"] for r in value] != p["seeds"] for value in rows.values()):
        raise DataError("Source selection needs all seeds in fixed order")
    candidates, summary = [], []
    natural_valid = all(r["source_gate_passed"] and not r["degenerate"] for r in base)
    for arm in ORDER:
        if arm not in rows:
            continue
        value = rows[arm]
        eligible = natural_valid and all(r["source_gate_passed"] and not r["degenerate"] for r in value)
        if arm != "natural":
            eligible &= (np.mean([r["f1"] for r in value]) >= np.mean([r["f1"] for r in base])
                         and all(b["recall"]-a["recall"] >= -p["source_maximum_seed_recall_drop"] for a, b in zip(base, value))
                         and sum(b["precision"] > a["precision"] for a, b in zip(base, value)) >= 2)
        item = {"arm": arm, "eligible": bool(eligible),
                "mean_precision": float(np.mean([r["precision"] for r in value])),
                "mean_f1": float(np.mean([r["f1"] for r in value]))}
        summary.append(item)
        if eligible:
            candidates.append(item)
    chosen = max(candidates, key=lambda r: (r["mean_precision"], r["mean_f1"], -ORDER.index(r["arm"]))) if candidates else None
    return {"selected_arm": chosen["arm"] if chosen else None,
            "improvement_candidate": chosen is not None and chosen["arm"] != "natural",
            "no_deployable_candidate": chosen is None,
            "natural_source_gate_passed": natural_valid, "arms": summary, "source_only": True}


def aligned_unchanged(reference, candidate):
    for key in ("t", "kettle_power", "kettle_state", "washing_machine_power", "washing_machine_state", "microwave_power"):
        if key not in reference or key not in candidate or not np.array_equal(reference[key], candidate[key], equal_nan=True):
            raise DataError(f"Common support / unchanged raw A output mismatch: {key}")
    return True


def development_gate(base, candidate, p):
    if [r["seed"] for r in base] != p["seeds"] or [r["seed"] for r in candidate] != p["seeds"]:
        raise DataError("Development comparison requires complete paired seed matrix")
    return bool(all(r["recall"] >= p["minimum_recall"] and not r["degenerate"] for r in candidate)
                and np.mean([r["precision"] for r in candidate]) >= p["minimum_precision"]
                and np.mean([b["f1"]-a["f1"] for a, b in zip(base, candidate)]) + p["f1_gain_comparison_tolerance"] >= p["minimum_f1_gain"]
                and sum(b["precision"] > a["precision"] and b["f1"] > a["f1"] for a, b in zip(base, candidate)) >= 2
                and all(b["f1"]-a["f1"] >= -p["maximum_seed_f1_drop"] for a, b in zip(base, candidate))
                and sum(r["fp"] for r in candidate) <= p["maximum_fp_ratio"]*sum(r["fp"] for r in base))


def evaluation_guard(run):
    """Must run before any future label open, including hash reads of truth."""
    verify(run)
    matrix = read_json(run/"matrix.json")
    if not matrix["preparation_passed"]:
        raise DataError("Insufficient source validation support; evaluation forbidden")
    expected = {f"{a}/{s}" for a in matrix["arms"] for s in matrix["seeds"]}
    completed = read_json(run/"training_complete.json")
    frozen = read_json(run/"freeze.json")
    if set(completed["completed_models"]) != expected or set(frozen["predictions"]) != expected:
        raise DataError("Incomplete model/prediction matrix; evaluation forbidden")
    if frozen["registration_sha256"] != sha256(run/"registration.json") or frozen["training_sha256"] != sha256(run/"training_complete.json"):
        raise DataError("Freeze registration/training binding mismatch")
    if frozen["source_selection_sha256"] != sha256(run/"source_selection.json"):
        raise DataError("Source selection changed after freeze")
    check_artifacts(run, frozen["artifacts"])
    if not frozen.get("ordinary_inference_matches") or not frozen.get("common_support_and_other_devices_match"):
        raise DataError("Inference equivalence and unchanged A outputs required")
    return True


def verify(run):
    r = read_json(run/"registration.json")
    check_artifacts(run, r["artifacts"])
    for house in read_json(run/"support.json").values():
        for artifact in house["inputs"].values():
            if sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise DataError("Original allowed source data changed")
    baseline = read_json(run/"baseline_identity.json")
    for relative, hashes in baseline["artifacts"].items():
        check_artifacts(Path(baseline["reference_run"])/relative, hashes)
    return {"status": "preparation_hashes_verified", "actual_fits_at_registration": r["actual_fits"]}


def report(run):
    verify(run)
    return {"matrix": read_json(run/"matrix.json"), "support": read_json(run/"support.json"),
            "actual_fits": 0 if not (run/"training_complete.json").exists() else None,
            "claim": "preparation_only_no_prediction_improvement_claim"}
