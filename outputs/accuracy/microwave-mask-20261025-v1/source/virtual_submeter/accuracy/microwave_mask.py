"""Independent preregistered missing-context experiment; preparation performs zero fits."""
import csv
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from ..data.common import DataError, inventory, iso, read_json, sha256, utc, write_json
from ..models.arrays import aggregate_arrays, numeric
from .microwave_diagnosis import load_npz, normalization, segments, split_mask
from .microwave_sampling import BatchStream, check_artifacts, support, validate_manifest
from .microwave_fullsource import select_checkpoint, aligned_unchanged, development_gate

ROOT = Path(__file__).resolve().parents[3]
ORDER = ["strict_mask", "partial_mask"]


def candidate_data(t, a, y, low, high, house):
    """Do not create grid rows, interpolate Watts, or reinterpret unobserved labels."""
    if house not in ("refit_2", "refit_19"):
        raise DataError("Only original House2/19 training is permitted")
    if (len(t) != len(a) or len(y) != len(t) or np.any(np.diff(t) <= 0)
            or np.any(t % 8) or t[0] < low or t[-1] >= high or high-low != 14*86400):
        raise DataError("Invalid original source grid or fourteen-day bounds")
    ids, events = segments(t, y)
    c = np.arange(15, len(t)-15)
    x = np.lib.stride_tricks.sliding_window_view(a, 31)
    times = np.lib.stride_tricks.sliding_window_view(t, 31)
    observed = np.isfinite(x)
    consecutive_missing = ((~observed[:, :-1]) & (~observed[:, 1:])).any(axis=1)
    rules = {
        "outside_training_or_cross_boundary_event": ~split_mask(t, c, ids, events, low, high),
        "timestamp_discontinuity": (np.diff(times, axis=1) != 8).any(axis=1),
        "center_aggregate_missing": ~np.isfinite(a[c]),
        "center_label_missing": ~np.isfinite(y[c]),
        "fewer_than_24_real_points": observed.sum(axis=1) < 24,
        "consecutive_missing_grid": consecutive_missing,
    }
    keep = np.ones(len(c), bool)
    reason = np.full(len(t), "edge_without_31_points", dtype="U48")
    for name, rejected in rules.items():
        reason[c[keep & rejected]] = name
        keep &= ~rejected
    strict = observed.all(axis=1) & keep
    reason[c[keep & ~strict]] = "added_partial"
    reason[c[strict]] = "strict"
    def take(mask):
        cc = c[mask]
        return {"x": np.array(x[mask], np.float32), "mask": observed[mask].copy(), "y": y[cc],
                "t": t[cc], "raw_index": cc, "event": ids[cc],
                "house": np.full(len(cc), ("refit_2", "refit_19").index(house), np.int64)}
    strict_data, partial = take(strict), take(keep)
    def summary(d):
        supported = set(d["event"][d["y"] >= 200].tolist())
        complete = [e["event_id"] for e in events if e["complete"]
                    and int((d["event"] == e["event_id"]).sum()) == e["samples"]]
        return {**support(d), "supported_threshold_segments": len(supported),
                "complete_observed_segments": len(complete), "complete_event_ids": complete,
                "supported_event_ids": sorted(supported)}
    ss, ps = summary(strict_data), summary(partial)
    daily = []
    for day in np.unique(t//86400):
        m = t//86400 == day
        daily.append({"date_utc": iso(int(day*86400))[:10],
                      "reasons": {str(k): int(v) for k, v in zip(*np.unique(reason[m], return_counts=True))},
                      "observed_positive_labels": int((np.isfinite(y[m]) & (y[m] >= 200)).sum()),
                      "strict": summary({k: v[strict_data["t"]//86400 == day] for k, v in strict_data.items()}),
                      "partial": summary({k: v[partial["t"]//86400 == day] for k, v in partial.items()})})
    stats = {"strict": ss, "partial": ps, "added_windows": ps["windows"]-ss["windows"],
             "added_positive_centers": ps["positive_windows"]-ss["positive_windows"],
             "newly_supported_event_ids": sorted(set(ps["supported_event_ids"])-set(ss["supported_event_ids"])),
             "threshold_segments": len(events), "label_complete_segments": sum(e["complete"] for e in events),
             "rejections_and_acceptances": {str(k): int(v) for k, v in zip(*np.unique(reason, return_counts=True))},
             "daily": daily}
    return strict_data, partial, stats, events


def prepare(run, plan_path, authorization=None):
    if run.exists():
        raise DataError("Run exists; never overwrite preparation")
    p = read_json(plan_path)
    if p != read_json(ROOT/"configs/microwave_mask_20261025.json"):
        raise DataError("Plan differs from registered fixed mask protocol")
    upstream = ROOT/p["fullsource_run"]
    upreg = read_json(upstream/"registration.json")
    names = ["train.npz", "support.json", "matrix.json", "normalization.json", "baseline_identity.json", "baseline_settings.json"]
    names += [f"validation/block_{i}.npz" for i in range(6)]
    check_artifacts(upstream, {k: upreg["artifacts"][k] for k in names})
    mpath = ROOT/p["audit_manifest"]
    parts, _ = validate_manifest(read_json(mpath), mpath.parent)
    run.mkdir(parents=True)
    # Stable sealed source of task4 is read-only. Only our new modules are overlaid.
    shutil.copytree(upstream/"source", run/"source")
    own = ["accuracy/microwave_mask.py", "accuracy/microwave_mask_execution.py", "models/mask_state_classifier.py"]
    for name in own:
        shutil.copyfile(ROOT/"src/virtual_submeter"/name, run/"source/virtual_submeter"/name)
    for name in names:
        dest = run/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream/name, dest)
    (run/"train.npz").rename(run/"original_strict.npz")
    shutil.copyfile(plan_path, run/"plan.json")
    shutil.copyfile(ROOT/"scripts/run_microwave_mask.py", run/"run_microwave_mask.py")
    shutil.copyfile(ROOT/"scripts/predict_microwave_mask.py", run/"predict_microwave_mask.py")
    shutil.copyfile(ROOT/"tests/test_microwave_mask.py", run/"test_microwave_mask.py")
    shutil.copyfile(ROOT/"docs/MICROWAVE_MASK_PLAN_20261025.md", run/"protocol.md")
    shutil.copyfile(mpath, run/"audit_manifest.json")
    all_strict, all_partial, stats, source_identity, event_rows = [], [], {}, {}, []
    for part in parts:
        h = part["household_id"]
        raw = {}
        source_identity[h] = {}
        for channel in ("aggregate", "microwave"):
            item = part["artifacts"][channel]
            path = (mpath.parent/item["path"]).resolve()
            # validate_manifest already restricts houses before opening source files.
            if sha256(path) != item["sha256"]:
                raise DataError("Source changed during preparation")
            source_identity[h][channel] = {"path": str(path), "sha256": item["sha256"]}
            with path.open(encoding="utf-8", newline="") as stream:
                raw[channel] = list(csv.DictReader(stream))
            if {r["household_id"] for r in raw[channel]} != {h}:
                raise DataError("Wrong source household")
        t, a = aggregate_arrays(raw["aggregate"], 8)
        lt = np.array([utc(r["timestamp"]) for r in raw["microwave"]])
        low, high = utc(part["start"]), utc(part["end"])
        if np.any(np.diff(lt) <= 0) or np.any(lt % 8) or lt[0] < low or lt[-1] >= high:
            raise DataError("Invalid label timestamps")
        mapping = {int(s): numeric(r["power_w"]) for s, r in zip(lt, raw["microwave"])}
        y = np.array([mapping.get(int(s), np.nan) for s in t], np.float32)
        strict, partial, stats[h], events = candidate_data(t, a, y, low, high, h)
        all_strict.append(strict)
        all_partial.append(partial)
        event_rows.extend({"house": h, **e} for e in events)
    joined = {}
    for arm, arrays in zip(ORDER, (all_strict, all_partial)):
        joined[arm] = {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}
        np.savez_compressed(run/f"{arm}.npz", **joined[arm])
    original = load_npz(run/"original_strict.npz")
    for k in original:
        np.testing.assert_array_equal(joined["strict_mask"][k], original[k])
    if len(original["y"]) != 150069 or int((original["y"] >= 200).sum()) != 646:
        raise DataError("Original strict support identity changed")
    if normalization(original["x"]) != read_json(run/"normalization.json"):
        raise DataError("Normalization differs from original strict windows")
    viable = (sum(s["added_positive_centers"] for s in stats.values()) > 0
              and sum(len(s["newly_supported_event_ids"]) for s in stats.values()) > 0)
    blocks = read_json(upstream/"matrix.json")["supported_validation_blocks"]
    if blocks != [f"block_{i}" for i in range(6)]:
        raise DataError("Expected six fixed supported House5 blocks")
    write_json(run/"mask_support.json", stats)
    write_json(run/"events.json", {"events": event_rows})
    write_json(run/"source_identity.json", source_identity)
    write_json(run/"upstream_identity.json", {"run": str(upstream.resolve()),
               "registration_sha256": sha256(upstream/"registration.json"),
               "artifacts": {k: upreg["artifacts"][k] for k in names}, "task4_results_read": False})
    write_json(run/"runtime.json", {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__})
    write_json(run/"matrix.json", {"arms": ORDER, "seeds": p["seeds"], "fits": 6 if viable else 0,
               "preparation_passed": viable, "supported_validation_blocks": blocks,
               "reason": None if viable else "no_added_positive_or_independent_threshold_segment"})
    replay = {}
    if viable:
        for arm in ORDER:
            d = joined[arm]
            for seed in p["seeds"]:
                stream = BatchStream(d, "event", seed, p)
                exposure = np.zeros(len(d["y"]), np.int64)
                digest = hashlib.sha256()
                for _ in range(3528):
                    idx = stream.next()
                    if int((d["y"][idx] >= 200).sum()) != 16 or len(idx) != 256:
                        raise DataError("Sampler deviated from 16/240 full batches")
                    np.add.at(exposure, idx, 1)
                    digest.update(idx.tobytes())
                seen = exposure > 0
                pos = d["y"] >= 200
                replay[f"{arm}/{seed}"] = {"updates": 3528, "indices_sha256": digest.hexdigest(),
                    "positive_exposure": int(exposure[pos].sum()), "negative_exposure": int(exposure[~pos].sum()),
                    "unique_windows": int(seen.sum()), "unique_positive_windows": int((seen & pos).sum()),
                    "unique_positive_events": len(set(zip(d["house"][seen & pos].tolist(), d["event"][seen & pos].tolist()))),
                    "repeated_exposures": int(exposure.sum()-seen.sum())}
                np.savez_compressed(run/f"dry_exposure_{arm}_{seed}.npz", count=exposure, t=d["t"], house=d["house"], event=d["event"])
    write_json(run/"dry_replay.json", {"actual_fits": 0, "matrix": replay})
    write_json(run/"registration.json", {"artifacts": inventory(run), "actual_fits": 0,
               "house17_labels_opened": False, "final_house_data_accessed": False,
               "execution_adapter_status": "awaiting_coordinator_message"})
    return {"status": "prepared_not_trained" if viable else "preparation_failed_zero_fits",
            "actual_fits": 0, "maximum_fits": 6 if viable else 0,
            "added_windows": sum(s["added_windows"] for s in stats.values()),
            "added_positive_centers": sum(s["added_positive_centers"] for s in stats.values()),
            "new_supported_segments": sum(len(s["newly_supported_event_ids"]) for s in stats.values())}


def select_source(rows, p):
    if set(rows) != set(ORDER) or any([r["seed"] for r in rows[a]] != p["seeds"] for a in ORDER):
        raise DataError("Complete paired mask seed matrix required")
    strict = rows["strict_mask"]
    valid = all(r["source_gate_passed"] and not r["degenerate"] for r in strict)
    candidates, summary = [], []
    for arm in ORDER:
        rr = rows[arm]
        ok = valid and all(r["source_gate_passed"] and not r["degenerate"] for r in rr)
        if arm == "partial_mask":
            ok &= (np.mean([r["f1"] for r in rr]) >= np.mean([r["f1"] for r in strict])
                   and all(b["recall"]-a["recall"] >= -.05 for a, b in zip(strict, rr))
                   and sum(b["precision"] > a["precision"] for a, b in zip(strict, rr)) >= 2)
        item = {"arm": arm, "eligible": bool(ok), "mean_precision": float(np.mean([r["precision"] for r in rr])),
                "mean_f1": float(np.mean([r["f1"] for r in rr]))}
        summary.append(item)
        if ok:
            candidates.append(item)
    chosen = max(candidates, key=lambda r: (r["mean_precision"], r["mean_f1"], -ORDER.index(r["arm"]))) if candidates else None
    selected = chosen["arm"] if chosen else None
    return {"selected_arm": selected, "partial_training_supported": selected == "partial_mask",
            "improvement_candidate": selected is not None, "no_deployable_candidate": selected is None,
            "arms": summary, "source_only": True}


def verify_preparation(run):
    r = read_json(run/"registration.json")
    check_artifacts(run, r["artifacts"])
    for channels in read_json(run/"source_identity.json").values():
        for item in channels.values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise DataError("Original source bytes changed")
    code = ROOT/("source" if (ROOT/"registration.json").exists() else "src")
    for name in ("accuracy/microwave_mask.py", "accuracy/microwave_mask_execution.py", "models/mask_state_classifier.py"):
        if sha256(code/"virtual_submeter"/name) != sha256(run/"source/virtual_submeter"/name):
            raise DataError("Use the sealed run entry point; implementation changed")
    return True


def evaluation_guard(run):
    verify_preparation(run)
    m = read_json(run/"matrix.json")
    if not m["preparation_passed"] or m["fits"] != 6:
        raise DataError("Nonviable preparation cannot evaluate")
    expected = {f"{a}/{s}" for a in ORDER for s in m["seeds"]}
    complete, frozen = read_json(run/"training_complete.json"), read_json(run/"freeze.json")
    if (len(complete["completed_models"]) != 6 or set(complete["completed_models"]) != expected
            or len(frozen["predictions"]) != 6 or set(frozen["predictions"]) != expected or complete["fits"] != 6):
        raise DataError("Incomplete six-model/frozen-prediction matrix")
    check_artifacts(run/"models", complete["artifacts"])
    check_artifacts(run, frozen["artifacts"])
    for key, file in (("registration_sha256", "registration.json"), ("training_sha256", "training_complete.json"),
                      ("source_selection_sha256", "source_selection.json")):
        if frozen[key] != sha256(run/file):
            raise DataError("Freeze dependency mismatch")
    needed = {f"predictions/{a}_{s}.npz" for a in ORDER for s in m["seeds"]}
    if not needed.issubset(frozen["artifacts"]) or not frozen.get("ordinary_inference_matches") or not frozen.get("common_support_and_other_devices_match"):
        raise DataError("Missing ordinary-inference/common-support bindings")
    if select_source(read_json(run/"source_rows.json"), read_json(run/"plan.json")) != read_json(run/"source_selection.json"):
        raise DataError("Source selection does not reproduce")
    return True


def train(run, authorization=None):
    from .microwave_mask_execution import train as execute
    return execute(run, authorization)


def freeze(run, authorization=None):
    from .microwave_mask_execution import freeze as execute
    return execute(run)


def evaluate(run, authorization=None):
    from .microwave_mask_execution import evaluate as execute
    return execute(run, authorization)


def verify(run, authorization=None):
    verify_preparation(run)
    if (run/"training_complete.json").exists():
        from .microwave_mask_execution import verify_training
        verify_training(run)
    if (run/"freeze.json").exists():
        evaluation_guard(run)
    if (run/"evaluation_complete.json").exists():
        check_artifacts(run/"evaluation", read_json(run/"evaluation_complete.json")["artifacts"])
    return {"status": "verified", "trained": (run/"training_complete.json").exists(),
            "effect_improvement_established": False if not (run/"evaluation_complete.json").exists() else read_json(run/"evaluation/summary.json")["declared_improvement"]}


def report(run, authorization=None):
    result = verify(run)
    result["support"] = read_json(run/"mask_support.json")
    if (run/"evaluation_complete.json").exists():
        result["development"] = read_json(run/"evaluation/summary.json")
    write_json(run/"report.json", result)
    return result
