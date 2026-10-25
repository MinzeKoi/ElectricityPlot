"""Finite, preregistered source-only capability and household transfer probes."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ..data.common import DataError, inventory, iso, read_json, sha256, source_fingerprint, utc, write_json
from ..models.arrays import aggregate_arrays, labels_at_centers, windows
from ..models.state_classifier import StateClassifier, predict_windows
from ..models.state_selection import checkpoint_rank, constrained_point, pr_curve

ROOT = Path(__file__).resolve().parents[3]


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data}


def segments(stamps, power):
    """Observed >=200W runs; missing readings break a segment, never mean off."""
    ids = np.full(len(stamps), -1, np.int64)
    events = []
    for i in np.flatnonzero(np.isfinite(power) & (power >= 200)):
        if i == 0 or ids[i - 1] < 0 or stamps[i] - stamps[i - 1] != 8:
            events.append({"event_id": len(events), "start": int(stamps[i]), "end": int(stamps[i] + 8),
                           "samples": 0, "complete": False})
        event = events[-1]
        ids[i] = event["event_id"]
        event["end"] = int(stamps[i] + 8)
        event["samples"] += 1
    for event in events:
        a, b = np.searchsorted(stamps, [event["start"], event["end"]])
        event["complete"] = bool(a > 0 and b < len(stamps) and stamps[a] - stamps[a-1] == 8
                                 and stamps[b] == event["end"] and np.isfinite(power[[a-1, b]]).all()
                                 and (power[[a-1, b]] < 200).all())
    return ids, events


def split_mask(stamps, centers, ids, events, low, high, width=31):
    radius = width // 2 * 8
    mask = (stamps[centers] - radius >= low) & (stamps[centers] + radius + 8 <= high)
    for event in events:
        if event["start"] < low or event["end"] > high:
            mask &= ids[centers] != event["event_id"]
    return mask


def support(data):
    y = data["y"] >= 200
    return {"windows": len(y), "positive_windows": int(y.sum()), "negative_windows": int((~y).sum()),
            "prevalence": float(y.mean()), "effective_seconds": len(y) * 8,
            "positive_segments_supported": len(np.unique(data["event"][y])),
            "first_center": iso(int(data["t"][0])), "last_center": iso(int(data["t"][-1]))}


def select_small(data, events, count):
    eligible = [e for e in events if e["complete"] and int((data["event"] == e["event_id"]).sum()) == e["samples"]]
    chosen = eligible[:count]
    if len(chosen) != count:
        raise DataError("Insufficient complete train events for registered small set")
    positive = np.flatnonzero(np.isin(data["event"], [e["event_id"] for e in chosen]))
    negative = np.flatnonzero(data["y"] < 200)
    easy_limit = float(np.quantile(data["x"][negative, 15], .25))
    used, records = set(), []
    for i in positive:
        records.append((int(i), "positive_complete_event", int(i)))
        for reason in ("center_power_matched_off", "low_power_time_matched_off"):
            candidates = np.array([j for j in negative if int(j) not in used], np.int64)
            if reason.startswith("center"):
                j = candidates[np.argmin(np.abs(data["x"][candidates, 15] - data["x"][i, 15]))]
            else:
                candidates = candidates[data["x"][candidates, 15] <= easy_limit]
                j = candidates[np.argmin(np.abs(data["t"][candidates] - data["t"][i]))]
            used.add(int(j))
            records.append((int(j), reason, int(i)))
    return records, chosen


def prepare(run, plan_path):
    if run.exists():
        raise DataError("Destination exists; use a new immutable run")
    plan = read_json(plan_path)
    if (plan["households"] != ["refit_2", "refit_19"] or plan["seeds"] != [42, 1019, 2026]
            or plan["window_samples"] != 31 or plan["maximum_fits"] != 9 or plan["additional_search_allowed"]):
        raise DataError("Unregistered household/model/search scope")
    parent = ROOT / plan["parent_protocol"]
    protocol = read_json(parent)
    parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
    if sorted(p["household_id"] for p in parts) != sorted(plan["households"]):
        raise DataError("Source household mismatch")
    run.mkdir(parents=True)
    shutil.copytree(ROOT / "src/virtual_submeter", run / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(plan_path, run / "plan.json")
    shutil.copyfile(ROOT / "docs/MICROWAVE_DIAGNOSIS_REGISTRATION_20261025.md", run / "protocol.md")
    shutil.copyfile(ROOT / "scripts/run_microwave_diagnosis.py", run / "run_microwave_diagnosis.py")
    source_meta, stats, all_events, all_rows, small_rows, small_arrays = {}, {}, [], [], [], []
    for part in parts:
        house = part["household_id"]
        raw = {}
        for channel in ("aggregate", "microwave"):
            artifact = part["artifacts"][channel]
            source = parent.parent / artifact["path"]
            if sha256(source) != artifact["sha256"]:
                raise DataError("Source data hash mismatch")
            target = run / f"inputs/{house}/{channel}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with target.open(encoding="utf-8", newline="") as stream:
                raw[channel] = list(csv.DictReader(stream))
        t, aggregate = aggregate_arrays(raw["aggregate"], 8)
        x, centers = windows(t, aggregate, 31, 8)
        # Original alignment API plus an independent direct timestamp mapping.
        y = labels_at_centers(raw, ["microwave"], t, centers)[:, 0]
        labelmap = {utc(r["timestamp"]): float(r["power_w"]) if r["power_w"] else np.nan for r in raw["microwave"]}
        full_y = np.array([labelmap.get(int(stamp), np.nan) for stamp in t], np.float32)
        np.testing.assert_array_equal(y, full_y[centers])
        np.testing.assert_array_equal(x[:, 15], aggregate[centers])
        ids, events = segments(t, full_y)
        start, end = utc(part["start"]), utc(part["end"])
        b1, b2, gap = start + 7 * 86400, start + 10 * 86400, plan["boundary_half_gap_seconds"]
        blocks = {"train": (start, b1-gap), "validation": (b1+gap, b2-gap), "outer": (b2+gap, end)}
        stats[house] = {"raw_samples": len(t), "missing_aggregate": int((~np.isfinite(aggregate)).sum()),
                        "missing_labels": int((~np.isfinite(full_y)).sum()), "valid_windows_before_label_mask": len(x),
                        "alignment_exact": True, "blocks": {}}
        for role, (low, high) in blocks.items():
            mask = split_mask(t, centers, ids, events, low, high) & np.isfinite(y)
            data = {"x": x[mask], "y": y[mask], "t": t[centers[mask]], "event": ids[centers[mask]], "raw_index": centers[mask]}
            if not (data["y"] >= 200).any() or not (data["y"] < 200).any():
                raise DataError(f"{house}/{role}: lacks both classes; registered comparison infeasible")
            stats[house]["blocks"][role] = {**support(data), "start": iso(low), "end": iso(high)}
            dest = run / f"arrays/{house}/{role}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if role == "outer":
                (run / "outer_labels").mkdir(exist_ok=True)
                np.savez_compressed(run / f"outer_labels/{house}.npz", y=data["y"])
                np.savez_compressed(dest, **{k: v for k, v in data.items() if k != "y"})
            else:
                np.savez_compressed(dest, **data)
            for j in range(len(data["y"])):
                all_rows.append({"household": house, "role": role, "window_id": f"{house}:{int(data['t'][j])}:31",
                                 "timestamp": iso(int(data["t"][j])), "raw_index": int(data["raw_index"][j]),
                                 "event_id": int(data["event"][j]), "truth_w": float(data["y"][j])})
            if role == "train":
                records, selected_events = select_small(data, events, plan["small_events_per_house"])
                idx = np.array([r[0] for r in records])
                small_arrays.append({k: v[idx] for k, v in data.items()})
                for j, reason, matched in records:
                    small_rows.append({"household": house, "timestamp": iso(int(data["t"][j])),
                                       "window_id": f"{house}:{int(data['t'][j])}:31", "train_array_index": j,
                                       "raw_index": int(data["raw_index"][j]), "event_id": int(data["event"][j]),
                                       "truth_w": float(data["y"][j]), "center_aggregate_w": float(data["x"][j, 15]),
                                       "reason": reason, "matched_positive_timestamp": iso(int(data["t"][matched]))})
                stats[house]["small_events"] = [e["event_id"] for e in selected_events]
        all_events.extend({"household": house, **e} for e in events)
        source_meta[house] = {"start": part["start"], "end": part["end"], "channel_mapping": part["origin"]["channel_mapping"]}
    np.savez_compressed(run / "arrays/small.npz", **{k: np.concatenate([a[k] for a in small_arrays]) for k in small_arrays[0]})
    save_csv(run / "window_manifest.csv", all_rows)
    save_csv(run / "small_manifest.csv", small_rows)
    write_json(run / "events.json", {"definition": "observed >=200W segments, not independent physical cycles", "events": all_events})
    write_json(run / "support.json", stats)
    write_json(run / "source_protocol.json", {"sources": source_meta, "historical_normalization": protocol["expansion_training"],
                                               "upstream_limitations": protocol["upstream_limitations"]})
    frozen = inventory(run)
    write_json(run / "registration.json", {"registered_at": datetime.now(timezone.utc).isoformat(), "inputs": frozen,
                                              "code_sha256": source_fingerprint(), "outer_results_used_for_selection": False,
                                              "accessed_households": plan["households"], "final_households_accessed": False})
    return {"status": "registered_before_training", "support": stats, "small_windows": len(small_rows)}


def registered(run):
    reg = read_json(run / "registration.json")
    if source_fingerprint() != reg["code_sha256"]:
        raise DataError("Current source differs from registered source; use sealed entrypoint")
    for p, h in reg["inputs"].items():
        if sha256(run / p) != h:
            raise DataError(f"Registered artifact changed: {p}")
    return read_json(run / "plan.json")


def metrics(scores, y, threshold):
    actual, predicted = y >= 200, scores >= threshold
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    curve = pr_curve(scores, actual)
    ap = float(np.sum(np.diff(np.r_[0., curve["recall"]]) * curve["precision"]))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp/(tp+fp) if tp+fp else 0.,
            "recall": tp/(tp+fn), "f1": 2*tp/(2*tp+fp+fn), "off_false_positive_rate": fp/(fp+tn),
            "ap": ap, "prevalence": float(actual.mean()), "n": len(actual), "threshold": float(threshold)}


def normalization(x):
    logged = np.log1p(x)
    return {"input_log_mean": float(logged.mean()), "input_log_std": max(float(logged.std()), 1e-6),
            "cpu_threads": 4, "batch_size": 256}


def configure(seed, plan):
    torch.set_num_threads(plan["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = StateClassifier(31, False)
    return model, torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])


def tensor_input(meta, x):
    return torch.from_numpy(((np.log1p(x) - meta["input_log_mean"]) / meta["input_log_std"]).astype(np.float32))


def replay(model, meta, data):
    scores = predict_windows(meta, model, data["x"])
    with torch.inference_mode():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tensor_input(meta, data["x"])),
                  torch.from_numpy((data["y"] >= 200).astype(np.float32)), pos_weight=torch.tensor(50.))
    result = metrics(scores, data["y"], .5)
    result.update(bce=float(loss), positive_score_quantiles=np.quantile(scores[data["y"] >= 200], [0, .5, 1]).tolist(),
                  negative_score_quantiles=np.quantile(scores[data["y"] < 200], [0, .5, 1]).tolist())
    return result, scores


def small_success(result, plan):
    return all(result[k] >= plan["small_success_" + k] for k in ("precision", "recall", "ap")) and result["bce"] <= plan["small_success_bce"]


def audit(run):
    plan = registered(run)
    data = load_npz(run / "arrays/small.npz")
    controls = read_json(run / "source_protocol.json")["historical_normalization"]
    meta = {**normalization(data["x"]), **{k: controls[k] for k in ("input_log_mean", "input_log_std")}}
    result = {}
    for seed in plan["seeds"]:
        model, _ = configure(seed, plan)
        row, _ = replay(model, meta, data)
        result[str(seed)] = {**row, "registered_learning_criterion_pass": small_success(row, plan)}
    return {"status": "untrained_capability_baseline", "results": result}


def seal(directory):
    write_json(directory / "complete.json", {"artifacts": inventory(directory)})


def small(run):
    plan = registered(run)
    if (run / "small_complete.json").exists():
        raise DataError("Small stage already completed")
    data = load_npz(run / "arrays/small.npz")
    controls = read_json(run / "source_protocol.json")["historical_normalization"]
    meta = {**normalization(data["x"]), **{k: controls[k] for k in ("input_log_mean", "input_log_std")}}
    results = []
    for seed in plan["seeds"]:
        dest = run / f"small/seed_{seed}"
        if dest.exists():
            raise DataError("Small model exists; refuse overwrite")
        dest.mkdir(parents=True)
        model, optimizer = configure(seed, plan)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        x, y = tensor_input(meta, data["x"]), torch.from_numpy((data["y"] >= 200).astype(np.float32))
        history, consecutive, started = [], 0, time.perf_counter()
        first, scores = replay(model, meta, data)
        history.append({"step": 0, **first, "gradient_l2": None, "parameter_change_l2": 0.})
        np.savez_compressed(dest / "initial_scores.npz", scores=scores)
        for step in range(1, plan["small_max_updates"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(plan["pos_weight"]))
            loss.backward()
            grad = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
            if not np.isfinite([float(loss.detach()), grad]).all():
                raise DataError("Non-finite small-set loss or gradient")
            before = {k: v.clone() for k, v in model.state_dict().items()} if step == 1 else None
            optimizer.step()
            if step == 1:
                write_json(dest / "first_update.json", {"gradient_l2": grad, "parameter_delta_l2": float(torch.sqrt(sum((v-before[k]).square().sum() for k, v in model.state_dict().items()))),
                    "normalized_min": float(x.min()), "normalized_max": float(x.max()), "finite": bool(torch.isfinite(x).all()),
                    "label_counts": [int((y == 0).sum()), int((y == 1).sum())]})
            if step % plan["small_check_every"] == 0:
                row, scores = replay(model, meta, data)
                delta = float(torch.sqrt(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items())))
                history.append({"step": step, **row, "gradient_l2": grad, "parameter_change_l2": delta})
                write_json(dest / "history.json", {"rows": history})
                np.savez_compressed(dest / f"scores_{step}.npz", scores=scores)
                consecutive = consecutive + 1 if small_success(row, plan) else 0
                print(f"small seed={seed} step={step} P={row['precision']:.4f} R={row['recall']:.4f} BCE={row['bce']:.5f}", flush=True)
                if consecutive >= plan["small_success_consecutive"]:
                    break
        torch.save(model.state_dict(), dest / "weights.pt")
        final, scores = replay(model, meta, data)
        np.savez_compressed(dest / "final_scores.npz", scores=scores, y=data["y"])
        result = {"seed": seed, "updates": step, "success": consecutive >= plan["small_success_consecutive"],
                  "stop_reason": "consecutive_success" if consecutive >= plan["small_success_consecutive"] else "registered_update_cap",
                  "seconds": time.perf_counter()-started, "normalization": meta, "initial": first, "final": final}
        write_json(dest / "result.json", result)
        seal(dest)
        results.append(result)
    write_json(run / "small_complete.json", {"results": results, "artifacts": inventory(run / "small")})
    return {"status": "small_completed", "results": results}


def train(run):
    plan = registered(run)
    if (run / "freeze.json").exists():
        raise DataError("Controlled predictions already frozen")
    if not (run / "small_complete.json").exists():
        raise DataError("Complete small capability stage first")
    for house in plan["households"]:
        data = load_npz(run / f"arrays/{house}/train.npz")
        val = load_npz(run / f"arrays/{house}/validation.npz")
        meta = normalization(data["x"])
        x, y = tensor_input(meta, data["x"]), torch.from_numpy((data["y"] >= 200).astype(np.float32))
        for seed in plan["seeds"]:
            dest = run / f"models/{house}/seed_{seed}"
            if dest.exists():
                raise DataError("Controlled model exists; refuse overwrite")
            dest.mkdir(parents=True)
            model, optimizer = configure(seed, plan)
            generator = torch.Generator().manual_seed(seed)
            def batches():
                while True:
                    yield from torch.randperm(len(x), generator=generator).split(plan["batch_size"])
            stream = batches()
            history, best, seen, positives, started = [], None, 0, 0, time.perf_counter()
            for epoch in range(1, plan["checkpoints"] + 1):
                model.train()
                loss_sum = 0.
                for _ in range(plan["steps_per_checkpoint"]):
                    idx = next(stream)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx], pos_weight=torch.tensor(plan["pos_weight"]))
                    loss.backward()
                    if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise DataError("Non-finite controlled loss/gradient")
                    optimizer.step()
                    seen += len(idx)
                    positives += int(y[idx].sum())
                    loss_sum += float(loss.detach())
                scores = predict_windows(meta, model, val["x"])
                curve = pr_curve(scores, val["y"] >= 200)
                point = constrained_point(curve, plan["minimum_validation_recall"])
                torch.save(model.state_dict(), dest / f"epoch_{epoch}.pt")
                np.savez_compressed(dest / f"epoch_{epoch}_validation.npz", scores=scores, **curve)
                history.append({"epoch": epoch, "bce": loss_sum / plan["steps_per_checkpoint"], "selection": point,
                                "validation_fixed_half": metrics(scores, val["y"], .5), "seen": seen, "positive_seen": positives})
                if best is None or checkpoint_rank(point, epoch) < best:
                    best = checkpoint_rank(point, epoch)
                    selected = {"epoch": epoch, "threshold": point["threshold"], "selection": point}
                write_json(dest / "history.json", {"rows": history})
                write_json(run / "progress.json", {"stage": "controlled_training", "house": house, "seed": seed, "checkpoint": epoch})
                print(f"controlled house={house} seed={seed} checkpoint={epoch}/12 val_P={point['precision']:.4f} R={point['recall']:.4f}", flush=True)
            shutil.copyfile(dest / f"epoch_{selected['epoch']}.pt", dest / "weights.pt")
            model.load_state_dict(torch.load(dest / "weights.pt", weights_only=True))
            info = {"household": house, "seed": seed, "normalization": meta, "selected": selected,
                    "steps": plan["steps_per_checkpoint"] * plan["checkpoints"], "seen": seen, "positive_seen": positives,
                    "train_support": support(data), "validation_support": support(val), "seconds": time.perf_counter()-started,
                    "parameter_count": sum(p.numel() for p in model.parameters()), "weights_sha256": sha256(dest / "weights.pt")}
            write_json(dest / "model.json", info)
            for target in plan["households"]:
                # Outer truth is physically separate and is not opened until evaluate.
                outer = load_npz(run / f"arrays/{target}/outer.npz")
                scores = predict_windows(meta, model, outer["x"])
                pred = run / f"predictions/train_{house}_seed_{seed}_target_{target}.npz"
                pred.parent.mkdir(exist_ok=True)
                np.savez_compressed(pred, scores=scores, t=outer["t"])
            replay_scores = predict_windows(meta, model, data["x"])
            np.savez_compressed(dest / "train_replay.npz", scores=replay_scores)
            seal(dest)
    frozen = {prefix + "/" + p: h for prefix in ("models", "predictions", "small") for p, h in inventory(run / prefix).items()}
    write_json(run / "freeze.json", {"artifacts": frozen, "registration_sha256": sha256(run / "registration.json"),
                                    "frozen_at": datetime.now(timezone.utc).isoformat(), "outer_labels_evaluated": False})
    return {"status": "all_models_thresholds_and_outer_predictions_frozen", "models": 6, "predictions": 12}


def check_freeze(run):
    freeze = read_json(run / "freeze.json")
    if freeze["registration_sha256"] != sha256(run / "registration.json"):
        raise DataError("Registration seal changed")
    for p, h in freeze["artifacts"].items():
        if sha256(run / p) != h:
            raise DataError(f"Frozen result changed: {p}")


def diagnostic_upper(scores, y):
    curve = pr_curve(scores, y >= 200)
    point = constrained_point(curve)
    return {"only_diagnostic_not_deployable": True, "p_at_r85": point["precision"], "recall": point["recall"],
            "fpr_at_r85": point["fp"] / int((y < 200).sum()), "maximum_f1": float(curve["f1"].max())}


def evaluate(run):
    plan = registered(run)
    check_freeze(run)
    if (run / "evaluation_complete.json").exists() or (run / "evaluation").exists():
        raise DataError("Evaluation already exists; refuse overwrite")
    dest = run / "evaluation"
    dest.mkdir()
    rows, cases = [], []
    for house in plan["households"]:
        for seed in plan["seeds"]:
            model_dir = run / f"models/{house}/seed_{seed}"
            meta = read_json(model_dir / "model.json")
            for role in ("train", "validation"):
                data = load_npz(run / f"arrays/{house}/{role}.npz")
                scores = load_npz(model_dir / ("train_replay.npz" if role == "train" else f"epoch_{meta['selected']['epoch']}_validation.npz"))["scores"]
                rows.append({"train_house": house, "target_house": house, "seed": seed, "role": role,
                             **metrics(scores, data["y"], meta["selected"]["threshold"]), "diagnostic": diagnostic_upper(scores, data["y"])})
            for target in plan["households"]:
                data = load_npz(run / f"arrays/{target}/outer.npz")
                y = load_npz(run / f"outer_labels/{target}.npz")["y"]
                scores = load_npz(run / f"predictions/train_{house}_seed_{seed}_target_{target}.npz")["scores"]
                kind = "same_house_outer" if house == target else "cross_house_outer"
                rows.append({"train_house": house, "target_house": target, "seed": seed, "role": kind,
                             **metrics(scores, y, meta["selected"]["threshold"]), "fixed_half": metrics(scores, y, .5),
                             "diagnostic": diagnostic_upper(scores, y),
                             "positive_segments_supported": len(np.unique(data["event"][y >= 200])),
                             "positive_segments_hit": len(np.unique(data["event"][(y >= 200) & (scores >= meta['selected']['threshold'])])),
                             "effective_seconds": len(y)*8})
                np.savez_compressed(dest / f"pr_{house}_{seed}_{target}.npz", **pr_curve(scores, y >= 200))
                # Deterministic score-ranked separated cases, not a training mining list.
                chosen = []
                for label, mask in (("false_positive", (y < 200) & (scores >= meta["selected"]["threshold"])),
                                    ("true_positive", (y >= 200) & (scores >= meta["selected"]["threshold"])),
                                    ("false_negative", (y >= 200) & (scores < meta["selected"]["threshold"]))):
                    kept = []
                    for j in np.argsort(-scores, kind="stable"):
                        if mask[j] and all(abs(int(data["t"][j])-int(data["t"][k])) >= 248 for k in kept):
                            kept.append(int(j))
                            chosen.append((label, int(j)))
                            if len(kept) == 3:
                                break
                for label, j in chosen:
                    cases.append({"train_house": house, "target_house": target, "seed": seed, "case": label,
                                  "timestamp": iso(int(data["t"][j])), "score": float(scores[j]), "truth_w": float(y[j]),
                                  "event_id": int(data["event"][j]), "aggregate_window_w": data["x"][j].tolist()})
    paired = []
    for target in plan["households"]:
        same = [r for r in rows if r["target_house"] == target and r["role"] == "same_house_outer"]
        cross = [r for r in rows if r["target_house"] == target and r["role"] == "cross_house_outer"]
        gap = float(np.mean([r["ap"] for r in same])-np.mean([r["ap"] for r in cross]))
        sf, cf = [float(np.mean([r["diagnostic"]["fpr_at_r85"] for r in group])) for group in (same, cross)]
        paired.append({"target_house": target, "mean_ap_same_minus_cross": gap, "mean_diagnostic_fpr_same": sf,
                       "mean_diagnostic_fpr_cross": cf, "registered_clear_advantage": bool(gap >= plan["same_house_advantage_ap"] and sf <= cf*plan["same_house_advantage_fpr_ratio"])})
    write_json(dest / "results.json", {"rows": rows, "paired_comparisons": paired,
                                      "consistent_same_house_advantage": all(p["registered_clear_advantage"] for p in paired)})
    save_csv(dest / "metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    write_json(dest / "error_cases.json", {"rule": "three highest score cases per type separated >=248s", "cases": cases})
    write_json(run / "evaluation_complete.json", {"artifacts": inventory(dest), "freeze_sha256": sha256(run / "freeze.json")})
    return {"status": "evaluated_once", "paired_comparisons": paired, "rows": len(rows)}


def verify(run):
    plan = registered(run)
    check_freeze(run)
    receipt = read_json(run / "evaluation_complete.json")
    if receipt["freeze_sha256"] != sha256(run / "freeze.json"):
        raise DataError("Evaluation refers to different prediction freeze")
    for p, h in receipt["artifacts"].items():
        if sha256(run / "evaluation" / p) != h:
            raise DataError("Evaluation seal changed")
    rows = read_json(run / "evaluation/results.json")["rows"]
    checked = 0
    for house in plan["households"]:
        for seed in plan["seeds"]:
            dest = run / f"models/{house}/seed_{seed}"
            meta = read_json(dest / "model.json")
            history = read_json(dest / "history.json")["rows"]
            expected = min(history, key=lambda r: checkpoint_rank(r["selection"], r["epoch"]))
            if expected["epoch"] != meta["selected"]["epoch"] or expected["selection"]["threshold"] != meta["selected"]["threshold"]:
                raise DataError("Internal validation selection mismatch")
            model = StateClassifier(31, False)
            model.load_state_dict(torch.load(dest / "weights.pt", weights_only=True))
            for target in plan["households"]:
                data = load_npz(run / f"arrays/{target}/outer.npz")
                scores = predict_windows(meta["normalization"], model, data["x"])
                stored = load_npz(run / f"predictions/train_{house}_seed_{seed}_target_{target}.npz")
                np.testing.assert_allclose(scores, stored["scores"], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(scores >= meta["selected"]["threshold"], stored["scores"] >= meta["selected"]["threshold"])
                np.testing.assert_array_equal(data["t"], stored["t"])
                y = load_npz(run / f"outer_labels/{target}.npz")["y"]
                row = next(r for r in rows if r["train_house"] == house and r["target_house"] == target and r["seed"] == seed and r["role"].endswith("outer"))
                for k, v in metrics(stored["scores"], y, meta["selected"]["threshold"]).items():
                    if row[k] != v:
                        raise DataError("Recomputed metric mismatch")
                checked += 1
    return {"status": "verified", "prediction_readbacks": checked, "registered_inputs": len(read_json(run / 'registration.json')["inputs"]),
            "final_households_accessed": False, "state_decisions_identical": True}
