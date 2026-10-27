"""Single preregistered background-composition candidate; no final-house access.

Preparation accepts only an explicit, hashed audit data manifest. Training reads
only sealed train/validation arrays. Outer truth is opened only after freeze.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from ..data.common import DataError, inventory, iso, read_json, sha256, utc, write_json
from ..models.arrays import aggregate_arrays, labels_at_centers, numeric, windows
from ..models.state_classifier import load_state_model, predict_state_rows, predict_windows
from ..models.state_selection import pr_curve
from .microwave_diagnosis import (configure, diagnostic_upper, load_npz, metrics,
                                  normalization, save_csv, segments, split_mask, tensor_input)

ROOT = Path(__file__).resolve().parents[3]
HOUSES = ["refit_2", "refit_19"]
ARMS = ["natural", "event", "event_background"]
SAMPLING_ARMS = ["natural", "event"]
SEEDS = [42, 1019, 2026]


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def candidate_dirs(run):
    return [("event_background", seed, run / f"models/event_background/seed_{seed}") for seed in SEEDS]


def training_labels(path, house, low, high):
    """Parse power only inside training. Do not interpret held-out label values."""
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stamp = utc(row["timestamp"])
            if stamp >= high:
                break
            if low <= stamp < high:
                if row["household_id"] != house or stamp in mapping or stamp % 8:
                    raise DataError("Training label identity/alignment mismatch")
                mapping[stamp] = numeric(row["power_w"])
    return mapping


def make_pools(data, mappings, bounds):
    """Use only original source train centers; missing context is never filled."""
    n = len(data["y"])
    power = np.full((n, 31), np.nan, np.float32)
    reasons = np.full(n, "eligible", dtype="U32")
    for h, house in enumerate(HOUSES):
        indices = np.flatnonzero(data["house"] == h)
        low, high = bounds[house]
        for i in indices:
            t = int(data["t"][i])
            if t-120 < low or t+128 > high:
                reasons[i] = "outside_train_context"
                continue
            power[i] = [mappings[house].get(t+offset*8, np.nan) for offset in range(-15, 16)]
    inside = reasons == "eligible"
    finite = np.isfinite(power).all(axis=1) & np.isfinite(data["x"]).all(axis=1)
    reasons[inside & ~finite] = "missing_or_nonfinite_context"
    valid = inside & finite
    nonnegative = (power >= 0).all(axis=1) & (data["x"] >= 0).all(axis=1)
    reasons[valid & ~nonnegative] = "negative_source_power"
    valid &= nonnegative
    if not np.array_equal(power[valid, 15], data["y"][valid]):
        raise DataError("Template center labels differ from frozen sampler labels")
    residual = data["x"]-power
    residual_ok = (residual >= 0).all(axis=1) & np.isfinite(residual).all(axis=1)
    # Check residual on templates too; never mask misalignment by clipping.
    reasons[valid & ~residual_ok] = "negative_background_residual"
    valid &= residual_ok
    positive = data["y"] >= 200
    template_ok = valid & positive
    background_ok = valid & (power < 200).all(axis=1)
    pools = {"power": power, "residual": residual, "template_ok": template_ok,
             "background_ok": background_ok, "reason": reasons}
    counts = {str(k): int(v) for k, v in zip(*np.unique(reasons, return_counts=True))}
    stats = {"all_train_windows": n, "template_centers": int(template_ok.sum()),
             "background_centers": int(background_ok.sum()), "exclusion_counts": counts,
             "eligible_template_threshold_segments": len(set(zip(data["house"][template_ok].tolist(),
                                                                 data["event"][template_ok].tolist()))),
             "houses": {}}
    for h, house in enumerate(HOUSES):
        mask = data["house"] == h
        pos = mask & positive
        ok = mask & template_ok
        stats["houses"][house] = {
            "positive_centers": int(pos.sum()), "template_centers": int(ok.sum()),
            "template_threshold_segments": len(np.unique(data["event"][ok])),
            "background_centers": int((mask & background_ok).sum()),
            "background_dates_utc": sorted(set((data["t"][mask & background_ok]//86400).tolist())),
            "positive_exclusions": {str(k): int(v) for k, v in zip(*np.unique(reasons[pos & ~template_ok], return_counts=True))}}
    return pools, stats


class BackgroundStream:
    """Separate RNG; exact original positive identity, no resampling on failure."""
    def __init__(self, data, pools, seed, plan):
        self.d, self.pool, self.seed, self.p = data, pools, seed, plan
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, plan["augmentation_rng_namespace"]]))
        self.queues, self.candidates = {}, {}
        self.attempts = self.successes = 0
        self.events, self.pairs = set(), set()
        bg = np.flatnonzero(pools["background_ok"])
        for h, day in sorted(set(zip(data["house"].tolist(), (data["t"]//86400).tolist()))):
            other_h, other_day = data["house"][bg] != h, data["t"][bg]//86400 != day
            tiers = (bg[other_h & other_day], bg[other_h], bg[other_day], bg)
            self.candidates[h, day] = next((a for a in tiers if len(a)), np.array([], np.int64))

    def apply(self, indices, update):
        x = self.d["x"][indices].copy()
        positive_slots = np.flatnonzero(self.d["y"][indices] >= 200)
        if len(positive_slots) != self.p["positive_per_batch"]:
            raise DataError("Original event batch positive quota changed")
        slots = self.rng.permutation(positive_slots)[:self.p["augmented_positive_per_batch"]]
        records = []
        for slot in slots:
            i = int(indices[slot])
            h, t, e = (int(self.d[k][i]) for k in ("house", "t", "event"))
            key = h, t//86400
            row = {"seed": self.seed, "update": update, "slot": int(slot), "template_index": i,
                   "template_house": HOUSES[h], "template_t": t, "template_event": e,
                   "template_center_w": float(self.d["y"][i]), "scale": 1., "time_shift": 0,
                   "batch_indices_sha256": array_hash(indices), "status": "fallback_original"}
            row["source_data_hashes"] = self.pool.get("source_hashes", np.array([], dtype="U64")).tolist()
            self.attempts += 1
            if not self.pool["template_ok"][i]:
                row["reason"] = str(self.pool["reason"][i])
            elif not len(self.candidates[key]):
                row["reason"] = "no_eligible_background"
            else:
                if key not in self.queues or not self.queues[key]:
                    self.queues[key] = self.rng.permutation(self.candidates[key]).tolist()
                j = int(self.queues[key].pop())
                combined = self.pool["residual"][j]+self.pool["power"][i]
                if not np.isfinite(combined).all() or (combined < 0).any():
                    raise DataError("Frozen eligible pool produced invalid synthetic power")
                if bool(self.pool["power"][i, 15] >= 200) != bool(self.d["y"][i] >= 200):
                    raise DataError("Synthetic power/label mismatch")
                x[slot] = combined
                row.update(status="synthesized", reason=None, background_index=j,
                           background_house=HOUSES[int(self.d["house"][j])], background_t=int(self.d["t"][j]),
                           template_power_sha256=array_hash(self.pool["power"][i]),
                           background_residual_sha256=array_hash(self.pool["residual"][j]))
                self.successes += 1
                self.events.add((h, e))
                self.pairs.add((h, int(self.d["house"][j])))
            row["output_sha256"] = array_hash(x[slot])
            records.append(row)
        digest = array_hash(x)
        for row in records:
            row["batch_output_sha256"] = digest
        return x, records

    def summary(self):
        return {"attempts": self.attempts, "synthesized": self.successes,
                "fallbacks": self.attempts-self.successes,
                "actual_fraction_of_positive_slots": self.successes/(self.attempts*2) if self.attempts else 0.,
                "unique_template_threshold_segments": len(self.events),
                "template_event_identities": sorted(self.events), "house_pairs": sorted(self.pairs)}


def control_path(run):
    return Path(read_json(run / "dependency.json")["control_run"])


def prepare(run, plan_path, control):
    if run.exists():
        raise DataError("Run exists; never overwrite preparation")
    p = read_json(plan_path)
    validate_plan(p)
    control = control.resolve()
    registration = read_json(control / "registration.json")
    # Only frozen preparation artifacts, never task2 scores or outer truth.
    readable = {name: digest for name, digest in registration["artifacts"].items()
                if not name.startswith("outer_labels/") and name not in ("events.json", "window_manifest.csv")}
    check_artifacts(control, readable)
    control_plan = read_json(control / "plan.json")
    for key, value in control_plan.items():
        if key not in ("experiment_id", "arms", "maximum_fits") and p.get(key) != value:
            raise DataError(f"Candidate changed control recipe: {key}")
    run.mkdir(parents=True)
    shutil.copytree(ROOT / "src/virtual_submeter", run / "source/virtual_submeter",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for src, name in ((plan_path, "plan.json"), (plan_path, "recipe.json"),
                      (ROOT / "scripts/run_microwave_background.py", "run_microwave_background.py"),
                      (ROOT / "docs/MICROWAVE_BACKGROUND_PLAN_20261025.md", "protocol.md")):
        shutil.copyfile(src, run / name)
    for name in ("normalization.json", "runtime.json", "support.json", "validation_support.json", "audit_receipt.json", "audit_manifest.json"):
        shutil.copyfile(control / name, run / name)
    write_json(run / "dependency.json", {"control_run": str(control),
               "registration_sha256": sha256(control / "registration.json"),
               "sealed_sampling_source_sha256": sha256(control / "source/virtual_submeter/accuracy/microwave_sampling.py")})
    for house in HOUSES:
        for role in ("train", "validation_0", "validation_1", "outer"):
            dest = run / f"arrays/{house}/{role}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(control / f"arrays/{house}/{role}.npz", dest)
        dest = run / f"inputs/{house}/aggregate.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(control / f"inputs/{house}/aggregate.csv", dest)
    audited = read_json(control / "audit_manifest.json")
    bounds = {part["household_id"]: (utc(part["start"]), utc(part["start"])+10*86400-p["boundary_half_gap_seconds"])
              for part in audited["sources"]}
    mapping = {h: training_labels(control / f"inputs/{h}/microwave.csv", h, *bounds[h]) for h in HOUSES}
    data = pooled(run)
    pools, stats = make_pools(data, mapping, bounds)
    pools["source_hashes"] = np.array([[part["artifacts"][channel]["sha256"] for channel in ("aggregate", "microwave")]
                                     for part in sorted(audited["sources"], key=lambda part: HOUSES.index(part["household_id"]))])
    np.savez_compressed(run / "pools.npz", **pools)
    write_json(run / "pool_support.json", stats)
    # Dry replay performs no model fits and consults no scores.
    previews = {}
    for seed in SEEDS:
        a, b = BatchStream(data, "event", seed, p), BatchStream(data, "event", seed, p)
        augment = BackgroundStream(data, pools, seed, p)
        for update in range(1, p["checkpoints"]*p["steps_per_checkpoint"]+1):
            idx = a.next()
            np.testing.assert_array_equal(idx, b.next())
            augment.apply(idx, update)
        previews[str(seed)] = augment.summary()
    write_json(run / "dry_replay.json", {"fits": 0, "seeds": previews})
    viable = bool(stats["template_centers"] and stats["background_centers"] and any(v["synthesized"] for v in previews.values()))
    write_json(run / "registration.json", {"artifacts": inventory(run), "fits": 3, "actual_fits": 0,
               "training_started": False, "preparation_passed": viable,
               "audit_sha256": registration["audit_sha256"], "households": HOUSES,
               "created_at": datetime.now(timezone.utc).isoformat()})
    return {"status": "prepared_not_trained" if viable else "preparation_failed_zero_augmentation",
            "actual_fits": 0, "support": stats, "dry_replay": previews, "awaiting_execution_message": True}


def require_controls_complete(run, authorization):
    if not read_json(run / "registration.json")["preparation_passed"]:
        raise DataError("No usable augmentation support: zero fits allowed")
    control = control_path(run)
    if sha256(control / "registration.json") != read_json(run / "dependency.json")["registration_sha256"]:
        raise DataError("Control preparation changed")
    for name in ("training_complete.json", "freeze.json", "evaluation_complete.json"):
        if not (control / name).exists() or authorization.get("control_"+name.removesuffix(".json")+"_sha256") != sha256(control / name):
            raise DataError("Task2 must be completed and bound by coordinator execution authorization")
    if read_json(control / "training_complete.json")["fits"] != 6:
        raise DataError("Expected exactly six reused control fits")
    check_artifacts(control / "models", read_json(control / "training_complete.json")["artifacts"])


def import_controls(run):
    control = control_path(run)
    for arm in ("natural", "event"):
        shutil.copytree(control / f"models/{arm}", run / f"models/{arm}")
    # Copy frozen inputs/weights only. No control results, outer metrics or cases.
    write_json(run / "control_reuse.json", {"models": inventory(run / "models"), "new_fits": 0})


def import_outer_truth(run):
    if (run / "evaluation").exists():
        raise DataError("Evaluation already exists")
    check_freeze(run)
    control = control_path(run)
    registration = read_json(control / "registration.json")
    for name, digest in registration["artifacts"].items():
        if name.startswith("outer_labels/"):
            src = control / name
            if sha256(src) != digest:
                raise DataError("Control outer truth changed")
            dst = run / name
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(src, dst)


def verify_trace(run, seed):
    data, p = pooled(run), read_json(run / "plan.json")
    stream = BatchStream(data, "event", seed, p)
    augment = BackgroundStream(data, load_npz(run / "pools.npz"), seed, p)
    path = run / f"models/event_background/seed_{seed}"
    with (path / "augmentation.jsonl").open(encoding="utf-8") as f:
        for update in range(1, p["checkpoints"]*p["steps_per_checkpoint"]+1):
            _, records = augment.apply(stream.next(), update)
            for expected in records:
                if json.loads(next(f)) != expected:
                    raise DataError("Augmentation identity/output replay differs")
        if f.read():
            raise DataError("Unexpected extra augmentation records")
    if augment.summary() != read_json(path / "augmentation_summary.json"):
        # JSON serializes tuples as lists.
        if json.loads(json.dumps(augment.summary())) != read_json(path / "augmentation_summary.json"):
            raise DataError("Augmentation summary differs")


def validate_plan(p):
    expected = read_json(ROOT / "configs/microwave_background_20261025.json")
    if p != expected:
        raise DataError("Background plan differs from the single registered candidate")


def require_source(house, role):
    if house not in HOUSES or role not in ("train", "validation_0", "validation_1", "outer"):
        raise DataError("Only source House2/19 and registered roles are allowed")


def validate_manifest(m, base):
    if m.get("schema") != "microwave_sampling_audit_v1" or m.get("status") != "ready":
        raise DataError("Completed audit data manifest required")
    audit = (base / m["audit"]["path"]).resolve()
    if sha256(audit) != m["audit"]["sha256"]:
        raise DataError("Audit receipt hash mismatch")
    parts = m["sources"]
    if sorted(p["household_id"] for p in parts) != sorted(HOUSES):
        raise DataError("Manifest must contain exactly House2/19")
    for p in parts:
        require_source(p["household_id"], "train")
        if utc(p["end"]) - utc(p["start"]) != 14 * 86400:
            raise DataError("Expected fixed original 14-day observation period")
        for channel in ("aggregate", "microwave"):
            a = p["artifacts"][channel]
            if sha256((base / a["path"]).resolve()) != a["sha256"]:
                raise DataError("Audited source hash mismatch")
    return parts, audit


def support(data, low=None, high=None):
    y = data["y"] >= 200
    n = len(y)
    return {"windows": n, "positive_windows": int(y.sum()), "negative_windows": int((~y).sum()),
            "events": len(np.unique(data["event"][y])), "effective_seconds": n * 8,
            "expected_grid_points": None if low is None else (high-low)//8,
            "excluded_grid_points": None if low is None else (high-low)//8-n,
            "positive_fraction": float(y.mean()) if n else None,
            "dates_utc": sorted(set((data["t"]//86400).tolist()))}



def check_artifacts(root, hashes):
    for name, digest in hashes.items():
        if sha256(root / name) != digest:
            raise DataError(f"Sealed artifact changed: {name}")


def registered(run):
    r = read_json(run / "registration.json")
    check_artifacts(run, r["artifacts"])
    # Compare just execution modules; unrelated work in this shared workspace is allowed.
    for name, digest in r["artifacts"].items():
        if name.startswith("source/") and name.endswith(".py"):
            code_root = ROOT / ("source" if (ROOT / "registration.json").exists() else "src")
            if sha256(code_root / name.removeprefix("source/")) != digest:
                raise DataError("Execution source changed; use sealed run_microwave_background.py")
    return read_json(run / "plan.json")


def source_array(run, house, role):
    require_source(house, role)
    if role == "outer":
        raise DataError("Outer arrays cannot enter training or source selection")
    return load_npz(run / f"arrays/{house}/{role}.npz")


def pooled(run):
    arrays = [source_array(run, h, "train") for h in HOUSES]
    return {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}


def validation_data(run):
    return {f"{h}/{role}": source_array(run, h, role)
            for h in HOUSES for role in ("validation_0", "validation_1")}


def supported_block(d, p):
    return (int((d["y"] >= 200).sum()) >= p["minimum_block_positive_windows"]
            and len(np.unique(d["event"][d["y"] >= 200])) >= p["minimum_block_events"]
            and bool((d["y"] < 200).any()))


class BatchStream:
    """Fixed full batches; date/event cycles prevent duration-proportional positives."""
    def __init__(self, data, arm, seed, plan):
        if arm not in SAMPLING_ARMS or not np.isfinite(data["y"]).all():
            raise DataError("Invalid sampler arm or missing label")
        self.d, self.arm, self.p = data, arm, plan
        self.rng = np.random.default_rng(seed)
        self.queues = {}
        self.positive = np.flatnonzero(data["y"] >= 200)
        self.negative = np.flatnonzero(data["y"] < 200)
        if not len(self.positive) or not len(self.negative):
            raise DataError("Sampling requires observed positives and negatives")
        self.groups = {}
        for i in self.positive:
            day = (int(data["house"][i]), int(data["t"][i]//86400))
            event = int(data["event"][i])
            if event < 0:
                raise DataError("Positive center has no event identity")
            self.groups.setdefault(day, {}).setdefault(event, []).append(int(i))

    def take(self, key, pool, n):
        result = []
        while len(result) < n:
            if key not in self.queues or not self.queues[key]:
                self.queues[key] = self.rng.permutation(len(pool)).tolist()
            result.append(pool[self.queues[key].pop()])
        return result

    def next(self):
        if self.arm == "natural":
            return np.array(self.take("natural", np.arange(len(self.d["y"])), self.p["batch_size"]), np.int64)
        chosen = []
        days = sorted(self.groups)
        for _ in range(self.p["positive_per_batch"]):
            day = self.take("days", days, 1)[0]
            event = self.take(("events", day), sorted(self.groups[day]), 1)[0]
            chosen.extend(self.take(("centers", day, event), self.groups[day][event], 1))
        chosen.extend(self.take("negative", self.negative, self.p["uniform_negative_per_batch"]))
        return self.rng.permutation(np.asarray(chosen, np.int64))


def selection(scores, blocks, p):
    """One common threshold; recall guard in every supported date/house block."""
    names = [k for k, d in blocks.items() if supported_block(d, p)]
    if len(names) < 2:
        return None
    # Global exact score union; no threshold grid search or outer-derived values.
    thresholds = np.unique(np.concatenate([scores[k] for k in blocks if len(scores[k])]))
    eligible = np.ones(len(thresholds), bool)
    precisions, f1s = [], []
    for k, d in blocks.items():
        s, y = np.asarray(scores[k]), d["y"] >= 200
        if len(s) != len(y) or not np.isfinite(s).all():
            raise DataError("Selection score alignment/finite check failed")
        if not len(s):
            continue
        order = np.argsort(s, kind="stable")
        ordered, truth = s[order], y[order]
        cut = np.searchsorted(ordered, thresholds, side="left")
        tp = np.r_[np.cumsum(truth[::-1])[::-1], 0][cut]
        predicted = len(s)-cut
        precision = tp/np.maximum(predicted, 1)
        f1 = 2*tp/np.maximum(predicted+int(y.sum()), 1)
        if k in names:
            # Zero-positive and unsupported blocks remain diagnostic, not guards.
            eligible &= (predicted > 0) & (predicted < len(s))
            eligible &= (tp/y.sum() >= p["minimum_validation_recall"])
            precisions.append(precision)
            f1s.append(f1)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return None
    ps, fs = np.array(precisions), np.array(f1s)
    i = max(indices, key=lambda j: (float(ps[:, j].min()), float(ps[:, j].mean()),
                                   float(fs[:, j].mean()), float(thresholds[j])))
    return {"threshold": float(thresholds[i]), "worst_precision": float(ps[:, i].min()),
            "mean_precision": float(ps[:, i].mean()), "mean_f1": float(fs[:, i].mean()),
            "supported_blocks": names, "limited_evidence": len(names) < 4}


def rank(point, epoch):
    return (-point["worst_precision"], -point["mean_precision"], -point["mean_f1"], epoch)


def model_dirs(run):
    return [(arm, seed, run / f"models/{arm}/seed_{seed}") for arm in ARMS for seed in SEEDS]


def train(run, authorization):
    p = registered(run)
    a = read_json(authorization)
    if (a.get("registration_sha256") != sha256(run / "registration.json")
            or a.get("audit_sha256") != read_json(run / "registration.json")["audit_sha256"]
            or a.get("maximum_fits") != 3 or a.get("execute") is not True
            or not a.get("coordinator_execution_message")):
        raise DataError("Post-audit coordinator execution authorization required")
    require_controls_complete(run, a)
    runtime = read_json(run / "runtime.json")
    if runtime != {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}:
        raise DataError("Runtime changed since preparation")
    if (run / "training_started.json").exists():
        raise DataError("Training already started; no automatic resume or overwrite")
    write_json(run / "training_started.json", {"authorization": a, "authorization_sha256": sha256(authorization)})
    import_controls(run)
    data, blocks = pooled(run), validation_data(run)
    meta = read_json(run / "normalization.json")
    x, y = tensor_input(meta, data["x"]), torch.from_numpy((data["y"] >= 200).astype(np.float32))
    for arm, seed, dest in candidate_dirs(run):
        dest.mkdir(parents=True, exist_ok=False)
        model, optimizer = configure(seed, p)
        torch.save(model.state_dict(), dest / "initial.pt")
        sampler = BatchStream(data, "event", seed, p)
        augmenter = BackgroundStream(data, load_npz(run / "pools.npz"), seed, p)
        trace = (dest / "augmentation.jsonl").open("x", encoding="utf-8")
        exposure = np.zeros(len(y), np.int64)
        histories, batch_rows, best, selected = [], [], None, None
        started = time.perf_counter()
        for epoch in range(1, p["checkpoints"]+1):
            model.train()
            for step in range(p["steps_per_checkpoint"]):
                idx = sampler.next()
                update = (epoch-1)*p["steps_per_checkpoint"]+step+1
                augmented, records = augmenter.apply(idx, update)
                for record in records:
                    trace.write(json.dumps(record, sort_keys=True)+"\n")
                batch_x = tensor_input(meta, augmented)
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(model(batch_x), y[idx],
                         pos_weight=torch.tensor(p["pos_weight"]), reduction="none")
                loss = losses.mean()
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise DataError("Nonfinite loss/gradient; preserve failed run")
                optimizer.step()
                np.add.at(exposure, idx, 1)
                positive = y[idx].bool()
                pl, nl = float(losses[positive].sum().detach()), float(losses[~positive].sum().detach())
                batch_rows.append({"update": (epoch-1)*p["steps_per_checkpoint"]+step+1,
                    "n": len(idx), "positive": int(positive.sum()), "negative": int((~positive).sum()),
                    "positive_loss_sum": pl, "negative_loss_sum": nl,
                    "positive_effective_loss_fraction": pl/(pl+nl) if pl+nl else 0.,
                    "unique_positive_events": len(set(zip(data["house"][idx[positive.numpy()]].tolist(),
                                                         data["event"][idx[positive.numpy()]].tolist())))})
            scores = {k: predict_windows(meta, model, d["x"]) for k, d in blocks.items()}
            point = selection(scores, blocks, p)
            torch.save(model.state_dict(), dest / f"epoch_{epoch}.pt")
            for k, s in scores.items():
                name = k.replace("/", "_")
                np.savez_compressed(dest / f"epoch_{epoch}_{name}.npz", scores=s, t=blocks[k]["t"])
            histories.append({"epoch": epoch, "selection": point})
            if point is not None and (best is None or rank(point, epoch) < best):
                best, selected = rank(point, epoch), {"epoch": epoch, **point}
            write_json(dest / "history.json", {"rows": histories})
            print(f"sampling arm={arm} seed={seed} checkpoint={epoch}/{p['checkpoints']} eligible={point is not None}", flush=True)
        trace.close()
        write_json(dest / "augmentation_summary.json", augmenter.summary())
        # No eligible source candidate: last checkpoint at .5 is diagnostic only.
        deployable = selected is not None
        if selected is None:
            selected = {"epoch": p["checkpoints"], "threshold": .5, "limited_evidence": True}
        shutil.copyfile(dest / f"epoch_{selected['epoch']}.pt", dest / "weights.pt")
        info = {"kind": "microwave_state_classifier_v1", "truth_threshold_w": 200,
                "window_samples": 31, "features": False, **meta,
                "decision_threshold_probability": selected["threshold"],
                "weights_sha256": sha256(dest / "weights.pt"), "selected": selected,
                "source_gate_passed": deployable, "arm": arm, "seed": seed,
                "parameter_count": sum(v.numel() for v in model.parameters()),
                "updates": p["checkpoints"]*p["steps_per_checkpoint"], "seconds": time.perf_counter()-started,
                "seen": int(exposure.sum()), "positive_seen": int(exposure[data["y"] >= 200].sum()),
                "positive_weight_mass_fraction": float(50*exposure[data['y'] >= 200].sum() /
                    (50*exposure[data['y'] >= 200].sum()+exposure[data['y'] < 200].sum()))}
        write_json(dest / "model.json", info)
        save_csv(dest / "batches.csv", batch_rows)
        save_csv(dest / "exposure.csv", [{"house": HOUSES[int(data['house'][i])], "t": int(data['t'][i]),
                 "date_utc": iso(int(data['t'][i]))[:10], "event": int(data['event'][i]),
                 "positive": bool(data['y'][i] >= 200), "count": int(n)} for i, n in enumerate(exposure)])
        grouped = {}
        for i, count in enumerate(exposure):
            key = (int(data["house"][i]), iso(int(data["t"][i]))[:10], int(data["event"][i]))
            row = grouped.setdefault(key, {"house": HOUSES[key[0]], "date_utc": key[1], "event": key[2],
                "available_centers": 0, "unique_centers_seen": 0, "total_exposure": 0})
            row["available_centers"] += 1
            row["unique_centers_seen"] += int(count > 0)
            row["total_exposure"] += int(count)
        save_csv(dest / "date_event_exposure.csv", list(grouped.values()))
        write_json(dest / "complete.json", {"artifacts": inventory(dest)})
    write_json(run / "training_complete.json", {"fits": 3, "artifacts": inventory(run / "models")})
    return {"status": "trained_three_new_fits_not_evaluated"}


def check_training(run):
    receipt = read_json(run / "training_complete.json")
    check_artifacts(run / "models", receipt["artifacts"])
    if receipt["fits"] != 3:
        raise DataError("Fit budget mismatch")


def freeze(run):
    p = registered(run)
    check_training(run)
    if (run / "predictions").exists() or (run / "freeze.json").exists():
        raise DataError("Predictions already started/frozen; no overwrite")
    dest = run / "predictions"
    dest.mkdir()
    eligibility = {}
    for arm, seed, path in model_dirs(run):
        meta, model = load_state_model(path)
        eligibility[f"{arm}/{seed}"] = meta["source_gate_passed"]
        for house in HOUSES:
            d = load_npz(run / f"arrays/{house}/outer.npz")
            scores = predict_windows(meta, model, d["x"])
            np.savez_compressed(dest / f"{arm}_{seed}_{house}.npz", scores=scores, t=d["t"],
                                state=scores >= meta["decision_threshold_probability"])
    # Candidate arm preselection is source only, before outer scoring.
    arm_rows = []
    for arm in ARMS:
        infos = [read_json(path / "model.json") for a, _, path in model_dirs(run) if a == arm]
        ok = all(m["source_gate_passed"] and not m["selected"]["limited_evidence"] for m in infos)
        arm_rows.append({"arm": arm, "eligible": ok,
                         "mean_precision": float(np.mean([m["selected"].get("mean_precision", 0) for m in infos])),
                         "mean_f1": float(np.mean([m["selected"].get("mean_f1", 0) for m in infos]))})
    # Preselect event only when source mean F1 does not fall below natural.
    arm_rows[1]["mean_f1_delta_vs_natural"] = arm_rows[1]["mean_f1"]-arm_rows[0]["mean_f1"]
    arm_rows[1]["eligible"] &= arm_rows[1]["mean_f1_delta_vs_natural"] >= p["minimum_source_event_f1_delta"]
    candidate = arm_rows[2]
    candidate["f1_delta_vs_event"] = candidate["mean_f1"]-arm_rows[1]["mean_f1"]
    candidate["f1_delta_vs_natural"] = candidate["mean_f1"]-arm_rows[0]["mean_f1"]
    candidate["eligible"] &= min(candidate["f1_delta_vs_event"], candidate["f1_delta_vs_natural"]) >= 0
    eligible = [r for r in arm_rows if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["mean_precision"], r["mean_f1"], -ARMS.index(r["arm"]))) if eligible else None
    write_json(run / "source_selection.json", {"arms": arm_rows, "selected": chosen,
               "no_deployable_candidate": chosen is None})
    write_json(run / "freeze.json", {"registration_sha256": sha256(run / "registration.json"),
               "training_sha256": sha256(run / "training_complete.json"),
               "source_selection_sha256": sha256(run / "source_selection.json"),
               "predictions": inventory(dest), "count": 18, "outer_labels_opened": False})
    return {"status": "all_18_predictions_frozen", "selection": chosen}


def check_freeze(run):
    # Check this before any outer truth access or evaluation directory creation.
    f = read_json(run / "freeze.json")
    if (f["registration_sha256"] != sha256(run / "registration.json")
            or f["training_sha256"] != sha256(run / "training_complete.json")
            or f["source_selection_sha256"] != sha256(run / "source_selection.json") or f["count"] != 18):
        raise DataError("Freeze references changed artifacts")
    check_artifacts(run / "predictions", f["predictions"])


def event_metrics(scores, y, event, threshold):
    pos = y >= 200
    return {"events_supported": len(np.unique(event[pos])),
            "events_hit": len(np.unique(event[pos & (scores >= threshold)]))}


def partition_metrics(scores, y, threshold):
    """Single-class dates still report false alarms; AP/recall may be undefined."""
    if (y >= 200).any() and (y < 200).any():
        return metrics(scores, y, threshold)
    actual, predicted = y >= 200, scores >= threshold
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    fn, tn = int((actual & ~predicted).sum()), int((~actual & ~predicted).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else 0.,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "off_false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "ap": None, "prevalence": float(actual.mean()) if len(y) else None,
            "n": len(y), "threshold": float(threshold), "status": "insufficient_classes_for_pr"}


def comparison(rows, p):
    decisions = []
    for house, control in ((h, c) for h in HOUSES for c in ("event", "natural")):
        natural = [r for r in rows if r["partition"] == f"{house}/outer" and r["arm"] == control]
        event = [r for r in rows if r["partition"] == f"{house}/outer" and r["arm"] == "event_background"]
        natural.sort(key=lambda r: r["seed"])
        event.sort(key=lambda r: r["seed"])
        paired = list(zip(natural, event))
        passed = (all(r["recall"] >= .85 for r in event)
            and np.mean([r["precision"] for r in event]) >= .08
            and np.mean([b["f1"]-a["f1"] for a, b in paired]) >= p["improvement_mean_f1"]
            and sum(r["fp"] for r in event) <= p["improvement_fp_ratio"]*sum(r["fp"] for r in natural)
            and sum(b["precision"] > a["precision"] and b["f1"] > a["f1"] for a, b in paired) >= 2
            and all(b["f1"]-a["f1"] >= -p["maximum_seed_f1_drop"] for a, b in paired))
        decisions.append({"house": house, "control": control, "registered_improvement_gate": bool(passed)})
    return decisions


def evaluate(run):
    p = registered(run)
    check_training(run)
    check_freeze(run)
    import_outer_truth(run)
    dest = run / "evaluation"
    if dest.exists():
        raise DataError("Evaluation already started/completed; no overwrite")
    dest.mkdir()
    rows = []
    for arm, seed, path in model_dirs(run):
        meta, model = load_state_model(path)
        partitions = {f"{h}/train": source_array(run, h, "train") for h in HOUSES}
        partitions.update(validation_data(run))
        for h in HOUSES:
            d = load_npz(run / f"arrays/{h}/outer.npz")
            truth = load_npz(run / f"outer_labels/{h}.npz")
            np.testing.assert_array_equal(d["t"], truth["t"])
            partitions[f"{h}/outer"] = {**d, **truth, **load_npz(run / f"outer_labels/{h}_events.npz")}
        for name, d in partitions.items():
            if name.endswith("/outer"):
                pred = load_npz(run / f"predictions/{arm}_{seed}_{name.split('/')[0]}.npz")
                np.testing.assert_array_equal(pred["t"], d["t"])
                scores = pred["scores"]
            else:
                scores = predict_windows(meta, model, d["x"])
            threshold = meta["decision_threshold_probability"]
            both = bool((d["y"] >= 200).any() and (d["y"] < 200).any())
            rows.append({"arm": arm, "seed": seed, "partition": name,
                         **partition_metrics(scores, d["y"], threshold), **event_metrics(scores, d["y"], d["event"], threshold),
                         "fixed_half": partition_metrics(scores, d["y"], .5),
                         "diagnostic": diagnostic_upper(scores, d["y"]) if both else None,
                         "effective_seconds": len(scores)*8})
            if both:
                np.savez_compressed(dest / f"pr_{arm}_{seed}_{name.replace('/', '_')}.npz", **pr_curve(scores, d["y"] >= 200))
    cases = []
    for house in HOUSES:
        data = load_npz(run / f"arrays/{house}/outer.npz")
        truth = load_npz(run / f"outer_labels/{house}.npz")["y"]
        for seed in SEEDS:
            scores = {arm: load_npz(run / f"predictions/{arm}_{seed}_{house}.npz")["scores"] for arm in ARMS}
            thresholds = {arm: read_json(run / f"models/{arm}/seed_{seed}/model.json")["decision_threshold_probability"] for arm in ARMS}
            for arm in ARMS:
                chosen = []
                for i in np.argsort(-scores[arm], kind="stable"):
                    if truth[i] < 200 and scores[arm][i] >= thresholds[arm] and all(abs(int(data["t"][i])-int(data["t"][j])) >= 248 for j in chosen):
                        chosen.append(int(i))
                        cases.append({"house": house, "seed": seed, "ranked_by": arm,
                            "timestamp": iso(int(data["t"][i])), "truth_w": float(truth[i]),
                            "aggregate_window_w": data["x"][i].tolist(),
                            "scores": {a: float(scores[a][i]) for a in ARMS}, "thresholds": thresholds})
                        if len(chosen) == 3:
                            break
    write_json(dest / "paired_false_positives.json", {"selection": "Top three FP scores per arm/seed/house, separated by 248 seconds; descriptive only", "cases": cases})
    decisions = comparison(rows, p)
    source = read_json(run / "source_selection.json")
    write_json(dest / "results.json", {"rows": rows, "comparisons": decisions,
               "improvement_demonstrated": all(r["registered_improvement_gate"] for r in decisions)
                   and source["selected"] is not None and source["selected"]["arm"] == "event_background",
               "scope": "previously viewed source temporal development diagnosis, not independent generalization"})
    write_json(run / "evaluation_complete.json", {"freeze_sha256": sha256(run / "freeze.json"), "artifacts": inventory(dest)})
    return {"status": "evaluated_once", "comparisons": decisions}


def verify(run):
    p = registered(run)
    check_training(run)
    check_freeze(run)
    receipt = read_json(run / "evaluation_complete.json")
    if receipt["freeze_sha256"] != sha256(run / "freeze.json"):
        raise DataError("Evaluation/freeze mismatch")
    check_artifacts(run / "evaluation", receipt["artifacts"])
    blocks = validation_data(run)
    data = pooled(run)
    if normalization(data["x"]) != read_json(run / "normalization.json"):
        raise DataError("Normalization differs from train-only estimate")
    checked = 0
    metric_rows = read_json(run / "evaluation/results.json")["rows"]
    raw_rows = {}
    for h in HOUSES:
        with (run / f"inputs/{h}/aggregate.csv").open(encoding="utf-8", newline="") as f:
            raw_rows[h] = list(csv.DictReader(f))
    for arm, seed, path in model_dirs(run):
        meta, model = load_state_model(path)
        candidates = []
        for epoch in range(1, p["checkpoints"]+1):
            scores = {k: load_npz(path / f"epoch_{epoch}_{k.replace('/', '_')}.npz")["scores"] for k in blocks}
            point = selection(scores, blocks, p)
            if point is not None:
                candidates.append((rank(point, epoch), {"epoch": epoch, **point}))
        expected = min(candidates)[1] if candidates else {"epoch": p["checkpoints"], "threshold": .5, "limited_evidence": True}
        if expected != meta["selected"]:
            raise DataError("Source checkpoint/threshold selection mismatch")
        for h in HOUSES:
            d = load_npz(run / f"arrays/{h}/outer.npz")
            stored = load_npz(run / f"predictions/{arm}_{seed}_{h}.npz")
            scores = predict_windows(meta, model, d["x"])
            np.testing.assert_allclose(scores, stored["scores"], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(scores >= meta["decision_threshold_probability"], stored["state"])
            np.testing.assert_array_equal(d["t"], stored["t"])
            ordinary = {utc(r["timestamp"]): r for r in predict_state_rows(meta, model, raw_rows[h])}
            ordinary_scores = np.array([ordinary[int(t)]["probability_on"] for t in d["t"]], np.float32)
            np.testing.assert_allclose(ordinary_scores, stored["scores"], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal([ordinary[int(t)]["state"] for t in d["t"]], stored["state"])
            truth = load_npz(run / f"outer_labels/{h}.npz")
            row = next(r for r in metric_rows if r["arm"] == arm and r["seed"] == seed and r["partition"] == f"{h}/outer")
            for k, value in metrics(stored["scores"], truth["y"], meta["decision_threshold_probability"]).items():
                if row[k] != value:
                    raise DataError("Recomputed outer metric differs")
            checked += 1
    for seed in SEEDS:
        a = torch.load(run / f"models/natural/seed_{seed}/initial.pt", weights_only=True)
        b = torch.load(run / f"models/event/seed_{seed}/initial.pt", weights_only=True)
        if any(not torch.equal(a[k], b[k]) for k in a):
            raise DataError("Paired initialization differs")
        c = torch.load(run / f"models/event_background/seed_{seed}/initial.pt", weights_only=True)
        if any(not torch.equal(a[k], c[k]) for k in a):
            raise DataError("Background/control initialization differs")
        verify_trace(run, seed)
    return {"status": "verified", "prediction_readbacks": checked, "source_only": True}


def report(run):
    result = verify(run)
    if (run / "report.md").exists():
        raise DataError("Report exists; refuse overwrite")
    r = read_json(run / "evaluation/results.json")
    curve_rows = []
    colors = {"natural": "#64748b", "event": "#2563eb", "event_background": "#c2410c"}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="840" viewBox="0 0 1000 840">',
           '<rect width="1000" height="840" fill="white"/>',
           '<text x="25" y="24" font-size="17">Source validation at selected threshold per checkpoint</text>']
    for row_index, metric in enumerate(("worst_precision", "mean_precision", "mean_f1")):
        for column, seed in enumerate(SEEDS):
            left, top, width, height = 50+column*330, 70+row_index*255, 270, 190
            svg.append(f'<text x="{left}" y="{top-14}" font-size="13">seed {seed}: {metric}</text>')
            svg.append(f'<path d="M {left} {top} V {top+height} H {left+width}" stroke="#999" fill="none"/>')
            for arm, _, path in [v for v in model_dirs(run) if v[1] == seed]:
                points = []
                for history in read_json(path / "history.json")["rows"]:
                    selected = history["selection"]
                    if selected is None:
                        continue
                    epoch, value = history["epoch"], selected[metric]
                    points.append(f'{left+(epoch-1)*width/max(1, read_json(run / "plan.json")["checkpoints"]-1):.2f},{top+height*(1-value):.2f}')
                    curve_rows.append({"arm": arm, "seed": seed, "epoch": epoch, "metric": metric, "value": value})
                svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[arm]}" stroke-width="2"/>')
    for i, (arm, color) in enumerate(colors.items()):
        svg.append(f'<text x="{40+i*290}" y="830" fill="{color}" font-size="15">{arm}</text>')
    svg.append('</svg>')
    (run / "validation_curves.svg").write_text("\n".join(svg), encoding="utf-8")
    save_csv(run / "validation_curves.csv", curve_rows)
    text = "# 微波炉背景组合受控实验\n\n"
    text += f"3次新增拟合（复用6个对照）；改善门槛：{r['improvement_demonstrated']}。来源时间留出属于开发诊断，不能称独立泛化证明。\n\n"
    text += "|采样|种子|分区|P|R|F1|FP|AP|\n|---|---|---|---|---|---|---|---|\n"
    for row in r["rows"]:
        if row.get("ap") is not None:
            text += f"|{row['arm']}|{row['seed']}|{row['partition']}|{row['precision']:.4f}|{row['recall']:.4f}|{row['f1']:.4f}|{row['fp']}|{row['ap']:.4f}|\n"
    text += "\n完整PR、混淆计数、事后P@R85及事件覆盖见evaluation；数据缺失与支持见support.json；实际曝光与损失比例见models。未满足门槛不追加搜索，不替换旧模型。\n"
    text += "\n源内增强前后曲线见validation_curves.svg/.csv，各检查点按同一选择规则确定阈值；无合格点不绘制，不插补。\n\n"
    text += "完整训练阈值段House2/19仅14/4，House2第二内验证块完整段0；最低块支持不能替代独立事件证据。\n\n"
    for _, seed, path in candidate_dirs(run):
        summary = read_json(path / "augmentation_summary.json")
        text += f"种子{seed}：{summary['synthesized']}次实际增强，{summary['fallbacks']}次回退；正槽实际增强比例{summary['actual_fraction_of_positive_slots']:.4%}；仅涉及{summary['unique_template_threshold_segments']}个不同阈值段，重复曝光不是新事件。\n\n"
    (run / "report.md").write_text(text, encoding="utf-8")
    write_json(run / "delivery_manifest.json", {"verification": result, "artifacts": inventory(run)})
    return {**result, "status": "reported"}
