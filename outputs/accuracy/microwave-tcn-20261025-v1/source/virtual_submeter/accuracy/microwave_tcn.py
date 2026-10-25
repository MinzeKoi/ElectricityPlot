"""Registered paired context experiment; preparation performs zero real fits."""
import csv
import hashlib
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from ..data.common import DataError, inventory, read_json, sha256, utc, write_json
from ..models.arrays import aggregate_arrays
from ..models.center_mask_tcn import CenterMaskTCN, expand_context
from .microwave_diagnosis import load_npz
from .microwave_sampling import BatchStream, check_artifacts
from .microwave_fullsource import select_checkpoint, aligned_unchanged, development_gate
ROOT = Path(__file__).resolve().parents[3]
ORDER = ["TCN31", "TCN127"]


def read_aggregate(item, house):
    path = Path(item["path"])
    if house not in ("refit_2", "refit_19", "refit_5") or sha256(path) != item["sha256"]:
        raise DataError("Invalid permitted source aggregate")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {r["household_id"] for r in rows} != {house}:
        raise DataError("Wrong household")
    return aggregate_arrays(rows, 8)


def select_source(rows, p):
    order = ["partial_mask"]+ORDER
    if set(rows) != set(order) or any([r["seed"] for r in rows[a]] != p["seeds"] for a in order):
        raise DataError("Complete paired source-only matrix required")
    old = rows["partial_mask"]
    summary = []
    for arm in order:
        rr = rows[arm]
        ok = all(r["source_gate_passed"] and not r["degenerate"] and r["recall"] >= .85 for r in rr)
        if arm != "partial_mask":
            ok &= (np.mean([r["f1"] for r in rr]) >= np.mean([r["f1"] for r in old])
                   and all(b["recall"]-a["recall"] >= -.05 for a, b in zip(old, rr))
                   and sum(b["precision"] > a["precision"] for a, b in zip(old, rr)) >= 2)
        summary.append({"arm": arm, "eligible": bool(ok),
                        "mean_precision": float(np.mean([r["precision"] for r in rr])),
                        "mean_f1": float(np.mean([r["f1"] for r in rr]))})
    candidates = [r for r in summary if r["eligible"]]
    best = max(candidates, key=lambda r: (r["mean_precision"], r["mean_f1"], -order.index(r["arm"]))) if candidates else None
    return {"selected_arm": best["arm"] if best else None, "arms": summary, "source_only": True,
            "improvement_candidate": best is not None, "no_deployable_candidate": best is None}


def prepare(run, plan_path, authorization=None):
    """Fail closed on unexpected source NPZ reads during preparation."""
    allowed = ROOT/read_json(plan_path)["mask_run"]
    npz = {str((allowed/"partial_mask.npz").resolve()).lower()}
    npz.update(str((allowed/f"validation/block_{i}.npz").resolve()).lower() for i in range(6))
    state = {"active": True, "source_npz_reads": set(), "forbidden_reads": []}
    def guard(event, args):
        if event != "open" or not state["active"] or not isinstance(args[0], (str, bytes)):
            return
        path = str(Path(args[0]).resolve()).lower()
        mode = args[1]
        if mode is not None and any(c in mode for c in "wax+"):
            return
        if mode is None:  # CPython low-level opens: ignore only explicit write/create flags.
            import os
            if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                return
        bad = ("truth_once" in path or "evaluation_labels" in path or
               (path.endswith(".csv") and any(h in path for h in ("refit_17", "house_17", "refit_8", "refit_9", "refit_20"))))
        if path.endswith(".npz"):
            local = Path(path).is_relative_to(run.resolve())
            bad |= path not in npz and not local
            state["source_npz_reads"].add(path)
        if bad:
            state["forbidden_reads"].append(path)
            raise DataError("Preparation attempted forbidden data read: "+path)
    sys.addaudithook(guard)
    try:
        return _prepare(run, plan_path, state)
    finally:
        state["active"] = False


def _prepare(run, plan_path, access):
    if run.exists():
        raise DataError("Run exists; cannot overwrite")
    p = read_json(plan_path)
    if p != read_json(ROOT/"configs/microwave_tcn_20261025.json"):
        raise DataError("Fixed plan differs")
    old = ROOT/p["mask_run"]
    reg = read_json(old/"registration.json")
    names = ["partial_mask.npz", "normalization.json", "baseline_identity.json", "baseline_settings.json", "source_identity.json"]
    names += [f"validation/block_{i}.npz" for i in range(6)]
    check_artifacts(old, {k: reg["artifacts"][k] for k in names})
    complete = read_json(old/"training_complete.json")
    if sha256(old/"source_rows.json") != complete["source_rows_sha256"]:
        raise DataError("Old source metrics changed")
    frozen = read_json(old/"freeze.json")
    if sha256(old/"source_selection.json") != frozen["source_selection_sha256"]:
        raise DataError("Old selection not frozen")
    run.mkdir(parents=True)
    shutil.copytree(old/"source", run/"source")
    for name in ["accuracy/microwave_tcn.py", "accuracy/microwave_tcn_execution.py", "models/center_mask_tcn.py"]:
        shutil.copyfile(ROOT/"src/virtual_submeter"/name, run/"source/virtual_submeter"/name)
    for name in ["run_microwave_tcn.py", "predict_microwave_tcn.py"]:
        shutil.copyfile(ROOT/"scripts"/name, run/name)
    shutil.copyfile(ROOT/"tests/test_microwave_tcn.py", run/"test_microwave_tcn.py")
    shutil.copyfile(ROOT/"docs/MICROWAVE_TCN_PLAN_20261025.md", run/"protocol.md")
    shutil.copyfile(plan_path, run/"plan.json")
    for name in names[:5]:
        shutil.copyfile(old/name, run/("TCN31.npz" if name == "partial_mask.npz" else name))
    write_json(run/"reference_source_rows.json", read_json(old/"source_rows.json"))
    # Bind old models without ever loading their weights into the new architecture.
    reference = {"run": str(old.resolve()), "registration_sha256": sha256(old/"registration.json"),
                 "freeze_sha256": sha256(old/"freeze.json"), "models": {}}
    for seed in p["seeds"]:
        path = old/f"models/partial_mask/seed_{seed}"
        meta = read_json(path/"model.json")
        if meta["kind"] != "microwave_mask_state_classifier_v1" or sha256(path/"weights.pt") != meta["weights_sha256"]:
            raise DataError("Invalid frozen old partial model")
        reference["models"][str(seed)] = {"model_json_sha256": sha256(path/"model.json"), "weights_sha256": meta["weights_sha256"]}
    write_json(run/"reference_identity.json", reference)
    write_json(run/"upstream_identity.json", {"run": str(old.resolve()), "registration_sha256": sha256(old/"registration.json")})
    core = load_npz(run/"TCN31.npz")
    if len(core["y"]) != 259806 or int((core["y"] >= 200).sum()) != 849:
        raise DataError("Fixed partial center support changed")
    identity = read_json(run/"source_identity.json")
    full = {k: v.copy() for k, v in core.items() if k not in ("x", "mask")}
    extended = []
    for hid, house in enumerate(("refit_2", "refit_19")):
        subset = core["house"] == hid
        t, a = read_aggregate(identity[house]["aggregate"], house)
        d = expand_context(t, a, core["raw_index"][subset], int(t[0]), int(t[-1])+8)
        np.testing.assert_array_equal(d["x"][:,48:79], core["x"][subset])
        np.testing.assert_array_equal(t[core["raw_index"][subset]], core["t"][subset])
        extended.append(d)
    full.update({k: np.concatenate([d[k] for d in extended]) for k in extended[0]})
    for key in core.keys()-{"x", "mask"}:
        np.testing.assert_array_equal(core[key], full[key])
    np.savez_compressed(run/"TCN127.npz", **full)
    # Frozen validation arrays provide centers/labels, source raw aggregate supplies outer context.
    upstream = ROOT/p["fullsource_run"]
    evidence = read_json(upstream/"support.json")["refit_5"]
    identity["refit_5"] = {"aggregate": evidence["inputs"]["aggregate"]}
    write_json(run/"source_identity.json", identity)
    t, a = read_aggregate(identity["refit_5"]["aggregate"], "refit_5")
    validation_identity = {}
    for i in range(6):
        key = f"block_{i}"
        d = load_npz(old/f"validation/{key}.npz")
        bounds = evidence["blocks"][key]
        ext = expand_context(t, a, d["raw_index"], utc(bounds["start"]), utc(bounds["end"]))
        np.testing.assert_array_equal(ext["x"][:,48:79], d["x"])
        np.testing.assert_array_equal(t[d["raw_index"]], d["t"])
        for arm in ORDER:
            (run/"validation"/arm).mkdir(parents=True, exist_ok=True)
            if arm == "TCN31":
                shutil.copyfile(old/f"validation/{key}.npz", run/f"validation/{arm}/{key}.npz")
            else:
                np.savez_compressed(run/f"validation/{arm}/{key}.npz", **{**d, **ext})
        validation_identity[key] = {"core_file_sha256": sha256(old/f"validation/{key}.npz"), **bounds}
    write_json(run/"validation_identity.json", validation_identity)
    positive = core["y"] >= 200
    support = {"centers": len(positive), "positive_centers": int(positive.sum()),
               "positive_centers_with_new_context": int((positive & (full["outer_observed"] > 0)).sum()),
               "centers_with_new_context": int((full["outer_observed"] > 0).sum()),
               "mean_new_observed_points": float(full["outer_observed"].mean()),
               "positive_mean_new_observed_points": float(full["outer_observed"][positive].mean()),
               "stop_counts": {side: {str(k): int(v) for k,v in zip(*np.unique(full[side+"_stop"], return_counts=True))} for side in ("left", "right")}}
    write_json(run/"context_support.json", support)
    viable = support["positive_centers_with_new_context"] > 0
    matrix = {"arms": ORDER, "seeds": p["seeds"], "fits": 6 if viable else 0,
              "preparation_passed": viable, "supported_validation_blocks": [f"block_{i}" for i in range(6)]}
    write_json(run/"matrix.json", matrix)
    replay, inits = {}, {}
    for seed in p["seeds"]:
        sampler = BatchStream(core, "event", seed, p)
        indices = np.stack([sampler.next() for _ in range(3528)])
        if not np.all((core["y"][indices] >= 200).sum(axis=1) == 16):
            raise DataError("Sampling budget changed")
        np.savez_compressed(run/f"sampling_{seed}.npz", indices=indices)
        exposure = np.bincount(indices.ravel(), minlength=len(positive))
        torch.manual_seed(seed)
        model = CenterMaskTCN()
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode()); digest.update(value.numpy().tobytes())
        inits[str(seed)] = digest.hexdigest()
        for arm in ORDER:
            replay[f"{arm}/{seed}"] = {"updates": 3528, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
            np.savez_compressed(run/f"dry_exposure_{arm}_{seed}.npz", count=exposure, t=core["t"], house=core["house"], event=core["event"])
    write_json(run/"initializations.json", inits)
    write_json(run/"dry_replay.json", {"actual_fits": 0, "matrix": replay})
    write_json(run/"runtime.json", {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__})
    write_json(run/"preparation_access.json", {"source_npz_reads": sorted(access["source_npz_reads"]),
               "forbidden_reads": access["forbidden_reads"], "guard_active_during_prepare": True})
    write_json(run/"registration.json", {"artifacts": inventory(run), "actual_fits": 0,
               "house17_labels_opened": False, "final_house_data_accessed": False,
               "execution_adapter_status": "awaiting_coordinator_message"})
    return {"status": "prepared_not_trained" if viable else "preparation_failed_zero_fits", "actual_fits": 0,
            "parameter_count": sum(v.numel() for v in model.parameters()), **support}

def verify_preparation(run):
    r = read_json(run/"registration.json")
    check_artifacts(run, r["artifacts"])
    core, full = load_npz(run/"TCN31.npz"), load_npz(run/"TCN127.npz")
    np.testing.assert_array_equal(core["x"], full["x"][:,48:79])
    np.testing.assert_array_equal(core["mask"], full["mask"][:,48:79])
    np.testing.assert_array_equal(np.isfinite(full["x"]), full["mask"])
    for key in ("t", "y", "house", "event", "raw_index"):
        np.testing.assert_array_equal(core[key], full[key])
    p = read_json(run/"plan.json")
    for seed in p["seeds"]:
        indices = load_npz(run/f"sampling_{seed}.npz")["indices"]
        stream = BatchStream(core, "event", seed, p)
        for saved in indices:
            np.testing.assert_array_equal(saved, stream.next())
        if indices.shape != (3528,256):
            raise DataError("Fixed sampling stream shape differs")
    old = Path(read_json(run/"upstream_identity.json")["run"])
    ref = read_json(run/"reference_identity.json")
    if sha256(old/"registration.json") != ref["registration_sha256"] or sha256(old/"freeze.json") != ref["freeze_sha256"]:
        raise DataError("Reference identity changed")
    for channels in read_json(run/"source_identity.json").values():
        for item in channels.values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise DataError("Original source bytes changed")
    # Bind the actual imported dependencies too, since shared live modules may evolve.
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("virtual_submeter") and path and path.endswith(".py"):
            relative = Path(*name.split("."))
            sealed = run/"source"/relative.with_suffix(".py")
            if Path(path).name == "__init__.py":
                sealed = run/"source"/relative/"__init__.py"
            if sealed.exists() and sha256(Path(path)) != sha256(sealed):
                raise DataError("Use the sealed run entry point; imported implementation changed: "+name)
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
    needed = {f"predictions/{a}_{s}.npz" for a in ["partial_mask"]+ORDER for s in m["seeds"]}
    if not needed.issubset(frozen["artifacts"]) or not frozen.get("ordinary_inference_matches") or not frozen.get("common_support_and_other_devices_match"):
        raise DataError("Missing ordinary-inference/common-support bindings")
    if select_source(read_json(run/"source_rows.json"), read_json(run/"plan.json")) != read_json(run/"source_selection.json"):
        raise DataError("Source selection does not reproduce")
    return True


def train(run, authorization=None):
    from .microwave_tcn_execution import train as execute
    return execute(run, authorization)


def freeze(run, authorization=None):
    from .microwave_tcn_execution import freeze as execute
    return execute(run)


def evaluate(run, authorization=None):
    from .microwave_tcn_execution import evaluate as execute
    return execute(run, authorization)


def benchmark(run, authorization=None):
    verify_preparation(run)
    if (run/"benchmark.json").exists():
        raise DataError("Synthetic benchmark already saved")
    p = read_json(run/"plan.json")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows = []
    for length in (31,127):
        torch.manual_seed(877)
        model = CenterMaskTCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x = torch.randn(256, 2, length)
        x[:,1] = 1
        y = torch.zeros(256); y[:16] = 1
        times = []
        for _ in range(p["synthetic_benchmark_updates"]):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(50.))
            loss.backward(); optimizer.step()
            times.append(time.perf_counter()-start)
        model.eval()
        with torch.inference_mode():
            start = time.perf_counter()
            for _ in range(8):
                model(x)
            inference = (time.perf_counter()-start)/8
        median = float(np.median(times[2:]))
        rows.append({"length": length, "synthetic_updates": len(times), "median_update_seconds": median,
                     "inference_batch_seconds": inference, "estimated_training_seconds_per_fit": median*3528,
                     "parameter_count": sum(v.numel() for v in model.parameters()),
                     "input_batch_bytes": x.numel()*4,
                     "hidden_activation_lower_bound_bytes": 256*16*length*7*4})
    # OS peak is whole process, includes Python/Torch and registration verification.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    c = Counters(); c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb)
    result = {"real_fits": 0, "synthetic_only": True, "cpu_threads": 4, "rows": rows,
              "process_peak_working_set_bytes": c.PeakWorkingSetSize,
              "six_fit_training_only_seconds": 3*sum(r["estimated_training_seconds_per_fit"] for r in rows),
              "estimate_excludes_validation_serialization_and_contention": True}
    write_json(run/"benchmark.json", result)
    return result


def report(run, authorization=None):
    result = verify(run)
    result["support"] = read_json(run/"context_support.json")
    result["benchmark"] = read_json(run/"benchmark.json") if (run/"benchmark.json").exists() else None
    result["actual_fits"] = read_json(run/"training_complete.json")["fits"] if result["trained"] else 0
    if (run/"evaluation_complete.json").exists():
        result["development"] = read_json(run/"evaluation/summary.json")
    write_json(run/"report.json", result)
    (run/"report.md").write_text("# 微波炉中心TCN受控对照\n\n离线模型使用未来上下文。默认不启用；功率沿用原A，无功率改善声明。\n\n```json\n"+
                                __import__("json").dumps(result, ensure_ascii=False, indent=2)+"\n```\n", encoding="utf-8")
    return result


def verify(run, authorization=None):
    verify_preparation(run)
    if (run/"training_complete.json").exists():
        from .microwave_tcn_execution import verify_training
        verify_training(run)
    if (run/"freeze.json").exists():
        evaluation_guard(run)
    if (run/"evaluation_complete.json").exists():
        from .microwave_tcn_execution import score_frozen, summarize
        receipt = read_json(run/"evaluation_complete.json")
        check_artifacts(run/"evaluation", receipt["artifacts"])
        if receipt["freeze_sha256"] != sha256(run/"freeze.json") or receipt["source_selection_sha256"] != sha256(run/"source_selection.json"):
            raise DataError("Evaluation dependency binding mismatch")
        truth = load_npz(run/"evaluation/truth_once.npz")
        rows, supports, power = score_frozen(run, truth["y"])
        if {"rows": rows, "support": supports, "power_baselines": power} != read_json(run/"evaluation/metrics.json"):
            raise DataError("Frozen prediction/truth metric replay differs")
        if summarize(run, rows, supports) != read_json(run/"evaluation/summary.json"):
            raise DataError("Development acceptance does not reproduce")
    return {"status": "verified", "trained": (run/"training_complete.json").exists(),
            "effect_improvement_established": False if not (run/"evaluation_complete.json").exists() else read_json(run/"evaluation/summary.json")["declared_improvement"]}


