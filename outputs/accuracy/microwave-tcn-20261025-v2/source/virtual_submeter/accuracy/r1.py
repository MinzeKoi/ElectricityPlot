"""R1 bounded source comparison on R0 dates; final households stay isolated."""

import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ..data.common import DataError, read_json, sha256, source_fingerprint, write_json
from .baseline import align_state, candidate_rank, evaluate_model
from .baseline_data import build_protocol, load_plan
from .runner import inventory


def evidence_inventory(output):
    return {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.relative_to(output).as_posix() not in ("run.json", "evidence.json")}


def merge_single_models(directories, output):
    """Pack independent CNNs into block diagonal layers, preserving predictions."""
    import torch
    from ..models.artifacts import load_model
    from ..models.network import SmallSeq2Point
    pairs = [load_model(d) for d in directories]
    reference = pairs[0][0]
    expanded = all(m.get("expansion_training") for m, _ in pairs)
    keys = ("input_log_mean", "input_log_std", "protocol_sha256")
    if not expanded:
        keys += ("source_train_windows", "source_validation_windows")
    if any(any(m[k] != reference[k] for k in keys) or len(m["appliances"]) != 1
           or m["settings"] != reference["settings"] for m, _ in pairs):
        raise DataError("Single-device packing requires identical inputs, normalization and settings")
    devices = [m["appliances"][0] for m, _ in pairs]
    if len(devices) != len(set(devices)) or output.exists():
        raise DataError("Duplicate packed devices or existing output")
    settings = copy.deepcopy(reference["settings"])
    count, channels, hidden = len(pairs), settings["channels"], settings["hidden_units"]
    width = settings["window_samples"]
    settings.update(channels=channels * count, hidden_units=hidden * count)
    packed = SmallSeq2Point(width, count, channels * count, hidden * count)
    with torch.no_grad():
        for p in packed.parameters():
            p.zero_()
        for index, (_, model) in enumerate(pairs):
            c, h = slice(index * channels, (index + 1) * channels), slice(index * hidden, (index + 1) * hidden)
            flat = slice(index * channels * width, (index + 1) * channels * width)
            packed.features[0].weight[c] = model.features[0].weight
            packed.features[0].bias[c] = model.features[0].bias
            packed.features[2].weight[c, c] = model.features[2].weight
            packed.features[2].bias[c] = model.features[2].bias
            packed.head[1].weight[h, flat] = model.head[1].weight
            packed.head[1].bias[h] = model.head[1].bias
            packed.head[3].weight[index, h] = model.head[3].weight[0]
            packed.head[3].bias[index] = model.head[3].bias[0]
    packed.eval()
    # A deterministic source-independent probe checks packing before publication.
    probe = torch.linspace(-2, 2, width * 17).reshape(17, width)
    with torch.inference_mode():
        expected = torch.cat([m(probe) for _, m in pairs], dim=1)
        torch.testing.assert_close(packed(probe), expected, atol=1e-6, rtol=1e-5)
    output.mkdir(parents=True)
    torch.save(packed.state_dict(), output / "weights.pt")
    metadata = copy.deepcopy(reference)
    metadata.update(appliances=devices, settings=settings,
                    power_scales_w=[m["power_scales_w"][0] for m, _ in pairs],
                    weights={"path": "weights.pt", "sha256": sha256(output / "weights.pt")},
                    single_device_components=[{"model_sha256": sha256(d / "model.json"),
                                               "appliance": m["appliances"][0], "selected_epoch": m["selected_epoch"]}
                                              for d, (m, _) in zip(directories, pairs)],
                    parameter_count=sum(p.numel() for p in packed.parameters()),
                    packed_independent_networks=True, settings_sha256=None,
                    validation_normalized_mae=sum(m["validation_normalized_mae"] for m, _ in pairs) / count,
                    selected_epoch=None)
    metadata["model_id"] = "r1-single-packed-" + metadata["weights"]["sha256"][:16]
    if expanded:
        metadata["training_households"] = sorted({h for m, _ in pairs for h in m["training_households"]})
        metadata["source_train_windows"] = {a: m["source_train_windows"] for a, (m, _) in zip(devices, pairs)}
        for component, (m, _) in zip(metadata["single_device_components"], pairs):
            component.update(training_households=m["training_households"], source_train_windows=m["source_train_windows"],
                             optimizer_steps=m["optimizer_steps"], timing_seconds=m["timing_seconds"])
        metadata["timing_seconds"] = {k: sum(m["timing_seconds"][k] for m, _ in pairs) for k in ("fit", "total")}
    write_json(output / "model.json", metadata)
    write_json(output / "settings.json", settings)
    write_json(output / "queries.json", {"records": [], "budget_used": 0})
    align_state(output)


def prepare(sources, development, plan_path, settings_path, output):
    from ..models.settings import load_settings
    plan, settings = load_plan(plan_path), load_settings(settings_path)
    assignments = {k: plan[k] for k in ("source_train_required", "source_train_optional", "source_validation", "development_households")}
    if assignments != {"source_train_required": ["2", "19"], "source_train_optional": [],
                       "source_validation": ["5"], "development_households": ["17"]}:
        raise DataError("R1 must preserve frozen R0 roles and exclude all final households")
    if [c["id"] for c in plan["candidates"]] != ["multi_mae", "single_mae", "multi_f1"]:
        raise DataError("R1 has exactly three predeclared candidates")
    if output.exists():
        raise DataError("R1 output exists; use a new directory")
    # Bind manifests to the saved R0 dates, even if a caller supplies another directory.
    root = Path(__file__).resolve().parents[3]
    date_hashes = {}
    for prepared, name in ((sources, "sources"), (development, "development")):
        selection_path = root / f"configs/r0_dates_{name}_20261022.json"
        selection = read_json(selection_path)
        audit = read_json(prepared / "manifest.json")
        for house, spec in selection["households"].items():
            info = audit["households"].get(house, {})
            if info.get("start") != spec["start"] or info.get("end") != spec["end"]:
                raise DataError("R1 input dates differ from frozen R0 selection")
        date_hashes[name] = sha256(selection_path)
    output.mkdir(parents=True)
    shutil.copyfile(plan_path, output / "plan.json")
    shutil.copyfile(settings_path, output / "base_settings.json")
    state = {"schema_version": 1, "status": "preparing", "created_at": datetime.now(timezone.utc).isoformat(),
             "formal_evaluation": False, "r0_date_hashes": date_hashes, "cases": []}
    write_json(output / "run.json", state)
    try:
        manifest = build_protocol(sources, development, output / "protocol", plan, settings)
        protocol = read_json(manifest)
        protocol.update(protocol_id="refit-r1-r0-dates-development-v1", data_kind="real_REFIT_complete_date_selected",
                        date_selection_protocol="r0-date-selected-strict8-v1", r0_date_hashes=date_hashes,
                        notice="R1 source 2/19, validation 5, inspected development 17 only; final 8/9/20 excluded.")
        write_json(manifest, protocol)
        for candidate in plan["candidates"]:
            write_json(output / f"settings/{candidate['id']}.json",
                       {**settings, "epochs": candidate["epochs"], "active_loss_weight": candidate["active_loss_weight"], "seed": plan["seeds"][0]})
        state.update(status="prepared", prepared_sha256=inventory(output / "protocol"),
                     plan_sha256=sha256(output / "plan.json"), settings_sha256=inventory(output / "settings"))
        write_json(output / "run.json", state)
    except Exception as error:
        state.update(status="failed_preparation", error=str(error))
        write_json(output / "run.json", state)
        raise
    return {"status": "r1_prepared", "admission": read_json(output / "protocol/admission.json")}


def fit_candidate(output, cid, seed, destination):
    from ..models.training import train
    settings = read_json(output / f"settings/{cid}.json")
    settings["seed"] = seed
    settings_path = destination.parent / (destination.name + "_settings.json")
    write_json(settings_path, settings)
    manifest = output / "protocol/protocol.json"
    if cid == "single_mae":
        components = []
        for device in read_json(manifest)["target_appliances"]:
            path = destination.parent / (destination.name + "_components") / device
            train(manifest, settings_path, path, appliances=[device])
            components.append(path)
        merge_single_models(components, destination)
    else:
        train(manifest, settings_path, destination, checkpoint_metric="primary_f1" if cid == "multi_f1" else "normalized_mae")
        align_state(destination)


def run_models(output):
    plan, state = load_plan(output / "plan.json"), read_json(output / "run.json")
    if (state["status"] != "prepared" or inventory(output / "protocol") != state["prepared_sha256"]
            or sha256(output / "plan.json") != state["plan_sha256"] or inventory(output / "settings") != state["settings_sha256"]):
        raise DataError("R1 preparation is incomplete or changed")
    state.update(status="training", code_sha256=source_fingerprint())
    write_json(output / "run.json", state)
    shutil.copytree(Path(__file__).resolve().parents[1], output / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        for candidate in plan["candidates"]:
            cid, seed = candidate["id"], plan["seeds"][0]
            path = output / f"candidates/{cid}"
            fit_candidate(output, cid, seed, path / "model")
            results = evaluate_model(path / "model", output / "protocol/protocol.json", path / "results")
            state["cases"].append({"candidate": cid, "seed": seed, "evaluations": results,
                                   "selection": candidate_rank(results, plan["primary_appliances"])})
            write_json(output / "run.json", state)
        eligible = [c for c in state["cases"] if c["selection"]["eligible"]]
        if not eligible:
            state.update(status="no_eligible_baseline", acceptance=False)
            write_json(output / "run.json", state)
            return state
        selected = min(eligible, key=lambda c: (c["selection"]["rank"], c["candidate"]))
        write_json(output / "selection.json", {"selected": selected["candidate"], "rule": plan["selection_rule"],
                                              "candidates": state["cases"], "frozen_before_additional_seeds": True})
        for seed in plan["seeds"]:
            target = output / f"selected/seed_{seed}"
            if seed == plan["seeds"][0]:
                shutil.copytree(output / f"candidates/{selected['candidate']}/model", target)
            else:
                fit_candidate(output, selected["candidate"], seed, target)
                results = evaluate_model(target, output / "protocol/protocol.json", output / f"selected_results/seed_{seed}")
                state["cases"].append({"candidate": selected["candidate"], "seed": seed, "evaluations": results,
                                       "selection": candidate_rank(results, plan["primary_appliances"])})
                write_json(output / "run.json", state)
        selected_cases = [selected] + [c for c in state["cases"] if c["seed"] != plan["seeds"][0]]
        accepted = all(c["selection"]["eligible"] for c in selected_cases)
        write_json(output / "baseline_freeze.json", {
            "status": "source_baseline_weights_frozen", "candidate": selected["candidate"],
            "protocol_sha256": sha256(output / "protocol/protocol.json"), "selection_sha256": sha256(output / "selection.json"),
            "models": {str(s): {"path": f"selected/seed_{s}", "model_manifest_sha256": sha256(output / f"selected/seed_{s}/model.json"),
                                "weights_sha256": sha256(output / f"selected/seed_{s}/weights.pt")} for s in plan["seeds"]},
            "primary_exit_passed": accepted, "formal_evaluation": False, "code_sha256": state["code_sha256"]})
        state.update(status="completed" if accepted else "selected_seed_gate_failed", acceptance=accepted,
                     selected_candidate=selected["candidate"])
        write_json(output / "summary.json", state)
        write_json(output / "evidence.json", {"artifacts_sha256": evidence_inventory(output)})
        state["evidence_sha256"] = sha256(output / "evidence.json")
        write_json(output / "run.json", state)
        return {"status": state["status"], "acceptance": accepted, "candidate": selected["candidate"]}
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(output / "run.json", state)
        raise


def verify(output):
    state = read_json(output / "run.json")
    if state["status"] not in ("completed", "selected_seed_gate_failed") or sha256(output / "evidence.json") != state["evidence_sha256"]:
        raise DataError("R1 incomplete or evidence changed")
    if evidence_inventory(output) != read_json(output / "evidence.json")["artifacts_sha256"]:
        raise DataError("R1 artifact hash mismatch")
    return {"status": "r1_verified", "acceptance": state["acceptance"], "source_code_matches_current": source_fingerprint() == state["code_sha256"],
            "formal_evaluation": False, "labels_reopened": False}
