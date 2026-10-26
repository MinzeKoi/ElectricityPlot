"""A1 bounded source-model comparison, selection and reusable weight freeze."""

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import shutil
import sys

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition
from ..environment import environment_report
from .baseline_data import build_protocol, load_plan
from .diagnosis import write_csv
from .runner import inventory


def event_detection(rows, labels, appliance, threshold, interval):
    """Overlap with observed threshold runs; censored fragments stay separate."""
    predictions = {r["timestamp"]: r["appliances"][appliance]["raw_state"] for r in rows}
    totals = {"complete_events": 0, "complete_detected": 0, "censored_fragments": 0,
              "censored_detected": 0, "events_without_prediction_support": 0}
    active, bounded, hit, supported, previous, previous_on = False, False, False, False, None, None

    def finish(complete):
        kind = "complete" if complete else "censored"
        totals["complete_events" if complete else "censored_fragments"] += 1
        totals[kind + "_detected"] += int(hit)
        totals["events_without_prediction_support"] += int(not supported)

    for row in labels:
        stamp = utc(row["timestamp"])
        value = float(row["power_w"]) if row["power_w"] != "" else None
        contiguous = previous is not None and stamp - previous == interval
        if not contiguous or value is None:
            if active:
                finish(False)
            active, previous_on = False, None
        if value is not None:
            on = value >= threshold
            if on:
                if not active:
                    active, bounded, hit, supported = True, contiguous and previous_on is False, False, False
                state = predictions.get(row["timestamp"])
                supported |= state is not None
                hit |= state is True
            elif active:
                finish(bounded)
                active = False
            previous_on = on
        previous = stamp
    if active:
        finish(False)
    totals["complete_event_recall"] = totals["complete_detected"] / totals["complete_events"] if totals["complete_events"] else None
    totals["notice"] = "Any true-state overlap with a fixed-threshold run; not physical cycles or one-to-one event F1."
    return totals


def candidate_rank(evaluations, primary):
    selected = [v for v in evaluations if v["appliance"] in primary]
    expected = len(primary) * len({v["partition"] for v in evaluations})
    valid = len(selected) == expected and expected > 0
    for item in selected:
        m = item["raw"]
        valid &= (m["valid_samples"] > 0 and m["true_positive"] > 0
                  and 0 < m["predicted_on_samples"] < m["valid_samples"]
                  and item["events"]["complete_detected"] > 0)
    if not valid:
        return {"eligible": False, "rank": None, "reason": "primary_device_activity_or_complete_event_detection_gate_failed"}
    f1 = sum(v["raw"]["f1"] for v in selected) / len(selected)
    active = sum(v["raw"]["active_mae_w"] / v["raw"]["on_threshold_w"] for v in selected) / len(selected)
    energy = sum(v["raw"]["energy"]["absolute_error_kwh"] / (v["raw"]["energy"]["truth_kwh"] + 0.01) for v in selected) / len(selected)
    return {"eligible": True, "rank": [-f1, active, energy], "mean_primary_f1": f1,
            "mean_normalized_active_mae": active, "mean_normalized_energy_error": energy}


def prepare(sources, development, plan_path, settings_path, historical, output):
    from ..models.artifacts import load_model
    from ..models.settings import load_settings
    if output.exists():
        raise DataError("A1 output exists; choose a fresh directory")
    plan, settings = load_plan(plan_path), load_settings(settings_path)
    load_model(historical)
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes(plan_path.read_bytes())
    (output / "base_settings.json").write_bytes(settings_path.read_bytes())
    state = {"schema_version": 1, "status": "preparing", "created_at": datetime.now(timezone.utc).isoformat(),
             "plan_sha256": sha256(output / "plan.json"), "historical_model_manifest_sha256": sha256(historical / "model.json"),
             "historical_model_path": str(historical.resolve()), "environment": environment_report()}
    write_json(output / "run.json", state)
    try:
        manifest = build_protocol(sources, development, output / "protocol", plan, settings)
        for candidate in plan["candidates"]:
            path = output / "settings" / f"{candidate['id']}.json"
            write_json(path, {**settings, **{k: candidate[k] for k in ("epochs", "active_loss_weight")}, "seed": plan["seeds"][0]})
            load_settings(path)
        state.update(status="prepared", protocol_sha256=sha256(manifest),
                     prepared_artifacts_sha256={"protocol": inventory(output / "protocol"), "settings": inventory(output / "settings")})
        write_json(output / "run.json", state)
        return {"status": "a1_data_prepared", "admission": read_json(output / "protocol/admission.json")["households"]}
    except Exception as error:
        state.update(status="failed_preparation", error=str(error))
        write_json(output / "run.json", state)
        raise


def align_state(model_dir):
    metadata = read_json(model_dir / "model.json")
    metadata.update(state_threshold_view="raw_power_w",
                    state_definition="state and raw_state = raw_power_w >= fixed source threshold; consistency-corrected power is independent",
                    accuracy_usage="development_only")
    metadata["model_id"] += "-a1-raw-state"
    write_json(model_dir / "model.json", metadata)


def evaluate_model(model_dir, manifest, output):
    from ..models.prediction import predict
    from ..evaluation.runner import evaluate
    protocol = read_json(manifest)
    results = []
    for partition in protocol["partitions"]:
        if partition["role"] not in ("source_validation", "frozen_test"):
            continue
        pid = partition["partition_id"]
        directory = output / pid
        predict(model_dir, directory / "prediction", manifest, pid)
        evaluate(directory / "prediction", manifest, directory / "evaluation")
        payload = read_json(directory / "prediction/predictions.json")
        metrics = read_json(directory / "evaluation/metrics.json")["metrics"]
        labels = read_partition(manifest, pid, "evaluation", include_labels=True)
        for appliance in payload["appliances"]:
            events = event_detection(payload["rows"], labels[appliance], appliance,
                                     payload["model_settings"]["on_thresholds_w"][appliance], payload["sample_interval_seconds"])
            results.append({"partition": pid, "household_id": partition["household_id"], "role": partition["role"],
                            "appliance": appliance, **metrics[appliance], "events": events})
    return results


def train_comparison(output):
    from ..models.training import train
    state = read_json(output / "run.json")
    if state["status"] != "prepared":
        raise DataError("Only a prepared, untrained A1 run can start training")
    if sha256(output / "plan.json") != state["plan_sha256"] or sha256(output / "protocol/protocol.json") != state["protocol_sha256"]:
        raise DataError("A1 plan or protocol changed after preparation")
    for area, expected in state["prepared_artifacts_sha256"].items():
        if inventory(output / area) != expected:
            raise DataError("A1 prepared data or candidate settings changed")
    plan, manifest = load_plan(output / "plan.json"), output / "protocol/protocol.json"
    state.update(status="training", code_sha256=source_fingerprint(), cases=[])
    write_json(output / "run.json", state)
    shutil.copytree(Path(__file__).resolve().parents[1], output / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        # Water kettle first, then the identical bounded settings for all devices.
        for stage, appliances in (("single_kettle", ["kettle"]), ("multi", None)):
            for candidate in plan["candidates"]:
                cid = candidate["id"]
                relative = f"candidates/{stage}/{cid}"
                directory = output / relative
                print(f"A1 train {stage}/{cid}", file=sys.stderr, flush=True)
                train(manifest, output / "settings" / f"{cid}.json", directory / "model", appliances=appliances)
                align_state(directory / "model")
                evaluations = evaluate_model(directory / "model", manifest, directory / "results")
                score = candidate_rank(evaluations, ["kettle"] if stage == "single_kettle" else plan["primary_appliances"])
                case = {"stage": stage, "candidate": cid, "seed": plan["seeds"][0], "path": relative,
                        "selection": score, "evaluations": evaluations}
                state["cases"].append(case)
                write_json(output / "run.json", state)
        multi = [c for c in state["cases"] if c["stage"] == "multi" and c["selection"]["eligible"]]
        if not multi:
            state.update(status="no_eligible_baseline", acceptance=False)
            write_json(output / "run.json", state)
            export_report(output, state)
            return {"status": "no_eligible_baseline", "acceptance": False}
        selected = min(multi, key=lambda c: (c["selection"]["rank"], c["candidate"]))
        selection = {"status": "configuration_frozen_before_additional_seeds", "candidate": selected["candidate"],
                     "reference_seed": plan["seeds"][0], "selection_rule_id": plan["selection_rule_id"],
                     "ranking": [{"candidate": c["candidate"], "selection": c["selection"]} for c in state["cases"] if c["stage"] == "multi"],
                     "settings_sha256": sha256(output / "settings" / f"{selected['candidate']}.json"),
                     "protocol_sha256": sha256(manifest), "code_sha256": source_fingerprint(),
                     "primary_appliances": plan["primary_appliances"], "extension_appliances": plan["extension_appliances"],
                     "no_final_data_or_confirmation_queries": True}
        write_json(output / "selection.json", selection)
        selected_dir = output / "selected" / f"seed_{plan['seeds'][0]}"
        shutil.copytree(output / selected["path"] / "model", selected_dir)
        state["selected_candidate"] = selected["candidate"]
        for seed in plan["seeds"][1:]:
            settings = read_json(output / "settings" / f"{selected['candidate']}.json")
            settings["seed"] = seed
            settings_path = output / "selected_settings" / f"seed_{seed}.json"
            write_json(settings_path, settings)
            model_dir = output / "selected" / f"seed_{seed}"
            print(f"A1 selected training seed {seed}", file=sys.stderr, flush=True)
            train(manifest, settings_path, model_dir)
            align_state(model_dir)
            evaluations = evaluate_model(model_dir, manifest, output / "selected_results" / f"seed_{seed}")
            state["cases"].append({"stage": "selected_seed", "candidate": selected["candidate"], "seed": seed,
                                   "path": f"selected/seed_{seed}", "selection": candidate_rank(evaluations, plan["primary_appliances"]),
                                   "evaluations": evaluations})
            write_json(output / "run.json", state)
        # Historical reference is evaluated on the same expanded inputs, with
        # raw and historical corrected views both retained, never retrained.
        historical = Path(state["historical_model_path"])
        if sha256(historical / "model.json") != state["historical_model_manifest_sha256"]:
            raise DataError("Historical model changed")
        reference = output / "historical_model"
        shutil.copytree(historical, reference)
        metadata = read_json(reference / "model.json")
        metadata.update(parent_model_manifest_sha256=state["historical_model_manifest_sha256"],
                        historical_reference=True, protocol_sha256=sha256(manifest))
        write_json(reference / "model.json", metadata)
        evaluations = evaluate_model(reference, manifest, output / "historical_results")
        state["cases"].append({"stage": "historical", "candidate": "m2_historical", "seed": metadata["settings"]["seed"],
                               "path": "historical_model", "evaluations": evaluations})
        selected_cases = [selected] + [c for c in state["cases"] if c["stage"] == "selected_seed"]
        accepted = all(c["selection"]["eligible"] for c in selected_cases)
        freeze = {"status": "source_baseline_weights_frozen", "candidate": selected["candidate"],
                  "selection_sha256": sha256(output / "selection.json"), "protocol_sha256": sha256(manifest),
                  "models": {str(s): {"path": f"selected/seed_{s}", "model_manifest_sha256": sha256(output / "selected" / f"seed_{s}/model.json"),
                                      "weights_sha256": read_json(output / "selected" / f"seed_{s}/model.json")["weights"]["sha256"]} for s in plan["seeds"]},
                  "primary_exit_passed": accepted, "extension_requires_separate_review": True,
                  "all_confirmation_strategies_must_reuse_these_exact_weights": True,
                  "code_sha256": state["code_sha256"], "formal_evaluation": False}
        write_json(output / "baseline_freeze.json", freeze)
        state.update(status="completed" if accepted else "selected_seed_gate_failed", acceptance=accepted,
                     baseline_freeze_sha256=sha256(output / "baseline_freeze.json"))
        export_report(output, state)
        evidence = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p != output / "run.json"}
        write_json(output / "evidence.json", {"artifacts_sha256": evidence})
        state["evidence_sha256"] = sha256(output / "evidence.json")
        write_json(output / "run.json", state)
        return verify(output)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(output / "run.json", state)
        raise


def export_report(output, state):
    rows = []
    for case in state["cases"]:
        for e in case["evaluations"]:
            for view in ("raw", "corrected"):
                m = e[view]
                rows.append({"stage": case["stage"], "candidate": case["candidate"], "seed": case["seed"],
                             "partition": e["partition"], "appliance": e["appliance"], "view": view,
                             **{k: m[k] for k in ("precision", "recall", "f1", "mae_w", "active_mae_w", "valid_samples", "predicted_on_samples", "actual_on_samples")},
                             "energy_absolute_error_kwh": m["energy"]["absolute_error_kwh"],
                             "complete_events": e["events"]["complete_events"], "complete_detected": e["events"]["complete_detected"]})
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    write_json(output / "summary.json", {"status": state["status"], "acceptance": state.get("acceptance", False),
                                        "selected_candidate": state.get("selected_candidate"), "cases": state["cases"],
                                        "formal_evaluation": False, "notice": "Source expansion and development model selection, not confirmation improvement."})
    headings = list(rows[0])
    table = '<table><tr>' + ''.join(f'<th>{escape(k)}</th>' for k in headings) + '</tr>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{escape(str(row[k]))}</td>' for k in headings) + '</tr>'
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A1 基础模型比较</title>'
            '<style>body{font:14px system-ui;margin:32px;color:#172033}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}th{position:sticky;top:0;background:#eef2ff}</style>'
            f'<h1>A1 基础模型比较</h1><p>阶段验收：{state.get("acceptance", False)}；配置：{escape(str(state.get("selected_candidate")))}</p>'
            '<p>来源训练与验证、已查看 House 3 开发评价。无目标确认，无最终盲测。新模型状态统一原始功率阈值；历史模型两视图分别保留。阈值事件命中不等于物理周期 F1。</p>'
            '<p><a href="protocol/admission.json">事件与数据支持</a> · <a href="summary.json">完整结果</a> · <a href="comparison.csv">比较表</a></p>' + table + '</table></html>')
    (output / "report.html").write_text(page, encoding="utf-8")


def verify(output):
    state = read_json(output / "run.json")
    if state["status"] not in ("completed", "selected_seed_gate_failed") or sha256(output / "evidence.json") != state["evidence_sha256"]:
        raise DataError("A1 run is incomplete or evidence changed")
    for relative, digest in read_json(output / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(output, relative)) != digest:
            raise DataError("A1 artifact evidence hash mismatch")
    freeze = read_json(output / "baseline_freeze.json")
    if sha256(output / "baseline_freeze.json") != state["baseline_freeze_sha256"]:
        raise DataError("A1 baseline freeze changed")
    return {"status": "a1_artifacts_verified", "acceptance": state["acceptance"],
            "selected_candidate": state["selected_candidate"], "training_seeds": len(freeze["models"]),
            "source_code_matches_current": state["code_sha256"] == source_fingerprint(),
            "labels_reopened": False, "formal_evaluation": False}
