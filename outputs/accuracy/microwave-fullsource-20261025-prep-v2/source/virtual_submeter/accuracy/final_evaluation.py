"""A5 unseen-household evaluation with a complete, immutable matrix barrier."""

from datetime import datetime, timezone
from html import escape
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition
from ..environment import environment_report
from . import dynamic_queries as dynamic
from . import dynamic_session as sessions
from . import fixed_queries as fixed
from .diagnosis import query_summary, write_csv
from .learners import composed_rows, fit_adapter, score_adapter, temporal_features
from .runner import inventory

POLICIES = ("random", "uncertainty", "proposed")
VARIANTS = (*POLICIES, "historical_m3")
NOTICE = "理想分表代理确认；种子不是独立家庭。H1/H2 只评价冻结的状态方案，功率保持不变；H3 未在 A5 状态矩阵评价。查看历史为操作者声明，不是操作系统访问审计。"


def validate_plan(plan, require_targets=False):
    expected = {"schema_version", "mode", "target_households", "excluded_households", "primary_strategy",
                "minimum_households", "minimum_observed_days", "maximum_days", "minimum_coverage",
                "adaptation_fraction", "h1_gain", "h2_gain", "maximum_device_drop"}
    if set(plan) != expected or plan["schema_version"] != 1 or plan["mode"] not in ("final", "engineering"):
        raise DataError("Invalid A5 plan schema/mode")
    if plan["primary_strategy"] != "proposed":
        raise DataError("A5 primary strategy is frozen as proposed, including negative results")
    for key in ("target_households", "excluded_households"):
        values = plan[key]
        if (not isinstance(values, list) or any(not isinstance(h, str) or not h.isdecimal() for h in values)
                or len(set(values)) != len(values)):
            raise DataError("A5 household identities must be unique numeric strings")
    if set(plan["target_households"]) & set(plan["excluded_households"]):
        raise DataError("A5 target households were already inspected or used as sources")
    if type(plan["minimum_households"]) is not int or plan["minimum_households"] < 3:
        raise DataError("A5 needs at least three distinct households, not seeds")
    if require_targets and len(plan["target_households"]) < plan["minimum_households"]:
        raise DataError("A5 target household roster is not frozen or has fewer than three households")
    for key in ("minimum_observed_days", "maximum_days", "minimum_coverage", "adaptation_fraction", "h1_gain", "h2_gain", "maximum_device_drop"):
        if type(plan[key]) not in (int, float) or not math.isfinite(plan[key]) or plan[key] <= 0:
            raise DataError("A5 numeric gates must be finite positive numbers")
    if not 0 < plan["adaptation_fraction"] < 1 or plan["minimum_coverage"] > 1 or plan["maximum_days"] < plan["minimum_observed_days"]:
        raise DataError("Invalid A5 duration, coverage or split")
    if plan["mode"] == "final" and (plan["minimum_observed_days"] < 14 or plan["h1_gain"] != 0.05
                                   or plan["h2_gain"] != 0.02 or plan["maximum_device_drop"] != 0.02):
        raise DataError("Final A5 cannot weaken the declared duration or hypothesis gates")
    return plan


def validate_protocol(protocol, parent, plan, learner, query):
    validate_plan(plan, require_targets=True)
    if protocol.get("a5_plan") != plan:
        raise DataError("A5 protocol was not prepared with the frozen plan")
    source = [p for p in parent["partitions"] if p["role"] in ("source_train", "source_validation")]
    current = [p for p in protocol["partitions"] if p["role"] in ("source_train", "source_validation")]
    if source != current:
        raise DataError("A5 source partitions differ from frozen A2 training")
    forbidden = {p["household_id"] for p in parent["partitions"]} | set(plan["excluded_households"])
    if forbidden & set(plan["target_households"]):
        raise DataError("A5 target overlaps prior source/development scope")
    for key in ("sample_interval_seconds", "target_appliances", "query_window_seconds", "max_model_context_seconds", "split_gap_seconds"):
        if protocol[key] != parent[key]:
            raise DataError("A5 changed frozen device, context, threshold timeline or window scope")
    if (learner["budgets"] != [0, 5, 10, 20] or learner["main_budget"] != 10
            or query["learner"] != "L2_max" or query["strategies"] != list(POLICIES)
            or not set(learner["budgets"]) <= set(protocol["query_budgets"])
            or not set(learner["query_seeds"]) <= set(protocol["random_seeds"])):
        raise DataError("A5 requires frozen L2_max, three policies, main budget 10 and 0/5/10/20 prefixes")
    if plan["mode"] == "final":
        statement = protocol.get("a5_unseen_declaration") or {}
        admitted = protocol.get("a5_admission", [])
        if (protocol["purpose"] != "final_evaluation" or protocol.get("accuracy_usage") != "final_evaluation"
                or statement.get("future_labels_unseen") is not True or statement.get("target_households") != plan["target_households"]
                or not str(statement.get("declared_by", "")).strip()
                or len(learner["source_seeds"]) < 3 or len(learner["query_seeds"]) < 3
                or {a["household_id"] for a in admitted} != set(plan["target_households"])
                or any((a.get("provenance") or {}).get("kind") != "real_complete" for a in admitted)):
            raise DataError("Final A5 needs complete real files, three source/query seeds and an unseen-label declaration")
        utc(statement.get("declared_at", ""))
    elif protocol["purpose"] != "engineering_pilot_only" or protocol.get("accuracy_usage") != "engineering_only":
        raise DataError("Engineering A5 must remain explicitly non-final")
    expected = {(h, role) for h in plan["target_households"] for role in ("target_adaptation", "frozen_test")}
    targets = [p for p in protocol["partitions"] if p["role"] not in ("source_train", "source_validation")]
    if ({(p["household_id"], p["role"]) for p in targets} != expected or len(targets) != len(expected)
            or len({p["partition_id"] for p in protocol["partitions"]}) != len(protocol["partitions"])):
        raise DataError("A5 must contain exactly one adaptation/future pair per frozen household")
    interval = protocol["sample_interval_seconds"]
    for house in plan["target_households"]:
        adapt = next(p for p in targets if p["household_id"] == house and p["role"] == "target_adaptation")
        future = next(p for p in targets if p["household_id"] == house and p["role"] == "frozen_test")
        if utc(future["start"]) - utc(adapt["end"]) < max(protocol["split_gap_seconds"], protocol["max_model_context_seconds"], protocol["query_window_seconds"]):
            raise DataError("A5 adaptation and future contexts are not isolated")
        for part in (adapt, future):
            if set(part["artifacts"]) != {"aggregate", *protocol["target_appliances"]}:
                raise DataError("A5 missing channel cannot be treated as all-off")
            if utc(part["end"]) - utc(part["start"]) < protocol["query_window_seconds"]:
                raise DataError("A5 target partition too short")
            for channel, artifact in part["artifacts"].items():
                area = "inputs" if channel == "aggregate" else "oracle_labels" if part["role"] == "target_adaptation" else "evaluation_labels"
                if (Path(artifact["path"]).parts[0] != area or artifact["rows"] <= 0
                        or artifact["valid_rows"] / artifact["rows"] < plan["minimum_coverage"]
                        or artifact["valid_rows"] * interval / 86400 < plan["minimum_observed_days"] / 2):
                    raise DataError("A5 channel fails frozen label area, coverage or duration gate")


def freeze(a2, manifest, plan_path, query_path, output):
    from ..data.oracle import answer_pending
    from ..models.artifacts import load_model
    from ..models.prediction import predict
    if output.exists():
        raise DataError("A5 output exists; use a fresh destination")
    plan = validate_plan(read_json(plan_path), require_targets=True)
    receipt = dynamic.validate_parent(a2)
    parent = a2 / "frozen"
    learner = fixed.load_plan(parent / "plan.json")
    query = sessions.load_plan(query_path)
    protocol, old_protocol = read_json(manifest), read_json(parent / "protocol/protocol.json")
    validate_protocol(protocol, old_protocol, plan, learner, query)
    if protocol.get("a5_parent_protocol_sha256") != sha256(parent / "protocol/protocol.json"):
        raise DataError("A5 source protocol identity differs")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    state = {"schema_version": 1, "status": "freezing", "created_at": datetime.now(timezone.utc).isoformat(),
             "code_sha256": source_fingerprint(), "formal_evaluation": plan["mode"] == "final",
             "future_labels_opened": False, "cases": [], "environment": environment_report()}
    write_json(output / "run.json", state)
    try:
        # Copy/hash hidden label bytes; only the evaluator later parses future CSV.
        for part in protocol["partitions"]:
            for artifact in part["artifacts"].values():
                source = safe_child(manifest.parent, artifact["path"])
                if sha256(source) != artifact["sha256"]:
                    raise DataError("A5 protocol artifact changed")
                target = safe_child(frozen / "protocol", artifact["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        shutil.copyfile(manifest, frozen / "protocol/protocol.json")
        for path, name in ((plan_path, "plan.json"), (query_path, "query_plan.json"),
                           (parent / "plan.json", "learner_plan.json"), (parent / "legacy_settings.json", "legacy_settings.json"),
                           (parent / "protocol/protocol.json", "source_protocol.json")):
            shutil.copyfile(path, frozen / name)
        write_json(frozen / "parent_receipt.json", {"verification": receipt, "a2_freeze_sha256": sha256(a2 / "freeze.json"),
                   "a2_evidence_sha256": sha256(a2 / "evidence.json")})
        shutil.copytree(Path(__file__).resolve().parents[1], frozen / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        local_manifest = frozen / "protocol/protocol.json"
        legacy = read_json(frozen / "legacy_settings.json")
        for seed in learner["source_seeds"]:
            base = frozen / "bases" / f"seed_{seed}"
            shutil.copytree(parent / "bases" / f"seed_{seed}", base)
            # Transfer the frozen weights to a new protocol without claiming a refit.
            metadata = read_json(base / "model.json")
            metadata.update(training_protocol_sha256=metadata["protocol_sha256"], protocol_sha256=sha256(local_manifest),
                            a5_transferred_without_training=True)
            write_json(base / "model.json", metadata)
            metadata, model = load_model(base)
            source_dir = frozen / "source_heads" / f"seed_{seed}"
            shutil.copytree(parent / "source_heads" / f"seed_{seed}", source_dir)
            head = read_json(source_dir / "source_head.json")
            if (head["base_model_sha256"] != sha256(parent / "bases" / f"seed_{seed}" / "model.json")
                    or head["weights_sha256"] != metadata["weights"]["sha256"]
                    or head["protocol_sha256"] != sha256(parent / "protocol/protocol.json")):
                raise DataError("A5 source head/weights do not belong to frozen A2 sources")
            with np.load(source_dir / "source_replay.npz", allow_pickle=False) as saved:
                replay = {a: (saved[a + "_x"].copy(), saved[a + "_y"].copy()) for a in metadata["appliances"]}
            for house in plan["target_households"]:
                parts = {p["role"]: p for p in protocol["partitions"] if p["household_id"] == house}
                payloads, hidden = {}, {}
                for role in ("target_adaptation", "frozen_test"):
                    part = parts[role]
                    cache = frozen / "cache" / f"seed_{seed}" / part["partition_id"]
                    predict(base, cache, local_manifest, part["partition_id"])
                    payloads[role] = read_json(cache / "predictions.json")
                    aggregates = read_partition(local_manifest, part["partition_id"], "selection" if role == "target_adaptation" else "inference")["aggregate"]
                    hidden[role] = temporal_features(metadata, model, aggregates)
                pool = sessions.build_pool(payloads["target_adaptation"], parts["target_adaptation"], protocol, legacy)
                for query_seed in learner["query_seeds"]:
                    for strategy in POLICIES:
                        trajectory = f"trajectories/seed_{seed}/house_{house}/query_{query_seed}/{strategy}"
                        directory = frozen / trajectory
                        sessions.create_session(directory, local_manifest, base, parts["target_adaptation"]["partition_id"],
                                                pool, query, strategy, query_seed, max(learner["budgets"]))
                        for count in range(0, max(learner["budgets"]) + 1, query["update_every"]):
                            records = sessions.read_session(directory)["records"]
                            started = time.perf_counter()
                            adapter, history = fit_adapter(payloads["target_adaptation"]["rows"], records, metadata, learner,
                                                           "L2_max", head, hidden["target_adaptation"], replay, legacy)
                            model_dir = directory / "models" / f"budget_{count}"
                            exported = fixed.export_model(base, metadata, adapter, records, {"learner": learner, "query": query}, history,
                                                          model_dir, sha256(source_dir / "source_head.json"), strategy, "a5")
                            sessions.commit_round(directory, model_dir, score_adapter(payloads["target_adaptation"]["rows"], metadata, adapter, hidden["target_adaptation"]))
                            seconds = time.perf_counter() - started
                            if count in learner["budgets"]:
                                _save_case(frozen, state, seed, query_seed, house, strategy, count, parts["frozen_test"],
                                           trajectory, model_dir, exported, payloads["frozen_test"],
                                           score_adapter(payloads["frozen_test"]["rows"], metadata, adapter, hidden["frozen_test"]), seconds)
                                if strategy == "random":
                                    # Historical MIL is an explicit separate structure, with identical random answers.
                                    old, hist = fit_adapter(payloads["target_adaptation"]["rows"], records, metadata, learner,
                                                            "L0", head, hidden["target_adaptation"], replay, legacy)
                                    old_dir = directory / "historical_models" / f"budget_{count}"
                                    old_meta = fixed.export_model(base, metadata, old, records, {"legacy": legacy}, hist, old_dir,
                                                                 sha256(source_dir / "source_head.json"), strategy, "a5")
                                    _save_case(frozen, state, seed, query_seed, house, "historical_m3", count, parts["frozen_test"],
                                               trajectory, old_dir, old_meta, payloads["frozen_test"],
                                               score_adapter(payloads["frozen_test"]["rows"], metadata, old, hidden["frozen_test"]), None)
                            for _ in range(count, min(count + query["update_every"], max(learner["budgets"]))):
                                if sessions.select_next(directory) is None:
                                    raise DataError("A5 complete candidate pool cannot fill shared household budget")
                                sessions.confirm(directory, **answer_pending(directory))
                        print(f"A5 frozen seed={seed} house={house} query_seed={query_seed} strategy={strategy}", file=sys.stderr, flush=True)
                        write_json(output / "run.json", state)
        write_json(output / "freeze.json", {"status": "all_households_queries_models_predictions_frozen", "future_labels_opened": False,
                   "artifacts_sha256": inventory(frozen), "cases": state["cases"]})
        state.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"))
        write_json(output / "run.json", state)
        verify_frozen(output)
        return {"status": "a5_frozen", "cases": len(state["cases"]), "formal_evaluation": state["formal_evaluation"], "future_labels_opened": False}
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed_freeze", error=str(error))
        write_json(output / "run.json", state)
        raise


def _save_case(frozen, state, seed, query_seed, house, variant, budget, future, trajectory, model_dir, metadata, payload, values, seconds):
    relative = f"cases/seed_{seed}/house_{house}/query_{query_seed}/{variant}/budget_{budget}"
    path = frozen / relative
    path.mkdir(parents=True)
    np.savez_compressed(path / "states.npz", scores=values)
    case = {"path": relative, "source_seed": seed, "query_seed": query_seed, "household_id": house, "variant": variant,
            "budget": budget, "future_partition": future["partition_id"], "trajectory": trajectory,
            "model_path": model_dir.relative_to(frozen).as_posix(), "model_id": metadata["model_id"],
            "base_prediction": f"cache/seed_{seed}/{future['partition_id']}/predictions.json", "adaptation_seconds": seconds}
    write_json(path / "case.json", case)
    state["cases"].append(case)


def verify_frozen(output):
    state, barrier = read_json(output / "run.json"), read_json(output / "freeze.json")
    frozen = output / "frozen"
    if (sha256(output / "freeze.json") != state["freeze_sha256"] or barrier["future_labels_opened"] is not False
            or state["cases"] != barrier["cases"] or inventory(frozen) != barrier["artifacts_sha256"]):
        raise DataError("A5 frozen evidence changed or is incomplete")
    plan, learner, query = (read_json(frozen / name) for name in ("plan.json", "learner_plan.json", "query_plan.json"))
    protocol = read_json(frozen / "protocol/protocol.json")
    validate_protocol(protocol, read_json(frozen / "source_protocol.json"), plan, learner, query)
    if state["formal_evaluation"] != (plan["mode"] == "final"):
        raise DataError("A5 engineering run cannot be relabeled final")
    expected = {(s, q, h, v, b) for s in learner["source_seeds"] for q in learner["query_seeds"]
                for h in plan["target_households"] for v in VARIANTS for b in learner["budgets"]}
    actual = {(c["source_seed"], c["query_seed"], c["household_id"], c["variant"], c["budget"]) for c in state["cases"]}
    if actual != expected or len(actual) != len(state["cases"]):
        raise DataError("A5 full household/seed/policy/budget matrix is incomplete")
    seen, supports = {}, {}
    for case in state["cases"]:
        trajectory = case["trajectory"]
        if trajectory not in seen:
            seen[trajectory] = sessions.read_session(safe_child(frozen, trajectory))
        session = seen[trajectory]
        model_dir = safe_child(frozen, case["model_path"])
        metadata = read_json(model_dir / "model.json")
        records = read_json(model_dir / "queries.json")["records"]
        if (session["plan"] != query or len(session["records"]) != max(learner["budgets"]) or session["pending"] is not None
                or records != session["records"][:case["budget"]] or metadata["model_id"] != case["model_id"]
                or metadata["queries_sha256"] != sha256(model_dir / "queries.json")
                or metadata["accuracy_state_head"]["variant"] != ("L0" if case["variant"] == "historical_m3" else "L2_max")):
            raise DataError("A5 model differs from visible trajectory prefix")
        if case["base_prediction"] not in supports:
            payload = read_json(safe_child(frozen, case["base_prediction"]))
            supports[case["base_prediction"]] = np.asarray([[r["appliances"][a]["raw_power_w"] is not None
                                                           for a in payload["appliances"]] for r in payload["rows"]])
        support = supports[case["base_prediction"]]
        with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
            values = saved["scores"]
            if (values.shape != support.shape or not np.array_equal(np.isfinite(values), support)
                    or (values[support] < 0).any() or (values[support] > 1).any()):
                raise DataError("A5 prediction scores changed valid support or score range")
    return state, plan, learner


def summarize(metrics, plan, learner):
    """Require every expected cell; seeds -> primary devices -> equal households."""
    expected = {(s, q, h, v, b, a) for s in learner["source_seeds"] for q in learner["query_seeds"]
                for h in plan["target_households"] for v in VARIANTS for b in learner["budgets"]
                for a in learner["primary_appliances"] + learner["extension_appliances"]}
    actual = {(r["source_seed"], r["query_seed"], r["household_id"], r["variant"], r["budget"], r["appliance"]) for r in metrics}
    if actual != expected or len(actual) != len(metrics):
        raise DataError("A5 metrics have missing/duplicate household, seed, device, policy or budget cells")
    groups = []
    mean = fixed.mean_complete
    for variant in VARIANTS:
        for budget in learner["budgets"]:
            households = []
            for house in plan["target_households"]:
                devices = []
                for appliance in learner["primary_appliances"]:
                    rows = [r for r in metrics if (r["variant"], r["budget"], r["household_id"], r["appliance"]) == (variant, budget, house, appliance)]
                    devices.append({"appliance": appliance, **{k: mean([r[k] for r in rows]) for k in ("f1", "precision", "recall", "f1_delta", "f1_vs_random")},
                                    "missing_f1": sum(r["f1"] is None for r in rows),
                                    "caused_degeneracy": sum(r["adaptation_caused_degeneracy"] for r in rows)})
                households.append({"household_id": house, "devices": devices,
                                   **{k: mean([d[k] for d in devices]) for k in ("f1", "precision", "recall", "f1_delta", "f1_vs_random")}})
            drops = [{"appliance": a, "f1_delta": mean([d["f1_delta"] for h in households for d in h["devices"] if d["appliance"] == a])}
                     for a in learner["primary_appliances"]]
            groups.append({"variant": variant, "budget": budget, "households": households,
                           **{k: mean([h[k] for h in households]) for k in ("f1", "precision", "recall", "f1_delta", "f1_vs_random")},
                           "device_deltas": drops, "missing_f1": sum(d["missing_f1"] for h in households for d in h["devices"]),
                           "caused_degeneracy": sum(d["caused_degeneracy"] for h in households for d in h["devices"])})
    main = next(g for g in groups if g["variant"] == plan["primary_strategy"] and g["budget"] == learner["main_budget"])
    complete = main["f1_delta"] is not None and main["f1_vs_random"] is not None
    # Stability covers every retained nonzero budget/policy of the main structure.
    compared = [g for g in groups if g["variant"] in POLICIES and g["budget"] > 0]
    stable = not any(g["caused_degeneracy"] or any(d["f1_delta"] is None or d["f1_delta"] < -plan["maximum_device_drop"] for d in g["device_deltas"]) for g in compared)
    positive = sum(h["f1_delta"] is not None and h["f1_delta"] > 0 for h in main["households"])
    h1 = complete and stable and main["f1_delta"] >= plan["h1_gain"] and positive >= math.ceil(2 * len(plan["target_households"]) / 3)
    h2 = complete and stable and main["f1_vs_random"] >= plan["h2_gain"]
    formal = plan["mode"] == "final"
    return {"schema_version": 1, "formal_evaluation": formal, "status": "final_evaluated" if formal else "engineering_only",
            "households": plan["target_households"], "main_budget": learner["main_budget"], "primary_strategy": plan["primary_strategy"],
            "groups": groups, "primary_metrics_complete": complete, "stable": stable, "positive_households": positive,
            "h1_numeric_gate_met": h1, "h2_numeric_gate_met": h2,
            "h1_final_target_met": h1 if formal and complete else None, "h2_final_target_met": h2 if formal and complete else None,
            "h3_final_target_met": None, "independent_member_rerun": False,
            "missing_f1_total": sum(r["f1"] is None for r in metrics),
            "historical_reference": "Legacy M3 MIL learner on identical random weak-answer prefixes; its own zero baseline; not the historical M3 policy or score.",
            "notice": NOTICE}


def evaluate(output):
    from ..evaluation.metrics import appliance_metrics
    state, plan, learner = verify_frozen(output)
    if state["status"] != "frozen" or state["future_labels_opened"]:
        raise DataError("Only a frozen unevaluated A5 matrix can open future labels")
    if state["code_sha256"] != source_fingerprint():
        raise DataError("A5 evaluation code changed after freeze; run the frozen source")
    state.update(status="evaluating", future_labels_opened=True)
    write_json(output / "run.json", state)
    frozen = output / "frozen"
    try:
        data, payloads, zero, metrics = {}, {}, {}, []
        for case in state["cases"]:
            pid = case["future_partition"]
            if pid not in data:
                data[pid] = read_partition(frozen / "protocol/protocol.json", pid, "evaluation", include_labels=True)
            path = case["base_prediction"]
            if path not in payloads:
                payloads[path] = read_json(safe_child(frozen, path))
            payload = payloads[path]
            with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
                rows = composed_rows(payload["rows"], payload["appliances"], saved["scores"])
            for appliance in payload["appliances"]:
                raw = appliance_metrics(rows, data[pid][appliance], appliance, payload["model_settings"]["on_thresholds_w"][appliance], payload["sample_interval_seconds"], "raw_power_w")
                corrected = appliance_metrics(rows, data[pid][appliance], appliance, payload["model_settings"]["on_thresholds_w"][appliance], payload["sample_interval_seconds"])
                key = case["source_seed"], case["query_seed"], case["household_id"], case["variant"], appliance
                if case["budget"] == 0:
                    zero[key] = raw
                before = zero[key]
                if any(raw[k] != before[k] for k in ("valid_samples", "mae_w", "active_mae_w", "energy")):
                    raise DataError("A5 state adaptation changed power or paired valid support")
                def degenerate(item):
                    return item["valid_samples"] > 0 and item["predicted_on_samples"] in (0, item["valid_samples"])
                metrics.append({**{k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget", "adaptation_seconds")},
                                "appliance": appliance, **{k: raw[k] for k in ("f1", "precision", "recall", "valid_samples", "evaluation_coverage", "actual_on_samples")},
                                "baseline_f1": before["f1"], "f1_delta": _delta(raw["f1"], before["f1"]),
                                "predicted_on_fraction": raw["predicted_on_samples"] / raw["valid_samples"] if raw["valid_samples"] else None,
                                "adaptation_caused_degeneracy": degenerate(raw) and not degenerate(before),
                                "degenerate": degenerate(raw), "raw": raw, "corrected": corrected})
        index = {(r["source_seed"], r["query_seed"], r["household_id"], r["budget"], r["appliance"]): r for r in metrics if r["variant"] == "random"}
        for row in metrics:
            other = index[(row["source_seed"], row["query_seed"], row["household_id"], row["budget"], row["appliance"])]
            if row["variant"] in POLICIES and (row["valid_samples"] != other["valid_samples"] or row["baseline_f1"] != other["baseline_f1"]):
                raise DataError("A5 policies have different paired support or zero-confirmation baseline")
            row["f1_vs_random"] = _delta(row["f1"], other["f1"]) if row["variant"] in POLICIES else None
        summary = summarize(metrics, plan, learner)
        distributions = []
        for trajectory in sorted({c["trajectory"] for c in state["cases"]}):
            records = sessions.read_session(safe_child(frozen, trajectory))["records"]
            for budget in learner["budgets"]:
                for appliance, counts in query_summary(records[:budget], learner["primary_appliances"] + learner["extension_appliances"]).items():
                    distributions.append({"trajectory": trajectory, "budget": budget, "appliance": appliance, **counts})
        write_json(output / "metrics.json", {"rows": metrics})
        write_json(output / "summary.json", summary)
        write_json(output / "query_distribution.json", {"rows": distributions})
        fields = [k for k in metrics[0] if k not in ("raw", "corrected")]
        write_csv(output / "comparison.csv", [{k: r[k] for k in fields} for r in metrics], fields)
        render_report(output, summary)
        names = ("metrics.json", "summary.json", "query_distribution.json", "comparison.csv", "report.html")
        write_json(output / "evidence.json", {"artifacts_sha256": {n: sha256(output / n) for n in names}})
        state.update(status="evaluated", evidence_sha256=sha256(output / "evidence.json"))
        write_json(output / "run.json", state)
        return verify(output)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed_evaluation", error=str(error))
        write_json(output / "run.json", state)
        raise


def _delta(value, before):
    return value - before if value is not None and before is not None else None


def render_report(output, summary):
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    groups = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['budget']}</td><td>{number(g['f1'])}</td><td>{number(g['precision'])}</td><td>{number(g['recall'])}</td><td>{number(g['f1_delta'])}</td><td>{number(g['f1_vs_random'])}</td><td>{g['missing_f1']}</td></tr>" for g in summary["groups"])
    households = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['budget']}</td><td>{escape(h['household_id'])}</td><td>{number(h['f1'])}</td><td>{number(h['f1_delta'])}</td><td>{number(h['f1_vs_random'])}</td></tr>" for g in summary["groups"] for h in g["households"])
    title = "A5 · 冻结方案与多户评价" if summary["formal_evaluation"] else "A5 · 合成／工程多户流程验证（非真实盲测）"
    (output / "report.html").write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#152a38}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff1d1;padding:18px}}</style>
<h1>{title}</h1><p class="notice">{NOTICE}</p><p>主预算：{summary['main_budget']} 次／户；主策略固定 proposed。先平均来源／查询种子，再主设备等权，最后家庭等权。缺失值不补零，不删除失败家庭。</p>
<p>主指标完整：{summary['primary_metrics_complete']}；稳定性：{summary['stable']}；H1 最终目标：{summary['h1_final_target_met']}；H2 最终目标：{summary['h2_final_target_met']}；H3：未评价。独立成员复跑：未完成。</p>
<p>historical_m3 仅为旧 MIL 学习器在同一随机确认前缀上的结构参照，单列自身零确认；不代表复现历史 M3 选片策略或成绩。</p>
<table><tr><th>方案</th><th>预算</th><th>F1</th><th>Precision</th><th>Recall</th><th>自身零确认 ΔF1</th><th>相对随机 ΔF1</th><th>缺失 F1</th></tr>{groups}</table>
<h2>逐户配对（全部预算与负结果）</h2><table><tr><th>方案</th><th>预算</th><th>家庭</th><th>F1</th><th>自身零确认 ΔF1</th><th>相对随机 ΔF1</th></tr>{households}</table>
<p><a href="comparison.csv">逐设备／种子指标</a> · <a href="metrics.json">原始与修正功率指标</a> · <a href="query_distribution.json">回答分布</a> · <a href="summary.json">验收与逐户汇总</a></p></html>''', encoding="utf-8")


def verify(output):
    state, plan, learner = verify_frozen(output)
    if state["status"] != "evaluated" or not state["future_labels_opened"] or sha256(output / "evidence.json") != state.get("evidence_sha256"):
        raise DataError("A5 evaluation incomplete or evidence changed")
    expected = {"metrics.json", "summary.json", "query_distribution.json", "comparison.csv", "report.html"}
    artifacts = read_json(output / "evidence.json")["artifacts_sha256"]
    if set(artifacts) != expected or any(sha256(output / n) != digest for n, digest in artifacts.items()):
        raise DataError("A5 evaluation artifact changed or missing")
    summary = read_json(output / "summary.json")
    if summary != summarize(read_json(output / "metrics.json")["rows"], plan, learner):
        raise DataError("A5 summary does not reproduce from complete per-case metrics")
    return {"status": "a5_verified", "cases": len(state["cases"]), "metric_rows": len(state["cases"]) * len(learner["primary_appliances"] + learner["extension_appliances"]),
            "formal_evaluation": state["formal_evaluation"], "h1_final_target_met": summary["h1_final_target_met"],
            "h2_final_target_met": summary["h2_final_target_met"], "current_source_matches_execution": source_fingerprint() == state["code_sha256"],
            "report": str(output / "report.html")}
