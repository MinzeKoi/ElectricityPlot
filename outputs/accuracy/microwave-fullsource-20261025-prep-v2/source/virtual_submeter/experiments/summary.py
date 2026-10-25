"""Deterministic reports derived only from persisted evaluation artifacts."""

from collections import defaultdict
import csv
import html
import math
from pathlib import Path
import statistics

from ..data.common import DataError, read_json, write_json

METRICS = ("mae_w", "active_mae_w", "precision", "recall", "f1", "energy_absolute_error_kwh",
           "evaluation_coverage", "attributed_energy_fraction", "empirical_coverage", "interval_width_w")
NOTICE = ("State adaptation leaves power frozen. P1 uses raw power; P2 is the same prediction with "
          "scale-down correction and source empirical intervals. Null scores are not zeros. "
          "Seeds are repeated runs, not independent households. Simulated answers are not a user study. "
          "Retain negative results; this report does not automatically establish algorithm superiority.")


def stats(values):
    valid = [v for v in values if v is not None]
    return {"mean": statistics.mean(valid) if valid else None,
            "sample_std": statistics.stdev(valid) if len(valid) > 1 else None,
            "valid_count": len(valid), "missing_count": len(values) - len(valid)}


def build_summary(output):
    design = read_json(output / "frozen/design.json")
    plan = read_json(output / "frozen/experiment.json")
    protocol = read_json(output / "frozen/protocol.json")
    rows, costs, stress = [], [], []
    for case in design["cases"]:
        metric = read_json(output / "evaluation" / case["id"] / "metrics.json")
        model = read_json(output / case["model"] / "model.json")
        prediction = read_json(output / case["prediction"] / "predictions.json")
        costs.append({"case_id": case["id"], "group": case["group"], "seed": case["seed"],
                      "household_id": case["household_id"], "budget": case["budget"],
                      "query_summary": model.get("query_summary", {"budget_used": 0}),
                      "adaptation_seconds": model.get("adaptation_seconds"),
                      "inference_seconds": prediction["inference_seconds"]})
        for group in ([case["group"], "P2"] if case["group"] == "P1" else [case["group"]]):
            version = design["group_output"][group]
            for appliance, versions in metric["metrics"].items():
                selected = versions[version]
                interval = metric["intervals"][appliance][version]
                rows.append({"case_id": case["id"], "group": group, "power_version": version,
                             "household_id": case["household_id"], "seed": case["seed"], "budget": case["budget"],
                             "appliance": appliance,
                             **{k: selected[k] for k in ("mae_w", "active_mae_w", "precision", "recall", "f1", "evaluation_coverage")},
                             "energy_absolute_error_kwh": selected["energy"]["absolute_error_kwh"],
                             "attributed_energy_fraction": metric["attribution"][version]["attributed_energy_fraction"],
                             "empirical_coverage": interval["empirical_coverage"],
                             "interval_width_w": interval["mean_width_w"]})
        stress_path = output / "stress" / case["id"] / "stress.json"
        if stress_path.is_file():
            value = read_json(stress_path)
            stress.append({"case_id": case["id"], "household_id": case["household_id"], "seed": case["seed"],
                           "budget": case["budget"], "paired": value["paired"],
                           "baseline_attribution": value["baseline"]["attribution"],
                           "injected_attribution": value["injected"]["attribution"]})
    by_house = defaultdict(list)
    for row in rows:
        by_house[(row["group"], row["budget"], row["appliance"], row["household_id"])].append(row)
    household = []
    for (group, budget, appliance, house), items in sorted(by_house.items()):
        household.append({"group": group, "budget": budget, "appliance": appliance, "household_id": house,
                          "metrics": {key: stats([r[key] for r in items]) for key in METRICS}})
    grouped = defaultdict(list)
    for row in household:
        grouped[(row["group"], row["budget"], row["appliance"])].append(row)
    macro = [{"group": g, "budget": b, "appliance": a,
              "metrics": {key: stats([r["metrics"][key]["mean"] for r in items]) for key in METRICS}}
             for (g, b, a), items in sorted(grouped.items())]
    # Paired differences: P1 minus its controls, always within household/seed.
    lookup = {(r["group"], r["budget"], r["household_id"], r["seed"], r["appliance"]): r for r in rows}
    paired = []
    for row in rows:
        if row["group"] != "P1":
            continue
        for control in ("B1", "B2", "B3"):
            other = lookup[(control, 0 if control == "B1" else row["budget"],
                            row["household_id"], row["seed"], row["appliance"])]
            paired.append({"control": control, "household_id": row["household_id"], "seed": row["seed"],
                           "budget": row["budget"], "appliance": row["appliance"],
                           "P1_minus_control": {key: row[key] - other[key] if row[key] is not None and other[key] is not None else None
                                               for key in METRICS}})
    pretraining = []
    for seed in plan["seeds"]:
        for group in ("B0", "B1"):
            model = read_json(output / f"frozen/seed_{seed}/{group}/model.json")
            pretraining.append({"seed": seed, "group": group, "timing_seconds": model["timing_seconds"],
                                "parameter_count": model.get("parameter_count")})
    return {"schema_version": 1, "mode": plan["mode"], "purpose": protocol["purpose"],
            "data_kind": protocol.get("data_kind", "dataset_measurements"),
            "test_history": plan["test_history"], "test_history_note": plan["test_history_note"],
            "formal_acceptance": False, "independent_member_reproduction": "not_attested",
            "formal_acceptance_note": "Engineering evidence only until independent reproduction and protocol/history review are supplied.",
            "notice": NOTICE, "aggregation": design["aggregation"], "rows": rows, "household": household,
            "macro": macro, "paired_differences": paired, "pretraining_cost": pretraining, "case_cost": costs,
            "background_stress": stress, "resource_note": "Environment memory/CPU capacity recorded; peak process RAM not measured."}


def export_summary(output, summary):
    write_json(output / "summary.json", summary)
    rows = summary["rows"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def cell(value):
        return html.escape("—" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value))
    header = ("Group", "Budget", "Appliance", "MAE W", "Active MAE W", "F1", "Energy error kWh", "Attribution")
    table = []
    for row in summary["macro"]:
        values = [row["group"], row["budget"], row["appliance"],
                  *[row["metrics"][k]["mean"] for k in ("mae_w", "active_mae_w", "f1", "energy_absolute_error_kwh", "attributed_energy_fraction")]]
        table.append("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in values) + "</tr>")
    links = "".join(f'<li><a href="evaluation/{html.escape(c["case_id"])}/report.html">{html.escape(c["case_id"])}</a></li>'
                    for c in summary["case_cost"])
    document = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>M5 frozen experiment</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px;border:1px solid #ddd;text-align:right}}th{{background:#eef3f8}}p{{line-height:1.6}}</style>
<h1>M5 冻结实验报告 · {cell(summary['mode'])}</h1>
<p>正式验收：未完成；独立成员复跑：未证明。测试历史：{cell(summary['test_history_note'])}</p>
<p>{html.escape(NOTICE)}</p><p>{html.escape(summary['aggregation'])}</p>
<p>下表为先种子、后家庭平均；逐户标准差、缺失数、配对差值、区间和背景压力结果见
<a href="summary.json">完整 JSON</a>；<a href="summary.csv">逐次结果 CSV</a>。完整日/周期电量、质量和原始/修正指标见各次报告。</p>
<table><tr>{''.join(f'<th>{h}</th>' for h in header)}</tr>{''.join(table)}</table>
<h2>可复核运行</h2><ul>{links}</ul></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def compare_runs(left: Path, right: Path, tolerance=1e-9):
    """Compare numeric outputs and queries; does not claim a second operator."""
    from .runner import verify_experiment
    if left.resolve() == right.resolve():
        raise DataError("Reproduction requires two distinct run directories")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise DataError("Comparison tolerance must be finite and nonnegative")
    for directory in (left, right):
        result = verify_experiment(directory)
        if result["phase"] != "evaluated":
            raise DataError("Compare requires two evaluated runs")
    for name in ("protocol", "experiment", "training", "adaptation", "trust"):
        if read_json(left / f"frozen/{name}.json") != read_json(right / f"frozen/{name}.json"):
            raise DataError(f"Reproduction settings differ: {name}")
    if read_json(left / "freeze.json")["code_sha256"] != read_json(right / "freeze.json")["code_sha256"]:
        raise DataError("Reproduction source versions differ")
    differences = []
    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}/{i}")
        elif type(a) in (int, float) and type(b) in (int, float):
            if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
                differences.append(path)
        elif a != b:
            differences.append(path)
    a, b = read_json(left / "summary.json"), read_json(right / "summary.json")
    for key in ("rows", "household", "macro", "paired_differences", "background_stress"):
        compare(a[key], b[key], key)
    designs = [read_json(p / "frozen/design.json") for p in (left, right)]
    if designs[0] != designs[1]:
        raise DataError("Reproduction case matrix differs")
    for case in designs[0]["cases"]:
        predictions = [read_json(p / case["prediction"] / "predictions.json")["rows"] for p in (left, right)]
        compare(*predictions, f"predictions/{case['id']}")
        if case["group"] in ("B2", "B3", "P1"):
            queries = []
            for root in (left, right):
                records = read_json(root / case["model"] / "queries.json")["records"]
                queries.append([{k: v for k, v in r.items() if k not in ("query_id", "model_version")} for r in records])
            compare(*queries, f"queries/{case['id']}")
    return {"status": "reproduction_matches" if not differences else "reproduction_differs",
            "tolerance": tolerance, "differences": differences, "independent_member_reproduction": "not_attested",
            "notice": "Numeric rerun comparison only; operator/machine independence is not inferred."}
