"""Small explicitly synthetic A5 end-to-end fixture; no internet or real data."""

import csv
from dataclasses import replace

from ..config import load_config
from ..data.common import DataError, read_json, sha256, write_json


def run(output, repository):
    from ..data.prepare import prepare
    from ..data.protocol import freeze as split
    from ..data.final_protocol import prepare as final_protocol
    from ..models.training import train
    from . import fixed_queries as fixed
    from . import final_evaluation as final
    if output.exists():
        raise DataError("A5 synthetic destination already exists")
    output.mkdir(parents=True)
    mapping = read_json(repository / "configs/refit_channels.json")
    for house in ("4", "6", "7"):
        mapping["households"][house] = mapping["households"]["3"]
    write_json(output / "channels.json", mapping)
    paths = {}
    for house in ("2", "3", "5", "4", "6", "7"):
        path = output / "raw" / f"house_{house}.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(240):
                values = {"kettle": 2200 if (i + int(house)) % 31 < 5 else 0,
                          "microwave": 700 if (i + int(house)) % 43 < 8 else 0,
                          "washing_machine": 150 if i % 47 < 17 else 0}
                row = {f"Appliance{a}": 0 for a in range(1, 10)}
                for a, value in values.items():
                    row[mapping["households"][house][a][0]] = value
                row.update(Unix=i * 8, Aggregate=500 + sum(values.values()), Issues=0)
                writer.writerow(row)
        paths[house] = path
    sources, targets = output / "sources", output / "targets"
    prepare({h: paths[h] for h in ("2", "3", "5")}, sources, output / "channels.json")
    prepare({h: paths[h] for h in ("4", "6", "7")}, targets, output / "channels.json")
    config = replace(load_config(repository / "configs/default.toml"), split_gap_seconds=56, query_window_seconds=16)
    split_plan = read_json(repository / "configs/m1_split.json")
    split_plan["max_model_context_seconds"] = 56
    write_json(output / "split.json", split_plan)
    a1 = output / "source_fixture"
    manifest = a1 / "protocol/protocol.json"
    split(sources, output / "split.json", manifest.parent, config)
    protocol = read_json(manifest)
    protocol["accuracy_usage"] = "development_only"
    write_json(manifest, protocol)
    settings = read_json(repository / "configs/m2_baseline.json")
    settings.update(window_samples=7, channels=2, hidden_units=4, epochs=2, batch_size=32, cpu_threads=1)
    write_json(output / "model_settings.json", settings)
    model = a1 / "selected/seed_42"
    train(manifest, output / "model_settings.json", model)
    metadata = read_json(model / "model.json")
    metadata["state_threshold_view"] = "raw_power_w"
    write_json(model / "model.json", metadata)
    write_json(a1 / "baseline_freeze.json", {"status": "source_baseline_weights_frozen", "primary_exit_passed": True,
               "protocol_sha256": sha256(manifest), "notice": "Synthetic engineering fixture, not an A1 real-data quality result",
               "models": {"42": {"path": "selected/seed_42", "model_manifest_sha256": sha256(model / "model.json"),
                                   "weights_sha256": metadata["weights"]["sha256"]}}})
    learner = read_json(repository / "configs/a2_learners.json")
    learner.update(source_seeds=[42], query_seeds=[42], source_epochs=5, target_epochs=5, variants=["L2_max"])
    write_json(output / "learner.json", learner)
    a2 = output / "source_learning"
    fixed.freeze(a1, output / "learner.json", repository / "configs/m3_adaptation.json", a2)
    fixed.evaluate(a2)
    plan = read_json(repository / "configs/a5_final.json")
    plan.update(mode="engineering", target_households=["4", "6", "7"], minimum_observed_days=0.001, maximum_days=1)
    write_json(output / "plan.json", plan)
    final_manifest = output / "protocol/protocol.json"
    final_protocol(a2 / "frozen/protocol/protocol.json", targets, plan, final_manifest.parent)
    run_dir = output / "experiment"
    final.freeze(a2, final_manifest, output / "plan.json", repository / "configs/a5_queries.json", run_dir)
    return final.evaluate(run_dir)
