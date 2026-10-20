"""Three-target-home M5 fixture demo; never real REFIT or final evidence."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.config import load_config
from virtual_submeter.data.common import DataError, read_json, write_json
from virtual_submeter.data.prepare import prepare
from virtual_submeter.data.protocol import freeze
from virtual_submeter.experiments.runner import freeze_experiment, evaluate_experiment, verify_experiment


def make_fixture(directory):
    if directory.exists():
        raise DataError("Fixture directory exists")
    directory.mkdir(parents=True)
    write_json(directory / "NOTICE.json", {"data_kind": "synthetic_contract_fixture", "real_refit": False})
    mapping = read_json(ROOT / "configs/refit_channels.json")["households"]
    inputs = {}
    for house in ("2", "3", "4", "5", "11"):
        source = directory / f"synthetic_{house}.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            fields = ["Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(480):
                on = (index + int(house)) % 20 >= 10
                powers = {"kettle": 2500 if on else 0, "microwave": 400 if on else 0,
                          "washing_machine": 40 if on else 0}
                row = {f"Appliance{i}": 0 for i in range(1, 10)}
                row.update({mapping[house][a][0]: value for a, value in powers.items()})
                row.update(Unix=index * 8, Aggregate=sum(powers.values()) + 300 + 30 * int(house), Issues=0)
                writer.writerow(row)
        inputs[house] = source
    prepare(inputs, directory / "prepared", ROOT / "configs/refit_channels.json")
    plan = read_json(ROOT / "configs/m1_split.json")
    plan.update(max_model_context_seconds=56, target_households=["3", "4", "11"])
    write_json(directory / "split.json", plan)
    config = replace(load_config(ROOT / "configs/default.toml"), split_gap_seconds=56, query_window_seconds=16)
    freeze(directory / "prepared", directory / "split.json", directory / "protocol", config)
    manifest = directory / "protocol/protocol.json"
    protocol = read_json(manifest)
    protocol["data_kind"] = "synthetic_contract_fixture"
    protocol["upstream_limitations"].append("SYNTHETIC CONTRACT FIXTURE ONLY.")
    write_json(manifest, protocol)
    training = read_json(ROOT / "configs/m2_baseline.json")
    training["window_samples"] = 7
    write_json(directory / "training.json", training)
    experiment = read_json(ROOT / "configs/m5_experiment.json")
    experiment.update(backbone="constant_mean", test_history="never_evaluated",
                      test_history_note="New synthetic contract fixture; not real performance evidence.")
    write_json(directory / "experiment.json", experiment)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = make_fixture(args.output)
        freeze_experiment(manifest, args.output / "experiment.json", args.output / "training.json",
                          ROOT / "configs/m3_adaptation.json", ROOT / "configs/m4_trust.json", args.output / "run")
        evaluate_experiment(args.output / "run")
        print(verify_experiment(args.output / "run"))
        return 0
    except (DataError, OSError, RuntimeError, ImportError) as error:
        print(f"M5 synthetic demo failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
