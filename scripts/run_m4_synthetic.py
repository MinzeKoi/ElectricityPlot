"""Reproducible M4 contract demo. No REFIT data or real-user evidence is used."""

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
from virtual_submeter.models.training import train
from virtual_submeter.trust.calibration import calibrate
from virtual_submeter.evaluation.stress import stress_test
from virtual_submeter.trust.verification import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    created = False
    try:
        if directory.exists():
            raise DataError("Synthetic demo destination exists")
        directory.mkdir(parents=True)
        created = True
        write_json(directory / "NOTICE.json", {"data_kind": "synthetic_contract_fixture", "real_refit": False, "real_user_study": False, "status": "running"})
        mapping = read_json(ROOT / "configs/refit_channels.json")["households"]
        inputs = {}
        for house in ("2", "3", "5"):
            source = directory / f"synthetic_house_{house}.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                fields = ["Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(480):
                    on = index % 20 >= 10
                    powers = {"kettle": 2500 if on else 0, "microwave": 400 if on else 0, "washing_machine": 40 if on else 0}
                    row = {f"Appliance{i}": 0 for i in range(1, 10)}
                    row.update({mapping[house][a][0]: power for a, power in powers.items()})
                    row.update({"Unix": index * 8, "Aggregate": sum(powers.values()) + 500, "Issues": 0})
                    writer.writerow(row)
            inputs[house] = source
        prepare(inputs, directory / "prepared", ROOT / "configs/refit_channels.json")
        plan = read_json(ROOT / "configs/m1_split.json")
        plan["max_model_context_seconds"] = 56
        write_json(directory / "plan.json", plan)
        config = replace(load_config(ROOT / "configs/default.toml"), split_gap_seconds=56, query_window_seconds=16)
        freeze(directory / "prepared", directory / "plan.json", directory / "protocol", config)
        manifest = directory / "protocol/protocol.json"
        protocol = read_json(manifest)
        protocol["data_kind"] = "synthetic_contract_fixture"
        protocol["upstream_limitations"].append("SYNTHETIC CONTRACT FIXTURE ONLY; not real REFIT measurements or a user study.")
        write_json(manifest, protocol)
        settings = read_json(ROOT / "configs/m2_baseline.json")
        settings["window_samples"] = 7
        write_json(directory / "training_settings.json", settings)
        train(manifest, directory / "training_settings.json", directory / "model", "constant_mean")
        calibrate(directory / "model", manifest, ROOT / "configs/m4_trust.json", directory / "calibration")
        stress_test(directory / "model", manifest, "frozen_test_house_3", ROOT / "configs/m4_trust.json", directory / "stress", directory / "calibration")
        verification = verify(directory / "stress")
        write_json(directory / "verification.json", verification)
        write_json(directory / "run.json", {"status": "m4_complete", "data_kind": "synthetic_contract_fixture", "stress": "stress/stress.json"})
        write_json(directory / "NOTICE.json", {"data_kind": "synthetic_contract_fixture", "real_refit": False, "real_user_study": False, "status": "complete"})
        print(f"Synthetic M4 verified: {directory / 'stress/report.html'}")
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        if created:
            write_json(directory / "NOTICE.json", {"data_kind": "synthetic_contract_fixture", "real_refit": False, "real_user_study": False, "status": "failed", "error": str(error)})
        print(f"Synthetic M4 failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
