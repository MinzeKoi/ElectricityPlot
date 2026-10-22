"""Synthetic fixtures test quality, energy, provenance and actual leakage boundaries."""

from collections import Counter
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.cli import main
from virtual_submeter.config import load_config
from virtual_submeter.data.common import DataError, iso, read_json, safe_child, utc, write_json
from virtual_submeter.data.prepare import Sample, prepare, read_samples, resample
from virtual_submeter.data.protocol import freeze, read_partition, validate_confirmation, validate_confirmation_history


class ResamplingTest(unittest.TestCase):
    def bins(self, points):
        samples = [Sample(t, {"aggregate": power}, {"aggregate": set()}) for t, power in points]
        return list(resample(samples, 8, {"raw_intervals_seconds": Counter()}))

    def test_time_weighted_energy_and_no_final_extrapolation(self):
        bins = self.bins([(0, 100), (2, 300), (8, 900)])
        self.assertEqual(len(bins), 1)
        self.assertEqual(bins[0][1]["aggregate"]["power_w"], 250)
        self.assertAlmostEqual(bins[0][1]["aggregate"]["power_w"] * 8, 100 * 2 + 300 * 6)

    def test_long_gap_is_null_instead_of_off_or_constant_energy(self):
        bins = self.bins([(0, 100), (24, 200), (32, 0)])
        self.assertEqual([item[1]["aggregate"]["power_w"] for item in bins], [100, None, None, 200])
        self.assertEqual(bins[1][1]["aggregate"]["coverage"], 0)

    def test_partial_boundary_and_missing_measurement_are_not_zero(self):
        bins = self.bins([(2, 100), (8, None), (16, 0)])
        self.assertIsNone(bins[0][1]["aggregate"]["power_w"])
        self.assertEqual(bins[0][1]["aggregate"]["coverage"], 0.75)
        self.assertIsNone(bins[1][1]["aggregate"]["power_w"])

    def test_timezone_is_required_and_dst_offsets_are_unambiguous(self):
        with self.assertRaises(DataError):
            utc("2024-10-27T01:30:00")
        self.assertEqual(utc("2024-10-27T01:30:00+01:00") + 3600, utc("2024-10-27T01:30:00+00:00"))


class DataProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.catalog = ROOT / "configs/refit_channels.json"
        self.inputs = {}
        for house in ("2", "3", "5"):
            source = self.directory / f"CLEAN_House{house}.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Time", "Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"])
                for timestamp in range(0, 512, 8):
                    writer.writerow(["ignored_local_time", timestamp, 3000, *[i * 10 for i in range(1, 10)], 0])
            self.inputs[house] = source
        self.prepared = self.directory / "prepared"
        self.protocol = self.directory / "frozen"
        self.plan_path = self.directory / "plan.json"
        self.plan = read_json(ROOT / "configs/m1_split.json")
        self.plan["max_model_context_seconds"] = 16
        self.config = replace(load_config(ROOT / "configs/default.toml"), split_gap_seconds=16, query_window_seconds=16)
        self.write_plan()

    def write_plan(self):
        write_json(self.plan_path, self.plan)

    def prepare(self):
        return prepare(self.inputs, self.prepared, self.catalog)

    def freeze(self):
        self.prepare()
        return freeze(self.prepared, self.plan_path, self.protocol, self.config)

    def confirmation(self):
        return {"query_id": "q1", "household_id": "3", "window_start": iso(0), "window_end": iso(16),
                "appliance_type": "kettle", "response": "used", "label_source": "simulated_from_submeter",
                "selection_strategy": "random", "model_version": "synthetic-test-only", "budget_before": 0, "budget_after": 1}

    def test_realistic_import_and_role_specific_access(self):
        self.freeze()
        manifest_path = self.protocol / "protocol.json"
        manifest = read_json(manifest_path)
        from virtual_submeter.data.common import sha256
        self.assertEqual(sha256(self.protocol / "plan.json"), manifest["plan_sha256"])
        self.assertEqual(len(manifest["partitions"]), 4)
        source = read_partition(manifest_path, "source_train_house_2", "training", True)
        self.assertEqual(set(source), {"aggregate", "kettle", "microwave", "washing_machine"})
        target = read_partition(manifest_path, "target_adaptation_house_3", "adaptation")
        self.assertEqual(set(target), {"aggregate"})
        prediction = read_partition(manifest_path, "frozen_test_house_3", "inference")
        evaluation = read_partition(manifest_path, "frozen_test_house_3", "evaluation", True)
        self.assertEqual(set(prediction), {"aggregate"})
        self.assertIn("kettle", evaluation)
        adapt, test = manifest["partitions"][2:]
        self.assertEqual(utc(test["start"]) - utc(adapt["end"]), 16)
        for row in target["aggregate"]:
            self.assertLess(utc(row["timestamp"]), utc(adapt["end"]))
        for row in prediction["aggregate"]:
            self.assertGreaterEqual(utc(row["timestamp"]), utc(test["start"]))

    def test_bounded_preparation_keeps_full_source_identity_and_exact_day_boundary(self):
        from virtual_submeter.data.common import sha256
        source = self.inputs["2"]
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Time", "Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"])
            for timestamp in range(0, 86400 + 24, 8):
                writer.writerow(["ignored", timestamp, 3000, *([10] * 9), 0])
        fingerprint = sha256(source)
        receipt = {"kind": "real_complete", "sha256": fingerprint}
        write_json(source.with_suffix(".receipt.json"), receipt)
        prepare({"2": source}, self.prepared, self.catalog, maximum_days=1)
        info = read_json(self.prepared / "manifest.json")["households"]["2"]
        self.assertEqual(info["bins"], 10800)
        self.assertEqual(info["end"], iso(86400))
        self.assertEqual(info["source_sha256"], fingerprint)
        self.assertEqual(info["provenance"], receipt)
        self.assertEqual(info["preparation_selection"]["maximum_days"], 1)
        self.assertIn("raw_rows", info)
        with (self.prepared / "aggregate/house_2.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]["timestamp"], iso(86392))
        self.assertEqual(len(rows), 10800)

    def test_bounded_preparation_rejects_nonpositive_and_fractional_days(self):
        for days in (0, -1, True, 0.5):
            with self.subTest(days=days), self.assertRaises(DataError):
                prepare(self.inputs, self.prepared, self.catalog, maximum_days=days)

    def date_selection(self, start=16, end=64):
        from virtual_submeter.data.common import sha256
        selection = self.directory / "selection.json"
        write_json(selection, {"schema_version": 1, "protocol_id": "r0-selected-dates-test",
                               "catalog_sha256": sha256(self.catalog), "households": {
                                   "2": {"start": iso(start), "end": iso(end), "source_sha256": sha256(self.inputs["2"])}}})
        return selection

    def test_selected_dates_keep_hashes_and_hide_activity_quality(self):
        selection = self.date_selection()
        prepare({"2": self.inputs["2"]}, self.prepared, self.catalog, selection=selection, support_only=True)
        audit = read_json(self.prepared / "manifest.json")
        house = audit["households"]["2"]
        self.assertEqual((house["start"], house["end"], house["bins"]), (iso(16), iso(64), 6))
        self.assertEqual(house["preparation_selection"]["kind"], "predeclared_continuous_dates")
        for summary in house["channels"].values():
            self.assertIsNone(summary["nonzero_bins"])
            self.assertIsNone(summary["observed_energy_kwh"])
        self.assertIn("未计算", (self.prepared / "quality.html").read_text(encoding="utf-8"))

    def test_date_selection_rejects_wrong_hash_boundaries_or_incomplete_source(self):
        from virtual_submeter.data.common import sha256
        for first, last in ((17, 64), (16, 16), (16, 1024)):
            with self.subTest(first=first, last=last), self.assertRaises(DataError):
                prepare({"2": self.inputs["2"]}, self.prepared, self.catalog, selection=self.date_selection(first, last))
        selected = self.date_selection()
        item = read_json(selected)
        item["households"]["2"]["source_sha256"] = "0" * 64
        write_json(selected, item)
        with self.assertRaises(DataError):
            prepare({"2": self.inputs["2"]}, self.prepared, self.catalog, selection=selected)
        with self.assertRaises(DataError):
            prepare({"2": self.inputs["2"]}, self.prepared, self.catalog, selection=self.date_selection(), maximum_days=1)

    def test_selection_preserves_original_hold_age_across_start_boundary(self):
        from virtual_submeter.data.prepare import selected_samples
        points = [Sample(t, {"aggregate": 100.0}, {"aggregate": set()}) for t in (0, 10, 16, 24, 32)]
        actual = list(resample(selected_samples(iter(points), 8, 24), 8, {"raw_intervals_seconds": Counter()}))
        inside = {t: channels["aggregate"]["power_w"] for t, channels in actual if 8 <= t < 24}
        self.assertIsNone(inside[8])
        self.assertEqual(inside[16], 100.0)

    def test_test_answers_and_oracle_labels_are_denied_to_model_consumers(self):
        self.freeze()
        for partition, consumer, labels in (
            ("frozen_test_house_3", "selection", False),
            ("frozen_test_house_3", "training", True),
            ("frozen_test_house_3", "inference", True),
            ("target_adaptation_house_3", "adaptation", True),
            ("target_adaptation_house_3", "selection", True),
            ("target_adaptation_house_3", "evaluation", True),
        ):
            with self.subTest(consumer=consumer, partition=partition), self.assertRaises(DataError):
                read_partition(self.protocol / "protocol.json", partition, consumer, labels)

    def test_target_aggregate_is_independent_of_submeter_answers_and_issues(self):
        self.prepare()
        original = (self.prepared / "aggregate/house_3.csv").read_bytes()
        source = self.inputs["3"]
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["Issues"] = "1"
            for key in row:
                if key.startswith("Appliance"):
                    row[key] = "999"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        other = self.directory / "other"
        prepare({"3": source}, other, self.catalog)
        self.assertEqual(original, (other / "aggregate/house_3.csv").read_bytes())

    def test_canonical_aggregate_import_keeps_nulls_and_string_household_ids(self):
        fixture = ROOT / "tests/fixtures/synthetic_aggregate.csv"
        prepare({"demo_home": fixture}, self.prepared, self.catalog, kind="aggregate")
        audit = read_json(self.prepared / "manifest.json")
        self.assertEqual(audit["dataset"], "USER_AGGREGATE")
        self.assertEqual(audit["households"]["demo_home"]["channel_mapping"], {})
        self.assertNotIn("labels", [p.name for p in self.prepared.iterdir()])
        with (self.prepared / "aggregate/house_demo_home.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[2]["aggregate_power_w"], "")
        self.assertIn("synthetic_fixture", rows[0]["quality_flags"])

    def test_duplicate_and_decreasing_timestamps_leave_no_success_artifacts(self):
        for last in (496, 488):
            source = self.inputs["2"]
            lines = source.read_text().splitlines()
            fields = lines[-1].split(",")
            fields[1] = str(last)
            lines[-1] = ",".join(fields)
            source.write_text("\n".join(lines) + "\n")
            with self.subTest(last=last), self.assertRaises(DataError):
                self.prepare()
            self.assertFalse(self.prepared.exists())

    def test_missing_and_invalid_power_remain_null_and_source_is_preserved(self):
        source = self.inputs["2"]
        text = source.read_text().replace(",3000,", ",nan,", 1).replace(",3000,", ",-2,", 1).replace(",3000,", ",inf,", 1)
        source.write_text(text)
        self.prepare()
        audit = read_json(self.prepared / "manifest.json")
        stats = audit["households"]["2"]["channels"]["aggregate"]
        self.assertEqual(stats["valid_bins"], 60)
        self.assertEqual(source.read_text(), text)
        with (self.prepared / "aggregate/house_2.csv").open() as stream:
            self.assertEqual([r["aggregate_power_w"] for r in list(csv.DictReader(stream))[:3]], ["", "", ""])

    def test_source_issues_excluded_and_multiple_washers_explicitly_summed(self):
        source = self.inputs["2"]
        rows = source.read_text().splitlines()
        rows[1] = rows[1][:-1] + "1"
        source.write_text("\n".join(rows) + "\n")
        samples = list(read_samples(source, "4", read_json(self.catalog)["households"]["4"], "refit"))
        self.assertEqual(samples[0].values["aggregate"], 3000)
        self.assertNotIn("upstream_aggregate_discrepancy", samples[0].flags["aggregate"])
        self.assertIsNone(samples[0].values["kettle"])
        self.assertEqual(samples[1].values["washing_machine"], 90)
        self.assertIn("multiple_meters_summed", samples[1].flags["washing_machine"])

    def test_missing_appliance_channel_is_not_treated_as_off(self):
        self.prepare()
        self.config = replace(self.config, target_appliances=("kettle", "unknown_appliance"))
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)
        self.assertFalse(self.protocol.exists())

    def test_household_overlap_and_insufficient_isolation_are_rejected(self):
        self.prepare()
        self.plan["target_households"] = ["2"]
        self.write_plan()
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)
        self.plan["target_households"] = ["3"]
        self.plan["max_model_context_seconds"] = 24
        self.write_plan()
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)

    def test_insufficient_duration_or_coverage_cannot_be_frozen(self):
        self.prepare()
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, replace(self.config, split_gap_seconds=3600))
        path = self.prepared / "aggregate/house_3.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["aggregate_power_w"] = ""
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        # Updating a declared hash simulates an audited, legitimately sparse input.
        from virtual_submeter.data.common import sha256
        audit = read_json(self.prepared / "manifest.json")
        audit["households"]["3"]["artifacts"]["aggregate"]["sha256"] = sha256(path)
        write_json(self.prepared / "manifest.json", audit)
        with self.assertRaisesRegex(DataError, "coverage"):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)
        self.assertFalse(self.protocol.exists())

    def test_data_mutation_is_detected_before_freeze_and_on_partition_read(self):
        self.freeze()
        path = self.protocol / "inputs/target_adaptation_house_3/aggregate.csv"
        path.write_text(path.read_text().replace("3000", "9999"))
        with self.assertRaisesRegex(DataError, "hash"):
            read_partition(self.protocol / "protocol.json", "target_adaptation_house_3", "selection")
        path = self.prepared / "aggregate/house_2.csv"
        path.write_text(path.read_text().replace("3000", "9999"))
        with self.assertRaisesRegex(DataError, "hash"):
            freeze(self.prepared, self.plan_path, self.directory / "different_protocol", self.config)

    def test_existing_dataset_and_protocol_cannot_be_replaced(self):
        self.freeze()
        with self.assertRaises(DataError):
            self.prepare()
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)

    def test_unverified_input_cannot_be_claimed_as_final_evaluation(self):
        self.prepare()
        self.plan["purpose"] = "final_evaluation"
        self.write_plan()
        with self.assertRaises(DataError):
            freeze(self.prepared, self.plan_path, self.protocol, self.config)

    def test_weak_labels_must_be_budgeted_windows_inside_adaptation(self):
        self.freeze()
        manifest = read_json(self.protocol / "protocol.json")
        record = self.confirmation()
        validate_confirmation(record, manifest)
        for change in ({"budget_after": 0}, {"budget_before": True}, {"window_start": iso(280), "window_end": iso(296)}, {"window_end": iso(8)}, {"power_w": 123}, {"response": "on_every_point"}):
            with self.subTest(change=change), self.assertRaises(DataError):
                validate_confirmation({**record, **change}, manifest)
        for response in ("unsure", "skipped"):
            with self.assertRaises(DataError):
                validate_confirmation({**record, "response": response, "budget_after": 0}, manifest)

    def test_budget_history_is_per_house_and_cannot_reuse_ids_or_hide_answers(self):
        self.freeze()
        manifest = read_json(self.protocol / "protocol.json")
        record = self.confirmation()
        with self.assertRaises(DataError):
            validate_confirmation_history([record, record], manifest)
        with self.assertRaises(DataError):
            validate_confirmation_history([record, {**record, "query_id": "q2"}], manifest)
        validate_confirmation_history([record, {**record, "query_id": "q2", "budget_before": 1, "budget_after": 2}], manifest)

    def test_manifest_paths_cannot_escape_and_import_requires_units(self):
        with self.assertRaises(DataError):
            safe_child(self.directory, "../secret.csv")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["prepare-aggregate", "--config", str(ROOT / "configs/default.toml"), "--json"])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "data_operation_failed")


if __name__ == "__main__":
    unittest.main()
