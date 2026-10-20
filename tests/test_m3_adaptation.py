"""Budget, leakage, MIL semantics and end-to-end M3 regression coverage."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.data.protocol import read_partition

try:
    import numpy as np
    NUMERIC = True
except ImportError:
    NUMERIC = False

import test_m2_baseline as baseline_tests


class ArtifactPublicationTest(unittest.TestCase):
    def test_windows_transient_rename_is_retried_without_partial_copy(self):
        from virtual_submeter.data.common import publish_directory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "stage"
            staging.mkdir()
            output = root / "output"
            rename = Path.rename
            calls = []

            def intermittent(path, target):
                calls.append(target)
                if len(calls) == 1:
                    raise PermissionError("transient sharing violation")
                return rename(path, target)

            with patch("virtual_submeter.data.common.sys.platform", "win32"), patch("virtual_submeter.data.common.time.sleep"), patch.object(Path, "rename", intermittent):
                publish_directory(staging, output)
            self.assertEqual(len(calls), 2)
            self.assertTrue(output.is_dir())
            self.assertFalse(staging.exists())
            staging.mkdir()
            with self.assertRaises(DataError):
                publish_directory(staging, output)

    def test_permanent_failure_leaves_staging_and_rejects_external_staging(self):
        from virtual_submeter.data.common import publish_directory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "stage"
            staging.mkdir()
            output = root / "output"
            with patch("virtual_submeter.data.common.sys.platform", "win32"), patch("virtual_submeter.data.common.time.sleep"), patch.object(Path, "rename", side_effect=PermissionError("permanent")) as rename:
                with self.assertRaises(PermissionError):
                    publish_directory(staging, output)
            self.assertEqual(rename.call_count, 6)
            self.assertTrue(staging.exists())
            self.assertFalse(output.exists())
            with self.assertRaises(DataError):
                publish_directory(staging, root / "different_parent/output")


@unittest.skipUnless(NUMERIC, "M3 state adaptation requires NumPy")
class AdaptationTest(unittest.TestCase):
    def setUp(self):
        # The M2 synthetic protocol is generated through prepare/freeze, with
        # source/target households isolated. Constants do not require PyTorch.
        baseline_tests.BaselineTest.setUp(self)
        baseline_tests.BaselineTest.train(self, "constant_mean")
        self.session = self.root / "session"
        self.adaptation_config = ROOT / "configs/m3_adaptation.json"

    def start(self, output=None, budget=5, strategy="proposed", seed=42):
        from virtual_submeter.adaptation.session import create_session
        return create_session(self.model, self.manifest, "target_adaptation_house_3", output or self.session,
                              self.adaptation_config, budget, strategy, seed)

    def answer(self, response="used", directory=None):
        from virtual_submeter.adaptation.session import select_next, confirm
        directory = directory or self.session
        query = select_next(directory)
        return confirm(directory, query["query_id"], response)

    def test_sessions_reject_test_partition_nonfrozen_budget_seed_and_adapted_base(self):
        from virtual_submeter.adaptation.session import create_session
        for partition, budget, seed in (("frozen_test_house_3", 5, 42), ("target_adaptation_house_3", 4, 42), ("target_adaptation_house_3", 5, 3)):
            with self.subTest(partition=partition, budget=budget, seed=seed), self.assertRaises(DataError):
                create_session(self.model, self.manifest, partition, self.session, self.adaptation_config, budget, seed=seed)
            self.assertFalse(self.session.exists())

    def test_pending_is_persisted_and_duplicate_stale_answers_do_not_spend_budget(self):
        from virtual_submeter.adaptation.session import select_next, confirm, read_session
        self.start()
        query = select_next(self.session)
        self.assertEqual(query, select_next(self.session))
        record = confirm(self.session, query["query_id"], "unsure")
        self.assertEqual(record["budget_after"], 1)
        with self.assertRaises(DataError):
            confirm(self.session, query["query_id"], "used")
        self.assertEqual(len(read_session(self.session)["records"]), 1)
        self.answer("skipped")
        self.answer("not_used")
        self.answer("used")
        self.answer("used")
        self.assertIsNone(select_next(self.session))
        self.assertEqual(len(read_session(self.session)["records"]), 5)

    def test_selection_and_adaptation_cannot_read_oracle_or_test_files(self):
        from virtual_submeter.adaptation.session import select_next
        from virtual_submeter.adaptation.training import adapt
        original = Path.open

        def guarded(path, *args, **kwargs):
            if any(part in ("oracle_labels", "evaluation_labels") for part in path.parts):
                raise AssertionError("Model-side code opened hidden labels")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            self.start()
            select_next(self.session)
            self.answer("used")
            adapt(self.session, self.root / "adapted")
        metadata = read_json(self.root / "adapted/model.json")
        self.assertFalse(metadata["target_pointwise_labels_read"])
        self.assertTrue(metadata["target_weak_labels_read"])

    def test_oracle_requires_selection_and_returns_only_weak_answer(self):
        from virtual_submeter.adaptation.session import select_next
        from virtual_submeter.data.oracle import answer_pending
        self.start()
        with self.assertRaises(DataError):
            answer_pending(self.session)
        select_next(self.session)
        answer = answer_pending(self.session)
        self.assertEqual(set(answer), {"query_id", "response", "label_source"})
        self.assertEqual(answer["label_source"], "simulated_from_submeter")
        self.assertIn(answer["response"], ("used", "not_used", "unsure"))

    def test_missing_oracle_support_is_unsure_and_still_costs_one(self):
        from virtual_submeter.adaptation.session import select_next, confirm
        from virtual_submeter.data.oracle import answer_pending
        self.start()
        query = select_next(self.session)
        protocol = read_json(self.manifest)
        part = next(p for p in protocol["partitions"] if p["partition_id"] == "target_adaptation_house_3")
        artifact = part["artifacts"][query["appliance_type"]]
        path = self.manifest.parent / artifact["path"]
        lines = path.read_text(encoding="utf-8").splitlines()
        # Remove all queried labels and explicitly refreeze just this fixture.
        import csv
        import io as text_io
        rows = list(csv.DictReader(text_io.StringIO("\n".join(lines))))
        rows = [r for r in rows if not query["window_start"] <= r["timestamp"] < query["window_end"]]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=lines[0].split(","))
            writer.writeheader()
            writer.writerows(rows)
        artifact["sha256"] = sha256(path)
        write_json(self.manifest, protocol)
        # A changed production protocol is rejected; build a fresh session on
        # the revised fixture/model rather than bypassing the protocol guard.
        with self.assertRaises(DataError):
            answer_pending(self.session)
        model = read_json(self.model / "model.json")
        model["protocol_sha256"] = sha256(self.manifest)
        write_json(self.model / "model.json", model)
        self.start(output=self.root / "second", strategy="random")
        # A refrozen fixture still cannot turn missing support into not_used.
        second = self.root / "second"
        select_next(second)
        # Patch only the label CSV reader, returning no samples to the oracle.
        with patch("virtual_submeter.data.oracle.csv.DictReader", return_value=iter([])):
            answer = answer_pending(second)
        self.assertEqual(answer["response"], "unsure")
        record = confirm(second, **answer)
        self.assertEqual(record["budget_after"], 1)

    def test_same_seed_replays_queries_and_each_appliance_window_is_disjoint(self):
        from virtual_submeter.adaptation.session import select_next
        self.start(strategy="random")
        other = self.root / "other"
        self.start(other, strategy="random")
        for _ in range(5):
            self.assertEqual(select_next(self.session), select_next(other))
            self.answer(directory=self.session)
            self.answer(directory=other)
        records = read_json(self.session / "session.json")["records"]
        for i, a in enumerate(records):
            for b in records[i + 1:]:
                if a["appliance_type"] == b["appliance_type"]:
                    self.assertFalse(a["window_start"] < b["window_end"] and b["window_start"] < a["window_end"])

    def test_mil_used_means_at_least_one_point_not_every_point(self):
        from virtual_submeter.adaptation.state import fit_head
        rows = [{"timestamp": f"1970-01-01T00:00:{i * 8:02d}Z", "aggregate_power_w": power,
                 "appliances": {"kettle": {"raw_power_w": power}}} for i, power in enumerate((0, 3000))]
        record = {"appliance_type": "kettle", "response": "used", "window_start": rows[0]["timestamp"], "window_end": "1970-01-01T00:00:16Z"}
        settings = {"epochs": 1, "learning_rate": 0.1, "l2": 0.0}
        head, _, counts = fit_head(rows, [record], ["kettle"], {"kettle": 2000}, settings)
        # Only the maximum-instance feature contributes to the first update.
        from virtual_submeter.adaptation.state import features, scores, INITIAL
        x = features(rows, "kettle", 2000)
        expected = np.array(INITIAL) - 0.1 * (scores(x)[1] - 1) * x[1]
        np.testing.assert_allclose(head["coefficients"]["kettle"], expected)
        self.assertEqual(counts["kettle"], 1)

    def test_adapter_preserves_power_energy_residual_and_nulls_while_exporting_state(self):
        from virtual_submeter.adaptation.training import adapt
        from virtual_submeter.models.prediction import predict
        from virtual_submeter.evaluation.runner import evaluate
        self.start()
        self.answer("used")
        adapt(self.session, self.root / "adapted")
        for name, model in (("before", self.model), ("after", self.root / "adapted")):
            predict(model, self.root / name, self.manifest, "frozen_test_house_3")
            evaluate(self.root / name, self.manifest, self.root / (name + "_evaluation"))
        before, after = [read_json(self.root / name / "predictions.json") for name in ("before", "after")]
        for a, b in zip(before["rows"], after["rows"]):
            for key in ("unexplained_power_w", "raw_excess_w", "quality_flags"):
                self.assertEqual(a[key], b[key])
            for appliance in before["appliances"]:
                for key in ("raw_power_w", "power_w", "correction_w", "confidence"):
                    self.assertEqual(a["appliances"][appliance][key], b["appliances"][appliance][key])
                if a["appliances"][appliance]["power_w"] is None:
                    self.assertIsNone(b["appliances"][appliance]["state"])
        html = (self.root / "after_evaluation/report.html").read_text(encoding="utf-8")
        self.assertNotIn("本模型未经目标家庭适配", html)
        self.assertIn("human_confirmed", html)
        self.assertIn("1 / 5", html)
        self.assertIn("state_score", (self.root / "after/predictions.csv").read_text())

    def test_zero_budget_and_all_unsure_keep_exact_base_predictions(self):
        from virtual_submeter.adaptation.training import adapt
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        self.start(budget=0)
        adapt(self.session, self.root / "adapted")
        rows = read_partition(self.manifest, "frozen_test_house_3", "inference")["aggregate"]
        base, base_model = load_model(self.model)
        metadata, model = load_model(self.root / "adapted")
        self.assertNotIn("state_adapter", metadata)
        self.assertEqual(predict_rows(base, base_model, rows), predict_rows(metadata, model, rows))
        other = self.root / "unsure"
        self.start(output=other)
        self.answer("unsure", other)
        adapt(other, self.root / "unsure_model")
        metadata, model = load_model(self.root / "unsure_model")
        self.assertEqual(predict_rows(base, base_model, rows), predict_rows(metadata, model, rows))

    def test_metrics_use_independent_state_instead_of_rethresholding_power(self):
        from virtual_submeter.evaluation.metrics import appliance_metrics
        row = {"household_id": "3", "timestamp": "1970-01-01T00:00:00Z", "appliances": {"kettle": {"power_w": 0, "state": True}}}
        labels = [{"household_id": "3", "timestamp": row["timestamp"], "power_w": "2500"}]
        metric = appliance_metrics([row], labels, "kettle", 2000, 8)
        self.assertEqual(metric["f1"], 1)
        self.assertEqual(metric["mae_w"], 2500)

    def test_corrupt_history_or_predictions_and_wrong_household_are_rejected(self):
        from virtual_submeter.adaptation.training import adapt
        from virtual_submeter.adaptation.session import read_session
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        self.start()
        self.answer()
        adapt(self.session, self.root / "adapted")
        metadata, model = load_model(self.root / "adapted")
        rows = read_partition(self.manifest, "source_validation_house_5", "inference")["aggregate"]
        with self.assertRaises(DataError):
            predict_rows(metadata, model, rows)
        queries = self.root / "adapted/queries.json"
        queries.write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            load_model(self.root / "adapted")
        path = self.session / "prediction/predictions.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            read_session(self.session)

    def test_runner_freezes_all_equal_budget_models_before_evaluation(self):
        from virtual_submeter.adaptation.runner import run_comparison
        from virtual_submeter.evaluation.runner import evaluate
        output = self.root / "comparison"
        seen = []

        def checked(*args, **kwargs):
            record = read_json(output / "run.json")
            self.assertEqual(len(record["cases"]), 6)
            self.assertTrue(all((output / c["path"] / "model/model.json").exists() for c in record["cases"]))
            seen.append(args[0])
            return evaluate(*args, **kwargs)

        with patch("virtual_submeter.evaluation.runner.evaluate", side_effect=checked):
            result = run_comparison(self.model, self.manifest, self.adaptation_config, output, [0, 5], [42])
        self.assertEqual(result["status"], "m3_pilot_evaluated")
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(c["budget"] == c["query_summary"]["budget_used"] for c in result["cases"]))
        from virtual_submeter.adaptation.verification import verify
        self.assertEqual(verify(output)["cases"], 6)

    def test_complete_query_windows_exclude_missing_data_and_boundaries(self):
        from virtual_submeter.adaptation.selection import candidates, load_settings
        self.start()
        payload = read_json(self.session / "prediction/predictions.json")
        protocol = read_json(self.manifest)
        partition = next(p for p in protocol["partitions"] if p["partition_id"] == "target_adaptation_house_3")
        settings = load_settings(self.adaptation_config)
        original = candidates(payload, partition, protocol, settings)
        self.assertTrue(original)
        damaged = original[0]
        for row in payload["rows"]:
            if damaged["window_start"] <= row["timestamp"] < damaged["window_end"]:
                row["appliances"][damaged["appliance_type"]]["raw_power_w"] = None
        remaining = candidates(payload, partition, protocol, settings)
        self.assertNotIn(damaged["query_id"], {c["query_id"] for c in remaining})

    def test_selection_is_independent_of_hidden_label_fingerprints(self):
        from virtual_submeter.adaptation.selection import candidates, load_settings
        self.start()
        payload = read_json(self.session / "prediction/predictions.json")
        protocol = read_json(self.manifest)
        partition = next(p for p in protocol["partitions"] if p["partition_id"] == "target_adaptation_house_3")
        settings = load_settings(self.adaptation_config)
        original = candidates(payload, partition, protocol, settings)
        payload["model_id"] = "model-referencing-different-hidden-label-digests"
        payload["protocol_sha256"] = "different-hidden-label-digests"
        self.assertEqual(original, candidates(payload, partition, protocol, settings))

    def test_strategy_scores_rank_visible_uncertainty_and_proposed_factors(self):
        from virtual_submeter.adaptation.selection import choose
        a = {"query_id": "a", "household_id": "3", "appliance_type": "kettle", "window_start": "1970-01-01T00:00:00Z", "window_end": "1970-01-01T00:00:16Z", "uncertainty": 0.9, "proposed_score": 0.4}
        b = {**a, "query_id": "b", "window_start": "1970-01-01T00:00:16Z", "window_end": "1970-01-01T00:00:32Z", "uncertainty": 0.5, "proposed_score": 0.8}
        self.assertEqual(choose([a, b], [], "uncertainty", 42)["query_id"], "a")
        self.assertEqual(choose([a, b], [], "proposed", 42)["query_id"], "b")

    def test_pending_answer_and_concurrent_session_lock_cannot_be_ignored(self):
        from virtual_submeter.adaptation.session import select_next
        from virtual_submeter.adaptation.training import adapt
        self.start()
        select_next(self.session)
        with self.assertRaises(DataError):
            adapt(self.session, self.root / "adapted")
        self.assertFalse((self.root / "adapted").exists())
        (self.session / ".session.lock").write_text("busy", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "busy"):
            select_next(self.session)

    def test_comparison_failure_does_not_open_test_answers_or_claim_success(self):
        from virtual_submeter.adaptation.runner import run_comparison
        output = self.root / "failed_comparison"
        with patch("virtual_submeter.adaptation.runner.select_next", return_value=None), patch("virtual_submeter.evaluation.runner.evaluate") as evaluator:
            with self.assertRaisesRegex(DataError, "enough"):
                run_comparison(self.model, self.manifest, self.adaptation_config, output, [5], [42])
        evaluator.assert_not_called()
        self.assertEqual(read_json(output / "run.json")["status"], "failed")
        self.assertFalse((output / "summary.json").exists())

    def test_unknown_adapter_fields_or_nonfinite_configuration_are_rejected(self):
        from virtual_submeter.adaptation.selection import load_settings
        from virtual_submeter.adaptation.state import validate_head
        settings = read_json(self.adaptation_config)
        settings["l2"] = -1
        path = self.root / "invalid-settings.json"
        write_json(path, settings)
        with self.assertRaises(DataError):
            load_settings(path)
        with self.assertRaises(DataError):
            validate_head({"method": "max_pool_mil_logistic_v1", "coefficients": {"kettle": [0, 1, 0]}, "active_appliances": ["foreign"]}, ["kettle"])

    def test_cli_can_resume_confirm_and_adapt_without_pointwise_labels(self):
        from virtual_submeter.cli import main
        self.start()
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["next-query", "--session", str(self.session)]), 0)
        query = json.loads(stream.getvalue())["query"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["confirm-query", "--session", str(self.session), "--query-id", query["query_id"], "--response", "not_used"]), 0)
            self.assertEqual(main(["adapt-model", "--session", str(self.session), "--output", str(self.root / "adapted")]), 0)


if __name__ == "__main__":
    unittest.main()
