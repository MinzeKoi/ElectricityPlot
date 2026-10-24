"""R2 policy identity, L1 ranking, honest exits and synthetic freeze barrier."""

import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json, sha256, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


@unittest.skipUnless(NUMERIC, "R2 requires the CPU dependencies")
class R2Test(unittest.TestCase):
    def setUp(self):
        from virtual_submeter.accuracy.dynamic_session import load_plan
        self.plan = load_plan(ROOT / "configs/r2_queries_20261022.json")
        self.learner = read_json(ROOT / "configs/r1_learners_20261022.json")

    def scope(self):
        return {"partitions": [{"role": role, "household_id": house} for role, houses in (
            ("source_train", ["2", "19"]), ("source_validation", ["5"]),
            ("target_adaptation", ["17"]), ("frozen_test", ["17"])) for house in houses],
            "protocol_id": "refit-r1-r0-dates-development-v1", "data_kind": "real_REFIT_complete_date_selected",
            "accuracy_usage": "development_only", "purpose": "engineering_pilot_only",
            "r0_date_hashes": {name: sha256(ROOT / f"configs/r0_dates_{name}_20261022.json")
                               for name in ("sources", "development")},
            "sample_interval_seconds": 8, "query_window_seconds": 1200, "split_gap_seconds": 3600}

    def parent_summary(self):
        return {"main_budget": 10, "development_exit_passed": True,
                "selected_development_learner": "L2_max", "groups": [
                    {"variant": variant, "budget": 10, "f1": f1, "f1_delta": gain, "missing_f1": 0,
                     "adaptation_degenerate_cases": 0, "device_drops_over_tolerance": 0}
                    for variant, f1, gain in (("L1", .426, .01), ("L2_max", .295, .051))]}

    def test_only_three_frozen_policies_and_unchanged_weights_are_accepted(self):
        from virtual_submeter.accuracy.dynamic_session import load_plan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            for key, value in (("learner", "L2_max"), ("strategies", ["random", "uncertainty", "proposed"]),
                               ("uncertainty_weight", .6), ("minimum_random_f1_gain", .01),
                               ("experiment_id", "old-a5")):
                write_json(path, {**self.plan, key: value})
                with self.subTest(key=key), self.assertRaises(DataError):
                    load_plan(path)
        self.assertEqual(load_plan(ROOT / "configs/a3_queries.json")["schema_version"], 1)

    def test_r1_absolute_choice_preserves_historical_own_gain_winner(self):
        from virtual_submeter.accuracy.r2 import selection_receipt
        summary = self.parent_summary()
        receipt = selection_receipt(summary, self.plan)
        self.assertEqual(receipt["learner"], "L1")
        self.assertEqual(receipt["historical_own_gain_selection"], "L2_max")
        self.assertEqual(summary, self.parent_summary())
        for key, value in (("missing_f1", 1), ("adaptation_degenerate_cases", 1),
                           ("device_drops_over_tolerance", 1), ("f1_delta", 0)):
            invalid = copy.deepcopy(summary)
            invalid["groups"][0][key] = value
            with self.subTest(key=key), self.assertRaises(DataError):
                selection_receipt(invalid, self.plan)

    def test_scope_rejects_reserved_households_and_budget_changes(self):
        from virtual_submeter.accuracy.r2 import validate_scope
        validate_scope(self.scope(), self.learner)
        for house in ("8", "9", "20"):
            protocol = self.scope()
            protocol["partitions"][-1]["household_id"] = house
            with self.subTest(house=house), self.assertRaises(DataError):
                validate_scope(protocol, self.learner)
        changed = copy.deepcopy(self.learner)
        changed["main_budget"] = 20
        with self.assertRaises(DataError):
            validate_scope(self.scope(), changed)

    def test_missing_inputs_do_not_create_run_or_claim_results(self):
        from virtual_submeter.accuracy.r2 import preflight
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "missing-r1"
            result = preflight(parent, ROOT / "configs/r2_queries_20261022.json")
            self.assertEqual(result["status"], "r2_inputs_missing")
            self.assertEqual((result["cases"], result["metric_rows"]), (108, 324))
            self.assertFalse(result["future_pointwise_labels_parsed"])
            self.assertFalse(parent.exists())

    def test_changed_scope_fails_before_general_parent_verification(self):
        from virtual_submeter.accuracy.r2 import preflight
        from virtual_submeter.accuracy.dynamic_queries import freeze
        with tempfile.TemporaryDirectory() as temporary:
            parent, output = Path(temporary) / "parent", Path(temporary) / "r2"
            (parent / "frozen/protocol").mkdir(parents=True)
            for name in ("run.json", "freeze.json", "evidence.json", "summary.json"):
                write_json(parent / name, {})
            write_json(parent / "frozen/plan.json", self.learner)
            protocol = self.scope()
            protocol["partitions"][-1]["household_id"] = "8"
            write_json(parent / "frozen/protocol/protocol.json", protocol)
            with patch("virtual_submeter.accuracy.dynamic_queries.validate_parent") as general:
                with self.assertRaises(DataError):
                    preflight(parent, ROOT / "configs/r2_queries_20261022.json")
                with self.assertRaises(DataError):
                    freeze(parent, ROOT / "configs/r2_queries_20261022.json", output)
                general.assert_not_called()
            self.assertFalse(output.exists())

    def test_l1_ranking_is_continuous_for_untouched_devices_with_same_gaps(self):
        from virtual_submeter.accuracy.r2 import selection_scores
        from virtual_submeter.accuracy.learners import score_adapter
        rows = [{"aggregate_power_w": 1000., "appliances": {"kettle": {
            "raw_power_w": raw, "raw_state": raw >= 200 if raw is not None else None}}}
            for raw in (0., 190., 210., None)]
        meta = {"appliances": ["kettle"], "settings": {"on_thresholds_w": {"kettle": 200}}}
        adapter = {"schema_version": 1, "variant": "L1", "coefficients": {"kettle": [0., 1.]},
                   "active_appliances": [], "normalizer": None}
        ranked = selection_scores(rows, meta, adapter)[:, 0]
        np.testing.assert_array_equal(score_adapter(rows, meta, adapter)[:3, 0], [0, 0, 1])
        self.assertTrue(0 < ranked[0] < ranked[1] < .5 < ranked[2] < 1)
        self.assertTrue(np.isnan(ranked[3]))
        adapter["coefficients"]["kettle"][0] = -.25
        adapter["active_appliances"] = ["kettle"]
        np.testing.assert_allclose(selection_scores(rows, meta, adapter), score_adapter(rows, meta, adapter), equal_nan=True)

    def test_h1_h2_and_absolute_strategy_choice_are_independent(self):
        from virtual_submeter.accuracy.r2 import summarize

        def outcome(gains):
            groups = [{"variant": strategy, "budget": 10, "f1": .4 + gain, "f1_delta": gain,
                       "f1_vs_random": gain - gains[0], "missing_f1": 0,
                       "adaptation_degenerate_cases": 0, "device_drops_over_tolerance": 0}
                      for strategy, gain in zip(self.plan["strategies"], gains)]
            metrics = [{"variant": g["variant"], "budget": 10, "appliance": "kettle", "query_seed": seed,
                        "f1_delta": g["f1_delta"], "f1_vs_random": g["f1_vs_random"]}
                       for g in groups for seed in self.learner["query_seeds"]]
            return summarize(metrics, self.plan, self.learner, {"groups": groups, "main_budget": 10})

        result = outcome([.06, .07, .065])
        self.assertEqual(result["selected_development_strategy"], "uncertainty_static")
        self.assertTrue(all(o["h1_development_target_met"] for o in result["strategy_outcomes"]))
        self.assertFalse(any(o["h2_development_target_met"] for o in result["strategy_outcomes"]))
        self.assertFalse(result["final_h1_h2_h3_evaluated"])
        result = outcome([.001, .03, .005])
        self.assertTrue(result["strategy_outcomes"][1]["h2_development_target_met"])
        self.assertFalse(result["strategy_outcomes"][1]["h1_development_target_met"])
        self.assertEqual(outcome([.01, .01, .01])["selected_development_strategy"], "random")


@unittest.skipUnless(NUMERIC, "R2 requires the CPU dependencies")
class R2ExperimentTest(unittest.TestCase):
    def setUp(self):
        from test_a2_learners import ExperimentTest
        ExperimentTest.setUp(self)
        self.plan.update(variants=["L1"], budgets=[0, 5, 10, 20], main_budget=10)
        write_json(self.plan_path, self.plan)

    def test_synthetic_three_policy_prefixes_resume_and_evaluation_barrier(self):
        from virtual_submeter.accuracy import fixed_queries as fixed, dynamic_queries as dynamic, dynamic_session as sessions
        from virtual_submeter.accuracy import r2
        from virtual_submeter.models.prediction import predict
        original_summary, original_open = fixed.summarize, Path.open

        def fixture_summary(*args):
            summary = original_summary(*args)
            summary.update(development_exit_passed=True, selected_development_learner="L1")
            # A synthetic model is for engineering, not an accuracy acceptance test.
            for g in summary["groups"]:
                if g["budget"] == 10:
                    g.update(f1=.4, f1_delta=.01, missing_f1=0, adaptation_degenerate_cases=0, device_drops_over_tolerance=0)
            return summary

        def synthetic_scope(protocol, learner):
            self.assertEqual(protocol["accuracy_usage"], "development_only")
            self.assertEqual({p["household_id"] for p in protocol["partitions"]}, {"2", "3", "5"})

        def deny_future(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "evaluation_labels" in path.parts and "r" in mode and "b" not in mode:
                raise AssertionError("Future labels parsed before the complete R2 freeze")
            return original_open(path, *args, **kwargs)

        output = self.root / "r2"
        query_plan = getattr(self, "query_plan_path", ROOT / "configs/r2_queries_20261022.json")
        with contextlib.redirect_stderr(io.StringIO()), patch.object(fixed, "summarize", fixture_summary):
            fixed.freeze(self.a1, self.plan_path, ROOT / "configs/m3_adaptation.json", self.output)
            fixed.evaluate(self.output)
            original_confirm = sessions.confirm
            interrupted = []

            def interrupt(directory, **answer):
                if not interrupted:
                    answer["response"] = "skipped"
                    interrupted.append(original_confirm(directory, **answer))
                    raise RuntimeError("R2 synthetic interruption")
                return original_confirm(directory, **answer)

            with patch.object(r2, "validate_scope", synthetic_scope), patch.object(Path, "open", deny_future):
                with patch.object(sessions, "confirm", interrupt), self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    dynamic.freeze(self.output, query_plan, output)
                result = dynamic.freeze(self.output, query_plan, output, resume=True)
                self.assertEqual(result["cases"], 12)
                self.assertFalse(result["development_future_labels_opened"])
        with patch.object(r2, "validate_scope", synthetic_scope):
            state, _, _ = dynamic.verify_frozen(output)
            frozen = output / "frozen"
            pools = []
            for strategy in ("random", "uncertainty_static", "proposed"):
                directory = frozen / "trajectories/seed_42/house_3/query_42" / strategy
                session = sessions.read_session(directory)
                self.assertEqual(len(session["records"]), 20)
                self.assertEqual(len(session["rounds"]), 5)
                pools.append(read_json(directory / "pool.json"))
                if strategy.endswith("_static"):
                    self.assertEqual(len({r["model_version"] for r in session["records"]}), 1)
            self.assertEqual(pools[0], pools[1])
            self.assertEqual(pools[1], pools[2])
            case = next(c for c in state["cases"] if c["variant"] == "proposed" and c["budget"] == 10)
            predict(frozen / case["model_path"], self.root / "reloaded", frozen / "inputs/protocol/protocol.json", case["future_partition"])
            payload = read_json(self.root / "reloaded/predictions.json")
            from virtual_submeter.accuracy.learners import score_adapter
            metadata = read_json(frozen / case["model_path"] / "model.json")
            with np.load(frozen / case["path"] / "states.npz", allow_pickle=False) as cache:
                # Untouched L1 devices retain the exact original boolean decisions.
                base = read_json(frozen / case["base_prediction"])
                np.testing.assert_allclose(score_adapter(base["rows"], metadata, metadata["accuracy_state_head"]), cache["scores"], equal_nan=True)
                for i, row in enumerate(payload["rows"]):
                    for j, appliance in enumerate(payload["appliances"]):
                        score = cache["scores"][i, j]
                        self.assertEqual(row["appliances"][appliance]["raw_state"], bool(score >= .5) if np.isfinite(score) else None)
            self.assertEqual(dynamic.evaluate(output)["metric_rows"], 36)
            with patch.object(Path, "open", deny_future):
                self.assertEqual(dynamic.verify(output)["status"], "a3_verified")
            with self.assertRaises(DataError):
                dynamic.evaluate(output)
            (output / "comparison.csv").write_text("corrupted", encoding="utf-8")
            with self.assertRaises(DataError):
                dynamic.verify(output)


if __name__ == "__main__":
    unittest.main()
