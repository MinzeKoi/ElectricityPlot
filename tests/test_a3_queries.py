"""Dynamic scoring, round replay, prefix budgets and freeze barrier regression."""

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


@unittest.skipUnless(NUMERIC, "A3 requires M2 CPU environment")
class SelectionTest(unittest.TestCase):
    def test_current_scores_change_uncertainty_and_hidden_labels_are_not_inputs(self):
        from virtual_submeter.accuracy.dynamic_session import choose, load_plan
        plan = load_plan(ROOT / "configs/a3_queries.json")
        pool = [{"query_id": str(i), "appliance_type": "kettle", "shape": [float(i), 0, 0, 0]} for i in range(3)]
        first = choose(pool, [], {"0": 0.5, "1": 0.99, "2": 0.1}, "uncertainty", 42, plan)
        second = choose(pool, [], {"0": 0.99, "1": 0.5, "2": 0.1}, "uncertainty", 42, plan)
        self.assertEqual(first["query_id"], "0")
        self.assertEqual(second["query_id"], "1")
        random = choose(pool, [], {"0": 0.5, "1": 0.99, "2": 0.1}, "random", 42, plan)
        self.assertEqual(random, choose(pool, [], {"0": 0, "1": 0, "2": 0}, "random", 42, plan))

    def test_coverage_and_diversity_ablations_are_independent(self):
        from virtual_submeter.accuracy.dynamic_session import choose, load_plan
        plan = load_plan(ROOT / "configs/a3_queries.json")
        plan.update(exploration_fraction=0, uncertainty_weight=0, diversity_weight=0.5, coverage_weight=0.5)
        pool = [{"query_id": str(i), "appliance_type": a, "shape": [x, 0, 0, 0]}
                for i, a, x in ((0, "kettle", 0), (1, "kettle", 10), (2, "microwave", 0))]
        records = [{"query_id": "0", "appliance_type": "kettle", "response": "skipped"}]
        scores = {str(i): 0.5 for i in range(3)}
        self.assertEqual(choose(pool, records, scores, "proposed_no_coverage", 42, plan)["query_id"], "1")
        self.assertEqual(choose(pool, records, scores, "proposed_no_diversity", 42, plan)["query_id"], "2")


@unittest.skipUnless(NUMERIC, "A3 requires M2 CPU environment")
class ExperimentTest(unittest.TestCase):
    def setUp(self):
        from test_a2_learners import ExperimentTest as A2Test
        A2Test.setUp(self)
        self.plan.update(budgets=[0, 5, 10], main_budget=10, variants=["L2_max"])
        write_json(self.plan_path, self.plan)

    def test_dynamic_session_resume_skip_round_versions_and_matrix_barrier(self):
        from virtual_submeter.accuracy import fixed_queries as fixed
        from virtual_submeter.accuracy import dynamic_queries as dynamic
        from virtual_submeter.accuracy import dynamic_session as sessions
        from virtual_submeter.data.oracle import answer_pending
        original_summary = fixed.summarize

        def synthetic_passing_summary(*args):
            summary = original_summary(*args)
            summary.update(development_exit_passed=True, selected_development_learner="L2_max")
            return summary

        original_open = Path.open
        def deny_future(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "evaluation_labels" in path.parts and "r" in mode and "b" not in mode:
                raise AssertionError("Future label parsed before full A3 freeze")
            return original_open(path, *args, **kwargs)

        a3 = self.root / "a3"
        query_plan = read_json(ROOT / "configs/a3_queries.json")
        path = self.root / "a3_plan.json"
        write_json(path, query_plan)
        # Synthetic fixture exercises engineering even when its A2 accuracy is
        # unsuitable; only the development-effect gate is replaced in this test.
        with patch.object(fixed, "summarize", synthetic_passing_summary), contextlib.redirect_stderr(io.StringIO()):
            fixed.freeze(self.a1, self.plan_path, ROOT / "configs/m3_adaptation.json", self.output)
            fixed.evaluate(self.output)
            self.assertEqual(dynamic.validate_parent(self.output)["status"], "a2_full_evidence_verified")
            cached = self.output / "frozen/cache/seed_42/house_3/target_adaptation_house_3/predictions.json"
            moved = self.root / "withheld-cache.json"
            self.assertTrue(cached.resolve().is_relative_to(self.root.resolve()))
            self.assertTrue(moved.resolve().is_relative_to(self.root.resolve()))
            cached.rename(moved)
            self.assertEqual(len(dynamic.validate_parent(self.output)["unavailable_derived_artifacts"]), 1)
            source = self.output / "frozen/source_heads/seed_42/source_head.json"
            original = source.read_bytes()
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(DataError):
                dynamic.validate_parent(self.output)
            source.write_bytes(original)
            original_confirm = sessions.confirm
            calls = []
            def interrupt_after_skip(directory, **answer):
                if not calls:
                    first = sessions.select_next(directory)
                    self.assertEqual(first, sessions.select_next(directory))
                    answer.update(response="skipped")
                    record = original_confirm(directory, **answer)
                    calls.append(record)
                    with self.assertRaises(DataError):
                        original_confirm(directory, **answer)
                    raise RuntimeError("simulated interruption after atomic answer")
                return original_confirm(directory, **answer)
            with patch.object(Path, "open", deny_future), patch.object(sessions, "confirm", interrupt_after_skip):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    dynamic.freeze(self.output, path, a3)
            self.assertEqual(read_json(a3 / "run.json")["status"], "failed_freeze")
            with patch.object(Path, "open", deny_future):
                result = dynamic.freeze(self.output, path, a3, resume=True)
            self.assertEqual(result["cases"], 21)
        frozen = a3 / "frozen"
        directory = frozen / "trajectories/seed_42/house_3/query_42/uncertainty"
        session = sessions.read_session(directory)
        self.assertEqual(len(session["records"]), 10)
        self.assertEqual(len(session["rounds"]), 3)
        versions = [r["model_version"] for r in session["records"]]
        self.assertEqual(len(set(versions[:5])), 1)
        self.assertEqual(len(set(versions[5:])), 1)
        self.assertNotEqual(versions[0], versions[5])
        static = sessions.read_session(directory.parent / "uncertainty_static")
        self.assertEqual(len({r["model_version"] for r in static["records"]}), 1)
        self.assertIsNone(sessions.select_next(directory))
        with self.assertRaises(DataError):
            answer_pending(directory)
        from virtual_submeter.models.prediction import predict
        case = next(c for c in read_json(a3 / "run.json")["cases"] if c["variant"] == "proposed" and c["budget"] == 10)
        predict(frozen / case["model_path"], self.root / "ordinary_prediction",
                frozen / "inputs/protocol/protocol.json", case["future_partition"])
        self.assertIn("kettle_state_score", (self.root / "ordinary_prediction/predictions.csv").read_text(encoding="utf-8").splitlines()[0])
        # Replay rejects a changed prefix even before the freeze inventory check.
        tampered = copy.deepcopy(session)
        tampered["records"][0]["model_version"] = "wrong-model"
        write_json(directory / "session.json", tampered)
        with self.assertRaises(DataError):
            sessions.read_session(directory)
        write_json(directory / "session.json", session)
        result = dynamic.evaluate(a3)
        self.assertEqual(result["metric_rows"], 63)
        with patch.object(Path, "open", deny_future):
            self.assertEqual(dynamic.verify(a3)["status"], "a3_verified")
        with self.assertRaises(DataError):
            dynamic.evaluate(a3)
        (a3 / "comparison.csv").write_text("corrupted", encoding="utf-8")
        with self.assertRaises(DataError):
            dynamic.verify(a3)


if __name__ == "__main__":
    unittest.main()
