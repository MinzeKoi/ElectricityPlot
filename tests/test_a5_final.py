"""A5 household isolation, all-matrix barrier, paired aggregation and seals."""

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


@unittest.skipUnless(NUMERIC, "A5 requires the M2 CPU environment")
class FinalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from virtual_submeter.accuracy import final_demo
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name) / "demo"
        with contextlib.redirect_stderr(io.StringIO()):
            cls.result = final_demo.run(cls.root, ROOT)
        cls.output = cls.root / "experiment"

    def test_three_household_engineering_roundtrip_is_never_final(self):
        from virtual_submeter.accuracy import final_evaluation as final
        from virtual_submeter.models.prediction import predict
        from virtual_submeter.models.artifacts import load_model
        self.assertEqual(self.result["cases"], 48)
        self.assertEqual(self.result["metric_rows"], 144)
        self.assertFalse(self.result["formal_evaluation"])
        self.assertIsNone(self.result["h1_final_target_met"])
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["households"], ["4", "6", "7"])
        self.assertIsNone(summary["h3_final_target_met"])
        frozen = self.output / "frozen"
        state = read_json(self.output / "run.json")
        case = next(c for c in state["cases"] if c["variant"] == "proposed" and c["budget"] == 10)
        ordinary = self.root / "ordinary"
        predict(frozen / case["model_path"], ordinary, frozen / "protocol/protocol.json", case["future_partition"])
        payload = read_json(ordinary / "predictions.json")
        self.assertEqual(payload["adaptation"], "a5_final_state_only")
        self.assertIn("kettle_state_score", (ordinary / "predictions.csv").read_text(encoding="utf-8").splitlines()[0])
        with np.load(frozen / case["path"] / "states.npz", allow_pickle=False) as saved:
            for i, row in enumerate(payload["rows"]):
                for j, appliance in enumerate(payload["appliances"]):
                    self.assertEqual(row["appliances"][appliance]["state_score"], float(saved["scores"][i, j]) if np.isfinite(saved["scores"][i, j]) else None)
        for seed in (42,):
            original = self.root / "source_learning/frozen/bases" / f"seed_{seed}"
            base = frozen / "bases" / f"seed_{seed}"
            self.assertEqual(sha256(base / "weights.pt"), sha256(original / "weights.pt"))
            self.assertTrue(load_model(base)[0]["a5_transferred_without_training"])
        with self.assertRaises(DataError):
            final.evaluate(self.output)

    def test_future_labels_are_never_parsed_before_complete_matrix(self):
        from virtual_submeter.accuracy import final_evaluation as final
        original_open = Path.open
        accesses = []
        run = self.root / "guarded"
        def guard(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "evaluation_labels" in path.parts and "r" in mode and "b" not in mode:
                self.assertEqual(read_json(run / "run.json")["status"], "evaluating")
                self.assertEqual(len(read_json(run / "freeze.json")["cases"]), 48)
                accesses.append(path)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard), contextlib.redirect_stderr(io.StringIO()):
            final.freeze(self.root / "source_learning", self.root / "protocol/protocol.json", self.root / "plan.json", ROOT / "configs/a5_queries.json", run)
            self.assertEqual(accesses, [])
            final.evaluate(run)
            self.assertTrue(accesses)
            accesses.clear()
            final.verify(run)
            self.assertEqual(accesses, [])

    def test_formal_admission_rejects_prefixes_seen_houses_and_missing_declaration(self):
        from virtual_submeter.data.final_protocol import prepare
        from virtual_submeter.accuracy.final_evaluation import validate_plan, validate_protocol
        plan = read_json(ROOT / "configs/a5_final.json")
        with self.assertRaises(DataError):
            validate_plan(plan, require_targets=True)
        plan["target_households"] = ["3", "6", "7"]
        with self.assertRaises(DataError):
            validate_plan(plan, require_targets=True)
        plan["target_households"] = ["4", "6", "7"]
        with self.assertRaisesRegex(DataError, "declaration"):
            prepare(self.root / "source_learning/frozen/protocol/protocol.json", self.root / "targets", plan, self.root / "no_statement")
        statement = self.root / "statement.json"
        write_json(statement, {"schema_version": 1, "target_households": plan["target_households"], "future_labels_unseen": True,
                              "declared_by": "synthetic test only", "declared_at": "2026-10-21T00:00:00Z"})
        with self.assertRaisesRegex(DataError, "synthetic"):
            prepare(self.root / "source_learning/frozen/protocol/protocol.json", self.root / "targets", plan, self.root / "synthetic_as_real", statement)
        frozen = self.output / "frozen"
        protocol = read_json(frozen / "protocol/protocol.json")
        parent = read_json(frozen / "source_protocol.json")
        learner = read_json(frozen / "learner_plan.json")
        query = read_json(frozen / "query_plan.json")
        engineering = read_json(frozen / "plan.json")
        altered = copy.deepcopy(protocol)
        next(p for p in altered["partitions"] if p["role"] == "frozen_test")["start"] = "1970-01-01T00:00:00Z"
        with self.assertRaisesRegex(DataError, "isolated"):
            validate_protocol(altered, parent, engineering, learner, query)

    def test_missing_duplicate_and_null_metrics_never_improve_summary(self):
        from virtual_submeter.accuracy.final_evaluation import summarize
        frozen = self.output / "frozen"
        plan, learner = read_json(frozen / "plan.json"), read_json(frozen / "learner_plan.json")
        metrics = read_json(self.output / "metrics.json")["rows"]
        for bad in (metrics[:-1], metrics + [metrics[0]]):
            with self.assertRaisesRegex(DataError, "missing/duplicate"):
                summarize(bad, plan, learner)
        missing = copy.deepcopy(metrics)
        row = next(r for r in missing if r["variant"] == "proposed" and r["budget"] == 10 and r["appliance"] == "kettle")
        row.update(f1=None, f1_delta=None, f1_vs_random=None)
        summary = summarize(missing, plan, learner)
        self.assertFalse(summary["primary_metrics_complete"])
        self.assertFalse(summary["h1_numeric_gate_met"])
        self.assertIsNone(summary["h1_final_target_met"])
        self.assertEqual(summary["missing_f1_total"], 1)

    def test_changed_seal_incomplete_matrix_and_source_drift_refuse_evaluation(self):
        from virtual_submeter.accuracy import final_evaluation as final
        run_path = self.output / "run.json"
        original = run_path.read_bytes()
        state = read_json(run_path)
        try:
            altered = copy.deepcopy(state)
            altered["cases"] = altered["cases"][:-1]
            write_json(run_path, altered)
            with self.assertRaises(DataError):
                final.verify_frozen(self.output)
            altered = copy.deepcopy(state)
            altered.update(status="frozen", future_labels_opened=False, code_sha256="changed")
            write_json(run_path, altered)
            with self.assertRaisesRegex(DataError, "code changed"):
                final.evaluate(self.output)
        finally:
            run_path.write_bytes(original)
        evidence = self.output / "comparison.csv"
        original = evidence.read_bytes()
        try:
            evidence.write_text("changed", encoding="utf-8")
            with self.assertRaises(DataError):
                final.verify(self.output)
        finally:
            evidence.write_bytes(original)

    def test_household_macro_average_and_stability_keep_all_budgets(self):
        from virtual_submeter.accuracy.final_evaluation import summarize
        frozen = self.output / "frozen"
        plan, learner = read_json(frozen / "plan.json"), read_json(frozen / "learner_plan.json")
        rows = read_json(self.output / "metrics.json")["rows"]
        baselines = {"4": 0.1, "6": 0.2, "7": 0.8}
        for row in rows:
            gain = 0.06 if row["variant"] == "proposed" and row["budget"] else 0.03 if row["budget"] else 0
            row.update(f1=baselines[row["household_id"]] + gain, precision=0.7, recall=0.6,
                       f1_delta=gain, f1_vs_random=0.03 if row["variant"] == "proposed" and row["budget"] else 0,
                       adaptation_caused_degeneracy=False, valid_samples=1 if row["household_id"] == "4" else 100000)
        summary = summarize(rows, plan, learner)
        main = next(g for g in summary["groups"] if g["variant"] == "proposed" and g["budget"] == 10)
        self.assertAlmostEqual(main["f1"], (0.1 + 0.2 + 0.8) / 3 + 0.06)
        self.assertTrue(summary["h1_numeric_gate_met"])
        self.assertTrue(summary["h2_numeric_gate_met"])
        self.assertIsNone(summary["h1_final_target_met"])
        next(r for r in rows if r["variant"] == "proposed" and r["budget"] == 20 and r["appliance"] == "kettle")["adaptation_caused_degeneracy"] = True
        summary = summarize(rows, plan, learner)
        self.assertFalse(summary["stable"])
        self.assertFalse(summary["h1_numeric_gate_met"])


if __name__ == "__main__":
    unittest.main()
