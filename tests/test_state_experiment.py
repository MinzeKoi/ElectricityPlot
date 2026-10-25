"""State training selection, leakage barriers, compatibility and missing inference."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from virtual_submeter.models.state_selection import pr_curve, constrained_point, checkpoint_rank
from virtual_submeter.models.state_classifier import StateClassifier, predict_state_rows, load_state_model
from virtual_submeter.models.training import train, dataset
from virtual_submeter.accuracy import state_experiment as experiment
from virtual_submeter.data.common import DataError, read_json, write_json, sha256, iso


class StateExperimentTest(unittest.TestCase):
    def fixture(self):
        from test_m2_baseline import BaselineTest
        f = BaselineTest()
        f.setUp()
        self.addCleanup(f.doCleanups)
        return f

    def test_pr_uses_whole_tie_groups_and_recall_guard(self):
        scores = np.array([.9, .9, .5, .2, np.nan])
        truth = np.array([1, 0, 1, 0, 1])
        curve = pr_curve(scores, truth)
        np.testing.assert_array_equal(curve["threshold"], [.9, .5, .2])
        p = constrained_point(curve, .85)
        self.assertEqual(p["threshold"], .5)
        self.assertEqual(p["tp"], 2)
        self.assertEqual(p["fp"], 1)
        self.assertEqual(p["recall"], 1)
        for scores, truth in (([1, 2], [0, 0]), ([1, 2], [1, 1]), ([np.nan], [1])):
            with self.assertRaises(DataError):
                pr_curve(scores, truth)
        self.assertLess(checkpoint_rank(p, 1), checkpoint_rank(p, 2))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curve.npz"
            np.savez_compressed(path, **curve)
            with np.load(path, allow_pickle=False) as lazy:
                self.assertEqual(constrained_point(lazy), p)

    def test_sampling_breaks_gaps_and_houses_preserves_class_mass(self):
        keys = [("2", 0), ("2", 8), ("2", 24), ("2", 32), ("19", 0), ("19", 8)]
        y = np.array([300, 300, 300, 0, 300, 0.])
        np.testing.assert_array_equal(experiment.event_groups(keys, y), [0, 0, 1, 2, 3, 4])
        p, audit = experiment.sampling_probabilities(keys, y, "C_EVENT")
        self.assertAlmostEqual(p.sum(), 1)
        self.assertAlmostEqual(p[y >= 200].sum(), 4 / 6)
        self.assertAlmostEqual(p[0] + p[1], p[2])
        self.assertEqual(audit["threshold_segments"], 3)
        p, audit = experiment.sampling_probabilities(keys, y, "C_HARD", np.array([.1, .2, .7, .9, .8, .1]))
        self.assertAlmostEqual(p[y >= 200].sum(), 4 / 6)
        self.assertGreater(p[0], p[1])
        with self.assertRaises(DataError):
            experiment.sampling_probabilities(keys, np.array([np.nan] * 6), "C_EVENT")

    def test_default_mae_and_opt_in_evidence_preserve_exact_training(self):
        f = self.fixture()
        a, b = f.root / "old", f.root / "evidence"
        train(f.manifest, f.settings_path, a, appliances=["microwave"])
        train(f.manifest, f.settings_path, b, appliances=["microwave"], checkpoint_evidence=True)
        from virtual_submeter.models.artifacts import load_model
        ma, old = load_model(a)
        mb, new = load_model(b)
        self.assertEqual(ma["selected_epoch"], mb["selected_epoch"])
        self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))
        history = read_json(b / "history.json")["epochs"]
        self.assertEqual(len(history), f.settings["epochs"])
        for r in history:
            with np.load(b / f"checkpoints/epoch_{r['epoch']}_microwave_pr.npz", allow_pickle=False) as curve:
                self.assertEqual(constrained_point(curve), r["state_evidence"][0]["constrained"])
        c = f.root / "precision"
        train(f.manifest, f.settings_path, c, appliances=["microwave"], checkpoint_metric="recall_precision")
        selected, _ = load_model(c)
        expected = min(history, key=lambda r: checkpoint_rank(r["state_evidence"][0]["constrained"], r["epoch"]))
        self.assertEqual(selected["selected_epoch"], expected["epoch"])
        self.assertEqual(selected["source_state_selection"]["point"], expected["state_evidence"][0]["constrained"])

    def test_training_rejects_target_labels_and_context_crossing_gap(self):
        f = self.fixture()
        protocol = read_json(f.manifest)
        target = [p for p in protocol["partitions"] if p["role"] == "frozen_test"]
        with self.assertRaises(DataError):
            dataset(f.manifest, target, ["microwave"], f.settings, 8)
        protocol["partitions"][0]["household_id"] = target[0]["household_id"]
        write_json(f.manifest, protocol)
        with self.assertRaises(DataError):
            train(f.manifest, f.settings_path, f.root / "bad", appliances=["microwave"], checkpoint_metric="recall_precision")

    def test_state_inference_missing_and_serialized_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            torch.manual_seed(42)
            model = StateClassifier(7, True)
            torch.save(model.state_dict(), path / "weights.pt")
            meta = {"kind": "microwave_state_classifier_v1", "truth_threshold_w": 200,
                    "decision_threshold_probability": .5, "weights_sha256": sha256(path / "weights.pt"),
                    "window_samples": 7, "features": True, "cpu_threads": 1, "batch_size": 4,
                    "input_log_mean": 0., "input_log_std": 1.}
            write_json(path / "model.json", meta)
            loaded_meta, loaded = load_state_model(path)
            rows = [{"household_id": "5", "timestamp": iso(i * 8), "aggregate_power_w": "500" if i != 10 else ""} for i in range(25)]
            result = predict_state_rows(loaded_meta, loaded, rows)
            self.assertEqual(len(result), 25)
            for i in range(25):
                expected_missing = i < 3 or i >= 22 or 7 <= i <= 13
                self.assertEqual(result[i]["state"] is None, expected_missing)
                self.assertEqual(result[i]["probability_on"] is None, expected_missing)
            (path / "weights.pt").write_bytes(b"changed")
            with self.assertRaises(DataError):
                load_state_model(path)

    def test_registration_and_source_selection_are_finite_and_support_paired(self):
        plan = experiment.load_plan(ROOT / "configs/state_20261024.json")
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "plan.json"
            write_json(p, {**plan, "seeds": [42]})
            with self.assertRaises(DataError):
                experiment.load_plan(p)
        rows = [{"candidate": g, "seed": s, "precision": .1, "recall": .9, "f1": .18,
                 "state_pattern": "mixed", "support_sha256": "same"} for g in experiment.GROUPS for s in experiment.SEEDS]
        self.assertEqual(experiment.choose_source(rows)["selected"], "R_PR")
        rows[0]["recall"] = .84
        self.assertEqual(experiment.choose_source(rows)["selected"], "C31")
        rows[0]["support_sha256"] = "different"
        with self.assertRaises(DataError):
            experiment.choose_source(rows)

    def test_evaluation_checks_freeze_before_reading_development_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path / "run.json", {"status": "frozen"})
            with patch.object(experiment, "registered"), patch.object(experiment, "validate_freeze", side_effect=DataError("tampered")), patch.object(experiment, "_truth") as reader:
                with self.assertRaises(DataError):
                    experiment.evaluate(path)
                reader.assert_not_called()

    def test_complete_synthetic_evaluation_separates_probability_and_power(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "run.json", {"status": "frozen"})
            write_json(run / "freeze.json", {"synthetic": True})
            write_json(run / "selection.json", {"selected": "C31"})
            write_json(run / "baseline_source_calibration.json", {str(s): {"threshold": 200} for s in experiment.SEEDS})
            stamps = np.arange(40) * 8
            truth = np.zeros((40, 3), np.float32)
            truth[10:15, 1] = 1000
            power = np.zeros_like(truth)
            power[10:15, 1] = 500
            scores = np.where(truth[:, 1] >= 200, .9, .1).astype(np.float32)
            cases = []
            for partition in (experiment.SOURCE_VALIDATION, experiment.DEVELOPMENT):
                for seed in experiment.SEEDS:
                    path = run / f"reference/predictions/seed_{seed}/{partition}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, stamps=stamps, aggregate=np.full(40, 1500), raw=power)
                    for candidate in experiment.GROUPS:
                        score = power[:, 1].copy() if candidate == "R_PR" else scores.copy()
                        if candidate == "C61":
                            score[:2] = score[-2:] = np.nan
                        path = run / f"predictions/{candidate}/seed_{seed}/{partition}.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, stamps=stamps, scores=score, power=power)
                        cases.append({"candidate": candidate, "seed": seed, "partition": partition,
                                      "threshold": 200 if candidate == "R_PR" else .8,
                                      "path": path.relative_to(run).as_posix()})
            with patch.object(experiment, "registered"), patch.object(experiment, "validate_freeze", return_value={"cases": cases}), patch.object(experiment, "_truth", return_value=truth):
                summary = experiment.evaluate(run)
            self.assertTrue(summary["other_devices_raw_unchanged"])
            rows = read_json(run / "metrics.json")["rows"]
            self.assertEqual(len(rows), 324)
            microwave = next(r for r in rows if r["candidate"] == "C31" and r["appliance"] == "microwave" and r["support"] == "common61")
            self.assertEqual(microwave["recall"], 1)
            self.assertEqual(microwave["precision"], 1)
            self.assertGreater(microwave["mae_w"], 0)
            self.assertEqual(microwave["decision_threshold"], .8)
            self.assertEqual(read_json(run / "run.json")["status"], "evaluated")

    def test_actual_freeze_hash_barrier_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "artifact.json", {"value": "original"})
            cases = [{"candidate": g, "seed": s, "partition": p, "path": "artifact.json", "sha256": sha256(run / "artifact.json")}
                     for g in experiment.GROUPS for s in experiment.SEEDS
                     for p in (experiment.SOURCE_VALIDATION, experiment.DEVELOPMENT)]
            write_json(run / "freeze.json", {"development_labels_parsed": False, "cases": cases, "models": 18,
                       "registered_inputs": {}, "artifacts": {"artifact.json": sha256(run / "artifact.json")}})
            write_json(run / "run.json", {"status": "frozen", "freeze_sha256": sha256(run / "freeze.json")})
            write_json(run / "artifact.json", {"value": "tampered"})
            with patch.object(experiment, "registered"), patch.object(experiment, "_inputs", return_value={}), patch.object(experiment, "_truth") as reader:
                with self.assertRaisesRegex(DataError, "Frozen state artifact changed"):
                    experiment.validate_freeze(run)
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
