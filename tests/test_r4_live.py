"""Shared live fits, restart recovery and portable replay without target labels."""

import contextlib
import io
from pathlib import Path
import shutil
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


@unittest.skipUnless(NUMERIC, "R4 requires numeric CPU environment")
class LiveTest(unittest.TestCase):
    def setUp(self):
        from test_a2_learners import ExperimentTest
        ExperimentTest.setUp(self)
        self.query = ROOT / "configs/r2_stability_20261023.json"
        self.directory = self.root / "live"

    def create(self, query=None, source=None):
        from virtual_submeter.accuracy import live
        return live.create(self.directory, self.model, self.manifest, "target_adaptation_house_3",
                           self.plan_path, query or self.query, ROOT / "configs/m3_adaptation.json",
                           "random", 42, 5, source)

    def test_same_fit_recovery_budget_and_no_hidden_label_access(self):
        from virtual_submeter.accuracy import live
        from virtual_submeter.accuracy.learners import fit_adapter
        from virtual_submeter.accuracy.dynamic_queries import _export
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict
        original = Path.open
        def guard(path, *args, **kwargs):
            if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")):
                raise AssertionError("Live service touched target submeter data")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            self.create()
            for index in range(5):
                current = live.status(self.directory)
                self.assertEqual(current["fitted_budget"], 0)
                pending = current["pending"]
                self.assertEqual(pending, live.status(self.directory)["pending"])
                response = ["used", "not_used", "unsure", "skipped", "not_used"][index]
                if index == 4:
                    with patch.object(live.sessions, "commit_round", side_effect=RuntimeError("interrupted fit")):
                        with self.assertRaisesRegex(RuntimeError, "interrupted"):
                            live.answer(self.directory, pending["query_id"], response)
                else:
                    live.answer(self.directory, pending["query_id"], response)
                with self.assertRaises(DataError):
                    live.answer(self.directory, pending["query_id"], response)
            current = live.status(self.directory)
            self.assertEqual((current["budget_used"], current["fitted_budget"], current["pending"]), (5, 5, None))
            metadata, _ = load_model(self.model)
            rows = read_json(self.directory / "prediction/predictions.json")["rows"]
            adapter, history = fit_adapter(rows, current["records"], metadata, self.plan, "L1_stable", legacy_settings=read_json(ROOT / "configs/m3_adaptation.json"))
            expected = _export(self.model, metadata, adapter, current["records"],
                               {"learner_settings": self.plan, "query_plan": read_json(self.query), "strategy": "random"},
                               history, self.root / "experiment_model", None, "random")
            actual = read_json(Path(current["model"]) / "model.json")
            self.assertEqual(actual["model_id"], expected["model_id"])
            self.assertEqual(actual["accuracy_state_head"], adapter)
            predict(Path(current["model"]), self.root / "after", self.directory / "inputs/protocol/protocol.json", current["partition_id"])
            result = read_json(self.root / "after/predictions.json")
            for before, after in zip(rows, result["rows"]):
                self.assertEqual(before["unexplained_power_w"], after["unexplained_power_w"])
                for a in metadata["appliances"]:
                    self.assertEqual(before["appliances"][a]["power_w"], after["appliances"][a]["power_w"])
            report = (self.root / "after/report.html").read_text(encoding="utf-8")
            self.assertIn("确认预算使用 5 次", report)
            self.assertNotIn("本模型未经目标家庭适配", report)
            self.assertFalse(any(p.is_file() for p in (self.directory / "inputs").rglob("*labels*")))
        path = self.directory / "inputs/learner.json"
        write_json(path, {})
        with self.assertRaises(DataError):
            live.status(self.directory)

    def test_l2_source_context_and_portable_release_relocated(self):
        from virtual_submeter.accuracy import live, release
        from virtual_submeter.accuracy.fixed_queries import train_source_head
        source = self.root / "source"
        source.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            train_source_head(self.model, self.manifest, self.plan, source)
        self.create(ROOT / "configs/a3_queries.json", source)
        for _ in range(5):
            pending = live.status(self.directory)["pending"]
            live.answer(self.directory, pending["query_id"], "skipped")
        bundle = self.root / "bundle"
        release.build(self.directory, bundle, ROOT)
        relocated = self.root / "relocated"
        shutil.copytree(bundle, relocated)
        self.directory.rename(self.root / "original_unavailable")
        receipt = release.replay(relocated, self.root / "replayed")
        self.assertEqual(receipt["status"], "portable_replay_verified")
        self.assertFalse(receipt["independent_member_reproduction"])
        (relocated / "records.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            release.replay(relocated, self.root / "bad_replay")

    def test_ui_uses_service_recovers_and_clears_invalid_model(self):
        from streamlit.testing.v1 import AppTest
        from virtual_submeter.accuracy import live
        inputs = self.root / "inputs"
        shutil.copytree(self.model, inputs / "model")
        shutil.copytree(self.manifest.parent, inputs / "protocol")
        for source, name in ((self.plan_path, "learner.json"), (self.query, "query.json"),
                             (ROOT / "configs/m3_adaptation.json", "legacy.json")):
            shutil.copyfile(source, inputs / name)
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(self.root)) + "))"
        script_path = self.root / "app_test.py"
        script_path.write_text(script, encoding="utf-8")
        app = AppTest.from_file(script_path, default_timeout=30).run()
        self.assertFalse(app.exception)
        def element(items, label):
            return next(x for x in items if x.label == label)
        element(app.selectbox, "确认预算").select(5).run()
        element(app.button, "新建确认会话").click().run()
        self.assertFalse(app.exception)
        directory = Path(app.session_state["r4_session_directory"])
        for response in ("used", "not_used", "unsure", "skipped", "not_used"):
            element(app.radio, "你的回答").set_value(response)
            next(b for b in app.button if b.label == "提交回答").click().run()
            self.assertFalse(app.exception)
        self.assertEqual(live.status(directory)["fitted_budget"], 5)
        element(app.button, "加载确认会话").click().run()
        self.assertFalse(app.exception)
        element(app.button, "查看当前适配前后状态并导出").click().run()
        self.assertFalse(app.exception)
        self.assertIn("r4_comparison", app.session_state)
        write_json(inputs / "model/model.json", {})
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("模型不可用" in x.value for x in app.error))
        self.assertNotIn("r4_comparison", app.session_state)

    def test_ui_no_data_or_model_never_shows_old_results(self):
        from streamlit.testing.v1 import AppTest
        empty = self.root / "empty"
        empty.mkdir()
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(empty)) + "))"
        script_path = self.root / "app_test.py"
        script_path.write_text(script, encoding="utf-8")
        app = AppTest.from_file(script_path, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn("prediction_directory", app.session_state)

    def test_user_aggregate_import_can_confirm_without_any_submeter(self):
        from virtual_submeter.accuracy import live
        from virtual_submeter.data.common import safe_child, sha256
        protocol = read_json(self.manifest)
        part = next(p for p in protocol["partitions"] if p["role"] == "target_adaptation")
        aggregate = safe_child(self.manifest.parent, part["artifacts"]["aggregate"]["path"])
        bound = live.bind_import(self.model, self.manifest, aggregate, self.root / "bound")
        imported = read_json(Path(bound["manifest"]))
        self.assertEqual(imported["purpose"], "user_import_no_ground_truth")
        self.assertEqual(set(imported["partitions"][0]["artifacts"]), {"aggregate"})
        self.assertEqual(sha256(Path(bound["model"]) / "weights.pt"), sha256(self.model / "weights.pt"))
        current = live.create(self.directory, Path(bound["model"]), Path(bound["manifest"]), bound["partition_id"],
                              self.plan_path, self.query, ROOT / "configs/m3_adaptation.json", "random", 42, 5)
        for _ in range(5):
            pending = current["pending"]
            live.answer(self.directory, pending["query_id"], "not_used")
            current = live.status(self.directory)
        self.assertEqual(current["fitted_budget"], 5)


if __name__ == "__main__":
    unittest.main()
