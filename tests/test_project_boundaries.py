"""Check unsupported scope, split guards, and truthful initialization status."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.cli import main
from virtual_submeter.config import ConfigError, load_config


class ProjectBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "configs/default.toml").read_text(encoding="utf-8")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "experiment.toml"

    def write_config(self, text):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_valid_configuration_does_not_claim_data_or_model_results(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["status", "--config", str(ROOT / "configs/default.toml"), "--json"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["stage"], "m4_trustworthy_output")
        self.assertEqual(result["project"], "ElectricityPlot")
        self.assertEqual(result["data_validation"], "not_run")
        self.assertEqual(result["training"], "available_not_run_by_status")
        self.assertEqual(result["experiments"], "not_run")

    def test_generation_storage_and_different_measurements_are_rejected(self):
        for old, new in (
            ("supports_pv = false", "supports_pv = true"),
            ("supports_storage = false", "supports_storage = true"),
            ('input_kind = "aggregate_active_power"', 'input_kind = "monthly_energy"'),
        ):
            with self.subTest(new=new), self.assertRaises(ConfigError):
                load_config(self.write_config(self.source.replace(old, new)))

    def test_adaptation_protocol_requires_zero_budget_and_isolation(self):
        for old, new in (
            ("[0, 5, 10, 20]", "[5, 10, 20]"),
            ("split_gap_seconds = 3600", "split_gap_seconds = 600"),
            ("sample_interval_seconds = 8", "sample_interval_seconds = true"),
        ):
            with self.subTest(new=new), self.assertRaises(ConfigError):
                load_config(self.write_config(self.source.replace(old, new)))

    def test_configuration_cannot_route_data_to_external_paths(self):
        for value in ("../private", "C:/private", "/private", "data/../../private"):
            with self.subTest(path=value), self.assertRaises(ConfigError):
                load_config(self.write_config(self.source.replace('"data/raw"', json.dumps(value))))

    def test_misspelled_fields_and_duplicate_targets_are_rejected(self):
        for text in (
            self.source.replace("query_budgets", "query_budget"),
            self.source.replace('["kettle", "microwave", "washing_machine"]', '["kettle", "kettle"]'),
        ):
            with self.subTest(config=text), self.assertRaises(ConfigError):
                load_config(self.write_config(text))

    def test_invalid_config_has_failure_status(self):
        path = self.write_config(self.source.replace("supports_pv = false", "supports_pv = true"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["validate-config", "--config", str(path), "--json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "invalid_config")


if __name__ == "__main__":
    unittest.main()
