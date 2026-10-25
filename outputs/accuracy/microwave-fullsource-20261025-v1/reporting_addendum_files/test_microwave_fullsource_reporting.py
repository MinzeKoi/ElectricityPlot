import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from virtual_submeter.accuracy.microwave_fullsource_reporting import improvement_flags


class ReportingTests(unittest.TestCase):
    def test_natural_can_meet_overall_goal_without_mechanism_claim(self):
        result = improvement_flags({"selected_before_development": "natural", "development_acceptance_passed": True, "improvement_candidate": False})
        self.assertTrue(result["declared_improvement"])
        self.assertFalse(result["sampling_or_background_improvement_supported"])

    def test_augmentation_claim_requires_pass_and_nonnatural_selection(self):
        for passed in (False, True):
            result = improvement_flags({"selected_before_development": "event", "development_acceptance_passed": passed, "improvement_candidate": True})
            self.assertEqual(result["declared_improvement"], passed)
            self.assertEqual(result["sampling_or_background_improvement_supported"], passed)

    def test_no_candidate_has_no_improvement(self):
        result = improvement_flags({"selected_before_development": None, "development_acceptance_passed": False, "improvement_candidate": False})
        self.assertFalse(result["declared_improvement"])


if __name__ == "__main__":
    unittest.main()
