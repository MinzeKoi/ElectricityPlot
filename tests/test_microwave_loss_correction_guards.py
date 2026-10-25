"""Task-specific negative lifecycle tests: no truth access on incomplete evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from virtual_submeter.accuracy import microwave_loss_correction as f


class GuardTests(unittest.TestCase):
    def test_incomplete_fit_matrix_rejected_before_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            (run/'plan.json').write_text('{}')
            (run/'training_complete.json').write_text(json.dumps({'fits':5,'completed_models':[]}))
            with patch.object(f,'verify_preparation',return_value={}),patch.object(f,'load_npz',side_effect=AssertionError('Array opened before incomplete matrix rejection')):
                with self.assertRaisesRegex(ValueError,'Incomplete six-fit matrix'):
                    f.verify_training(run,predictions=True)

    def test_incomplete_evaluation_rejected_before_truth(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            with patch.object(f,'authorize',return_value={}),patch.object(f,'verify_training',side_effect=f.DataError('Incomplete nine-prediction matrix')):
                with self.assertRaisesRegex(ValueError,'Incomplete nine-prediction matrix'):
                    f.evaluate(run,run/'authorization.json')
            self.assertFalse((run/'evaluation').exists())

    def test_changed_freeze_denies_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            (run/'registration.json').write_text('{}')
            (run/'freeze.json').write_text('{}')
            a=dict(registration_sha256=f.sha256(run/'registration.json'),maximum_fits=6,evaluate=True,
                   coordinator_execution_message='task8',freeze_sha256='wrong')
            (run/'authorization.json').write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'not bound to freeze'):
                f.evaluate(run,run/'authorization.json')
            self.assertFalse((run/'evaluation').exists())


if __name__=='__main__': unittest.main()
