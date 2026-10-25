import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from virtual_submeter.accuracy import microwave_regression_calibration as c
from virtual_submeter.data.common import DataError


class RegressionCalibrationTests(unittest.TestCase):
    def test_list_valued_rows_can_be_saved_reopened_and_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [{'candidate': 'W50', 'seed': 42, 'threshold': 7.5}]
            c.save(run/'source_rows.json', rows)
            self.assertEqual(c.source_rows(run), rows)
            c.save(run/'source_rows.json', rows)
            with self.assertRaises(DataError):
                c.save(run/'source_rows.json', [])

    def test_full_matrix_freeze_required_for_label_barrier(self):
        f = {'cases': [{'candidate': g, 'seed': s, 'partition': c.DEV}
                       for g in c.FAMILIES for s in c.SEEDS],
             'development_labels_parsed': False, 'source_complete_sha256': 'source'}
        def stage(run, name):
            return f if name == 'freeze' else {}
        with patch.object(c, 'context'), patch.object(c, 'stage_check', side_effect=stage), patch.object(c, 'sha256', return_value='source'):
            self.assertEqual(c.validate_freeze(Path('run')), f)
            f['cases'].pop()
            with self.assertRaises(DataError):
                c.validate_freeze(Path('run'))

    def test_labels_align_by_house_and_timestamp_with_missing(self):
        rows = [{'household_id': 'refit_5', 'timestamp': '1970-01-01T00:00:16Z', 'power_w': '240'},
                {'household_id': 'refit_5', 'timestamp': '1970-01-01T00:00:00Z', 'power_w': ''}]
        actual = c.aligned_truth(rows, np.array([0, 8, 16]), 'refit_5')
        np.testing.assert_equal(actual, [np.nan, np.nan, 240])
        with self.assertRaises(DataError):
            c.aligned_truth(rows, np.array([0, 8, 16]), 'refit_17')
        with self.assertRaises(DataError):
            c.aligned_truth(rows, np.array([8, 0]), 'refit_5')

    def test_missing_is_unavailable_not_off(self):
        np.testing.assert_array_equal(c.decisions(np.array([np.nan, 0, 3]), 2), [-1, 0, 1])
        np.testing.assert_array_equal(c.decisions(np.array([0, 3]), None), [-1, -1])

    def test_exact_pr_includes_ties_and_block_constraint(self):
        score = np.array([0., 2., 2., 3., 0., 1., 2., 4.])
        truth = np.array([0, 200, 200, 0, 0, 200, 200, 0.])
        masks = np.array([np.arange(8) < 4, np.arange(8) >= 4])
        result, pr = c.calibrate(score, truth, masks)
        feasible = []
        for t in np.unique(score):
            ms = [c.state_metrics(score[m], truth[m], t) for m in masks]
            if all(r['recall'] >= .85 and r['state_pattern'] == 'mixed' for r in ms):
                feasible.append((min(r['precision'] for r in ms), np.mean([r['precision'] for r in ms]),
                                 np.mean([r['f1'] for r in ms]), t))
        self.assertEqual(result['threshold'], max(feasible)[-1])
        self.assertEqual(pr['block_0_tp'][2], 2)

    def test_reject_degenerate_only_recall_solution(self):
        result, _ = c.calibrate(np.ones(4), np.array([0, 0, 200, 200]), np.ones((1, 4), bool))
        self.assertFalse(result['passed'])
        self.assertIsNone(result['threshold'])

    def test_complete_matrix_and_source_selection(self):
        with self.assertRaises(DataError):
            c.choose([])
        rows = []
        for g in c.FAMILIES:
            for s in c.SEEDS:
                p = .2 if g == 'W1' else .1
                rows.append({'candidate': g, 'seed': s, 'passed': True,
                             'metrics': {'precision': p, 'recall': .9, 'f1': p}})
        self.assertEqual(c.choose(rows)['selected'], 'W1')
        rows[6]['passed'] = False
        self.assertEqual(c.choose(rows)['selected'], 'W50')
        with self.assertRaises(DataError):
            c.choose(rows[:-1]+[rows[0]])

    def test_evaluation_cannot_open_labels_before_complete_freeze(self):
        with patch.object(c, 'validate_freeze', side_effect=DataError('incomplete')), patch.object(c.np, 'load') as load:
            with self.assertRaises(DataError):
                c.evaluate(Path('not_a_run'))
            load.assert_not_called()

    def test_fixed_baseline_remains_and_every_seed_recall_guard(self):
        rows = []
        for s in c.SEEDS:
            for g, p, r, f, fp in [('W50', .04, .92, .077, 8000), ('W1_CAL', .1, .9, .18, 3000)]:
                rows.append({'candidate': g, 'seed': s, 'partition': c.DEV, 'appliance': 'microwave',
                             'view': 'raw', 'precision': p, 'recall': r, 'f1': f, 'false_positive': fp,
                             'false_negative': 10, 'support_sha256': 'same', 'valid_samples': 137523, 'state_pattern': 'mixed'})
        self.assertTrue(c.gate(rows, 'W1')['precision_exit_passed'])
        rows[1]['recall'] = .84
        self.assertFalse(c.gate(rows, 'W1')['precision_exit_passed'])
        rows[1]['recall'] = .9
        rows[1]['state_pattern'] = 'all_on'
        self.assertFalse(c.gate(rows, 'W1')['precision_exit_passed'])
        rows[1]['state_pattern'] = 'mixed'
        rows[1]['valid_samples'] = 137522
        self.assertFalse(c.gate(rows, 'W1')['precision_exit_passed'])


if __name__ == '__main__':
    unittest.main()
