"""Non-training checks for Task24 draft budget gate."""
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_microwave_task25.py'
spec = importlib.util.spec_from_file_location('task24_budget_draft', SCRIPT)
task24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task24)
sys.path.insert(0, str(SCRIPT.parents[1] / 'src'))
from virtual_submeter.accuracy.microwave_source5_loss import qualified
from virtual_submeter.accuracy.microwave_task21 import isolation
from virtual_submeter.data.common import DataError
import torch


class BudgetGate(unittest.TestCase):
    def test_task24_receipt_cannot_authorize_task25(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            reg = run / 'registration.json'
            reg.write_text(json.dumps({'status':'sealed_pending_review','authorized':False}), encoding='utf-8')
            approval = run / 'old_budget.json'
            approval.write_text(json.dumps({'authorized':True,'approval_origin':'human',
                'experiment':'microwave_task24_multihouse_v1','registration_sha256':task24.sha(reg),
                'additional_new_fits':3,'maximum_updates':10584,'seeds':[42,1019,2026],
                'human_authorization_quote':'fixture','human_authorization_reference':'fixture'}),encoding='utf-8')
            with self.assertRaisesRegex(PermissionError,'exact Task25 budget'):
                task24.require_budget(run,approval)

    def test_preparation_isolation_rejects_unlisted_read_network_and_backward(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / 'run'
            with isolation(run, []):
                with self.assertRaises(DataError):
                    (Path(td) / 'unlisted.npz').open('rb')
                with self.assertRaises(DataError):
                    socket.getaddrinfo('example.invalid', 443)
                with self.assertRaises(DataError):
                    torch.tensor(1., requires_grad=True).backward()

    def test_source_guard_rejects_missing_degenerate_tie_and_recall_drop(self):
        seeds = (42, 1019, 2026)
        control = [{'seed': s, 'f1': .5, 'recall': .97, 'precision': .2,
                    'degenerate': False, 'source_gate_passed': True} for s in seeds]
        candidate = [dict(row, precision=.21) for row in control]
        checkpoints = [{'seed': s, 'source_gate_passed': True} for s in seeds]
        self.assertTrue(task24.source_eligible(control, candidate, checkpoints, qualified))
        self.assertFalse(task24.source_eligible(control, candidate[:2], checkpoints, qualified))
        self.assertFalse(task24.source_eligible(control, candidate, checkpoints[:2], qualified))
        self.assertFalse(task24.source_eligible(control, [dict(r, degenerate=True) for r in candidate], checkpoints, qualified))
        self.assertFalse(task24.source_eligible(control, [dict(r, precision=.2) for r in candidate], checkpoints, qualified))
        candidate[0]['recall'] = .91
        self.assertFalse(task24.source_eligible(control, candidate, checkpoints, qualified))

    def test_source_ranking_uses_mean_precision_then_f1_with_control_ties(self):
        seeds = (42, 1019, 2026)
        control = [dict(seed=s, precision=.25, f1=.5, recall=.97, degenerate=False,
                        source_gate_passed=True) for s in seeds]
        checkpoints = [dict(seed=s, source_gate_passed=True) for s in seeds]
        def candidate(precisions, f1):
            return [dict(row, precision=p, f1=f1) for row, p in zip(control, precisions)]
        self.assertFalse(task24.source_eligible(control, candidate([.375, .375, 0.0], .5), checkpoints, qualified))
        self.assertTrue(task24.source_eligible(control, candidate([.375, .375, 0.0], .51), checkpoints, qualified))
        self.assertFalse(task24.source_eligible(control, candidate([.3, .3, 0.0], .51), checkpoints, qualified))

    def test_missing_budget_is_rejected_before_opening_run(self):
        with self.assertRaisesRegex(PermissionError, 'new human budget'):
            task24.require_budget(Path('does-not-exist'), None)

    def test_wrong_budget_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            reg = run / 'registration.json'
            reg.write_text(json.dumps({'status': 'sealed_pending_review', 'authorized': False}), encoding='utf-8')
            approval = run / 'approval.json'
            approval.write_text(json.dumps({'authorized': True, 'approval_origin': 'human',
                'experiment': 'microwave_task25_positive_share_v1', 'registration_sha256': task24.sha(reg),
                'additional_new_fits': 3, 'maximum_updates': 10585, 'seeds': [42, 1019, 2026],
                'human_authorization_quote': 'fixture', 'human_authorization_reference': 'fixture'}), encoding='utf-8')
            with self.assertRaisesRegex(PermissionError, 'exact Task25 budget'):
                task24.require_budget(run, approval)

    def test_unsealed_preparation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            reg = run / 'registration.json'
            reg.write_text(json.dumps({'status': 'draft', 'authorized': False}), encoding='utf-8')
            approval = run / 'approval.json'
            approval.write_text(json.dumps({'authorized': True, 'approval_origin': 'human',
                'experiment': 'microwave_task25_positive_share_v1', 'registration_sha256': task24.sha(reg),
                'additional_new_fits': 3, 'maximum_updates': 10584, 'seeds': [42, 1019, 2026],
                'human_authorization_quote': 'fixture', 'human_authorization_reference': 'fixture'}), encoding='utf-8')
            with self.assertRaisesRegex(PermissionError, 'sealed before independent review'):
                task24.require_budget(run, approval)

    def test_registered_mutation_and_review_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            adapter = run / SCRIPT.name
            shutil.copyfile(SCRIPT, adapter)
            source = run / 'source/virtual_submeter/example.py'
            source.parent.mkdir(parents=True)
            source.write_text('# fixture\n', encoding='utf-8')
            spec = importlib.util.spec_from_file_location('sealed_task24_fixture', adapter)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            reg = run / 'registration.json'
            reg.write_text(json.dumps({'status': 'sealed_pending_review', 'authorized': False,
                'artifacts': {adapter.name: module.sha(adapter), source.relative_to(run).as_posix(): module.sha(source)},
                'inputs': {}}), encoding='utf-8')
            reviews = ['independent_review_1.json', 'independent_review_2.json']
            for name in reviews:
                (run / name).write_text(json.dumps({'passed': True, 'reviewed_registration_sha256': module.sha(reg)}), encoding='utf-8')
            approval = run / 'approval.json'
            payload = {'authorized': True, 'approval_origin': 'human', 'experiment': 'microwave_task25_positive_share_v1',
                'registration_sha256': module.sha(reg), 'additional_new_fits': 3, 'maximum_updates': 10584,
                'seeds': [42, 1019, 2026], 'human_authorization_quote': 'fixture', 'human_authorization_reference': 'fixture',
                'review_sha256': {name: module.sha(run / name) for name in reviews}}
            approval.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(module.require_budget(run, approval), payload)
            source.write_text('# changed\n', encoding='utf-8')
            with self.assertRaisesRegex(PermissionError, 'artifact changed'):
                module.require_budget(run, approval)
            source.write_text('# fixture\n', encoding='utf-8')
            (run / reviews[0]).write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(PermissionError, 'bind both review'):
                module.require_budget(run, approval)


if __name__ == '__main__':
    unittest.main()
