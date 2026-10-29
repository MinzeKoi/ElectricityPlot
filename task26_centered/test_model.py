import unittest
import numpy as np
import torch
from virtual_submeter.models.center_mask_tcn import CenterMaskTCN, tensor_input
from task26_centered.model import CenteredTCN, centered_input
from task26_centered.train_core import fit_seed


class RepresentationTest(unittest.TestCase):
    def test_representation_and_start(self):
        torch.set_num_threads(1)
        meta = {'input_log_mean': 3.0, 'input_log_std': 2.0}
        rng = np.random.default_rng(6)
        x = rng.uniform(100, 2500, (3, 127)).astype(np.float32)
        x[0, 0] = np.nan
        x[1, 50] = np.nan
        old = CenterMaskTCN()
        rng_state = torch.get_rng_state().clone()
        new = CenteredTCN(old)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        for name, value in old.state_dict().items():
            other = new.state_dict()[name]
            self.assertTrue(torch.equal(value, other[:, :2] if name == "project.weight" else other))
        z = centered_input(meta, x)
        self.assertEqual(sum(p.numel() for p in old.parameters()), 5329)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 5345)
        self.assertTrue(torch.equal(z[:, :2], tensor_input(meta, x)))
        self.assertEqual(z[0, 2, 0].item(), 0)
        self.assertEqual(z[1, 2, 50].item(), 0)
        shifted = np.where(np.isfinite(x), x + 500, x)
        np.testing.assert_allclose(centered_input(meta, shifted)[:, 2].numpy(), z[:, 2].numpy(), atol=5e-7)
        with torch.no_grad():
            torch.testing.assert_close(old(tensor_input(meta, x)), new(z), rtol=0, atol=0)
        self.assertTrue(torch.count_nonzero(new.project.weight[:, 2]) == 0)

    def test_synthetic_first_update(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        original = CenterMaskTCN()
        candidate = CenteredTCN(original)
        data = np.random.default_rng(9).uniform(10, 3000, (256, 127)).astype(np.float32)
        labels = torch.cat([torch.ones(16), torch.zeros(240)])
        meta = {"input_log_mean": 3.0, "input_log_std": 2.0}
        inputs = centered_input(meta, data)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
        before = candidate.project.weight[:, 2].detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(candidate(inputs), labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(candidate.project.weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, candidate.project.weight[:, 2]))

    def test_fit_core_synthetic(self):
        seed = 42
        torch.manual_seed(seed)
        initial = CenterMaskTCN().state_dict()
        x = np.random.default_rng(10).uniform(0, 3000, (256, 127)).astype(np.float32)
        train = {'x': x, 'mask': np.isfinite(x), 'y': np.r_[np.full(16, 200), np.zeros(240)]}
        blocks = {'block_4': {'x': x[:3], 'mask': np.isfinite(x[:3])}}
        plan = {'cpu_threads': 1, 'learning_rate': .001, 'checkpoints': 1, 'steps_per_checkpoint': 1, 'batch_size': 256}
        events = []
        model, scores = fit_seed(seed, initial, train, blocks, np.arange(256)[None, :],
                                 {'input_log_mean': 3, 'input_log_std': 2}, plan,
                                 lambda epoch, model, scores: events.append((epoch, len(scores['block_4']))))
        self.assertEqual(events, [(1, 3)])
        self.assertEqual(len(scores), 1)
        self.assertGreater(torch.count_nonzero(model.project.weight[:, 2]).item(), 0)

    def test_fit_core_rejects_partial_validation(self):
        torch.manual_seed(42)
        initial = CenterMaskTCN().state_dict()
        x = np.ones((256, 127), np.float32)
        train = {'x': x, 'mask': np.isfinite(x), 'y': np.r_[np.full(16, 200), np.zeros(240)]}
        val = x[:1].copy()
        val[0, 55] = np.nan
        with self.assertRaisesRegex(ValueError, 'complete core31'):
            fit_seed(42, initial, train, {'block_4': {'x': val, 'mask': np.isfinite(val)}},
                     np.arange(256)[None, :], {'input_log_mean': 0, 'input_log_std': 1},
                     {'cpu_threads': 1, 'learning_rate': .001, 'checkpoints': 1,
                      'steps_per_checkpoint': 1, 'batch_size': 256}, lambda *args: None)

    def test_empty_batch(self):
        result = centered_input({"input_log_mean": 0, "input_log_std": 1}, np.empty((0, 127), np.float32))
        self.assertEqual(tuple(result.shape), (0, 3, 127))

    def test_short_core_rejected(self):
        x = np.ones((1, 127), dtype=np.float32)
        x[:, 48:56] = np.nan
        with self.assertRaises(ValueError):
            centered_input({'input_log_mean': 0, 'input_log_std': 1}, x)

if __name__ == '__main__':
    unittest.main()
