"""Read back real expansion models through ordinary prediction, without opening labels."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.data_expansion import verify
from virtual_submeter.data.common import write_json, sha256
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Readback evidence output exists')
    verify(args.run)
    checks = []
    for seed in (42, 1019, 2026):
        original, network = load_model(ROOT / f'outputs/accuracy/r1-20261022/selected/seed_{seed}')
        reproduced, current = load_model(args.run / f'models/A/seed_{seed}/packed')
        for key in ('input_log_mean', 'input_log_std', 'power_scales_w', 'sample_interval_seconds', 'state_threshold_view'):
            if original[key] != reproduced[key]:
                raise ValueError('A metadata differs from historical R1: ' + key)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
        checks.append({'seed': seed, 'historical_R1_A_weights_and_physical_scaling_identical': True})
    for group in ('A', 'B', 'C'):
        manifest = args.run / f'protocols/{group}/protocol.json'
        pid = 'frozen_test_refit_17_0'
        # Input-only consumer; deliberately never asks for evaluation/oracle labels.
        inputs = read_partition(manifest, pid, 'inference')['aggregate'][:4000]
        for seed in (42, 1019, 2026):
            path = args.run / f'models/{group}/seed_{seed}/packed'
            meta, model = load_model(path)
            predictions = predict_rows(meta, model, inputs)
            compact = args.run / f'predictions/{group}/seed_{seed}/{pid}.npz'
            width = meta['settings']['window_samples'] // 2
            # The saved full series has context beyond this probe's right edge.
            stop = len(inputs) - width
            with np.load(compact, allow_pickle=False) as saved:
                valid = 0
                for index, appliance in enumerate(meta['appliances']):
                    for array, field in (('raw', 'raw_power_w'), ('corrected', 'power_w')):
                        values = np.array([r['appliances'][appliance][field] for r in predictions[:stop]], dtype=float)
                        np.testing.assert_allclose(values, saved[array][:stop, index], atol=1e-5, rtol=1e-5, equal_nan=True)
                        valid += int(np.isfinite(values).sum())
            checks.append({'group': group, 'seed': seed, 'model_sha256': sha256(path / 'model.json'),
                           'ordinary_prediction_matches_compact': True, 'finite_values_checked': valid})
    write_json(args.output, {'status': 'real_models_readback_verified', 'checks': checks, 'labels_reopened': False})
    print('Real model readback verified; no labels reopened')


if __name__ == '__main__':
    main()
