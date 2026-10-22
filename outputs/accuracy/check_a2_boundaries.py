import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from virtual_submeter.data.common import read_json, sha256, write_json
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows

run = root / 'outputs/accuracy/a2-20261021'
frozen = run / 'frozen'
cases = read_json(run / 'freeze.json')['cases']
plan = read_json(frozen / 'plan.json')
checks = dict(cases=0, power_weights_unchanged=0, no_answer_exact_source_devices=0,
              bounded_devices=0, single_class_slope_preserved_devices=0)
for case in cases:
    directory = frozen / case['path'] / 'model'
    metadata = read_json(directory / 'model.json')
    base = read_json(frozen / f"bases/seed_{case['source_seed']}/model.json")
    source = read_json(frozen / f"source_heads/seed_{case['source_seed']}/source_head.json")
    assert sha256(directory / 'weights.pt') == base['weights']['sha256']
    checks['power_weights_unchanged'] += 1
    head = metadata['accuracy_state_head']
    records = read_json(directory / 'queries.json')['records']
    assert len(records) == case['budget']
    for appliance, coefficients in head['coefficients'].items():
        variant = case['variant']
        prior = np.array(source['coefficients'][appliance] if variant.startswith('L2') else [0, 1, 0] if variant == 'L0' else [0, 1])
        answers = [r for r in records if r['appliance_type'] == appliance and r['response'] in ('used', 'not_used')]
        delta = np.array(coefficients) - prior
        if not answers:
            np.testing.assert_array_equal(delta, np.zeros(len(prior)))
            checks['no_answer_exact_source_devices'] += 1
        if answers and variant not in ('L0', 'L2_max_unprotected'):
            classes = {r['response'] for r in answers}
            if len(classes) == 1 or variant == 'L1':
                np.testing.assert_array_equal(delta[1:], np.zeros(len(prior) - 1))
                bound = plan['single_class_bias_bound'] if len(classes) == 1 else plan['bias_bound']
                assert abs(delta[0]) <= bound + 1e-12
                checks['single_class_slope_preserved_devices'] += int(len(classes) == 1)
            else:
                assert np.linalg.norm(delta) <= plan['parameter_radius'] + 1e-12
            checks['bounded_devices'] += 1
    checks['cases'] += 1

case = next(c for c in cases if c['source_seed'] == 1019 and c['query_seed'] == 42 and c['variant'] == 'L2_max' and c['budget'] == 10)
model_dir = frozen / case['path'] / 'model'
metadata, model = load_model(model_dir)
aggregate = read_partition(frozen / 'protocol/protocol.json', case['future_partition'], 'inference')['aggregate']
rows = predict_rows(metadata, model, aggregate)
with np.load(frozen / case['path'] / 'states.npz', allow_pickle=False) as stored:
    actual = np.array([[r['appliances'][a]['state_score'] if r['appliances'][a]['state_score'] is not None else np.nan
                        for a in metadata['appliances']] for r in rows])
    np.testing.assert_allclose(actual, stored['scores'], rtol=0, atol=1e-12, equal_nan=True)
checks.update(status='passed', real_model_reload_score_parity=True, future_pointwise_labels_read=False,
              inference_rows=len(rows), code_sha256=read_json(run / 'run.json')['code_sha256'])
write_json(root / 'outputs/accuracy/a2-boundary-check.json', checks)
print(json.dumps(checks, indent=2))
