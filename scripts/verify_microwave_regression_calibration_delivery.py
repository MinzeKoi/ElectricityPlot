"""Additional direct checks and recipe integration; never fits a model."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from virtual_submeter.accuracy import microwave_regression_calibration as c


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    c.validate_freeze(run)
    c.stage_check(run, 'evaluation_complete')
    reg, parent = c.context(run)
    data = c.read_partition(parent/'protocol/protocol.json', c.SV, 'calibration', include_labels=True)
    stamps, _ = c.aggregate_arrays(data['aggregate'], 8)
    truth = c.aligned_truth(data['microwave'], stamps, 'refit_5')
    support_parent = (ROOT/reg['plan']['block_support']).parent
    masks = []
    for i in range(6):
        with np.load(support_parent/f'validation/block_{i}.npz', allow_pickle=False) as a:
            masks.append(np.isin(stamps, a['t']))
    rows = json.loads((run/'source_rows.json').read_text(encoding='utf-8'))
    direct_points = 0
    baseline_support = None
    for r in rows:
        g, s = r['candidate'], r['seed']
        original = c.old_arrays(parent, g, s, c.SV)
        assert np.array_equal(stamps, original['stamps'])
        valid = np.isfinite(original['raw'][:, 1]) & np.isfinite(truth)
        if baseline_support is None:
            baseline_support = valid
        assert np.array_equal(valid, baseline_support)
        for i, mask in enumerate(masks):
            direct = c.state_metrics(original['raw'][mask, 1], truth[mask], r['threshold'])
            assert direct == r['blocks'][i]
        with np.load(run/f'source_pr/{g}_{s}.npz', allow_pickle=False) as pr:
            # Check distributed thresholds and the selected threshold, including score ties.
            indices = set(np.linspace(0, len(pr['threshold'])-1, 11).astype(int))
            indices.add(int(np.searchsorted(pr['threshold'], r['threshold'])))
            for idx in indices:
                for i, mask in enumerate(masks):
                    direct = c.state_metrics(original['raw'][mask, 1], truth[mask], float(pr['threshold'][idx]))
                    for short, full in [('tp', 'true_positive'), ('fp', 'false_positive'), ('fn', 'false_negative')]:
                        assert direct[full] == int(pr[f'block_{i}_{short}'][idx])
                    direct_points += 1
    ancestry = []
    for g in c.FAMILIES:
        for s in c.SEEDS:
            model_path = parent/f'models/{g}/seed_{s}'
            meta, model = c.load_model(model_path)
            record = {k: meta[k] for k in ('selected_epoch', 'checkpoint_metric', 'protocol_sha256',
                      'training_households', 'validation_households', 'source_train_windows',
                      'input_log_mean', 'input_log_std', 'power_scales_w', 'code_sha256')}
            record.update(candidate=g, seed=s, model_sha256=c.sha256(model_path/'model.json'),
                          weights_sha256=c.sha256(model_path/'weights.pt'))
            if g == 'W50':
                _, reference = c.load_model(parent/f'reference/models/seed_{s}')
                assert all(np.array_equal(v.numpy(), reference.state_dict()[k].numpy()) for k, v in model.state_dict().items())
                record['original_A_parameters_equal'] = True
            ancestry.append(record)
    result = c.read_json(run/'results.json')
    previous = c.read_json(parent/'metrics.json')['rows']
    for s in c.SEEDS:
        actual = next(r for r in result['rows'] if r['candidate'] == 'W50' and r['seed'] == s)
        old = next(r for r in previous if r['candidate'] == 'W50' and r['seed'] == s and
                   r['partition'] == c.DEV and r['appliance'] == 'microwave' and r['view'] == 'raw')
        for key in ('valid_samples', 'true_positive', 'false_positive', 'false_negative', 'precision', 'recall', 'f1', 'support_sha256'):
            assert actual[key] == old[key], (s, key)
    # Ordinary exported-recipe path, with genuine aggregate rows and no supplied labels.
    sample_dir = run/'recipe_readback'
    sample_dir.mkdir(exist_ok=True)
    input_path, output = sample_dir/'aggregate_2048.csv', sample_dir/'W50_42.csv'
    if not input_path.exists():
        with input_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data['aggregate'][0]))
            writer.writeheader()
            writer.writerows(data['aggregate'][:2048])
    if not output.exists():
        c.predict_recipe(run/'recipe.json', 'W50', 42, input_path, output)
    with output.open(encoding='utf-8-sig', newline='') as stream:
        sample = list(csv.DictReader(stream))
    with np.load(run/f'replays/W50_42_{c.SV}.npz', allow_pickle=False) as a:
        score = a['score'][:2033]
        state = a['state'][:2033]
    maximum_difference = 0.
    for i, row in enumerate(sample[:2033]):
        assert (row['raw_score_w'] == '') == (not np.isfinite(score[i]))
        if np.isfinite(score[i]):
            difference = abs(float(row['raw_score_w'])-score[i])
            maximum_difference = max(maximum_difference, difference)
            assert np.isclose(float(row['raw_score_w']), score[i], rtol=1e-6, atol=2e-4)
            assert int(row['state']) == state[i]
        else:
            assert row['state'] == ''
    validation = {'status': 'passed', 'fits': 0, 'source_PR_direct_confusion_checks': direct_points,
                  'source_twelve_model_time_masks_equal': True, 'source_block_selected_metrics_equal': True,
                  'original_A_W50_parameter_checks': 3, 'fixed_200W_development_baseline_exactly_reproduced': True,
                  'recipe_ordinary_inference_checked_rows': 2033, 'recipe_input_rows': 2048,
                  'recipe_truncated_right_boundary_rows_excluded': 15,
                  'recipe_raw_maximum_difference_w': maximum_difference,
                  'recipe_raw_tolerance': {'rtol': 1e-6, 'atol_w': 2e-4},
                  'recipe_state_and_missing_exactly_equal': True,
                  'source_selection_sha256': c.sha256(run/'source_selection.json'),
                  'registration_sha256': c.sha256(run/'registration.json'), 'freeze_sha256': c.sha256(run/'freeze.json'),
                  'evaluation_complete_sha256': c.sha256(run/'evaluation_complete.json'),
                  'verifier_sha256': c.sha256(Path(__file__)),
                  'verification_is_local_automated_not_independent_member': True,
                  'ancestry': ancestry,
                  'protocol_deviation': {'pre_freeze_development_cache_array_materialized_for_schema': True,
                      'observed_fields': 't/y shapes and dtypes only; no values or statistics examined',
                      'development_labels_used_for_source_selection': False,
                      'strict_no_pre_freeze_label_read_condition_met': False},
                  'overall_acceptance_passed': False}
    c.save(run/'delivery_validation.json', validation)
    print(json.dumps({k: v for k, v in validation.items() if k != 'ancestry'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
