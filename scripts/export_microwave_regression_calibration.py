"""Export original-A powers with the frozen independent microwave decisions."""
import argparse
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
    _, parent = c.context(run)
    selected = c.read_json(run/'source_selection.json')['selected']
    if selected is None:
        raise RuntimeError('No source preselection to export')
    records = []
    for seed in c.SEEDS:
        reference = parent/f'reference/predictions/seed_{seed}/{c.DEV}.npz'
        with np.load(reference, allow_pickle=False) as a:
            stamps, aggregate, raw = a['stamps'], a['aggregate'], a['raw']
        with np.load(run/f'replays/{selected}_{seed}_{c.DEV}.npz', allow_pickle=False) as a:
            if not np.array_equal(a['stamps'], stamps):
                raise RuntimeError('Combined export time identity mismatch')
            microwave_state = a['state']
        states = np.column_stack([c.decisions(raw[:, 0], 2000), microwave_state, c.decisions(raw[:, 2], 20)])
        target = run/f'combined_predictions/seed_{seed}.npz'
        payload = {'stamps': stamps, 'aggregate': aggregate, 'raw_power_w': raw, 'state': states}
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, **payload)
        with np.load(target, allow_pickle=False) as a:
            if any(not np.array_equal(a[k], v, equal_nan=True) for k, v in payload.items()):
                raise RuntimeError('Combined export changed')
        records.append({'seed': seed, 'path': target.relative_to(run).as_posix(), 'sha256': c.sha256(target),
                        'original_A_reference_path': str(reference), 'original_A_reference_sha256': c.sha256(reference),
                        'all_three_raw_power_columns_exactly_original_A': True,
                        'other_device_states_exactly_original_A': True, 'microwave_state_exactly_frozen_calibration': True})
    c.save(run/'combined_export.json', {'selected': selected, 'fits': 0, 'columns': ['kettle', 'microwave', 'washing_machine'],
           'state_missing_value': -1, 'default_ui_changed': False, 'effect_acceptance_passed': False, 'cases': records,
           'source_selection_sha256': c.sha256(run/'source_selection.json'), 'freeze_sha256': c.sha256(run/'freeze.json'),
           'exporter_sha256': c.sha256(Path(__file__))})
    files = {k: v for k, v in c.inventory(run).items() if k != 'final_delivery_manifest.json'}
    c.save(run/'final_delivery_manifest.json', {'artifacts': files,
           'note': 'Adds original-A power combination export to the immutable delivery manifest', 'fits': 0})
    print('Exported and checked three original-A power / calibrated-state combinations; no model fitting.')


if __name__ == '__main__':
    main()
