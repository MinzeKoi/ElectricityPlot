"""Source-only Task26 checkpoint and frozen relative-gate decision.

The caller supplies source arrays and registration binding; this module performs no I/O.
Full-grid scores must come from the checkpoint selected by select_checkpoints.
"""
import numpy as np

from virtual_submeter.accuracy import microwave_event_gate as gate
from virtual_submeter.accuracy import microwave_recall_guard95 as guard
from virtual_submeter.accuracy import microwave_source5_loss as loss
from virtual_submeter.accuracy import microwave_source5_temporal as temporal

SEEDS = (42, 1019, 2026)
BLOCKS = ('block_4', 'block_5')
PLAN = {'minimum_block_positive_windows': 5, 'minimum_block_events': 2,
        'minimum_validation_recall': .85}


def select_checkpoints(checkpoints, blocks):
    if set(checkpoints) != set(SEEDS) or set(blocks) != set(BLOCKS):
        raise ValueError('Three seeds and exactly blocks 4 and 5 are required')
    return {seed: temporal.reselect_checkpoints(checkpoints[seed], blocks, PLAN) for seed in SEEDS}


def decide(control_rows, rows, checkpoints):
    """Apply original relative protection and then mean precision/F1 ranking."""
    eligible = (len(rows) == 3 and all(checkpoints[s] is not None for s in SEEDS)
                and loss.qualified(control_rows, rows))
    selected = 'control_guard95'
    if eligible and tuple(gate.mean_rows(rows)[k] for k in ('precision', 'f1')) > tuple(
            gate.mean_rows(control_rows)[k] for k in ('precision', 'f1')):
        selected = 'task26'
    return {'eligible': bool(eligible), 'selected': selected,
            'source_selected_candidate': selected == 'task26'}


def select_source(checkpoints, blocks, full_scores, control_rows, registration_id):
    """Return a source-only, registration-bound decision without writing files.

    blocks: block_4/5 -> t, y, event; checkpoints: seed -> 12 mappings of
    block names to center probabilities. full_scores: seed -> block -> t,p,
    the full time grid from the selected checkpoint, including missing NaNs.
    control_rows are the frozen source-only control_guard95 rows in seed order.
    """
    if not isinstance(registration_id, str) or not registration_id:
        raise ValueError('Registration binding is required')
    if [r['seed'] for r in control_rows] != list(SEEDS) or set(full_scores) != set(SEEDS):
        raise ValueError('Complete paired seeds required')
    if any(not np.isfinite([r[k] for k in ('precision', 'recall', 'f1')]).all()
           or any(not 0 <= r[k] <= 1 for k in ('precision', 'recall', 'f1')) for r in control_rows):
        raise ValueError('Invalid control metrics')
    chosen = select_checkpoints(checkpoints, blocks)
    recipes, rows = [], []
    for seed in SEEDS:
        checkpoint = chosen[seed]
        if checkpoint is None:
            recipes.append({'seed': seed, 'checkpoint': None, 'L': None, 'H': None,
                            'feasible': False, 'blocks': []})
            continue
        if set(full_scores[seed]) != set(BLOCKS):
            raise ValueError('Full score blocks incomplete')
        scores = {}
        for key in BLOCKS:
            t, p = np.asarray(full_scores[seed][key]['t']), np.asarray(full_scores[seed][key]['p'])
            if t.ndim != 1 or p.ndim != 1 or len(t) != len(p) or np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
                raise ValueError('Invalid full score grid')
            _, start, end = next(v for v in temporal.VALIDATION if key == f'block_{v[0]}')
            if not len(t) or np.any(t < start) or np.any(t >= end) or np.any(np.diff(t) <= 0):
                raise ValueError('Full score grid must stay inside its fixed block in increasing order')
            ix = gate.align(t, blocks[key]['t'])
            np.testing.assert_allclose(p[ix], checkpoints[seed][checkpoint['epoch'] - 1][key], rtol=1e-5, atol=1e-7)
            scores[key] = {'t': t, 'p': p, 'h': gate.local_max(t, p), 'ix': ix,
                           'y': np.asarray(blocks[key]['y']) >= 200}
        low, _ = gate.select_threshold({k: {'values': d['p'][d['ix']], 'y': d['y']} for k, d in scores.items()},
            np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in scores.values()])), .95)
        high = guard.calibrate95(scores, low) if low is not None else None
        metrics = [dict(block=k, **gate.confusion(gate.gate_state(d['p'], d['h'], low, high)[d['ix']], d['y']))
                   for k, d in scores.items()] if high is not None else []
        recipes.append({'seed': seed, 'checkpoint': checkpoint, 'L': low, 'H': high,
                        'feasible': high is not None, 'blocks': metrics})
        if metrics:
            rows.append({'seed': seed, **gate.mean_rows(metrics),
                         'degenerate': any(v['degenerate'] for v in metrics),
                         'source_gate_passed': all(v['recall'] >= .95 for v in metrics)})
    return {'schema': 'task26_source_decision_v1', 'source_only': True,
            'registration_id': registration_id, **decide(control_rows, rows, chosen),
            'recipes': recipes, 'rows': rows, 'control_rows': control_rows}



