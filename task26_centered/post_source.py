"""Pure Task26 post-source building blocks; no filesystem or execution entry point.

A future entry point must authenticate the receipt, registration, source artifacts,
model weights and all six baseline identities before using these functions.
"""
import numpy as np
import torch

from virtual_submeter.accuracy import microwave_event_gate as gate
from virtual_submeter.models.center_mask_tcn import context_at_complete_centers
from task26_centered.model import centered_input
from task26_centered.source_gate import SEEDS, select_source


def source_precondition(checkpoints, blocks, full_scores, control_rows, sealed, registration_id,
                        *, receipt_verified, registration_verified, source_artifacts_verified,
                        development_in_scope):
    """Recompute a sealed decision; caller authenticates all four prerequisites."""
    if not all((receipt_verified, registration_verified, source_artifacts_verified, development_in_scope)):
        return False
    replay = select_source(checkpoints, blocks, full_scores, control_rows, registration_id)
    return replay == sealed and replay['selected'] == 'task26' and replay['source_selected_candidate']


def ordinary(meta, model, t, aggregate, low, high):
    """Compute Task26 scores on strict core31, preserving the full time axis."""
    if (meta['kind'] != 'task26_centered_tcn_v1' or meta['window_samples'] != 127
            or meta['inference_support'] != 'strict_complete_31'):
        raise ValueError('Task26 model metadata mismatch')
    t, aggregate = np.asarray(t), np.asarray(aggregate)
    x, centers = context_at_complete_centers(meta, t, aggregate)
    z = centered_input(meta, x)
    model.eval()
    with torch.inference_mode():
        values = np.concatenate([torch.sigmoid(model(batch)).cpu().numpy()
                                 for batch in z.split(meta['batch_size'])]) if len(z) else np.empty(0, np.float32)
    scores = np.full(len(t), np.nan, np.float32)
    scores[centers] = values
    return {'t': t, 'scores': scores,
            'state': gate.gate_state(scores, gate.local_max(t, scores), low, high)}


def development_summary(rows, supports):
    """The original eight acceptance gates; input already frozen and checked."""
    groups = {arm: [next(r for r in rows if r['arm'] == arm and r['seed'] == seed)
                    for seed in SEEDS] for arm in ('A', 'W50', 'task26')}
    base, new = groups['W50'], groups['task26']
    means = {arm: gate.mean_rows(v) | {'fp': float(np.mean([r['fp'] for r in v]))}
             for arm, v in groups.items()}
    checks = {'mean_P_at_least_008': means['task26']['precision'] >= .08,
        'each_R_at_least_085': all(r['recall'] >= .85 for r in new),
        'mean_F1_gain_at_least_002': means['task26']['f1'] - means['W50']['f1'] + 1e-12 >= .02,
        'at_least_two_seed_P_F1_up': sum(b['precision'] > a['precision'] and b['f1'] > a['f1'] for a,b in zip(base,new)) >= 2,
        'each_F1_drop_at_most_002': all(b['f1'] - a['f1'] >= -.02 for a,b in zip(base,new)),
        'mean_FP_reduction_at_least_30_percent': means['task26']['fp'] <= .7 * means['W50']['fp'],
        'no_new_all_on_or_off': all(not r['degenerate'] for r in new),
        'support_137523_exact': len(supports) == 3 and all(v['n'] == 137523 for v in supports)}
    return {'selected_before_development': 'task26', 'acceptance_gates': checks,
            'development_acceptance_passed': all(checks.values()), 'formal_means': means}
