"""Exact tied-score PR points and source-only recall-constrained selection."""
import numpy as np

from ..data.common import DataError


def pr_curve(scores, truth):
    scores, truth = np.asarray(scores), np.asarray(truth)
    valid = np.isfinite(scores) & np.isfinite(truth)
    s, y = scores[valid], truth[valid].astype(bool)
    if not len(s) or not y.any() or y.all():
        raise DataError("PR selection requires observed positive and negative source labels")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    n = ends + 1
    fp, fn = n - tp, int(y.sum()) - tp
    return {"threshold": s[ends], "precision": tp / n, "recall": tp / y.sum(),
            "f1": 2 * tp / (n + y.sum()), "tp": tp, "fp": fp, "fn": fn}


def constrained_point(curve, minimum_recall=.85):
    if not 0 < minimum_recall <= 1:
        raise DataError("Recall guard must be in (0, 1]")
    eligible = np.flatnonzero(curve["recall"] >= minimum_recall)
    if not len(eligible):
        raise DataError("No source threshold satisfies recall guard")
    index = max(eligible, key=lambda i: (curve["precision"][i], curve["f1"][i], curve["threshold"][i]))
    return {k: float(v[index]) if k not in ("tp", "fp", "fn") else int(v[index]) for k, v in curve.items()}


def checkpoint_rank(point, epoch):
    # Earliest epoch wins an exact tie; MAE is deliberately a separate outcome.
    return (-point["precision"], -point["f1"], epoch)
